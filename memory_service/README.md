# memory_service

The assistant's memory service (core + archival). The package runs in ROS 2 but
is **independent of the other packages of the architecture**: it does not import
`shared_utils` or `db_adapters`, and it does not contact an LLM or ChromaDB until
it really needs to. This makes it possible to test it seriously, without API keys
and without a network.

## Structure

| Module | Role | External dependencies |
| --- | --- | --- |
| `config.py` | configuration (limits, Chroma path, `LLM_CONFIG`) | only `python-dotenv` |
| `backends.py` | **lazy** construction of chat model and vector store, replaceable | imports the provider (OpenAI, Mistral) and Chroma only if used |
| `consolidation.py` | memory data model and consolidation operations | `pydantic`, `langchain-core` |
| `eviction.py` | removal from the archive: tombstones and, over the limit, active memories by score | none besides the vector store |
| `memory_manager_llm.py` | LangGraph graph and `MemoryAgent` | `langgraph`, `langchain-core` |
| `memory_server.py` | ROS node with the two services | `rclpy`, `memory_service_interfaces` |
| `memory_client.py` | example ROS client | `rclpy`, `memory_service_interfaces` |

## ROS interface

The two services (`memory_service_interfaces`) expose the consolidated memory:

**`GetMemory`** — request: `user_input`, the message the memory has to answer.
Passing it is what lets the retrieval branch search the archive based on what the
user is asking **right now**; left empty, the service falls back on the last user
message already in memory, which is the question of the previous round.

| field (response) | content |
| --- | --- |
| `memory_list` | content of the **active** core memories (superseded and deleted stay out) |
| `memory_ids` | id of each one, same order as `memory_list`: the two lists zip together |
| `last_messages` | messages still in the conversation window |
| `retrieved_memories` | archive memories retrieved for **this** request, one per line in the form `ID: <id>, Content: <text>`. Empty when the core memory was enough or when nothing was found |
| `operation_log` | consolidation operations of **this** call, one JSON per line |

**`UpdateMemory`** — request `user_input`, `queries`, `results`, `explanation`; the
response has `memory_list`, `memory_ids` and `operation_log` with the same semantics.

Each `operation_log` entry is the JSON of an `OperationLogEntry`:

```json
{"op_type": "update", "item_id": "c8f90d30-...", "related_item_id": "7ca3ad4f-...",
 "content": "The user eats fish", "score": null, "score_terms": null,
 "ratings": {"utility": 4}, "timestamp": "2026-08-24T18:23:39.059368"}
```

`op_type` is one of `create`, `redundant`, `update`, `delete`, `archive`, `evict`,
`prune`. `score` and `score_terms` are set only in `prune` entries: the score with
which the memory left the archive and the terms that make it up, by name (for
example `{"utility": 0.8, "time_decay": 0.126}`). `ratings` is set only in `create`
and `update` entries with `consolidation_utility` on: the classifier's judgment of
the memory, that is `utility` or `need` and `gain`, and `safety` only when it is
true (then the ratings are not there).

The log reports **only the operations of the current call**, not the whole
history: `state["operation_log"]` grows for the whole life of the node and
`get_memory` is invoked often, so sending the whole history at every call would
cost more and more. The full history stays available in-process via
`MemoryAgent.state["operation_log"]`.

`memory_list` and `last_messages` keep their previous name and semantics, so the
packages that read them keep working: only a `colcon build` is needed to
regenerate the interfaces.

`intent_recognition.listener_callback` passes `user_input`, so the archive
retrieval starts from the current question:

```python
self.get_response = self.memory_client.send_get_request(user_input)
```

The other calls - explainability, inner_speech, intent_recognition at startup -
leave it empty, and the fallback applies.

## Tests

```bash
python memory_service/run_tests.py -v
```

The offline tests all run on the **real LangGraph graph** with simulated backends
(`test/fakes.py`): no network calls, no model, no ChromaDB.

`test/test_consolidation.py` is the turn-based scenario: each turn exercises a
different classification (`new`, `redundant`, `update`, `delete`, archiving,
retrieval) and prints the full state of the memory - core, messages, archive with
the metadata, retrieval, operation log. `test_memory_agent.py` holds instead the
unit checks (limits, malformed tool calls, retrieve cache, tombstones never
retrieved), `test_eviction.py` the removal from the archive and `test_config.py`
the configuration. `test_fama.py`, `test_operation_matrix.py` and
`test_calibrate_utility.py` check the long test's metrics and the calibration
script, without any model.

The node tests (`test/test_memory_server.py`) build `MemoryServer` with a stub
agent and invoke the callbacks with real requests/responses: they need rclpy and
the generated interfaces, so they must run inside the ROS workspace (outside it
they skip themselves).

With pytest installed the whole package can be collected (`conftest.py` fixes the
`sys.path`):

```bash
pytest memory_service
colcon test --packages-select memory_service
```

`test/test_tool_calling_gate.py` is the **gate to pass before everything
else**: it checks that the configured model produces valid tool calls for the
real schemas (`InsertCoreMemories`, `SplitCoreAndArchivalMemory`,
`retrieve_memory` / `NoSearchNeeded`) and that it copies the ids exactly. They are
two different failures and it measures them separately, because they lead to
different remedies: the wrong structure is cured with constrained structured
output, the wrong ids with prompt aliasing. Every check is repeated several times,
because with an LLM a single right answer says nothing.

```bash
pytest memory_service/test/test_tool_calling_gate.py -v -s

# more reliable reading before deciding
MEMORY_GATE_ATTEMPTS=10 pytest memory_service/test/test_tool_calling_gate.py -v -s
```

`test/test_memory_llm.py` is instead the **integration** test with the real
stack (chat model, embedding, ChromaDB): it checks that a deleted fact never
comes out of a real Chroma archive again, which only holds if the `status` filter
works inside the real store — the offline tests run on a double. It is skipped
if `LLM_CONFIG` is not set, and also if the models do not answer — with a message
saying which of the two.
`LANGSMITH_API_KEY` stays optional, as in the rest of the architecture: if present
it only enables tracing.

```bash
pytest memory_service/test/test_memory_llm.py -v -s
```

`test/test_long_term_interaction.py` is instead the simulation of a **long
session**: 232 user messages injected one at a time — 176 facts and 56
questions — mixed as they would really happen. The first 34 are the
presentation; from there on new facts, repetitions, refinements, contradictions,
deletion requests and chatter alternate without blocks, respecting only
causality (a fact is introduced before it is modified).

Every message goes through the real pipeline, whatever it says: a `get_memory`
(the `retrieve` branch) and then an `update_memory` with the answer (the `insert`
branch). The service has no notion of a question. **Questions** exist only in the
evaluation: they are the 56 messages listed in `test/fama.py` (`fama.QUESTIONS`),
not the ones ending with `?` — the last one, the final summary, has no question
mark. Many are placed right after an update, a contradiction or a deletion — "sai
dirmi dove abito?" comes after "dimentica il mio indirizzo" —, others recall facts
written much earlier, and ten are assistant tasks that use the facts without
naming them ("mi consigli cosa cucinare per cena?"). The ones about facts that
ended up in the archive force the retrieve to really query it.

For every question the report shows the answer, what came from the archive and
the **FAMA** score (Memora, arXiv:2604.20006), computed twice with the same
criteria: on the answer, and on the active memories right after the message
before the question is consolidated. The gap between the two says whether
consolidation or the answer gave way. Every message is also a row of the
confusion matrix of the operations (`test/operation_matrix.py`), which sums the
saved runs of the same commit.

During the run it prints only one progress line every 10 messages; at the end it
prints the report: full operation log, core memory, archive, questions and
answers, the FAMA table, the confusion matrix and the cost per node. The asserts
are structural (every item that left the core memory is in the archive or has an
`evict` or `prune` entry in the log, core memory with only active items, unique
ids, no question without an answer); FAMA is measured and printed, never asserted.

Every message costs 3-4 real calls to the model, so the full run takes a while:
start with a low `MEMORY_LONGRUN_MESSAGES`. The archive is in the service's folder
(`MEMORY_CHROMA_PATH`) but in a collection of its own
(`MEMORY_LONGRUN_COLLECTION`), so the production collection is not touched. The runs
add up, though: clean it between runs with `reset_archive.py` — which deletes the
whole folder, production collection included — or point
`MEMORY_LONGRUN_CHROMA_PATH` to another folder.

```bash
pytest memory_service/test/test_long_term_interaction.py -v -s

# reduced version, for a quick try
MEMORY_LONGRUN_MESSAGES=20 pytest memory_service/test/test_long_term_interaction.py -v -s

# with a pause between messages, if the endpoint is shared with other loads
MEMORY_LONGRUN_DELAY=1 pytest memory_service/test/test_long_term_interaction.py -v -s
```

**Calibration of the utility tables.** With `MEMORY_LONGRUN_CALIBRATION` the long
test appends a trace of the run to that file: for every FAMA question the memories
active at its snapshot, with the ratings. In that run eviction is off, because a
memory removed before a question could never turn out served.
`consolidation_utility` in `config.py` decides which table the run calibrates:
`need_gain` or `single`.

`test/calibrate_utility.py` reads the traces of the last commit and, for every
(memory, question) pair, removes the memory and recomputes FAMA-memory: the memory
served if the score drops, and how much it drops is its gain. From there need and
gain per level, pulled toward the mean of all the levels when the data are few and
made monotone, and the two tables already normalized, to paste into `eviction.py`.
For `need_gain` it also prints the free table, cell by cell, and with at least two
runs it compares the two on each run left out. No model call: FAMA-memory is
deterministic.

```bash
MEMORY_LONGRUN_CALIBRATION=memory_service/test/longrun_calibration.jsonl pytest memory_service/test/test_long_term_interaction.py -v -s
python memory_service/test/calibrate_utility.py
```

## The models on the cluster

The models run on a Kubernetes cluster behind an **OpenAI compatible** endpoint,
so `model_provider` is `openai` and `MEMORY_LLM_BASE_URL` points to the service's
address. No new dependency is needed: `langchain_openai` and `openai` were already
in `requirements.txt`.

A few things worth knowing, checked against the sources:

- **Without a key the client is not even built.** `openai-python` raises
  `OpenAIError` in the constructor, not at the first call, if it finds no key —
  even towards a server that does not check it. That is why `backends.py` passes
  the placeholder `PLACEHOLDER_API_KEY = "EMPTY"` when `MEMORY_API_KEY_ENV` is not
  configured. If one day the endpoint asked for a real token, it is enough to
  point `MEMORY_API_KEY_ENV` to the variable that holds it.
- **The embeddings must send the text, not the tokens.** `OpenAIEmbeddings` by
  default tokenizes before sending, and for a model `tiktoken` does not know it
  falls back on `cl100k_base` — OpenAI's vocabulary. The ids of one vocabulary are
  different words in another: the server would embed noise and return it
  **without any error**. That is why `backends.py` passes
  `check_embedding_ctx_length=False`.
- **No greedy decoding.** Qwen advises against `temperature=0`: *"can lead to
  performance degradation and endless repetitions"*. `.config` carries the preset
  recommended for thinking (`1.0 / 0.95 / 20`); `top_k` is not a parameter of the
  OpenAI API and travels in `extra_body`, which vLLM honours. All the values are
  tuned from `.config`, without touching the code, and whatever you omit stays at
  the server's default.
- **Thinking is switched per node.** The `enable_thinking` entry of the
  `memory_agent` entry in `.config` is the base for every node, together with the
  sampling parameters right below it — which must be changed **together** with
  it, because the two presets are different (on `1.0 / 0.95 / 20`, off
  `0.7 / 0.8 / 20`). `NODE_SAMPLING` in `config.py` overrides it node by node
  (`retrieval`, `generate_answer`, `consolidation`, `core_split`): whatever an
  override does not mention is inherited, and `None` leaves a parameter to the
  server. Qwen switches reasoning through the chat template, not with a sampling
  parameter, so `backends.py` sends it as `chat_template_kwargs` inside
  `extra_body`. The gate prints which mode it ran
  in, so two runs can be compared without relying on memory.
- **The tool calls that are always expected are forced.** With a temperature
  deliberately different from zero the model now and then answers in prose
  instead of emitting the call — measured by the gate: 9 times out of 10, not 10.
  Where a tool call is the only sensible outcome that tenth time is **silent data
  loss**: the node saves nothing and the messages are cut anyway. That is why
  `InsertCoreMemories` and `SplitCoreAndArchivalMemory` are bound with
  `tool_choice="required"`. No flexibility is lost: the model can always return an
  empty list. On the retrieval branch `retrieve_memory` and `NoSearchNeeded` are
  bound together, also with `required`: not searching stays possible, but it is
  an answer that can be seen and not a silence.
- **`api_key_env` must not be `GROQ_API_KEY`.** That key — which is in the
  `.env`, because the other five nodes of the architecture need it — would end up
  in the `Authorization` header towards the cluster: leave it empty, or point it
  to a variable meant for the cluster. With a hosted provider like Groq the
  variable is not needed anyway: the
  provider does the lookup itself, which is why `backends.py` never passes
  `api_key=None` explicitly (it would suppress it).

## The core memory limit

The limit in characters is **asked of the model, not imposed by the code**: there
is no deterministic eviction behind it. When the core memory overflows, the
`summarize_core_memories` node passes the split everything needed to respect it:

- every memory with its own length (`id: content (N characters)`)
- the current total, the limit, and **how many characters must be freed**
- the explicit statement that the constraint is hard and that keeping everything
  in core is not a valid answer
- the fact that archiving is not deleting: the archive stays searchable, so when
  in doubt it is better to archive

If the model still decides not to archive enough, **its decision is respected**
and the core memory stays over the limit. It does not go unnoticed though: a
warning appears in the logs.

```
WARNING: core memory still over the limit after the split (500/100 characters)
```

## Who is a source of facts

The consolidation prompt receives the two voices in **separate fields**: what the
user said, from which facts are extracted, and what the assistant answered, only
as context — it serves to give meaning to a "yes, that one" that means nothing on
its own.

It is not a subtlety. The assistant builds its answers from the memory, so it
repeats facts already known: putting the two voices in a single block,
consolidation extracts them again and classifies them `redundant`. In a long run a
quarter of the operations ended up being reprocessing of the system's own output,
and now and then a paraphrase drifted enough to become a near-duplicate memory.

For the same reason **the archive search uses only the user's words**: searching
with the assistant's answer would bring up exactly the memories it was quoting,
offering them as candidates.

## The archive candidates

When a fact is classified, the model compares it with the active core memories
**plus the first K of the archive** (`ARCHIVE_CANDIDATES_K = 10`), merged into a
single `id: content` list by `build_candidate_memories`. From both sources **only
the active ones** go through: superseded and deleted stay out.

The filter **does not happen after the search, but inside the index**:
`search_archive` passes `filter={"status": "active"}` to Chroma, which searches
only among the active ones.

It is not a detail. Filtering after a top-K search would yield fewer than K
candidates, because the tombstones take the first places — and the gap **gets
worse with use**, since with eviction off the tombstones are never removed:
exactly on a heavily used archive the classifier would find itself without
targets, with nothing signalling it. Asking for K active ones, K are received,
always, as long as K active ones exist.

Every document in the archive has a `status`, written by `archive_metadata`: the
case "document without status" does not exist, and the code does not consider it.

One thing that stays as it is: the query used to search the archive is **the block
of messages being consolidated**, not the single extracted fact. The K candidates
are therefore chosen once per batch. That is why an `update` towards an archived
memory may not find its target when the batch also talks about something else.

## Replaceable backends

Chat model and vector store are created at the first request and can be injected
from outside:

```python
from memory_service import backends
from memory_service.memory_manager_llm import MemoryAgent

backends.configure(llm=my_chat_model, vector_store=my_store)
agent = MemoryAgent()
```

The same mechanism is used by the tests and makes it possible to try a different
provider without touching the graph's code. The node also accepts an explicit
agent (`MemoryServer(agent=...)`), otherwise it uses the `MemoryAgent` singleton.

## Configuration

The values are read from environment variables (from `.env` / `.config`):

| Variable | Meaning |
| --- | --- |
| `LLM_CONFIG` | dictionary of the models; the `memory_agent` entry is used |
| `EMBEDDING_CONFIG` | dictionary of the embedding models; the `memory_agent` entry is used |
| `MEMORY_LLM_BASE_URL` | endpoint of the chat model (for an OpenAI compatible server it usually ends in `/v1`) |
| `MEMORY_EMBEDDING_BASE_URL` | endpoint of the embedding model |
| `MEMORY_API_KEY_ENV` | name of the variable that holds the key, or empty. Empty means "no key": towards an `openai` endpoint the placeholder `EMPTY` is sent. It is **not** `GROQ_API_KEY`: that one is for the other nodes and must not reach the cluster |
| `MEMORY_LLM_NODE` | entry of `LLM_CONFIG` to use |
| `MEMORY_MAX_HISTORICAL_MESSAGES` | `2` (default): messages kept before the summary. **Keep it even**: with an odd number the tail of the window is an assistant answer, and every consolidation receives the end of one exchange plus the start of the next instead of a whole (user, assistant) pair |
| `MEMORY_CORE_MEMORY_LIMIT` | maximum characters of the core memory |
| `MEMORY_ARCHIVE_LIMIT` | **active** memories in the archive beyond which eviction steps in. It does not block insertion: the graph's `exceed_archive_limit` router checks it at the end of every `insert` and, if exceeded, in the `prune_archive` node the 10% of the active memories with the lowest score leaves, all together (`eviction.prune_archive`, `prune` entry in the log). Memories with `safety` true are never candidates and get no score, but the share stays 10% of all the active ones. The score combines utility (see `consolidation_utility` below), the decay over time and the frequency. The decay is a Weibull or a power law according to `eviction_time_decay_function` (`weibull` or `power`), both with a one-hour scale, computed on `updated_at` or `retrieved_at` according to `eviction_time_decay_field`. The frequency is `n_used` when uses are counted (`track_used`, see below) and `n_retrieve` otherwise, divided by the highest among the candidates. `eviction_combination` chooses how: `weighted_sum` is 0.5 utility + 0.25 decay + 0.25 frequency, `product` is utility × (0.5 decay + 0.5 frequency); without utility both are the mean of decay and frequency. It runs only with `MemoryConfig.eviction` on (the default): the switches are changed only in the code (`config.py`), with no environment variable |
| `MEMORY_GENERATE_ANSWER` | `true` (default) or `false`: whether the `retrieve` branch also **composes an answer** for the user. No field of the service returns it — the archive reaches the caller through `retrieved_memories` — so it only ends up in `last_messages`, as a turn of the conversation. `false` saves one call per `retrieve`, leaves one (user, assistant) pair per exchange instead of three messages, and keeps the memory from quoting itself inside consolidation. `track_used` needs it on, because the list of used memories comes from the answer |
| `MEMORY_CHROMA_PATH` | archive folder (resolved to an absolute path) |
| `MEMORY_COLLECTION_NAME` | ChromaDB collection |
| `MEMORY_ENV_FILE` | explicit path of the environment file |

`retrieval_mode` has no environment variable and is changed only in `config.py`.
It decides whether the archive search of the `retrieve` branch is the model's
choice:

- `decide` (default): the model chooses between `retrieve_memory` and `NoSearchNeeded`;
- `always_llm_query`: it always searches, the model writes only the query and `k`.

In both a retrieval increments `n_retrieve` of the returned memories. The long test
prints and saves the run's mode.

`track_used` (on or off, also only in `config.py`) makes the answer say which
memories it used. `generate_answer` switches to a second prompt, the same as the
first plus the request for the list, and answers with the `AnswerWithUsedMemories`
tool call: first `answer`, which reaches the user as always, then
`used_memory_ids`. The core memories appear in the prompt with their id, in the
same format as the archive ones. Every memory in the list, core or archive, gets
`n_used` + 1 and `retrieved_at` now; retrieval keeps counting `n_retrieve` but no
longer touches `retrieved_at`, which becomes the last use. The eviction's frequency
term moves from `n_retrieve` to `n_used`. With `generate_answer` off there is no
list to read and the switch has no effect.

`consolidation_utility` (only in `config.py`) makes the classifier judge, in the
same call, every `new` fact and every `update`:

- `none`: no judgment, the usual prompt and tool;
- `single`: `safety` and a `utility` rating from 1 to 5;
- `need_gain`: `safety` and two ratings from 1 to 3, `need` (how likely the fact is
  to be needed) and `gain` (how much it improves the answer when it is); `utility`
  stays empty.

The classification part of the prompt stays identical in the three modes: the two
new ones append a step for `safety` and one for the ratings, and the tool's schema
changes while keeping the name `InsertCoreMemories`. `safety` is true only if
forgetting the fact can put the user at risk (an allergy, a medical condition): in
that case the ratings are not given and, if the model gives them anyway, they are
discarded. An `update` is judged again, a `redundant` keeps the ratings it had, a
`delete` has none. An off-scale rating counts as not given. The four fields live in
`CoreMemoryItem` and in the archive metadata, where a missing value is absent
rather than `None`.

In eviction the judgment enters the score as the utility term, high meaning keep
as for the other terms: with `single` the rating's entry in `UTILITY_TABLE`, with
`need_gain` the (need, gain) cell of `NEED_GAIN_TABLE`, both in `eviction.py`.
Until they are calibrated they are rating / 5 and (need / 3) · (gain / 3); the
calibrated values are printed by `test/calibrate_utility.py` (see the Tests
section). A memory without a rating, because the model did not give it or gave it
off the scale, takes the middle one (`NEUTRAL_RATINGS`: 3, or need 2 and gain 2),
so the prune does not stop. In all modes a memory with `safety` true never leaves
the archive because of the limit.

`.env` and `.config` are searched for walking up the directories starting from the
working directory and from the package's location. If the node is launched from an
install space that does not contain those files, it is better to point to them
explicitly with `MEMORY_ENV_FILE`.

Watch out for `MEMORY_CHROMA_PATH`: a relative path is resolved against the
working directory, so launching the node from different folders means using
different archives. Set an absolute path to have a stable memory.

## Running

```bash
ros2 run memory_service memory_server
ros2 run memory_service memory_client
```
