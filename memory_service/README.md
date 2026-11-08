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
| `retrieved_memories` | archive memories retrieved for **this** request, one per line in the form `ID: <id>, Content: <text>` (`ID: <id>, Date: <day>, Content: <text>` with `answer_with_dates`). Empty when the core memory was enough or when nothing was found |
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
the configuration. `test_fama.py` and `test_operation_matrix.py`
check the long test's metrics, `test_calibrate_utility.py` the calibration of the
utility tables, `test_halumem_adapter.py` the benchmark adapter and
`test_eviction_replay.py` the eviction replay, without any model.

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
real schemas (`InsertCoreMemories`, `retrieve_memory` / `NoSearchNeeded`) and that it
copies the ids exactly. They are
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
session**: 264 user messages injected one at a time — 200 facts and 64
questions — mixed as they would really happen. The first 34 are the
presentation; from there on new facts, repetitions, refinements, contradictions,
deletion requests and chatter alternate without blocks, respecting only
causality (a fact is introduced before it is modified). The last 20 add events:
things that happened once, with a detail a later question asks for (a concert, a
repair, a loan, a dinner, a lottery win, an exhibition), so that a fact that
happened once can also turn out useful.

Every message goes through the real pipeline, whatever it says: a `get_memory`
(the `retrieve` branch) and then an `update_memory` with the answer (the `insert`
branch). The service has no notion of a question. **Questions** exist only in the
evaluation: they are the 64 messages listed in `test/fama.py` (`fama.QUESTIONS`),
not the ones ending with `?` — message 244, the summary, has no question mark.
Many are placed right after an update, a contradiction or a deletion — "sai
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
prints the report: full operation log, with how many new and updated facts lack the
ratings of the mode and how many times each level of each rating was given, core memory, archive, questions and answers, the FAMA table, the
confusion matrix and the cost per node. The asserts
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

**The HaluMem benchmark.** `test/halumem_adapter.py` replays HaluMem-Medium through
the real agent and hands the result to HaluMem's `evaluation.py`. It needs a clone of
[HaluMem](https://github.com/MemTensor/HaluMem) outside the ROS workspace, with
`HaluMem-Medium.jsonl` in its `data/` folder. Do not install HaluMem's own dependencies:
`evaluation.py` only uses packages already in `requirements.txt`.

The first five users of the dataset (`CALIBRATION_USERS`) are kept for the calibration
of the utility tables: `run` replays them only with `--calibration-users`, which replays
nothing else, in a version of its own (`run.json` records which set it holds, and a
version never mixes the two). So no user is both calibrated on and evaluated, and
`--users N` takes the first N of the set, for the evaluated ones the first N after the
calibration users. `run` replays one user per
process, each with an archive of its own, exchange by exchange with the dataset's clock
(`backends.set_clock`). It fixes what the experiments
fix whatever `config.py` says (`always` with the question as the query, the window of 2, the power law on `updated_at`,
`count_candidate_search` off, `retrieve_k` 20 as HaluMem gives the other systems,
`answer_with_dates`) and takes from the command line what they vary. It stops at
once if the models do not answer. A user that finished is skipped when the same
`--version` is launched again, one that did not starts over. To launch a version again
from scratch, `clean --all` deletes it.

`evaluate` runs the judge, read from `eval/.env` in the clone: `OPENAI_BASE_URL`,
`OPENAI_API_KEY`, `OPENAI_MODEL`, `RETRY_TIMES`, `WAIT_TIME_LOWER`, `WAIT_TIME_UPPER`.
A judge call that keeps failing, or answers with a label HaluMem does not know, stays
empty in the user's cached result and is never asked again: the (all) scores and the F1
count it as wrong. At the end `evaluate` prints these judgments user by user; deleting
the user's `tmp2/<uuid>.json` and launching `evaluate` again judges that user again.

```bash
# a pilot on one user
python3 memory_service/test/halumem_adapter.py run --halumem PATH --version pilot --users 1
python3 memory_service/test/halumem_adapter.py evaluate --halumem PATH --version pilot

# the archives of a run, which evaluate does not need; --all deletes the whole run
python3 memory_service/test/halumem_adapter.py clean --halumem PATH --version pilot

# what the experiments vary
python3 memory_service/test/halumem_adapter.py run --halumem PATH --version NAME --workers 4     --utility need_gain --combination product --frequency rate --core-limit N --archive-limit N     --prune-fraction F --model MODEL
```

Everything lands in `PATH/eval/results/memory_service-NAME/`: `run.json` with the
configuration and the commit, `memory_service_eval_results.jsonl` for the evaluation,
`memory_service_eval_stat_result.json` with its scores, `logs/` with the agent's output
for each user and `archives/` with the Chroma folders. Next to what HaluMem reads, every
session also keeps its operations, each with the exchange and the insert it belongs to (the
flush is an insert of its own), and how many memories were active at its end. Every user keeps in `unrated` how many new and updated
facts came without the ratings of the mode, and in `ratings` how many times each level of
each rating was given, counted as in the long test: the run prints both when the user
finishes, and summed over the finished users at the end.

What the memory costs the robot is kept too. Every session has `archive_bytes`, the size
on disk of the user's Chroma folder when it ends: Chroma writes its index in batches and
does not always give back the space of a removed memory, so the size grows in steps and
may not shrink after a prune. Every question has `vector_search_ms`, the archive search
alone with the embedding of the question, next to `search_duration_ms`, the whole
retrieve branch with the answer. In `node_stats`, next to the model calls, `archive_search`
holds the calls and seconds of every archive search of the user, those of consolidation
included.

**Calibration of the utility tables.** `test/calibration_ratings.py` has the model judge
the memory points of the calibration users as consolidation judges a new fact: the same
descriptions of safety and of the ratings of `--utility` (`single` or `need_gain`), the
settings of the consolidation node, one call per session, the points it skipped asked
again. The interference points stay out, since a memory should never hold them, and the
person is "the user" as in the adapter. Every session leaves a line with the mode, the
model and the commit in `test/calibration_work/halumem_ratings.jsonl` (`--ratings` to
change it, ignored by git); launched again, the sessions already judged with the same
mode, model and commit are skipped. `--workers N` judges N sessions at the same time,
`--model` changes the Qwen model as in `run`.

`test/calibrate_utility.py`, which `calibration_ratings.py` also runs at the end, reads
the ratings of the last commit and model. A (point, question) pair is a point written in
a session before the question's, and the point served if it is among the question's
evidence: a question of the same session asks about what was just said, which recency
keeps anyway. An update does not end the point it updates, since many HaluMem updates
add to the old point or only rephrase it and the questions keep citing it; safety facts
stay out, as they do from eviction. For each level, the fraction of its pairs that served,
pulled toward the fraction of all the levels when the data are few and made monotone:
`UTILITY_TABLE` is that fraction per utility level, `NEED_GAIN_TABLE` the fraction of the
need level times that of the gain level, both normalized to paste into `eviction.py`. For
`need_gain` it also prints the free table, cell by cell. Every table, and need and gain
alone, is tried on each user left out: the AUC says how often a pair that served scores
above one that did not, user by user and then averaged. The users are not pooled: each
is scored by a table estimated on the others, with a scale of its own, and eviction
ranks one archive at a time. For `need_gain` the line `provisional` is the `NEED_GAIN_TABLE`
`eviction.py` holds, which learns nothing from the pairs, beside the calibrated ones.
With two series or more it also prints the AUC of the mean vote: for every point the
mean of the ratings the series gave it, and of need × gain. Beside the AUC of one vote it
says how much of the weakness of a rating is noise that averaging takes away.

```bash
python3 memory_service/test/calibration_ratings.py --halumem PATH --utility need_gain --workers 5
python3 memory_service/test/calibration_ratings.py --halumem PATH --utility single --workers 5

# the tables again, from the ratings already saved
python3 memory_service/test/calibrate_utility.py --halumem PATH

# three series of the same model, their pairs together
python3 memory_service/test/calibrate_utility.py --halumem PATH --ratings A.jsonl B.jsonl C.jsonl --model MODEL
```

`calibrate_utility.py` reads the last model saved; `--model` picks another, so a file
can hold the 4B and the 9B. `calibration_ratings.py` passes it the model it just used.
With several `--ratings` files, each a series of the same points, the pairs of all of
them go together: a point counts once per series, with that series' judgment, so the
fraction of a level averages the series and the rare levels get more pairs. A file with
no ratings of the model is left out, and said so.

**Stability of the ratings.** A series launched again with another `--ratings` file
judges the same points once more. `test/rating_alpha.py` takes two series or more, each
file a rater and each point a unit, and prints Krippendorff's alpha of every rating:
safety nominal, utility, need, gain and need × gain ordinal. A rating a series did not
give (a safety point, a point never judged) is missing. Next to alpha it prints a
bootstrap interval over the points and the exact agreement. Of each file and mode it
reads the last commit and model saved for that mode, or with `--model` that model's.

```bash
python3 memory_service/test/rating_alpha.py --ratings A.jsonl B.jsonl C.jsonl --model MODEL
```

**The eviction replay.** `test/eviction_replay.py` compares eviction rules on the same
memories. Its input is a trace: a `run` whose archive never reaches the limit
(`--archive-limit 100000`). The replay walks the trace's operations again, insert by
insert, with a limit and a rule and no model: the core split is `choose_core_split`, a
redundant brings an archived memory back to the core as the service does, the service's
rule is `choose_prune_targets`, the search of every question is computed again
on the smaller archive from the vectors of the trace. Next to the service's score, with
utility `none` or the trace's, the three combinations, three prune fractions and the
trace's frequency function, it replays the calibrated rules: the trace's utility with a
calibrated table in place of the provisional one, for `weighted_sum` and `product`, since
`expected_gain` reads no table. With `--calibration-trace` the table comes from the
ratings given in context (below); otherwise from the context-free series of the trace's
model, their pairs together (`--ratings` with one file or more, the calibration file by
default). Without a calibration of that model and mode they are left out, and `simulate`
says so. As references it replays FIFO, LRU, LFU,
utility alone, with the calibrated table too, random and an oracle that knows the
questions to come, at the limits 25, 50, 100, 150 and 200. Two rules that would choose
the same are replayed once: with `none` the combination does not count, with `single`
`expected_gain` is `weighted_sum`, and utility alone with a calibrated table that ranks
the levels as the provisional one is utility alone. When an operation targets a memory the rule already
removed, the real system would have gone another way: the replay counts it (divergence)
and does what the system would do, an update keeps its new memory, a redundant stores its
fact again, a delete finds nothing.

`embed` computes the vectors with the service's embedding model, also of the evidence
sentences, `label` asks HaluMem's judge (`eval/.env`, as `evaluate`) which memories each
question needs, among the core, the 40 archived memories closest to the question and the
10 closest to each sentence of its evidence: a question is worded unlike the fact it asks
about, its evidence is not. It prints where the needed memories were in the question's
ranking: many beyond the 40th would mean the question alone misses them. `simulate`
replays and prints, for every rule and limit,
the share of needed memories still kept and still found, where it lies between random and
the oracle, by question type and by age, the divergence and the prunes, with how many a
later question needed (HPR) and the NPS, (HPR of random − HPR) / (HPR of random − HPR of
the oracle). For the rules at the trace's prune fraction it prints who leaves: the share
of prunes on memories written in the same session, their mean age in sessions, and the
share by rating level of the trace, each with how many a later question needed. And the AUC of
the utility term (need × gain, or utility in `single`) against being needed by a later
session, with the calibrated table too. Before the rules it checks the replay
with no limit against the trace: the split must archive the same memories at the same
insert, the search must find what the questions found. Everything lands in the run's
`replay/` folder, and `label` and `embed` skip what they already did.

```bash
python3 memory_service/test/halumem_adapter.py run --halumem PATH --version trace --archive-limit 100000
python3 memory_service/test/eviction_replay.py embed --halumem PATH --version trace
python3 memory_service/test/eviction_replay.py label --halumem PATH --version trace
python3 memory_service/test/eviction_replay.py simulate --halumem PATH --version trace --workers 4 --ratings A.jsonl B.jsonl C.jsonl
```

**Calibration in context.** The series of `calibration_ratings.py` judge each session's
points alone, while in a run the ratings are given in the middle of consolidation, with
the core, the candidates and the dialogue, on the facts the model wrote. To calibrate
the ratings eviction really uses, the calibration users are replayed as a trace
(`run --calibration-users --archive-limit 100000`, same mode, model and sampling as the
traces to replay), labelled by the judge, and `calibrate` derives the tables from it: a
pair is a rated memory, safety apart, still active when a question of a later session
comes, and it served when the judge chose it for that question. It prints the same
tables and AUCs as `calibrate_utility.py`, the provisional table included, and saves the
counts in `replay/calibration.json`. `simulate --calibration-trace CAL ...` then takes
the table of those traces together; a calibration of another mode, model or sampling is
left out, and one that shares a user with the trace replayed stops it. The context-free
series stay for the stability of the ratings, which only identical texts can measure.

How much the context itself changes the ratings is measured on the same texts:
`calibration_ratings.py --trace CAL` judges with the calibration prompt, session by
session, the memories the trace wrote (every create and update, by id, with its last
text), and `compare` sets them beside the trace's own ratings: the levels of each counted
once per memory, the mean shift and the crossing of the two for every rating, their
agreement (Krippendorff, to set beside the one between context-free series, which is the
noise alone), and the AUC of each on the same pairs. Both reports also count the levels
once per fact (`by point`, `by memory`), so the shares do not weigh a fact by its pairs.

**The mean vote.** One rating is a noisy draw: on the calibration points the mean of five
series ranks better than one vote. Whether a better rating keeps better memories is
measured on a trace: rate its memories again without context two or three times
(`calibration_ratings.py --trace`, one file per series). `compare` with all the files
adds their agreement and the AUC of their mean vote; `simulate --rerated` with them adds
the rules `rerated`, the trace's utility with the term of one vote without context (the
first series), and `mean-vote`, with the mean of the terms of all of them, every
combination at the trace's prune fraction. A memory the series did not rate keeps the
trace's rating, and safety is always the trace's, so only the utility term changes.
Beside the rating AUC, `simulate` prints the AUC of the vote with context and of each
variant on the memories all of them rated. `eviction.choose_prune_targets` takes these
terms by id (`given`), in place of those of the ratings.

```bash
python3 memory_service/test/halumem_adapter.py run --halumem PATH --version cal --calibration-users --archive-limit 100000 --utility need_gain
python3 memory_service/test/eviction_replay.py embed --halumem PATH --version cal
python3 memory_service/test/eviction_replay.py label --halumem PATH --version cal
python3 memory_service/test/eviction_replay.py calibrate --halumem PATH --version cal
python3 memory_service/test/calibration_ratings.py --halumem PATH --utility need_gain --trace cal --workers 4
python3 memory_service/test/eviction_replay.py compare --halumem PATH --version cal --ratings memory_service/test/calibration_work/cal_without_context.jsonl

# the mean vote on a trace: three series without context, then the replay with them
python3 memory_service/test/calibration_ratings.py --halumem PATH --utility need_gain --trace trace --ratings R1.jsonl
python3 memory_service/test/eviction_replay.py compare --halumem PATH --version trace --ratings R1.jsonl R2.jsonl R3.jsonl
python3 memory_service/test/eviction_replay.py simulate --halumem PATH --version trace --workers 4 --rerated R1.jsonl R2.jsonl R3.jsonl
python3 memory_service/test/eviction_replay.py simulate --halumem PATH --version trace --workers 4 --calibration-trace cal
```

**What a limit costs the robot.** After `simulate`, `test/resource_frontier.py` reads the
trace and its replay report. From `archive_bytes` and `vector_search_ms` it fits the
bytes and the search milliseconds of one more memory, with the empty folder and the size
of a vector alone. It then gives every limit of the replay in MB and ms and prints the
retention and of oracle of every rule at every limit. A point is on the frontier when no
rule keeps more with an archive no larger; the oracle is shown but is never on it. The
same table, one row per rule and limit, with the NPS and the share of prunes in the
same session, goes to `replay/frontier.csv` for the plots.

```bash
python3 memory_service/test/resource_frontier.py --halumem PATH --version trace
```

**What consolidation writes.** `test/consolidation_report.py` reads one run or several
and prints what consolidation made of the dialogue. From HaluMem's judgments in `tmp2/`,
after `evaluate`, it prints:
- recall and weighted recall of the memory points, and the interference points kept out;
- target and weighted accuracy of what was extracted, and their F1;
- the updates that were correct, hallucinated or omitted;
- the same broken down by type of point (persona, event, relationship) and by the
  session's place in the user's history (1-10, 11-25, 26-50, 51 on), to see whether
  consolidation holds as the memory grows.

From the run alone, before `evaluate` too, it prints:
- creates, updates, redundants and deletes per exchange;
- the share of redundants that brought an archived memory back to the core;
- the share of exchanges that wrote nothing;
- the memories written per memory point;
- the cost of the consolidation calls: per exchange, the seconds and tokens of a call,
  and their share of the time the dialogue took.

Every figure is the mean over users of the user's own figure, and the judgments HaluMem
could not use are left out, so the figures are not the pooled scores `evaluate` prints.
With several versions only the users all of them hold are read. Each version after the
first gets a column with its difference from the first, user by user, and the 95%
bootstrap interval over the users. Seconds depend on how many users ran at once, so
compare runs launched with the same `--workers`. `--csv` writes every row, version by
version, for the plots.

```bash
python3 memory_service/test/consolidation_report.py --halumem PATH --versions trace-ng trace-none trace-single --csv consolidation.csv
```

**Questions that lean on the last exchange.** HaluMem's questions name what they ask
about, and the adapter asks them with an empty window, so `query_rewrite` has nothing to
resolve there. `test/followup_questions.py` builds that case offline, on a trace already
run through `embed` and `label`, with no new run. In a trace no search changes what the
memory holds, so every question can be answered again on the core and the archive the
replay gives it.

For every question, HaluMem's judge writes four texts once: the question as the user would
ask it about themselves, a sentence of the user that brings up its subject, a reply of the
assistant, and the follow-up that leans on them without naming the subject. Seven searches
follow, each answered by the service's own `retrieval_node` and `generate_answer` with the
trace's model and settings:
- `Q`: the question with an empty window, searched with its words, as in a HaluMem run;
- `Q-rw`: the same, searched with the model's query, as in a HaluMem run with
  `query_rewrite`;
- `Qw`: the question after the exchange, which is in the window;
- `Qw-rw`: the same, searched with the model's query, which sees the exchange;
- `F`: the follow-up after the exchange, searched with its words;
- `F-rw`: the same, searched with the model's query;
- `F+said`: the same, searched with the user's sentence and the follow-up, with no model.

The answer reads the user's words, the window and what the search found, so two searches
that find the same memories share one answer. The judge reads the original question.

`report` prints, per search, the needed memories found (in core or among the 20, as the
replay counts them) and the answers correct, hallucinated and omitted, on the questions
every search has. Each search is set against a reference that differs in one thing only:
- `F` against `Qw` is what leaning on the exchange costs, with the same window;
- `F-rw` and `F+said` against `F` are what each remedy gives back;
- `Qw-rw` against `Qw` is what the rewrite costs where nothing needs resolving, as on the
  robot, whose window is seldom empty, and `Q-rw` against `Q` the same as on HaluMem;
- `Qw` against `Q` is what the window alone changes in the answer, since the search is
  the same.

The differences are taken user by user, with the 95% bootstrap interval over the users. It
also prints the correct answers by question type, the seconds and tokens of a rewrite and
of an answer, and a few exchanges with their rewritten queries.

The steps alternate the models: `generate` and `judge` need the judge of `eval/.env`, and
`answer` needs the trace's model and the embedding model. Each step saves its work per user
in the trace's `followup/` folder, and a step launched again skips what it already did,
except the calls that failed.

```bash
python3 memory_service/test/followup_questions.py generate --halumem PATH --version trace
python3 memory_service/test/followup_questions.py answer --halumem PATH --version trace --workers 4
python3 memory_service/test/followup_questions.py judge --halumem PATH --version trace
python3 memory_service/test/followup_questions.py report --halumem PATH --version trace
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
  (`retrieval`, `generate_answer`, `consolidation`): whatever an
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
  `InsertCoreMemories` is bound with
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

The limit in characters is **imposed by the code**, with no model call. When the core
memory overflows, the `split_core` node moves the least recent memories to the archive,
one at a time, until the rest fits (`consolidation.split_core_by_recency`). Recent is
`updated_at`, the last write or confirmation, so a `redundant` keeps a memory in core;
a single memory longer than the limit leaves too. Archiving is not deleting: the memory
stays searchable, and only its seat in the always-on context is lost.

A `redundant` on an archived memory brings it back (`reinforce_archived_item`): it leaves
the archive with its counts, ratings and `safety`, enters the core with `updated_at` of
now, and the split sends it back when newer memories push it out. So the core always
holds the most recently written or confirmed memories, as working memory holds what was
last rehearsed. An update of an archived memory already starts its new version in core.
A memory the archive cannot delete stays there with its `updated_at` refreshed, so a
memory is never in both places.

Recency, and not a judgment of importance, because the model's split kept the newest
memories anyway: on a HaluMem user, in four splits out of five every memory it kept was
newer than every memory it archived, the ratings and `safety` made no difference, and it
freed about twice what was needed, leaving the core half empty.

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
candidates, because the tombstones take the first places: they leave the archive
only at the end of the insert that retired them, so within it they are there, and
the classifier would find itself with fewer targets, with nothing signalling it.
Asking for K active ones, K are received, always, as long as K active ones exist.

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

The clock is replaceable too. Every timestamp the service writes (`updated_at`,
`retrieved_at`, the log's `timestamp`) and the `now` of the eviction come from
`backends.now()`, which is the wall clock until a replay moves it:

```python
backends.set_clock(session_start)   # at the start of each session of a dataset
```

From there the clock advances with real time, so the memories of a session carry
its date and keep the order in which they were written. It never goes back: a
start earlier than the current reading is ignored. `set_clock(None)` and
`backends.reset()` return to the wall clock.

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
| `MEMORY_ARCHIVE_LIMIT` | **active** memories in the archive beyond which eviction steps in. It does not block insertion: the graph's `exceed_archive_limit` router checks it at the end of every `insert` and, if exceeded, in the `prune_archive` node the share `prune_fraction` of the active memories with the lowest score leaves, all together (`eviction.prune_archive`, `prune` entry in the log). Memories with `safety` true are never candidates and get no score, but the share is computed on all the active ones. The score combines utility (see `consolidation_utility` below), the decay over time and the frequency. The decay is a Weibull or a power law according to `eviction_time_decay_function` (`weibull` or `power`), both with the scale `eviction_time_decay_scale_hours` (91 days by default), computed on `updated_at` or `retrieved_at` according to `eviction_time_decay_field`. The frequency is `n_used` when uses are counted (`track_used`, see below) and `n_retrieve` otherwise, on a log scale relative to the highest among the candidates: log(1 + n) / log(1 + highest); with `eviction_frequency_function` `rate` instead of `log` it is the count per search run while the memory was in the archive, with Laplace's rule: (n + 1) / (n_search + 2), 0.5 for a memory never searched (every counted search adds 1 to `n_search` of every active archived memory, and an update inherits it as it inherits `n_retrieve`). `eviction_combination` chooses how: `weighted_sum` is 0.4 utility + 0.3 decay + 0.3 frequency, `product` is utility × (0.5 decay + 0.5 frequency), `expected_gain` is gain × (0.4 need + 0.3 decay + 0.3 frequency), the probability that the memory is needed times what it is worth then, as in Anderson's rational analysis of memory, with need and gain as two terms of their own; without utility the three are the mean of decay and frequency, and with `single`, which has no gain, `expected_gain` is the weighted sum. The prune runs only with `MemoryConfig.eviction` on (the default); the tombstones leave at the end of every insert whatever it says, so a superseded or deleted memory never outlives the insert that retired it and a forgotten fact leaves the disk. The switches are changed only in the code (`config.py`), with no environment variable |
| `MEMORY_GENERATE_ANSWER` | `true` (default) or `false`: whether the `retrieve` branch also **composes an answer** for the user. No field of the service returns it — the archive reaches the caller through `retrieved_memories` — so it only ends up in `last_messages`, as a turn of the conversation. `false` saves one call per `retrieve`, leaves one (user, assistant) pair per exchange instead of three messages, and keeps the memory from quoting itself inside consolidation. `track_used` needs it on, because the list of used memories comes from the answer |
| `MEMORY_CHROMA_PATH` | archive folder (resolved to an absolute path) |
| `MEMORY_COLLECTION_NAME` | ChromaDB collection |
| `MEMORY_ENV_FILE` | explicit path of the environment file |

`retrieval_mode` has no environment variable and is changed only in `config.py`.
It decides whether the archive search of the `retrieve` branch is the model's
choice:

- `decide` (default): the model chooses between `retrieve_memory` and `NoSearchNeeded`,
  and writes only `k`;
- `always`: it always searches, with no model call.

In both the archive is searched with the user's own words (the `user_input` of the
request, or else the user's last message), never with a text the model writes: on
HaluMem the model's keyword queries missed more of the memories a question needed than
the question itself, even with the question in Italian and the facts in English. A
retrieval increments `n_retrieve` of the returned memories. The long test prints and
saves the run's mode.

`query_rewrite` (off by default, also only in `config.py`) brings back the query the
model writes. In `always` the model is called with the facts kept at hand and the
messages still in the window, and writes the query and `k`. In `decide` it writes them
when it chooses to search. Both prompts ask for the query in the language of the facts.
The archive is searched with that query alone, and with the user's words only when the
model writes an empty one. It is the case HaluMem never tests: a question that leans on
the last exchange ("and what did I like there?") does not name what it asks for, while
the model sees that exchange. The benchmark adapter keeps it off, and the eviction replay
refuses a trace run with it on, since it searches again with the questions. The long
test prints and saves it next to the mode.

`count_candidate_search` (on or off, also only in `config.py`) makes the archive search
that consolidation runs for the classifier's candidates count as a retrieval too: no
model call, and `n_retrieve` grows on every consolidated message. The benchmark adapter
turns it off: there only the questions count.

`retrieve_k` (also only in `config.py`) is `None` or a number: with `None` the model
writes `k` in `decide` and `always` returns 20, a number replaces both, whatever the
model wrote. `answer_with_dates` (on or off, also only in `config.py`) puts in front of every
fact and memory the answer reads the day of its `updated_at`, that is of its last write
or confirmation, and tells the model what that day means. The benchmark adapter sets
`retrieve_k` to 20 and turns `answer_with_dates` on.

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
`delete` has none. With `need_gain` the schema says so itself: an operation has three
shapes, `new` and `update` with `safety` false and `need` and `gain` required, `new` and
`update` with `safety` true and no rating, `redundant` and `delete` with none. An
`update` that arrives without any judgment all the same (a server that does not hold the
model to the schema, or `single`, where the fields stay optional) takes its target's,
and the agent prints it. An off-scale rating counts as not given. The four fields live
in `CoreMemoryItem` and in the archive metadata, where a missing value is absent rather
than `None`.

When the classifier gives a `target_item_id` that exists neither in core memory nor in
the archive, the target is looked for by text among the core memories and the five
archived memories closest to the fact (`CONTENT_MATCH_K`), a search that does not count
as a retrieval. A memory matches when one text contains the other, ignoring case and
the closing punctuation. Only a single match is taken: when memories with different
texts match, none is chosen, since a guess could change the wrong one; the same text
stored twice counts as one memory, and the core copy wins. A `redundant` or an `update`
without a target is stored as a new memory, a `delete` without a target is ignored. A
`redundant` whose id names a memory that is no longer active is dropped: tombstones leave
the archive at the end of every insert, so that memory was superseded or deleted in the
same call, and storing the fact as new would bring it back. An `update` or a `delete` with
such an id is resolved by its text, as one with an unknown id.

The classifier is asked one question of the memory about the same thing: does what the
user said make any part of it false? Only then it is an `update`; telling more is a
`new`, even on the same topic, and nothing more is a `redundant`. Two rules in the code
hold whatever the model writes, comparing texts as the content match does: an `update`
whose text is the one of its target is a `redundant`, with the ratings it had; and one
text is one memory per call. A second `new` with a text the call already wrote is
dropped, and a second `update` with it supersedes its target with that same memory,
which keeps the higher counts and ratings of its parts, and `safety` if any has it.

In eviction the judgment enters the score as the utility term, high meaning keep
as for the other terms: with `single` the rating's entry in `UTILITY_TABLE`, with
`need_gain` the (need, gain) cell of `NEED_GAIN_TABLE`, both in `eviction.py`.
Until they are calibrated they are rating / 5 and (need / 3) · (gain / 3); the
calibrated values are printed by `test/calibrate_utility.py` (see the Tests
section). With `need_gain` and the `expected_gain` combination, need and gain enter
apart instead: the need's entry in `NEED_TABLE` (need / 3) inside the mean, the gain's
in `GAIN_TABLE` ((6 + gain) / 9) as the factor, narrow because it multiplies. A memory
without a rating, because the model did not give it or gave it off the scale, takes
the middle one (`NEUTRAL_RATINGS`: 3, or need 2 and gain 2), so the prune does not stop.
In all modes a memory with `safety` true never leaves the archive because of the limit.

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
