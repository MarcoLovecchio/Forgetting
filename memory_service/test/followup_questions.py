#!/usr/bin/env python3
"""Questions that lean on the last exchange: does the model's rewritten query find more?

HaluMem's questions name what they ask about, and the adapter asks them with no messages
in the window, so the query the model writes (MemoryConfig.query_rewrite) has nothing to
resolve there. This builds the case it is for, offline, on a trace already embedded and
labelled by eviction_replay.py, with no new run: in a trace no search changes what the
memory holds, so every question can be answered again on the core and the archive it met.

For every question the judge's model writes, once, the question as the user would ask it
about themselves, a sentence of the user that brings up its subject, a reply of the
assistant, and the follow-up that leans on them without naming the subject. Seven
searches, each answered by the service's own nodes (retrieval_node, generate_answer):

    Q        the question, with an empty window, searched with its words: a HaluMem run
    Q-rw     the same, searched with the model's query: a HaluMem run with query_rewrite
    Qw       the question after the exchange, which is in the window
    Qw-rw    the same, searched with the model's query, which sees the exchange
    F        the follow-up after the exchange, searched with its words
    F-rw     the same, searched with the model's query
    F+said   the same, searched with the user's sentence and the follow-up, no model

F against Qw is what leaning on the exchange costs, with the same window; F-rw and
F+said against F what each remedy gives back; Qw-rw against Qw what the rewrite costs
where nothing needs resolving, as on the robot, and Q-rw against Q as on HaluMem; Qw
against Q what the window alone changes in the answer, since the search is the same.

The answer reads the user's words, the window and what the search found: two searches
that find the same memories for the same words share the answer.

    generate  the exchanges, with HaluMem's judge (eval/.env)
    answer    the rewritten queries, the vectors of every query, the search on the archive
              of the trace at that question and the answer: the trace's model, settings
              and embedding model
    judge     every answer, with HaluMem's question judge and the original question
    report    per search, the needed memories found (in core or among the searched, as
              eviction_replay.py counts them) and the answers correct, hallucinated and
              omitted, on the questions every search has; each against its reference,
              user by user, with the 95% bootstrap interval over the users; and what a
              rewrite costs

Every step keeps a cache per user and skips what it already did, failures apart.

    python3 memory_service/test/followup_questions.py generate --halumem PATH --version TRACE
    python3 memory_service/test/followup_questions.py answer --halumem PATH --version TRACE
    python3 memory_service/test/followup_questions.py judge --halumem PATH --version TRACE
    python3 memory_service/test/followup_questions.py report --halumem PATH --version TRACE

Everything lands in PATH/eval/results/memory_service-TRACE/followup/.
"""

import argparse
import contextlib
import importlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import consolidation_report as kpi  # noqa: E402
import eviction_replay as replay  # noqa: E402
import halumem_adapter as adapter  # noqa: E402
from memory_service import backends  # noqa: E402
from memory_service import memory_manager_llm as nodes  # noqa: E402
from memory_service.config import NODE_SAMPLING, MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    NO_ARCHIVAL_RESULTS,
    CoreMemoryItem,
    memory_line,
)

FOLDER = "followup"
EXCHANGE = ("question", "said", "reply", "follow_up")
RESULT_TYPES = ("Correct", "Hallucination", "Omission")
METRICS = ("found", "complete", "correct", "hallucinated", "omitted")
VERDICTS = {"correct": "Correct", "hallucinated": "Hallucination", "omitted": "Omission"}
SHARE = kpi.SHARE


@dataclass(frozen=True)
class Search:
    name: str
    follow_up: bool
    window: bool
    rewritten: bool = False
    with_said: bool = False


SEARCHES = (
    Search("Q", False, False),
    Search("Q-rw", False, False, rewritten=True),
    Search("Qw", False, True),
    Search("Qw-rw", False, True, rewritten=True),
    Search("F", True, True),
    Search("F-rw", True, True, rewritten=True),
    Search("F+said", True, True, with_said=True),
)
# (search, its reference), one difference each: the follow-up against the question with
# the same window, the two remedies, the rewrite with and without a window, the window.
COMPARISONS = (("F", "Qw"), ("F-rw", "F"), ("F+said", "F"), ("Qw-rw", "Qw"), ("Q-rw", "Q"),
               ("Qw", "Q"))

GENERATE_PROMPT = """A user talks with their assistant robot, which remembers what the user told it
in earlier conversations. Below is a question about the user. Turn it into a short exchange.

Question: {question}

Write four texts. All but the reply are said by the user, in the first person ("I", "my").
- question: the same question, asked by the user about themselves, every detail kept.
- said: one short sentence that brings up the subject of the question without asking it
  and without stating any fact about the user, such as "Let's talk about my wardrobe."
- reply: one short sentence of the assistant that invites the question and states no fact
  about the user.
- follow_up: the question as the user would ask it right after the reply. It leans on what
  was just said: the subject is not named again (a pronoun, "there", "that", or nothing),
  while every other detail of the question, dates included, is kept.

Reply with a JSON block and nothing else:
```json
{{"question": "...", "said": "...", "reply": "...", "follow_up": "..."}}
```"""


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #

def user_path(run_dir: str, part: str, uuid: str, suffix: str = ".json") -> str:
    folder = os.path.join(run_dir, FOLDER, part)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, uuid + suffix)


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def trace_questions(result: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [(replay.question_key(index, number), question)
            for index, session in enumerate(result["sessions"])
            for number, question in enumerate(session.get("questions", []))]


def halumem_module(arguments: argparse.Namespace, name: str) -> Any:
    """A module of HaluMem's eval/, from its folder: it reads eval/.env from there."""
    eval_dir = os.path.join(os.path.abspath(os.path.expanduser(arguments.halumem)), "eval")
    os.chdir(eval_dir)
    sys.path.insert(0, eval_dir)
    return importlib.import_module(name)


def error_text(error: Exception) -> Dict[str, str]:
    return {"error": f"{type(error).__name__}: {error}"}


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #

def parse_exchange(reply: Any) -> Dict[str, str]:
    if not isinstance(reply, dict):
        raise ValueError(f"not a JSON object: {reply!r}")
    exchange = {name: str(reply.get(name) or "").strip() for name in EXCHANGE}
    missing = [name for name, text in exchange.items() if not text]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    return exchange


def generate_user(result: Dict[str, Any], stored: Dict[str, Any],
                  ask: Callable[[str], Any], workers: int) -> int:
    """The exchanges of the questions not built yet, into `stored`: how many were asked."""
    pending = [(key, question["asked"]) for key, question in trace_questions(result)
               if "follow_up" not in stored.get(key, {})]

    def build(job: Tuple[str, str]) -> Tuple[str, Dict[str, str]]:
        key, asked = job
        try:
            return key, parse_exchange(ask(GENERATE_PROMPT.format(question=asked)))
        except Exception as error:
            return key, error_text(error)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, outcome in pool.map(build, pending):
            stored[key] = outcome
    return len(pending)


def generate(arguments: argparse.Namespace) -> int:
    run_dir = adapter.run_folder(arguments)
    results = replay.trace_results(run_dir)[:arguments.users]
    llms = halumem_module(arguments, "llms")
    shown = 0
    for result in results:
        path = user_path(run_dir, "exchanges", result["uuid"])
        stored = load_json(path)
        asked = generate_user(result, stored, llms.llm_request_for_json, arguments.judge_workers)
        replay.save_json(path, stored)
        failed = sum(1 for entry in stored.values() if "error" in entry)
        print(f"{result['uuid']}: {asked} questions turned into exchanges, {failed} failed "
              "(launch generate again to retry them)", flush=True)
        for key, question in trace_questions(result):
            entry = stored.get(key, {})
            if shown < 3 and "follow_up" in entry:
                shown += 1
                print(f"    {question['asked']}\n      question:  {entry['question']}\n"
                      f"      said:      {entry['said']}\n      reply:     {entry['reply']}\n"
                      f"      follow-up: {entry['follow_up']}")
    return 0


# --------------------------------------------------------------------------- #
# answer
# --------------------------------------------------------------------------- #

@dataclass
class Moment:
    """The memory a question met: the core in its order, the archive with each day."""

    core: List[CoreMemoryItem]
    archive: List[Tuple[str, str, datetime]]


def question_moments(run_dir: str, result: Dict[str, Any], user: Dict[str, Any],
                     config: MemoryConfig) -> Tuple[Dict[str, Moment], Dict[str, Any]]:
    """The memory at every question, from the replay with no limit, and its checks."""
    closeness = replay.load_closeness(run_dir, result)
    moments: Dict[str, Moment] = {}

    def keep(state: replay.Replay, event: replay.Question, ranked: List[str]) -> None:
        core = sorted(state.core, key=lambda item_id: (state.entered[item_id], item_id))
        moments[event.key] = Moment(
            [state.items[item_id].model_copy() for item_id in core],
            [(item_id, state.items[item_id].content, state.items[item_id].updated_at)
             for item_id in state.archive])

    unlimited = replay.Replay(config, None, None, closeness.rank, on_question=keep).run(
        replay.trace_events(result, user))
    return moments, replay.trace_checks(unlimited, result)


def asked_text(exchange: Dict[str, str], search: Search) -> str:
    return exchange["follow_up"] if search.follow_up else exchange["question"]


def window_texts(exchange: Dict[str, str], search: Search) -> List[str]:
    return [exchange["said"], exchange["reply"]] if search.window else []


def rewrite_kind(search: Search) -> str:
    """What the rewrite sees: the words asked, and whether the exchange is in the window."""
    if search.follow_up:
        return "follow-up"
    return "question+window" if search.window else "question"


def search_query(search: Search, exchange: Dict[str, str],
                 rewrites: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """The text the search embeds; None while its rewrite is missing."""
    asked = asked_text(exchange, search)
    if search.rewritten:
        done = rewrites.get(rewrite_kind(search), {})
        if "written" not in done:
            return None
        # As the tool node does: an empty query falls back on the user's words.
        return done["written"] or asked
    if search.with_said:
        return f"{exchange['said']} {exchange['follow_up']}"
    return asked


def node_state(moment: Moment, asked: str, window: Sequence[str],
               retrieved: str = "") -> Dict[str, Any]:
    """The state a retrieve call of the service would hold at that question."""
    messages = ([HumanMessage(content=window[0]), AIMessage(content=window[1])]
                if window else [])
    return {"messages": messages, "tool_calls": [], "operation_log": [],
            "core_memory": [item.model_copy() for item in moment.core],
            "retrieved_memory": retrieved, "current_query": asked,
            "current_interaction": "retrieve"}


def usage(message: Any) -> Dict[str, int]:
    metadata = getattr(message, "usage_metadata", None) or {}
    return {"input_tokens": int(metadata.get("input_tokens") or 0),
            "output_tokens": int(metadata.get("output_tokens") or 0)}


def rewritten_query(config: MemoryConfig, moment: Moment, asked: str,
                    window: Sequence[str]) -> Dict[str, Any]:
    """The query retrieval_node writes with query_rewrite, and what the call cost."""
    runtime = SimpleNamespace(context=replace(config, retrieval_mode="always",
                                              query_rewrite=True))
    started = time.perf_counter()
    response = adapter.with_retries(lambda: nodes.retrieval_node(
        node_state(moment, asked, window), runtime))["tool_calls"][-1]
    calls = getattr(response, "tool_calls", None) or [{}]
    return {"written": str(calls[0].get("args", {}).get("query") or "").strip(),
            "seconds": time.perf_counter() - started, **usage(response)}


def retrieved_text(found: Sequence[str], moment: Moment, dated: bool) -> str:
    """What the tool node hands the answer, as retrieve_active_archival_memories writes it."""
    if not found:
        return NO_ARCHIVAL_RESULTS
    archived = {item_id: (content, day) for item_id, content, day in moment.archive}
    return "\n".join(memory_line(archived[item_id][0], item_id,
                                 archived[item_id][1] if dated else None)
                     for item_id in found)


def answered(config: MemoryConfig, moment: Moment, asked: str, window: Sequence[str],
             retrieved: str) -> Dict[str, Any]:
    """The answer of generate_answer, and what the call cost."""
    runtime = SimpleNamespace(context=config)
    started = time.perf_counter()
    response = adapter.with_retries(lambda: nodes.generate_answer(
        node_state(moment, asked, window, retrieved), runtime))["messages"][-1]
    return {"answer": str(response.content).strip(),
            "seconds": time.perf_counter() - started, **usage(response)}


def archive_matrix(archive: Sequence[Tuple[str, str, datetime]], memories: Dict[str, Any]
                   ) -> Any:
    """The normalized vectors of the archive, in its order; None when it is empty."""
    import numpy

    if not archive:
        return None
    try:
        return numpy.stack([memories[replay.text_key(content)] for _, content, _ in archive])
    except KeyError as missing:
        raise SystemExit(f"No vector for a memory text ({missing}): launch "
                         "eviction_replay.py embed again.")


def closest(query: Any, archive: Sequence[Tuple[str, str, datetime]], matrix: Any,
            k: int) -> List[str]:
    """The k archived memories closest to the query, as the replay ranks them."""
    import numpy

    if matrix is None:
        return []
    scores = matrix @ (query / numpy.linalg.norm(query))
    order = sorted(range(len(archive)), key=lambda index: (-float(scores[index]),
                                                           archive[index][0]))
    return [archive[index][0] for index in order[:k]]


def memory_vectors(run_dir: str, uuid: str) -> Dict[str, Any]:
    import numpy

    path = replay.vectors_path(run_dir, uuid)
    if not os.path.exists(path):
        raise SystemExit(f"No vectors for {uuid}: launch eviction_replay.py embed first.")
    with numpy.load(path) as stored:
        return {key: vector / numpy.linalg.norm(vector)
                for key, vector in zip(stored["memory_keys"], stored["memory_vectors"])}


def query_vectors(path: str, texts: Sequence[str], embed_query: Callable[[str], Any],
                  workers: int) -> Dict[str, Any]:
    """The vector of every query, by text_key: the ones already saved, the others embedded."""
    import numpy

    vectors = {}
    if os.path.exists(path):
        with numpy.load(path) as stored:
            vectors = dict(zip(stored["keys"], stored["vectors"]))
    missing = list(dict.fromkeys(text for text in texts if replay.text_key(text) not in vectors))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for text, vector in zip(missing, pool.map(embed_query, missing)):
            vectors[replay.text_key(text)] = numpy.array(vector, dtype="float32")
    if missing:
        numpy.savez_compressed(path, keys=numpy.array(list(vectors)),
                               vectors=numpy.array(list(vectors.values()), dtype="float32"))
    return vectors


def answer_user(run_dir: str, result: Dict[str, Any], user: Dict[str, Any],
                config: MemoryConfig, exchanges: Dict[str, Any], stored: Dict[str, Any],
                embed_query: Callable[[str], Any], workers: int,
                save: Callable[[], None] = lambda: None) -> Dict[str, Any]:
    """Rewrites, searches and answers of one user, into `stored`: the trace's checks."""
    moments, checks = question_moments(run_dir, result, user, config)
    records = dict(trace_questions(result))
    ready = {key: exchange for key, exchange in exchanges.items()
             if "follow_up" in exchange and key in moments}
    stored["model"] = config.llm_config.get("model_name")
    stored["node_sampling"] = NODE_SAMPLING
    rewrites = stored.setdefault("rewrites", {})

    jobs = [(key, search) for key in ready for search in SEARCHES if search.rewritten
            and "written" not in rewrites.get(key, {}).get(rewrite_kind(search), {})]

    def rewrite(job: Tuple[str, Search]) -> Tuple[str, str, Dict[str, Any]]:
        key, search = job
        exchange = ready[key]
        try:
            outcome = rewritten_query(config, moments[key], asked_text(exchange, search),
                                      window_texts(exchange, search))
        except Exception as error:
            outcome = error_text(error)
        return key, rewrite_kind(search), outcome

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, kind, outcome in pool.map(rewrite, jobs):
            rewrites.setdefault(key, {})[kind] = outcome
    save()

    planned: Dict[str, Dict[str, str]] = {}
    for key, exchange in ready.items():
        for search in SEARCHES:
            query = search_query(search, exchange, rewrites.get(key, {}))
            if query is not None:
                planned.setdefault(key, {})[search.name] = query
    vectors = query_vectors(user_path(run_dir, "vectors", result["uuid"], ".npz"),
                            [query for queries in planned.values() for query in queries.values()],
                            embed_query, workers)
    memories = memory_vectors(run_dir, result["uuid"])
    k = config.retrieve_k or nodes.RETRIEVE_K

    questions = stored.setdefault("questions", {})
    inputs: Dict[str, Tuple[str, str, List[str], List[str]]] = {}
    for key, queries in planned.items():
        moment = moments[key]
        entry = questions.setdefault(key, {})
        entry.update(type=records[key].get("question_type", ""),
                     core=[item.id for item in moment.core])
        searches = entry.setdefault("searches", {})
        matrix = archive_matrix(moment.archive, memories)
        for search in SEARCHES:
            if search.name not in queries:
                continue
            query = queries[search.name]
            found = closest(vectors[replay.text_key(query)], moment.archive, matrix, k)
            asked = asked_text(ready[key], search)
            window = window_texts(ready[key], search)
            digest = replay.text_key(json.dumps([key, asked, window, found]))
            searches[search.name] = {"query": query, "found": found, "answer": digest}
            inputs[digest] = (key, asked, window, found)

    answers = stored.setdefault("answers", {})
    pending = [(digest, job) for digest, job in inputs.items()
               if "answer" not in answers.get(digest, {})]

    def respond(item: Tuple[str, Tuple[str, str, List[str], List[str]]]
                ) -> Tuple[str, Dict[str, Any]]:
        digest, (key, asked, window, found) = item
        moment = moments[key]
        try:
            return digest, answered(config, moment, asked, window,
                                    retrieved_text(found, moment, config.answer_with_dates))
        except Exception as error:
            return digest, error_text(error)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for digest, outcome in pool.map(respond, pending):
            answers[digest] = outcome
    save()
    return checks


def query_embedder() -> Callable[[str], Any]:
    """embed_query of the service's embedding model, as the search embeds a question."""
    embeddings = backends._build_embeddings(MemoryConfig.from_environment())
    return lambda text: adapter.with_retries(lambda: embeddings.embed_query(text))


def answer(arguments: argparse.Namespace) -> int:
    run_dir = adapter.run_folder(arguments)
    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    manifest = replay.trace_manifest(run_dir)
    if manifest.get("node_sampling") != NODE_SAMPLING:
        print("Warning: NODE_SAMPLING is not the one of the trace, so the answers are not "
              "given with its settings.")
    # First: it loads the workspace's environment the models need.
    embed_query = query_embedder()
    config = replay.trace_config(run_dir)
    backends.configure(config=config)
    users = {user["uuid"]: user for user in adapter.dataset_users(root)}
    for result in replay.trace_results(run_dir)[:arguments.users]:
        uuid = result["uuid"]
        exchanges = load_json(user_path(run_dir, "exchanges", uuid))
        if not exchanges:
            print(f"{uuid}: no exchanges, launch generate first", flush=True)
            continue
        path = user_path(run_dir, "answers", uuid)
        stored = load_json(path)
        with open(user_path(run_dir, "logs", uuid, ".log"), "w", encoding="utf-8",
                  buffering=1) as log, contextlib.redirect_stdout(log):
            checks = answer_user(run_dir, result, users[uuid], config, exchanges, stored,
                                 embed_query, arguments.workers,
                                 lambda: replay.save_json(path, stored))
        failed = (sum(1 for kinds in stored.get("rewrites", {}).values()
                      for outcome in kinds.values() if "error" in outcome),
                  sum(1 for outcome in stored.get("answers", {}).values() if "error" in outcome))
        print(f"{uuid}: search agreement with the trace "
              f"{replay.percent(checks['search agreement'])}%, "
              f"{len(stored.get('questions', {}))} questions, "
              f"{len(stored.get('answers', {}))} answers; {failed[0]} rewrites and "
              f"{failed[1]} answers failed (launch answer again to retry them)", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# judge
# --------------------------------------------------------------------------- #

def judge_user(result: Dict[str, Any], stored: Dict[str, Any], judged: Dict[str, Any],
               evaluate: Callable[[str, str, str, str], Any], workers: int) -> int:
    """HaluMem's verdict on every answer not judged yet, into `judged`: how many were asked.

    The judge reads the original question and evidence: a follow-up asks the same thing.
    """
    name = result["user_name"]
    records = dict(trace_questions(result))
    jobs: Dict[str, Tuple[Dict[str, Any], str]] = {}
    for key, entry in stored.get("questions", {}).items():
        for search in entry["searches"].values():
            digest = search["answer"]
            text = stored.get("answers", {}).get(digest, {}).get("answer")
            if text is not None and "result_type" not in judged.get(digest, {}):
                jobs[digest] = (records[key], text)

    def ask(item: Tuple[str, Tuple[Dict[str, Any], str]]) -> Tuple[str, Dict[str, Any]]:
        digest, (record, text) = item
        try:
            reply = evaluate(record["question"], record["answer"],
                             "\n".join(replay.evidence_of(record)),
                             adapter.personalized(text, name))
        except Exception as error:
            return digest, error_text(error)
        return digest, {"result_type": reply.get("evaluation_result")
                        if isinstance(reply, dict) else None}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for digest, outcome in pool.map(ask, list(jobs.items())):
            judged[digest] = outcome
    return len(jobs)


def judge(arguments: argparse.Namespace) -> int:
    run_dir = adapter.run_folder(arguments)
    results = replay.trace_results(run_dir)[:arguments.users]
    eval_tools = halumem_module(arguments, "eval_tools")
    for result in results:
        stored = load_json(user_path(run_dir, "answers", result["uuid"]))
        if not stored.get("questions"):
            print(f"{result['uuid']}: no answers, launch answer first", flush=True)
            continue
        path = user_path(run_dir, "judgments", result["uuid"])
        judged = load_json(path)
        asked = judge_user(result, stored, judged, eval_tools.evaluation_for_question,
                           arguments.judge_workers)
        replay.save_json(path, judged)
        failed = sum(1 for outcome in judged.values() if "error" in outcome)
        unusable = sum(1 for outcome in judged.values()
                       if "result_type" in outcome and outcome["result_type"] not in RESULT_TYPES)
        print(f"{result['uuid']}: {asked} answers judged, {failed} failed (launch judge again "
              f"to retry them), {unusable} with a label HaluMem does not know", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def verdict(judged: Dict[str, Any], digest: str) -> Optional[str]:
    found = judged.get(digest, {}).get("result_type")
    return found if found in RESULT_TYPES else None


def user_report(stored: Dict[str, Any], judged: Dict[str, Any],
                needs: Dict[str, List[str]]) -> Dict[str, Any]:
    """One user's values per search, on the questions every search has.

    found is the needed memories in core or found over the needed, complete the labelled
    questions with all of them; the answer shares are over the questions with a usable
    verdict in every search.
    """
    names = [search.name for search in SEARCHES]
    every = {key: entry for key, entry in stored.get("questions", {}).items()
             if all(name in entry["searches"] for name in names)}
    verdicts = {key: [verdict(judged, entry["searches"][name]["answer"]) for name in names]
                for key, entry in every.items()}
    judged_keys = [key for key, found in verdicts.items() if None not in found]
    values: Dict[str, Dict[str, Optional[float]]] = {}
    for position, name in enumerate(names):
        needed = reached = complete = labelled = 0
        for key, entry in every.items():
            need = needs.get(key)
            if not need:
                continue
            seen = set(entry["core"]) | set(entry["searches"][name]["found"])
            hits = sum(1 for item_id in need if item_id in seen)
            needed, reached = needed + len(need), reached + hits
            labelled, complete = labelled + 1, complete + (hits == len(need))
        given = [verdicts[key][position] for key in judged_keys]
        values[name] = {"found": kpi.ratio(reached, needed),
                        "complete": kpi.ratio(complete, labelled),
                        **{metric: kpi.ratio(given.count(label), len(given))
                           for metric, label in VERDICTS.items()}}
    types: Dict[str, Dict[str, Optional[float]]] = {}
    for kind in sorted({every[key]["type"] for key in judged_keys}):
        keys = [key for key in judged_keys if every[key]["type"] == kind]
        types[kind] = {name: kpi.ratio(sum(verdicts[key][position] == "Correct" for key in keys),
                                       len(keys)) for position, name in enumerate(names)}
    return {"values": values, "types": types, "questions": len(stored.get("questions", {})),
            "every": len(every), "judged": len(judged_keys),
            "labelled": sum(1 for key in every if needs.get(key)),
            "changed": sum(1 for entry in every.values() if entry["searches"]["F-rw"]["found"]
                           != entry["searches"]["F"]["found"]),
            "rewrites": [outcome for kinds in stored.get("rewrites", {}).values()
                         for outcome in kinds.values() if "written" in outcome],
            "answers": [outcome for outcome in stored.get("answers", {}).values()
                        if "answer" in outcome]}


def by_user(reports: Dict[str, Dict[str, Any]], name: str, metric: str
            ) -> Dict[str, Optional[float]]:
    return {uuid: found["values"][name][metric] for uuid, found in reports.items()}


def cost_text(calls: Sequence[Dict[str, Any]], what: str) -> str:
    if not calls:
        return f"no {what}"
    return (f"{len(calls)} {what}, {kpi.mean([call['seconds'] for call in calls]):.2f} s, "
            f"{kpi.mean([call['input_tokens'] for call in calls]):.0f} input and "
            f"{kpi.mean([call['output_tokens'] for call in calls]):.0f} output tokens each")


def print_examples(run_dir: str, results: Sequence[Dict[str, Any]], count: int) -> None:
    examples = []
    for result in results:
        exchanges = load_json(user_path(run_dir, "exchanges", result["uuid"]))
        stored = load_json(user_path(run_dir, "answers", result["uuid"]))
        for key, entry in stored.get("questions", {}).items():
            if all(name in entry["searches"] for name in ("Q-rw", "Qw-rw", "F-rw")):
                examples.append((exchanges[key], entry["searches"]))
    for exchange, searches in random.Random(0).sample(examples, min(count, len(examples))):
        print(f"\n  question:  {exchange['question']}\n    rewritten alone: "
              f"{searches['Q-rw']['query']}\n    rewritten after the exchange: "
              f"{searches['Qw-rw']['query']}\n  said:      {exchange['said']}"
              f"\n  reply:     {exchange['reply']}\n  follow-up: {exchange['follow_up']}"
              f"\n    rewritten: {searches['F-rw']['query']}")


def report(arguments: argparse.Namespace) -> int:
    run_dir = adapter.run_folder(arguments)
    results = replay.trace_results(run_dir)[:arguments.users]
    reports = {}
    for result in results:
        stored = load_json(user_path(run_dir, "answers", result["uuid"]))
        if stored.get("questions"):
            reports[result["uuid"]] = user_report(
                stored, load_json(user_path(run_dir, "judgments", result["uuid"])),
                replay.needs_of(replay.load_labels(run_dir, result["uuid"])))
    if not reports:
        raise SystemExit(f"No answers in {run_dir}: launch answer first.")

    total = {name: sum(found[name] for found in reports.values())
             for name in ("questions", "every", "labelled", "judged", "changed")}
    print("Q the question with an empty window, Qw after the exchange, F the follow-up after "
          "the exchange; -rw searched with the model's query, F+said with the user's sentence "
          "too. Means over users of the user's own share.")
    print(f"\n{len(reports)} users, {total['questions']} questions, {total['every']} with every "
          f"search, {total['labelled']} of them labelled and {total['judged']} judged in every "
          "search")
    print(f"\n{'':<10}" + "".join(f"{metric:>14}" for metric in METRICS))
    for search in SEARCHES:
        cells = [kpi.mean(kpi.present(by_user(reports, search.name, metric)))
                 for metric in METRICS]
        print(f"  {search.name:<8}" + "".join(f"{kpi.shown(cell, SHARE):>14}" for cell in cells))

    compared = ("found", "correct", "hallucinated", "omitted")
    print("\nEach search minus its reference, user by user, in percentage points, with the 95% "
          "interval over the users and the users above/below/at zero:")
    print(f"{'':<16}" + "".join(f"{metric:>31}" for metric in compared))
    for name, reference in COMPARISONS:
        cells = [kpi.difference_text(kpi.paired(by_user(reports, reference, metric),
                                                by_user(reports, name, metric)), SHARE)
                 for metric in compared]
        print(f"  {name + ' - ' + reference:<14}" + "".join(f"{cell:>31}" for cell in cells))

    kinds = sorted({kind for found in reports.values() for kind in found["types"]})
    print("\nAnswers correct by question type:")
    print(f"{'':<34}" + "".join(f"{search.name:>9}" for search in SEARCHES))
    for kind in kinds:
        cells = [kpi.mean(kpi.present({uuid: found["types"].get(kind, {}).get(search.name)
                                       for uuid, found in reports.items()}))
                 for search in SEARCHES]
        print(f"  {kind:<32}" + "".join(f"{kpi.shown(cell, SHARE):>9}" for cell in cells))

    rewrites = [call for found in reports.values() for call in found["rewrites"]]
    answers = [call for found in reports.values() for call in found["answers"]]
    print(f"\nRewrites: {cost_text(rewrites, 'calls')}. Answers: {cost_text(answers, 'calls')}.")
    print(f"F-rw found other memories than F in {total['changed']} of {total['every']} "
          "questions.")
    print_examples(run_dir, results, arguments.examples)
    return 0


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Questions that lean on the last exchange, "
                                                 "with and without the model's query.")
    steps = parser.add_subparsers(dest="step", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    common.add_argument("--version", required=True, type=adapter.version_name,
                        help="the trace, embedded and labelled by eviction_replay.py")
    common.add_argument("--users", type=int, help="only the first users of the trace")

    built = steps.add_parser("generate", parents=[common], help="the exchanges, by the judge")
    built.add_argument("--judge-workers", type=int, default=8, dest="judge_workers")
    built.set_defaults(step_function=generate)

    replied = steps.add_parser("answer", parents=[common],
                               help="rewrites, searches and answers, by the service's models")
    replied.add_argument("--workers", type=int, default=4, help="model calls at the same time")
    replied.set_defaults(step_function=answer)

    judged = steps.add_parser("judge", parents=[common], help="the answers, by the judge")
    judged.add_argument("--judge-workers", type=int, default=10, dest="judge_workers")
    judged.set_defaults(step_function=judge)

    printed = steps.add_parser("report", parents=[common], help="the tables")
    printed.add_argument("--examples", type=int, default=3,
                         help="exchanges to print with their rewritten queries")
    printed.set_defaults(step_function=report)
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_arguments()
    sys.exit(parsed.step_function(parsed))
