#!/usr/bin/env python3
"""The memory service on HaluMem-Medium: replay the dataset, then let HaluMem judge it.

Every user is replayed from an empty memory and leaves what HaluMem's evaluation.py
reads: the memories extracted in each session, the memories found for every update,
the answer to every question.

    run        replays the dialogues through the real agent, with the models on the
               cluster and an archive on disk for each user; it never calls the judge
    evaluate   runs HaluMem's evaluation.py on the result, with the judge of eval/.env

A session goes like this:

- exchange by exchange, as in production: both messages, then an insert. The clock is
  set to the timestamp of the exchange, so the ages the eviction sees are the dataset's;
- count_candidate_search is on, because here nothing calls get_memory on every message;
- at the end the last exchange is still in the window: flush_window consolidates it,
  so it is scored in its own session, and the next session starts with no messages;
- the questions go through the retrieve branch: the answer is the service's. They leave
  `messages` right after, so they are never consolidated, but like any search they
  count in n_retrieve.

The dataset calls people by name and the service calls them "the user": neutral rewrites
what goes to the service, personalized what goes to the judge.

What the experiments fix is set here whatever the defaults of config.py say: see
run_config. What they vary comes from the command line.

A user is restarted from scratch if it did not finish, and skipped if it did, so a run
can be launched again with the same --version. The output of the agent goes to
logs/<uuid>.log, every operation with the exchange it belongs to to the result file.

    python3 memory_service/test/halumem_adapter.py run --halumem PATH --version NAME
    python3 memory_service/test/halumem_adapter.py evaluate --halumem PATH --version NAME

Results are in PATH/eval/results/memory_service-NAME.
"""

import argparse
import contextlib
import copy
import dataclasses
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from memory_service import backends  # noqa: E402
from memory_service.config import NODE_SAMPLING, MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    OperationLogEntry,
    search_archive,
    serialize_core_memory_for_prompt,
)
from memory_service.memory_manager_llm import (  # noqa: E402
    NODE_STATS,
    MemoryAgent,
    reset_node_stats,
)

FRAME = "memory_service"
DATASET = os.path.join("data", "HaluMem-Medium.jsonl")
TIME_FORMAT = "%b %d, %Y, %H:%M:%S"
EXTRACTING_OPERATIONS = ("create", "redundant", "update")
UPDATE_SEARCH_K = 10
ATTEMPTS = 5
RETRY_SECONDS = 30


def dataset_time(text: str) -> datetime:
    return datetime.strptime(text, TIME_FORMAT)


def user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if not match:
        raise ValueError("No name found in the persona")
    return match.group(1).strip()


def neutral(text: str, name: str) -> str:
    """The dataset's text as the service would write it: the person is "the user".

    Most questions use the full name, the others one part of it.
    """
    parts = sorted({name, *name.split()}, key=len, reverse=True)
    return re.sub(r"\b(?:" + "|".join(map(re.escape, parts)) + r")\b", "the user", text)


def personalized(text: str, name: str) -> str:
    """The service's text as the judge expects it: "the user" is the person."""
    return re.sub(r"\b[Tt]he user\b", name, text)


def with_retries(call: Callable[[], Any]) -> Any:
    """A call to the cluster, again after a failure: a long run must survive a hiccup."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return call()
        except Exception as error:
            if attempt == ATTEMPTS:
                raise
            print(f"\tAttempt {attempt} of {ATTEMPTS} failed: {error!r}")
            time.sleep(RETRY_SECONDS * attempt)


def insert(agent: MemoryAgent) -> None:
    with_retries(lambda: agent.run_memory_agent("insert"))


def flush_window(agent: MemoryAgent) -> None:
    """Consolidate the exchange still in the window and leave no messages.

    With a window of 1 the user's message is the one that leaves: the facts come only
    from there, so the assistant's reply is not missed.
    """
    window = agent.config
    agent.config = dataclasses.replace(window, maximum_historical_messages=1)
    try:
        insert(agent)
    finally:
        agent.config = window
    agent.state["messages"] = []


def logged(entries: Sequence[OperationLogEntry], exchange: int) -> List[Dict[str, Any]]:
    return [{"exchange": exchange, **entry.model_dump(mode="json")} for entry in entries]


def extracted_memories(operations: Sequence[Dict[str, Any]], name: str) -> List[str]:
    """What the session stored, or confirmed it already had, once each."""
    found = (personalized(operation["content"], name) for operation in operations
             if operation["op_type"] in EXTRACTING_OPERATIONS and operation["content"])
    return list(dict.fromkeys(found))


def memories_about(agent: MemoryAgent, text: str, name: str) -> List[str]:
    """What the memory holds on a fact: the core, plus the closest archived ones.

    search_archive, because the evaluation is not a retrieval.
    """
    core = serialize_core_memory_for_prompt(agent.state["core_memory"])
    found = [content for _, content in search_archive(neutral(text, name), k=UPDATE_SEARCH_K)]
    return [personalized(memory, name) for memory in core + found]


def answer_question(agent: MemoryAgent, question: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The question through the retrieve branch, leaving no trace in the messages."""
    agent.up_to_date = False
    asked = neutral(question["question"], name)
    started = time.time()
    state = with_retries(lambda: agent.run_memory_agent("retrieve", query=asked))
    elapsed = (time.time() - started) * 1000
    answer = str(state["messages"][-1].content).strip()
    agent.state["messages"] = []

    answered = copy.deepcopy(question)
    answered["asked"] = asked
    answered["context"] = "\n".join(
        serialize_core_memory_for_prompt(state["core_memory"]) + [state["retrieved_memory"]])
    answered["system_response"] = personalized(answer, name)
    answered["search_duration_ms"] = elapsed
    return answered


def active_in_archive() -> int:
    stored = backends.get_vector_store().get(where={"status": "active"}) or {}
    return len(stored.get("ids") or [])


def replay_session(agent: MemoryAgent, session: Dict[str, Any], name: str) -> Dict[str, Any]:
    """One session: the dialogue, then what the evaluation asks of the memory."""
    started = time.time()
    dialogue = session["dialogue"]
    operations: List[Dict[str, Any]] = []
    exchange = 0
    for exchange, (said, replied) in enumerate(zip(dialogue[::2], dialogue[1::2])):
        backends.set_clock(dataset_time(said["timestamp"]))
        agent.append_message(said["content"], "user")
        agent.append_message(replied["content"], "assistant")
        insert(agent)
        operations += logged(agent.last_operations(), exchange)
    backends.set_clock(dataset_time(session["end_time"]))
    flush_window(agent)
    operations += logged(agent.last_operations(), exchange)
    elapsed = (time.time() - started) * 1000

    points = copy.deepcopy(session["memory_points"])
    for point in points:
        if point["is_update"] == "True" and point["original_memories"]:
            point["memories_from_system"] = memories_about(agent, point["memory_content"], name)

    replayed = {
        "memory_points": points,
        "dialogue": dialogue,
        "extracted_memories": extracted_memories(operations, name),
        "add_dialogue_duration_ms": elapsed,
        "operations": operations,
        "active_core": len(serialize_core_memory_for_prompt(agent.state["core_memory"])),
        "active_archive": active_in_archive(),
    }
    if "questions" in session:
        replayed["questions"] = [answer_question(agent, question, name)
                                 for question in session["questions"]]
    return replayed


def replay_user(agent: MemoryAgent, user: Dict[str, Any],
                progress: Callable[[str], None] = lambda text: None) -> Dict[str, Any]:
    """Every session of a user, in order, on the same memory."""
    name = user_name(user["persona_info"])
    sessions = []
    for index, session in enumerate(user["sessions"], 1):
        sessions.append(replay_session(agent, session, name))
        progress(f"{name}: session {index} of {len(user['sessions'])}, "
                 f"{sessions[-1]['active_archive']} active in the archive")
    return {"uuid": user["uuid"], "user_name": name, "sessions": sessions,
            "node_stats": {node: dict(stats) for node, stats in NODE_STATS.items()}}


def run_config(arguments: argparse.Namespace) -> MemoryConfig:
    """The configuration of the experiments, on top of the environment's models."""
    config = MemoryConfig.from_environment()
    varying = {name: value for name, value in (
        ("consolidation_utility", arguments.utility),
        ("eviction_combination", arguments.combination),
        ("core_memory_limit", arguments.core_limit),
        ("archive_memory_limit", arguments.archive_limit),
        ("prune_fraction", arguments.prune_fraction),
    ) if value is not None}
    if arguments.model:
        varying["llm_config"] = {**config.llm_config, "model_name": arguments.model}
    return dataclasses.replace(
        config,
        maximum_historical_messages=2,
        retrieval_mode="always",
        generate_answer=True,
        track_used=False,
        eviction=True,
        eviction_time_decay_field="updated_at",
        eviction_time_decay_function="power",
        count_candidate_search=True,
        **varying,
    )


def git_commit() -> str:
    """The commit of the code, with a '+' if the tree has uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PACKAGE_ROOT,
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PACKAGE_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return commit + ("+" if commit and dirty else "")


def check_manifest(run_dir: str, config: MemoryConfig) -> None:
    """Write what the run is, or refuse to mix it with a different one."""
    manifest = {"config": dataclasses.asdict(config), "node_sampling": NODE_SAMPLING,
                "commit": git_commit()}
    path = os.path.join(run_dir, "run.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise SystemExit(f"{run_dir} holds a run with another configuration or "
                                 "another commit: choose another --version.")
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def result_path(run_dir: str, uuid: str) -> str:
    return os.path.join(run_dir, "tmp", f"{uuid}.json")


def process_user(user: Dict[str, Any], config: MemoryConfig, run_dir: str) -> Tuple[str, str]:
    """A user in a process of its own, on an archive of its own: (uuid, outcome)."""
    uuid = user["uuid"]
    archive = os.path.join(run_dir, "archives", uuid)
    shutil.rmtree(archive, ignore_errors=True)
    user_config = dataclasses.replace(config, chroma_path=archive, collection_name="halumem")

    def progress(text: str) -> None:
        print(text, file=sys.__stdout__, flush=True)

    try:
        with open(os.path.join(run_dir, "logs", f"{uuid}.log"), "w", encoding="utf-8",
                  buffering=1) as log, contextlib.redirect_stdout(log):
            backends.reset()
            backends.configure(config=user_config)
            MemoryAgent.reset_instance()
            reset_node_stats()
            replayed = replay_user(MemoryAgent(config=user_config), user, progress)
        partial = result_path(run_dir, uuid) + ".partial"
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump(replayed, handle, ensure_ascii=False)
        os.replace(partial, result_path(run_dir, uuid))
        return uuid, "ok"
    except Exception:
        with open(os.path.join(run_dir, "tmp", f"{uuid}_error.log"), "w",
                  encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        return uuid, "failed"


def load_users(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    users = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                users.append(json.loads(line))
            if limit and len(users) == limit:
                break
    return users


def merge_results(users: Sequence[Dict[str, Any]], run_dir: str) -> int:
    """The file evaluation.py reads, in the dataset's order: how many users it holds."""
    merged = 0
    path = os.path.join(run_dir, f"{FRAME}_eval_results.jsonl")
    with open(path, "w", encoding="utf-8") as output:
        for user in users:
            finished = result_path(run_dir, user["uuid"])
            if os.path.exists(finished):
                with open(finished, encoding="utf-8") as handle:
                    output.write(json.dumps(json.load(handle), ensure_ascii=False) + "\n")
                merged += 1
    return merged


def run_folder(arguments: argparse.Namespace) -> str:
    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    return os.path.join(root, "eval", "results", f"{FRAME}-{arguments.version}")


def run(arguments: argparse.Namespace) -> int:
    """Replay the users that are not done yet, then write the results file."""
    # Here and not on top: it loads the workspace's .env, which evaluate must not see.
    from live_model import live_stack_unavailable

    config = run_config(arguments)
    backends.reset()
    backends.configure(config=config)
    reason = live_stack_unavailable()
    if reason:
        raise SystemExit(f"The models do not answer: {reason}")

    run_dir = run_folder(arguments)
    for folder in ("tmp", "logs", "archives"):
        os.makedirs(os.path.join(run_dir, folder), exist_ok=True)
    check_manifest(run_dir, config)

    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    users = load_users(os.path.join(root, DATASET), arguments.users)
    pending = [user for user in users
               if not os.path.exists(result_path(run_dir, user["uuid"]))]
    print(f"{len(users)} users, {len(users) - len(pending)} already done, "
          f"{arguments.workers} at a time", flush=True)

    # Spawn: a forked process would inherit the parent's models and clients.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=arguments.workers, mp_context=context) as pool:
        futures = [pool.submit(process_user, user, config, run_dir) for user in pending]
        for future in as_completed(futures):
            uuid, outcome = future.result()
            print(f"User {uuid}: {outcome}", flush=True)

    merged = merge_results(users, run_dir)
    print(f"{merged} of {len(users)} users in {run_dir}", flush=True)
    return 0 if merged == len(users) else 1


def evaluate(arguments: argparse.Namespace) -> int:
    """HaluMem's evaluation, from its own folder: that is where it finds .env and results."""
    run_dir = run_folder(arguments)
    results = os.path.join(run_dir, f"{FRAME}_eval_results.jsonl")
    if not os.path.exists(results):
        raise SystemExit(f"No results in {run_dir}: launch run first.")
    users = len(load_users(results))

    eval_dir = os.path.dirname(os.path.dirname(run_dir))
    os.chdir(eval_dir)
    sys.path.insert(0, eval_dir)
    import evaluation

    evaluation.main(FRAME, arguments.version, user_num=users,
                    max_workers=arguments.judge_workers)
    return 0


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The memory service on HaluMem-Medium.")
    steps = parser.add_subparsers(dest="step", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    common.add_argument("--version", required=True, help="the name of this run")

    replay = steps.add_parser("run", parents=[common], help="replay the dataset")
    replay.add_argument("--users", type=int, help="only the first users, for a pilot")
    replay.add_argument("--workers", type=int, default=1, help="users at the same time")
    replay.add_argument("--utility", choices=("none", "single", "need_gain"))
    replay.add_argument("--combination", choices=("weighted_sum", "product"))
    replay.add_argument("--core-limit", type=int, dest="core_limit")
    replay.add_argument("--archive-limit", type=int, dest="archive_limit")
    replay.add_argument("--prune-fraction", type=float, dest="prune_fraction")
    replay.add_argument("--model", help="the Qwen model, instead of the one in LLM_CONFIG")
    replay.set_defaults(step_function=run)

    judge = steps.add_parser("evaluate", parents=[common], help="judge a finished run")
    judge.add_argument("--judge-workers", type=int, default=10, dest="judge_workers",
                       help="judge calls at the same time, per user")
    judge.set_defaults(step_function=evaluate)
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_arguments()
    sys.exit(parsed.step_function(parsed))
