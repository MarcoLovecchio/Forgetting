#!/usr/bin/env python3
"""How many calls at once the model server takes before each call slows down.

Two parts, one after the other, each at every concurrency level: a level sends level ×
rounds requests, never more than level at a time.

    consolidation  the consolidation model with the service's settings, on stretches of
                   HaluMem dialogue of about --words words, each a different one so that
                   no prompt is cached
    answer         the answer as the service gives it in the experiments: its prompt and
                   the settings of generate_answer, on the questions of HaluMem, with the
                   latest memory points as the core (CORE_CHARS), the QUESTION_K before
                   them as the memories recalled, and the last two messages of the session

With --part judge it sends instead the calls of HaluMem's judge, with the settings of
eval/.env and without its retries, on the results of a finished run (--version):

    evaluate       the requests of halumem_adapter.py evaluate, its four kinds in the
                   share it sends them: one per memory point (integrity, or update), one
                   per memory the service wrote (accuracy, with the session's dialogue),
                   one per question
    label          the requests of eviction_replay.py label, one per question with
                   evidence, on about as many of the memories written before it as label
                   shows (LABEL_SHOWN)

Per level it prints the median and 90th percentile of the seconds per call, the output
tokens per call, the output tokens per second of one call and of the whole server, and
both against the first level. More workers pay only while the server's throughput grows
with them: when it stays flat, each call gets slower by as much as the calls added, and
the extra workers only queue. With 1 as the first level, "s vs first" is how much slower
a call is at each level than on a robot with one user.

Launch it with nothing else on the server, or read each level as that many calls on top
of what is already running there.

    python3 memory_service/test/load_probe.py --halumem PATH [--model qwen3.5-9b]
                                              [--levels 1 5 10 15] [--rounds 2]
                                              [--answer-rounds 10]
                                              [--part consolidation|answer|both]
                                              [--csv FILE]
    python3 memory_service/test/load_probe.py --halumem PATH --part judge --version RUN
                                              [--levels 1 8 15] [--judge-rounds 4]
                                              [--csv FILE]

--csv also writes one row per part and level, with the time the probe started: the
numbers of each table, for the plots and to set probes of different moments side by side.
"""

import argparse
import csv
import dataclasses
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
for path in (PACKAGE_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import eviction_replay as replay  # noqa: E402
import halumem_adapter as adapter  # noqa: E402
from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import NO_ARCHIVAL_RESULTS, memory_line  # noqa: E402

INSTRUCTION = ("\n\nList every fact about the user in these messages, one per line, "
               "written as a short sentence about the user.")
CORE_CHARS = 400  # the core limit of the experiments
# The core and the memories closest to the evidence mostly fall among the 40 closest.
LABEL_SHOWN = replay.LABEL_CANDIDATES + replay.EVIDENCE_CANDIDATES


def turns(users: Sequence[Dict[str, Any]]) -> Iterator[str]:
    for user in users:
        for session in user["sessions"]:
            for message in session["dialogue"]:
                yield f"{message['role']}: {message['content']}"


def dialogue_prompts(users: Sequence[Dict[str, Any]], words: int, count: int) -> List[str]:
    """`count` prompts of about `words` words each, on consecutive stretches of dialogue."""
    prompts: List[str] = []
    stretch: List[str] = []
    size = 0
    for turn in turns(users):
        stretch.append(turn)
        size += len(turn.split())
        if size >= words:
            prompts.append("\n".join(stretch) + INSTRUCTION)
            stretch, size = [], 0
            if len(prompts) == count:
                return prompts
    return reused(prompts, count, "Not enough dialogue for one prompt")


def reused(found: List[Any], count: int, empty: str) -> List[Any]:
    if not found:
        raise ValueError(empty)
    # A small dataset is reused: the same requests again, later in the probe.
    return [found[index % len(found)] for index in range(count)]


def day(timestamp: str) -> datetime:
    return datetime.strptime(timestamp, adapter.TIME_FORMAT)


def answer_inputs(users: Sequence[Dict[str, Any]], count: int, dates: str
                  ) -> List[Dict[str, Any]]:
    """`count` inputs of the answer prompt, one per HaluMem question, in the dataset's order."""
    found: List[Dict[str, Any]] = []
    for user in users:
        points: List[Tuple[str, datetime]] = []
        for session in user["sessions"]:
            points += [(point["memory_content"], day(point["timestamp"]))
                       for point in session.get("memory_points", [])]
            messages = "\n".join(("Human: " if message["role"] == "user" else "AI: ")
                                 + message["content"] for message in session["dialogue"][-2:])
            core: List[Tuple[str, datetime]] = []
            for point in reversed(points):
                if len("\n".join(content for content, _ in [point] + core)) > CORE_CHARS:
                    break
                core.insert(0, point)
            older = points[:len(points) - len(core)][-adapter.QUESTION_K:]
            recalled = "\n".join(memory_line(content, f"{number:08x}", written)
                                 for number, (content, written) in enumerate(older))
            for question in session.get("questions", []):
                found.append({
                    "input": question["question"],
                    "core_memory": [memory_line(content, updated_at=written)
                                    for content, written in core],
                    "retrieved_memory": recalled or NO_ARCHIVAL_RESULTS,
                    "messages": messages,
                    "dates": dates})
                if len(found) == count:
                    return found
    return reused(found, count, "No question in the dataset")


def evaluation_dialogue(dialogue: Sequence[Dict[str, Any]]) -> str:
    """A session's dialogue as HaluMem's evaluation writes it."""
    lines = []
    for turn in dialogue:
        lines.append(f"[{turn['timestamp']}]{turn['role']}: {turn['content']}")
        if turn["role"] == "assistant":
            lines.append("")
    return "\n".join(lines)


def evaluate_requests(users: Sequence[Dict[str, Any]]) -> List[Tuple[str, Dict[str, str]]]:
    """(kind, fields of its prompt) of every judge call of evaluate on a run's results,
    built as HaluMem's process_user builds them."""
    found: List[Tuple[str, Dict[str, str]]] = []
    for user in users:
        for session in user["sessions"]:
            if session.get("is_generated_qa_session", False):
                continue
            extracted = session["extracted_memories"]
            written = "\n".join(extracted)
            points = session["memory_points"]
            for point in points:
                if point["is_update"] == "True" and point.get("memories_from_system", []):
                    found.append(("update", {
                        "memories": "\n".join(point["memories_from_system"]),
                        "updated_memory": point["memory_content"],
                        "original_memory": "\n".join(point["original_memories"])}))
                elif written.strip():
                    # With nothing written, evaluate scores the point 0 without a call.
                    found.append(("integrity", {"memories": written,
                                                "expected_memory_point": point["memory_content"]}))
            dialogue = evaluation_dialogue(session["dialogue"])
            golden = "\n".join(point["memory_content"] for point in points
                               if point["memory_source"] != "interference")
            found += [("accuracy", {"dialogue": dialogue, "golden_memories": golden,
                                    "candidate_memory": memory}) for memory in extracted]
            found += [("question", {
                "question": qa["question"], "reference_answer": qa["answer"],
                "key_memory_points": "\n".join(item["memory_content"] for item in qa["evidence"]),
                "response": qa["system_response"]}) for qa in session.get("questions", [])]
    return found


def evaluate_templates(tools: Any) -> Dict[str, str]:
    """The prompt of each kind, from HaluMem's eval_tools."""
    return {"integrity": tools.EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY,
            "accuracy": tools.EVALUATION_PROMPT_FOR_MEMORY_ACCURACY,
            "update": tools.EVALUATION_PROMPT_FOR_UPDATE_MEMORY,
            "question": tools.EVALUATION_PROMPT_FOR_QUESTION}


def label_requests(users: Sequence[Dict[str, Any]]
                   ) -> List[Tuple[Dict[str, Any], List[str], str]]:
    """(question, memories shown, user's name) of every question with evidence: the last
    LABEL_SHOWN memories the service wrote up to the question's session."""
    found = []
    for user in users:
        written: List[str] = []
        for session in user["sessions"]:
            written += session.get("extracted_memories", [])
            found += [(question, written[-LABEL_SHOWN:], user["user_name"])
                      for question in session.get("questions", [])
                      if replay.evidence_of(question)]
    return found


def spread(found: List[Any], count: int, empty: str) -> List[Any]:
    """`count` of them evenly spaced, so that each kind keeps its share of the calls."""
    if len(found) <= count:
        return reused(found, count, empty)
    return [found[index * len(found) // count] for index in range(count)]


class Judge:
    """HaluMem's judge as evaluate and label call it, with the settings of eval/.env, but
    without its retries: a call that fails is counted as failed, not sent again."""

    def __init__(self, llms: Any):
        self.llms = llms

    def invoke(self, prompt: str) -> Any:
        response = self.llms.client.chat.completions.create(
            model=self.llms.MODEL, messages=[{"role": "user", "content": prompt}],
            **self.llms.common_params)
        usage = getattr(response, "usage", None)
        return SimpleNamespace(usage_metadata={
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0})


def percentile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def level_stats(seconds: Sequence[float], outputs: Sequence[int], wall: float
                ) -> Dict[str, float]:
    return {
        "median s": statistics.median(seconds),
        "p90 s": percentile(seconds, 0.9),
        "tokens per call": sum(outputs) / len(outputs),
        "tokens/s per call": statistics.mean(out / took for out, took in zip(outputs, seconds)
                                             if took > 0),
        "tokens/s in all": sum(outputs) / wall if wall > 0 else 0.0,
    }


def probe_level(llm: Any, prompts: Sequence[str], level: int) -> Dict[str, Any]:
    """Every prompt sent, `level` at a time: the stats of the calls that answered."""

    def call(prompt: str):
        started = time.perf_counter()
        try:
            response = llm.invoke(prompt)
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"
        usage = getattr(response, "usage_metadata", None) or {}
        return time.perf_counter() - started, int(usage.get("output_tokens", 0))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=level) as pool:
        outcomes = list(pool.map(call, prompts))
    wall = time.perf_counter() - started
    answered = [(took, out) for took, out in outcomes if took is not None]
    errors = [out for took, out in outcomes if took is None]
    found: Dict[str, Any] = {"level": level, "calls": len(answered), "errors": errors,
                             "wall s": wall}
    if answered:
        found.update(level_stats([took for took, _ in answered],
                                 [out for _, out in answered], wall))
    return found


def probe_levels(llm: Any, requests: Sequence[Any], levels: Sequence[int], rounds: int
                 ) -> List[Dict[str, Any]]:
    """Every level in turn, each on requests of its own."""
    found, offset = [], 0
    for level in levels:
        count = level * rounds
        print(f"Level {level}: {count} requests...", flush=True)
        found.append(probe_level(llm, requests[offset:offset + count], level))
        offset += count
    return found


def print_levels(title: str, levels: Sequence[Dict[str, Any]]) -> None:
    first = next((found for found in levels if found["calls"]), None)
    print(f"\n{title}")
    print(f"{'level':>5} {'calls':>5} {'median s':>9} {'p90 s':>7} {'tok/call':>9} "
          f"{'tok/s 1 call':>13} {'tok/s all':>10} {'all vs first':>13} {'s vs first':>11}")
    for found in levels:
        if not found["calls"]:
            print(f"{found['level']:>5} {0:>5}  every call failed: {found['errors'][0]}")
            continue
        print(f"{found['level']:>5} {found['calls']:>5} {found['median s']:>9.1f} "
              f"{found['p90 s']:>7.1f} {found['tokens per call']:>9.0f} "
              f"{found['tokens/s per call']:>13.1f} {found['tokens/s in all']:>10.1f} "
              f"{found['tokens/s in all'] / first['tokens/s in all']:>12.2f}x "
              f"{found['median s'] / first['median s']:>10.2f}x")
        if found["errors"]:
            print(f"      {len(found['errors'])} failed, the first: {found['errors'][0]}")


CSV_HEADER = ["started", "part", "model", "level", "calls", "errors", "wall s", "median s",
              "p90 s", "tokens per call", "tokens/s per call", "tokens/s in all",
              "all vs first", "s vs first"]


def csv_rows(started: str, part: str, model: str, levels: Sequence[Dict[str, Any]]
             ) -> List[List[Any]]:
    """One row per level, as print_levels shows it; empty cells for a level that failed."""
    first = next((found for found in levels if found["calls"]), None)
    rows = []
    for found in levels:
        row = [started, part, model, found["level"], found["calls"], len(found["errors"]),
               found["wall s"]]
        if found["calls"]:
            row += [found["median s"], found["p90 s"], found["tokens per call"],
                    found["tokens/s per call"], found["tokens/s in all"],
                    found["tokens/s in all"] / first["tokens/s in all"],
                    found["median s"] / first["median s"]]
        else:
            row += [""] * 7
        rows.append(row)
    return rows


Parts = List[Tuple[str, List[Dict[str, Any]]]]


def service_parts(root: str, arguments: argparse.Namespace) -> Tuple[str, Parts]:
    """The service model's calls: consolidation, answer or both."""
    # Here and not on top: it loads the workspace's .env, which the judge must not see.
    from live_model import chat_model_unavailable

    config = MemoryConfig.from_environment()
    if arguments.model:
        config = dataclasses.replace(
            config, llm_config={**config.llm_config, "model_name": arguments.model})
    backends.reset()
    backends.configure(config=config)
    reason = chat_model_unavailable()
    if reason:
        raise SystemExit(f"The model does not answer: {reason}")

    # Here too: it builds the graph.
    from memory_service.memory_manager_llm import ANSWER_PROMPT, DATES_NOTE

    users = adapter.dataset_users(root)
    model = config.llm_config.get("model_name")
    parts: Parts = []
    if arguments.part in ("consolidation", "both"):
        needed = sum(level * arguments.rounds for level in arguments.levels)
        print(f"Model {model}, consolidation settings, {needed} requests of about "
              f"{arguments.words} words of dialogue.")
        parts.append(("Consolidation", probe_levels(
            backends.get_llm("consolidation"), dialogue_prompts(users, arguments.words, needed),
            arguments.levels, arguments.rounds)))
    if arguments.part in ("answer", "both"):
        needed = sum(level * arguments.answer_rounds for level in arguments.levels)
        print(f"Model {model}, answer settings, {needed} questions.")
        parts.append(("Answer", probe_levels(
            ANSWER_PROMPT | backends.get_llm("generate_answer"),
            answer_inputs(users, needed, DATES_NOTE), arguments.levels,
            arguments.answer_rounds)))
    return model, parts


def judge_parts(root: str, arguments: argparse.Namespace) -> Tuple[str, Parts]:
    """The judge's calls, evaluate's and label's, on the results of a finished run."""
    results = os.path.join(adapter.run_folder(arguments), f"{adapter.FRAME}_eval_results.jsonl")
    if not os.path.exists(results):
        raise SystemExit(f"No results in {os.path.dirname(results)}: launch "
                         "halumem_adapter.py run first.")
    users = adapter.load_users(results)

    # From HaluMem's folder, as evaluate and label do: llms.py reads eval/.env from there.
    eval_dir = os.path.join(root, "eval")
    os.chdir(eval_dir)
    sys.path.insert(0, eval_dir)
    import eval_tools
    import llms

    judge = Judge(llms)
    templates = evaluate_templates(eval_tools)
    levels, rounds = arguments.levels, arguments.judge_rounds
    needed = sum(level * rounds for level in levels)
    print(f"Judge {llms.MODEL} (eval/.env), {needed} requests per part, on the results of "
          f"{arguments.version}.")
    evaluate = [templates[kind].format(**fields) for kind, fields in
                spread(evaluate_requests(users), needed, "No evaluation request in the run")]
    label = [replay.judge_prompt(question, [(None, memory, 0) for memory in memories], name)
             for question, memories, name in
             spread(label_requests(users), needed, "No question with evidence in the run")]
    parts: Parts = [("Evaluate", probe_levels(judge, evaluate, levels, rounds)),
                    ("Label", probe_levels(judge, label, levels, rounds))]
    return llms.MODEL, parts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="How many calls at once the model server "
                                                 "takes before each call slows down.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--model", help="the Qwen model, instead of the one in LLM_CONFIG")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 5, 10, 15],
                        help="calls at the same time, one level after the other")
    parser.add_argument("--rounds", type=int, default=2,
                        help="requests per level, as a multiple of the level")
    parser.add_argument("--words", type=int, default=2500,
                        help="words of dialogue in each prompt, about a consolidation call")
    parser.add_argument("--answer-rounds", type=int, default=10, dest="answer_rounds",
                        help="answer requests per level, as a multiple of the level: they are "
                             "short, so more of them")
    parser.add_argument("--part", choices=("consolidation", "answer", "both", "judge"),
                        default="both",
                        help="which calls to send: the service model's, or HaluMem's judge's")
    parser.add_argument("--version", type=adapter.version_name,
                        help="with --part judge, the finished run the requests are built on")
    parser.add_argument("--judge-rounds", type=int, default=4, dest="judge_rounds",
                        help="judge requests per level and part, as a multiple of the level")
    parser.add_argument("--csv", help="also write every level, part by part, to this file")
    arguments = parser.parse_args(argv)
    if min(arguments.levels) < 1 or min(arguments.rounds, arguments.answer_rounds,
                                        arguments.judge_rounds) < 1:
        parser.error("--levels and the rounds must be at least 1")
    if arguments.part == "judge" and not arguments.version:
        parser.error("--part judge needs --version, a finished run")

    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    # The judge's part changes folder.
    path = os.path.abspath(arguments.csv) if arguments.csv else None
    started = datetime.now().isoformat(timespec="seconds")
    if arguments.part == "judge":
        model, parts = judge_parts(root, arguments)
    else:
        model, parts = service_parts(root, arguments)
    for title, levels in parts:
        print_levels(title, levels)
    print("\nA level is worth it while 'all vs first' grows with it; when it stops, the "
          "calls added only queue and 's vs first' grows instead.")
    if path:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            for title, levels in parts:
                writer.writerows(csv_rows(started, title.lower(), model, levels))
        print(f"\nEvery level, part by part: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
