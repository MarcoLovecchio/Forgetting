#!/usr/bin/env python3
"""Where each answer of a trace failed: the memory was not stored, not found, or found.

The split of EvalMem (arXiv:2609.22231), on what a trace already holds, after
halumem_adapter.py evaluate and eviction_replay.py label:

    stored      label found at least one active memory that holds the information,
                among the core, the archived memories closest to the question and those
                closest to its evidence; none means never extracted, or lost in an update
    found       the needed memories that were in the answer's context: the core at that
                question, as the replay rebuilds it, and the ids the search returned
    answered    HaluMem's verdict on the answer

A question with several needed memories is found in part when only some were in context.
EvalMem's third examiner, the answer given again from the gold evidence, is not here: a
wrong answer with the memory in context may come from the answer or from a memory that
holds only part of the fact. A question without evidence (Memory Boundary) can only be
answered right by abstaining, and is counted apart. A correct answer with no needed memory
in context is one the labels cannot explain: its share says how far to trust them.

Counts are pooled over the users. Only traces: a run with eviction has no labels, and the
replay gives its memory side.

    python3 memory_service/test/error_analysis.py --halumem PATH --versions TRACE [TRACE ...]
                                                  [--csv FILE]
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import eviction_replay as replay
import halumem_adapter as adapter

RESULT_TYPES = ("Correct", "Hallucination", "Omission")
CATEGORIES = ("not stored", "not found", "found in part", "found")
IN_CONTEXT = ("found in part", "found")
NO_EVIDENCE = "no evidence"

Row = Dict[str, Any]


def verdicts(judged: Dict[str, Any]) -> Dict[Tuple[int, str], Optional[str]]:
    """(session, question) -> HaluMem's verdict: a question is asked once per session."""
    return {(record["ssession_id"], record["question"]): record.get("result_type")
            for record in judged.get("question_answering_records", [])}


def core_at_questions(config: Any, result: Dict[str, Any], user: Dict[str, Any]
                      ) -> Tuple[Dict[str, set], Optional[float]]:
    """The core ids at every question, and the split agreement with the trace."""
    cores: Dict[str, set] = {}

    def keep(state: replay.Replay, event: replay.Question, ranked: List[str]) -> None:
        cores[event.key] = set(state.core)

    unlimited = replay.Replay(config, None, None,
                              lambda key, memories: [item_id for item_id, _ in memories],
                              on_question=keep).run(replay.trace_events(result, user))
    return cores, replay.trace_checks(unlimited, result)["split agreement"]


def category(needed: Sequence[str], in_context: set) -> str:
    if not needed:
        return "not stored"
    reached = sum(1 for item_id in needed if item_id in in_context)
    if not reached:
        return "not found"
    return "found" if reached == len(needed) else "found in part"


def user_rows(result: Dict[str, Any], user: Dict[str, Any], labels: Dict[str, Any],
              judged: Dict[str, Any], config: Any) -> Tuple[List[Row], Counter, Optional[float]]:
    """One row per question that was judged and labelled; what was left out, and why."""
    cores, agreement = core_at_questions(config, result, user)
    given = verdicts(judged)
    rows, left_out = [], Counter()
    for index, session in enumerate(result["sessions"]):
        for number, question in enumerate(session.get("questions", [])):
            key = replay.question_key(index, number)
            verdict = given.get((index, question["question"]))
            if verdict not in RESULT_TYPES:
                left_out["not judged"] += 1
                continue
            row = {"uuid": result["uuid"], "question": key,
                   "type": question.get("question_type", ""), "verdict": verdict}
            if not replay.evidence_of(question):
                rows.append({**row, "category": NO_EVIDENCE})
                continue
            outcome = labels.get(key)
            if not outcome or "error" in outcome or "needed" not in outcome:
                left_out["not labelled"] += 1
                continue
            in_context = cores.get(key, set()) | set(
                replay.CONTEXT_ID.findall(question.get("context", "")))
            rows.append({**row, "category": category(outcome["needed"], in_context)})
    return rows, left_out, agreement


def load_version(root: str, name: str) -> Tuple[List[Row], Counter, List[float], int]:
    """Every row of a trace, what was left out, the split agreements, the users read."""
    run_dir = os.path.join(root, "eval", "results", f"{adapter.FRAME}-{name}")
    if not os.path.exists(os.path.join(run_dir, "run.json")):
        raise SystemExit(f"No run in {run_dir}.")
    config = replay.trace_config(run_dir)
    users = {user["uuid"]: user for user in adapter.dataset_users(root)}
    rows: List[Row] = []
    left_out: Counter = Counter()
    agreements: List[float] = []
    read = 0
    for result in replay.trace_results(run_dir):
        judgments = os.path.join(run_dir, "tmp2", f"{result['uuid']}.json")
        labels = replay.load_labels(run_dir, result["uuid"])["labels"]
        if not os.path.exists(judgments) or not labels:
            left_out["users not judged or not labelled"] += 1
            continue
        with open(judgments, encoding="utf-8") as handle:
            judged = json.load(handle)
        found, missing, agreement = user_rows(result, users[result["uuid"]], labels, judged,
                                              config)
        rows += found
        left_out += missing
        if agreement is not None:
            agreements.append(agreement)
        read += 1
    return rows, left_out, agreements, read


def share(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "    -"


def wrong(rows: Sequence[Row]) -> List[Row]:
    return [row for row in rows if row["verdict"] != "Correct"]


def layers(rows: Sequence[Row]) -> Dict[str, int]:
    """The wrong answers with evidence, by where the memory failed."""
    failed = wrong([row for row in rows if row["category"] != NO_EVIDENCE])
    counts = Counter(row["category"] for row in failed)
    return {"wrong": len(failed), "not stored": counts["not stored"],
            "not found": counts["not found"],
            "in context": counts["found in part"] + counts["found"],
            "in part": counts["found in part"]}


def print_version(name: str, rows: Sequence[Row], left_out: Counter,
                  agreements: Sequence[float], read: int) -> None:
    evidence = [row for row in rows if row["category"] != NO_EVIDENCE]
    without = [row for row in rows if row["category"] == NO_EVIDENCE]
    lowest = f"{100 * min(agreements):.1f}%" if agreements else "-"
    print(f"\n{name}: {read} users, {len(evidence)} questions with evidence and "
          f"{len(without)} without, judged and labelled; left out: "
          + (", ".join(f"{count} {why}" for why, count in left_out.items()) or "none")
          + f"; split agreement with the trace, lowest user: {lowest}")

    print(f"\n  {'with evidence':<16}{'questions':>10}{'correct':>9}{'halluc.':>9}"
          f"{'omitted':>9}")
    for name_ in CATEGORIES:
        group = [row for row in evidence if row["category"] == name_]
        counts = Counter(row["verdict"] for row in group)
        print(f"  {name_:<16}{len(group):>10}" + "".join(
            f"{share(counts[verdict], len(group)):>9}" for verdict in RESULT_TYPES))
    counts = Counter(row["verdict"] for row in without)
    print(f"  {'no evidence':<16}{len(without):>10}" + "".join(
        f"{share(counts[verdict], len(without)):>9}" for verdict in RESULT_TYPES))

    found = layers(rows)
    print(f"\n  The {found['wrong']} wrong answers with evidence: not stored "
          f"{share(found['not stored'], found['wrong']).strip()}, not found "
          f"{share(found['not found'], found['wrong']).strip()}, in context "
          f"{share(found['in context'], found['wrong']).strip()} (in part "
          f"{share(found['in part'], found['wrong']).strip()})")
    print(f"\n  {'wrong answers by type':<32}{'wrong':>7}{'not stored':>12}{'not found':>11}"
          f"{'in context':>12}")
    for kind in sorted({row["type"] for row in evidence}):
        found = layers([row for row in evidence if row["type"] == kind])
        print(f"  {kind:<32}{found['wrong']:>7}{share(found['not stored'], found['wrong']):>12}"
              f"{share(found['not found'], found['wrong']):>11}"
              f"{share(found['in context'], found['wrong']):>12}")
    unexplained = [row for row in evidence if row["verdict"] == "Correct"
                   and row["category"] in ("not stored", "not found")]
    correct = sum(1 for row in evidence if row["verdict"] == "Correct")
    print(f"\n  Correct with no needed memory in context: {len(unexplained)} of {correct} "
          f"correct answers ({share(len(unexplained), correct).strip()}), answers the labels "
          "cannot explain")


def csv_rows(name: str, rows: Sequence[Row]) -> List[List[Any]]:
    """version, question type, category, verdict, count."""
    counts = Counter((row["type"], row["category"], row["verdict"]) for row in rows)
    return [[name, kind, group, verdict, count]
            for (kind, group, verdict), count in sorted(counts.items())]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Where each answer of a trace failed: not "
                                                 "stored, not found, or found.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--versions", nargs="+", required=True, type=adapter.version_name,
                        help="traces already evaluated and labelled")
    parser.add_argument("--csv", help="also write the counts, version by version, to this file")
    arguments = parser.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    found = []
    for name in arguments.versions:
        rows, left_out, agreements, read = load_version(root, name)
        print_version(name, rows, left_out, agreements, read)
        found += csv_rows(name, rows)
    if arguments.csv:
        with open(arguments.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["version", "question_type", "category", "verdict", "count"])
            writer.writerows(found)
        print(f"\nThe counts, version by version: {arguments.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
