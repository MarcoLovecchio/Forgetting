#!/usr/bin/env python3
"""How faithfully consolidation turns the dialogue into memory, and what it costs the robot.

From one or more runs of halumem_adapter.py; the judged sections after its evaluate:

    fidelity       HaluMem's judgments: recall and weighted recall of the memory points,
                   the interference points kept out, target and weighted accuracy of what
                   was extracted, their F1, the updates correct, hallucinated, omitted
    answers        the answers correct, hallucinated and omitted, and correct by question
                   type: what the whole service gives back, consolidation to answer
    by type        recall and updates correct of persona, event and relationship points
    by session     recall, target accuracy and updates correct by the session's place in
                   the user's history: whether consolidation holds as the memory grows
    operations     per exchange, the creates, updates, redundants and deletes; the share
                   of redundants that brought an archived memory back to the core; the
                   exchanges that wrote nothing; the memories written per memory point
    cost           the consolidation calls: per exchange, seconds and tokens per call,
                   their share of the time the dialogue took
    answer cost    the answer calls, seconds and tokens each, and per question the ms of
                   the archive search and of the whole retrieve branch: what the user
                   waits for

Every figure is the mean over users of the user's own figure; a user with nothing to
count in a cell stays out of it. A judgment HaluMem could not use is left out, as in its
(valid) scores. With several versions only the users all of them hold are read, and each
version after the first is set beside it: the mean difference user by user, with its 95%
bootstrap interval over the users. Seconds depend on how many users ran at once: compare
runs launched with the same --workers.

    python3 memory_service/test/consolidation_report.py --halumem PATH --versions A [B ...]
                                                        [--csv FILE]
"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import halumem_adapter as adapter

CONSOLIDATION_OPERATIONS = ("create", "update", "redundant", "delete")
UPDATE_LABELS = ("Correct", "Hallucination", "Omission", "Other")
ANSWER_LABELS = ("Correct", "Hallucination", "Omission")
QUESTION_TYPES = ("Basic Fact Recall", "Memory Boundary", "Memory Conflict",
                  "Generalization & Application", "Multi-hop Inference", "Dynamic Update")
TYPES = (("persona", "Persona Memory"), ("event", "Event Memory"),
         ("relationship", "Relationship Memory"))
SESSION_BUCKETS = ((1, 10), (11, 25), (26, 50), (51, None))
BOOTSTRAP_SAMPLES = 1000
LABEL_WIDTH = 34
SHARE, NUMBER, TOKENS, MILLISECONDS = "share", "number", "tokens", "ms"

Record = Dict[str, Any]
Value = Tuple[str, str, Optional[float]]


class Judged(NamedTuple):
    """The usable judgments of a user."""

    points: List[Record]
    interference: List[Record]
    extracted: List[Record]
    updates: List[Record]
    answers: List[Record]


def judged_records(judged: Dict[str, Any]) -> Judged:
    integrity = [record for record in judged.get("memory_integrity_records", [])
                 if record.get("memory_integrity_score") is not None]
    return Judged(
        [record for record in integrity if record["memory_source"] != "interference"],
        [record for record in integrity if record["memory_source"] == "interference"],
        [record for record in judged.get("memory_accuracy_records", [])
         if record.get("memory_accuracy_score") is not None],
        [record for record in judged.get("memory_update_records", [])
         if record.get("memory_update_type") in UPDATE_LABELS],
        [record for record in judged.get("question_answering_records", [])
         if record.get("result_type") in ANSWER_LABELS])


def ratio(part: float, whole: float) -> Optional[float]:
    return part / whole if whole else None


def share(records: Sequence[Record], hit: Callable[[Record], bool]) -> Optional[float]:
    return ratio(sum(1 for record in records if hit(record)), len(records))


def recall(points: Sequence[Record]) -> Optional[float]:
    return share(points, lambda record: record["memory_integrity_score"] == 2)


def weighted_recall(points: Sequence[Record]) -> Optional[float]:
    return ratio(sum(0.5 * record["memory_integrity_score"] * record["importance"]
                     for record in points),
                 sum(record["importance"] for record in points))


def accuracy(extracted: Sequence[Record]) -> Optional[float]:
    return ratio(sum(0.5 * record["memory_accuracy_score"] for record in extracted),
                 len(extracted))


def is_target(record: Record) -> bool:
    return str(record.get("is_included_in_golden_memories")).lower() == "true"


def updated(updates: Sequence[Record], label: str) -> Optional[float]:
    return share(updates, lambda record: record["memory_update_type"] == label)


def f1(precision: Optional[float], found: Optional[float]) -> Optional[float]:
    if precision is None or found is None:
        return None
    return 2 * precision * found / (precision + found) if precision + found else 0.0


def fidelity(judged: Judged) -> List[Value]:
    target = accuracy([record for record in judged.extracted if is_target(record)])
    found = recall(judged.points)
    return [
        ("recall", SHARE, found),
        ("weighted recall", SHARE, weighted_recall(judged.points)),
        ("interference kept out", SHARE,
         share(judged.interference, lambda record: record["memory_integrity_score"] == 0)),
        ("target accuracy", SHARE, target),
        ("weighted accuracy", SHARE, accuracy(judged.extracted)),
        ("F1", SHARE, f1(target, found)),
        ("updates correct", SHARE, updated(judged.updates, "Correct")),
        ("updates hallucinated", SHARE, updated(judged.updates, "Hallucination")),
        ("updates omitted", SHARE, updated(judged.updates, "Omission")),
    ]


def group(name: str, judged: Judged, member: Callable[[Record], bool],
          extracted: bool = False) -> List[Value]:
    """Recall and updates correct of the records of a group; with `extracted` the target
    accuracy too, which only the session can group: an extracted memory has no type."""
    values = [(f"recall, {name}", SHARE,
               recall([record for record in judged.points if member(record)]))]
    if extracted:
        values.append((f"target accuracy, {name}", SHARE,
                       accuracy([record for record in judged.extracted
                                 if is_target(record) and member(record)])))
    values.append((f"updates correct, {name}", SHARE,
                   updated([record for record in judged.updates if member(record)],
                           "Correct")))
    return values


def by_type(judged: Judged) -> List[Value]:
    values = []
    for name, memory_type in TYPES:
        values += group(name, judged,
                        lambda record, memory_type=memory_type:
                        record["memory_type"] == memory_type)
    return values


def session_bucket(index: int) -> str:
    """The bucket of a session, from HaluMem's ssession_id, which counts from 0."""
    number = index + 1
    for low, high in SESSION_BUCKETS:
        if number >= low and (high is None or number <= high):
            return f"sessions {low}-{high}" if high else f"sessions {low}+"
    raise ValueError(f"No bucket for session {number}")


def by_session(judged: Judged) -> List[Value]:
    values = []
    for low, high in SESSION_BUCKETS:
        bucket = session_bucket(low - 1)
        values += group(bucket, judged,
                        lambda record, bucket=bucket:
                        session_bucket(record["ssession_id"]) == bucket, extracted=True)
    return values


def exchanges(result: Record) -> int:
    return sum(len(session["dialogue"]) // 2 for session in result["sessions"])


def from_archive(result: Record) -> Tuple[int, int]:
    """(redundants on an archived memory, redundants), following the log: such a redundant
    brings the memory back to the core, and the split may archive it again."""
    archived: set = set()
    hits = total = 0
    for session in result["sessions"]:
        for operation in session["operations"]:
            kind, item_id = operation["op_type"], operation["item_id"]
            if kind == "archive":
                archived.add(item_id)
            elif kind == "redundant":
                total += 1
                hits += item_id in archived
                archived.discard(item_id)
            elif kind == "update":
                archived.discard(operation.get("related_item_id"))
            elif kind in ("delete", "evict", "prune"):
                archived.discard(item_id)
    return hits, total


def operations(result: Record) -> List[Value]:
    count = exchanges(result)
    done = [operation for session in result["sessions"] for operation in session["operations"]
            if operation["op_type"] in CONSOLIDATION_OPERATIONS]
    # With the window of 2 every exchange is consolidated by one insert: the next, or the flush.
    written = sum(len({operation["insert"] for operation in session["operations"]
                       if operation["op_type"] in CONSOLIDATION_OPERATIONS})
                  for session in result["sessions"])
    points = sum(1 for session in result["sessions"] for point in session["memory_points"]
                 if point["memory_source"] != "interference")
    kinds = {kind: sum(1 for operation in done if operation["op_type"] == kind)
             for kind in CONSOLIDATION_OPERATIONS}
    return ([(f"{kind} per exchange", NUMBER, ratio(kinds[kind], count))
             for kind in CONSOLIDATION_OPERATIONS]
            + [("redundants from the archive", SHARE, ratio(*from_archive(result))),
               ("exchanges with no operation", SHARE, ratio(count - written, count)),
               ("written per memory point", NUMBER,
                ratio(kinds["create"] + kinds["update"], points))])


def cost(result: Record) -> List[Value]:
    stats = result.get("node_stats", {}).get("consolidation", {})
    calls, seconds = stats.get("calls", 0), stats.get("seconds", 0.0)
    count = exchanges(result)
    dialogue = sum(session.get("add_dialogue_duration_ms", 0)
                   for session in result["sessions"]) / 1000
    return [
        ("calls per exchange", NUMBER, ratio(calls, count)),
        ("seconds per call", NUMBER, ratio(seconds, calls)),
        ("input tokens per call", TOKENS, ratio(stats.get("input_tokens", 0), calls)),
        ("output tokens per call", TOKENS, ratio(stats.get("output_tokens", 0), calls)),
        ("seconds per exchange", NUMBER, ratio(seconds, count)),
        ("share of the dialogue time", SHARE, ratio(seconds, dialogue)),
    ]


def answered(answers: Sequence[Record], label: str) -> Optional[float]:
    return share(answers, lambda record: record["result_type"] == label)


def answers(judged: Judged) -> List[Value]:
    return ([("correct", SHARE, answered(judged.answers, "Correct")),
             ("hallucinated", SHARE, answered(judged.answers, "Hallucination")),
             ("omitted", SHARE, answered(judged.answers, "Omission"))]
            + [(f"correct, {kind}", SHARE,
                answered([record for record in judged.answers
                          if record.get("question_type") == kind], "Correct"))
               for kind in QUESTION_TYPES])


def answer_cost(result: Record) -> List[Value]:
    stats = result.get("node_stats", {}).get("generate_answer", {})
    calls = stats.get("calls", 0)
    questions = [question for session in result["sessions"]
                 for question in session.get("questions", [])]

    def milliseconds(name: str) -> Optional[float]:
        found = [question[name] for question in questions if question.get(name) is not None]
        return ratio(sum(found), len(found))

    return [
        ("seconds per answer", NUMBER, ratio(stats.get("seconds", 0.0), calls)),
        ("input tokens per answer", TOKENS, ratio(stats.get("input_tokens", 0), calls)),
        ("output tokens per answer", TOKENS, ratio(stats.get("output_tokens", 0), calls)),
        ("archive search per question", MILLISECONDS, milliseconds("vector_search_ms")),
        ("retrieve branch per question", MILLISECONDS, milliseconds("search_duration_ms")),
    ]


# (title, whether it reads HaluMem's judgments, the values of a user)
SECTIONS = (
    ("fidelity", True, fidelity),
    ("answers", True, answers),
    ("by type", True, by_type),
    ("by session", True, by_session),
    ("operations", False, operations),
    ("cost", False, cost),
    ("answer cost", False, answer_cost),
)


@dataclass
class Version:
    """A run: its manifest, its results and HaluMem's judgments, by uuid."""

    name: str
    run_dir: str
    manifest: Dict[str, Any]
    results: Dict[str, Record]
    judged: Dict[str, Record]


def load_version(root: str, name: str) -> Version:
    run_dir = os.path.join(root, "eval", "results", f"{adapter.FRAME}-{name}")
    path = os.path.join(run_dir, f"{adapter.FRAME}_eval_results.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"No results in {run_dir}: launch halumem_adapter.py run first.")
    results = {result["uuid"]: result for result in adapter.load_users(path)}
    judged = {}
    for uuid in results:
        judgments = os.path.join(run_dir, "tmp2", f"{uuid}.json")
        if os.path.exists(judgments):
            with open(judgments, encoding="utf-8") as handle:
                judged[uuid] = json.load(handle)
    manifest = {}
    if os.path.exists(os.path.join(run_dir, "run.json")):
        with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
    return Version(name, run_dir, manifest, results, judged)


def describe(version: Version) -> str:
    config = version.manifest.get("config", {})
    sampling = version.manifest.get("node_sampling", {}).get("consolidation")
    left_out = sum(sum(counts.values())
                   for counts in adapter.unusable_judgments(version.run_dir).values())
    return (f"{version.name}: {len(version.results)} users, {len(version.judged)} judged "
            f"({left_out} judgments left out); {version.manifest.get('users', '?')} users, "
            f"commit {version.manifest.get('commit') or '?'}, utility "
            f"{config.get('consolidation_utility', '?')}, model "
            f"{config.get('llm_config', {}).get('model_name', '?')}, core "
            f"{config.get('core_memory_limit', '?')}, archive "
            f"{config.get('archive_memory_limit', '?')}, consolidation sampling "
            f"{json.dumps(sampling) if sampling else 'of the model'}")


def common_users(versions: Sequence[Version], held: Callable[[Version], Dict[str, Any]]
                 ) -> List[str]:
    """The uuids every version holds, in the first version's order."""
    return [uuid for uuid in held(versions[0])
            if all(uuid in held(version) for version in versions[1:])]


Table = Tuple[List[Tuple[str, str]], List[Dict[str, Dict[str, Optional[float]]]]]


def section_table(versions: Sequence[Version], users: Sequence[str], judgments: bool,
                  compute: Callable[[Any], List[Value]]) -> Table:
    """(label, format) of every row, and for each version: label -> uuid -> value."""
    rows: List[Tuple[str, str]] = []
    tables = []
    for version in versions:
        table: Dict[str, Dict[str, Optional[float]]] = {}
        for uuid in users:
            source = (judged_records(version.judged[uuid]) if judgments
                      else version.results[uuid])
            for label, kind, value in compute(source):
                if (label, kind) not in rows:
                    rows.append((label, kind))
                table.setdefault(label, {})[uuid] = value
        tables.append(table)
    return rows, tables


def present(values: Dict[str, Optional[float]]) -> List[float]:
    return [value for value in values.values() if value is not None]


def mean(values: Sequence[float]) -> Optional[float]:
    return ratio(sum(values), len(values))


def paired(first: Dict[str, Optional[float]], other: Dict[str, Optional[float]]
           ) -> List[float]:
    """Other minus first, user by user, where both have a value."""
    return [other[uuid] - value for uuid, value in first.items()
            if value is not None and other.get(uuid) is not None]


def bootstrap(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
              ) -> Optional[Tuple[float, float]]:
    """The 95% percentile interval of the mean, the users drawn with replacement."""
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    means = sorted(sum(generator.choice(values) for _ in values) / len(values)
                   for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def shown(value: Optional[float], kind: str, signed: bool = False) -> str:
    if value is None:
        return "-"
    sign = "+" if signed else ""
    if kind == SHARE:
        return f"{100 * value:{sign}.1f}" + ("" if signed else "%")
    return f"{value:{sign}.0f}" if kind in (TOKENS, MILLISECONDS) else f"{value:{sign}.2f}"


def difference_text(values: Sequence[float], kind: str) -> str:
    if not values:
        return "-"
    interval = bootstrap(values)
    text = shown(mean(values), kind, signed=True)
    if interval:
        text += f" [{shown(interval[0], kind, True)}, {shown(interval[1], kind, True)}]"
    return text


def print_section(title: str, users: int, names: Sequence[str], table: Table) -> None:
    rows, tables = table
    width = max([LABEL_WIDTH] + [len(label) + 4 for label, _ in rows])
    headers = list(names) + [f"{name} diff" for name in names[1:]]
    columns = ([max(len(name), 8) + 2 for name in names]
               + [max(len(name) + 5, 22) + 2 for name in names[1:]])
    print(f"\n{title} ({users} users)")
    print(f"{'':<{width}}" + "".join(f"{header:>{column}}"
                                     for header, column in zip(headers, columns)))
    for label, kind in rows:
        cells = [shown(mean(present(found.get(label, {}))), kind) for found in tables]
        cells += [difference_text(paired(tables[0].get(label, {}), found.get(label, {})), kind)
                  for found in tables[1:]]
        print(f"  {label:<{width - 2}}" + "".join(f"{cell:>{column}}"
                                                  for cell, column in zip(cells, columns)))


def csv_rows(title: str, names: Sequence[str], table: Table) -> List[List[Any]]:
    """section, kpi, version, users, mean, and the difference from the first version with
    its interval; shares as fractions."""
    rows, tables = table
    found = []
    for label, _ in rows:
        for position, (name, values) in enumerate(zip(names, tables)):
            row = [title, label, name, len(present(values.get(label, {}))),
                   mean(present(values.get(label, {})))]
            if position:
                difference = paired(tables[0].get(label, {}), values.get(label, {}))
                interval = bootstrap(difference) or (None, None)
                row += [mean(difference), interval[0], interval[1]]
            else:
                row += [None, None, None]
            found.append(["" if cell is None else cell for cell in row])
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Fidelity, operations and cost of consolidation, run by run."""
    parser = argparse.ArgumentParser(description="How faithfully consolidation turns the "
                                                 "dialogue into memory, and what it costs.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--versions", nargs="+", required=True, type=adapter.version_name,
                        help="the runs; the differences are from the first")
    parser.add_argument("--csv", help="also write every row, version by version, to this file")
    arguments = parser.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    versions = [load_version(root, name) for name in arguments.versions]
    for version in versions:
        print(describe(version))
    names = [version.name for version in versions]
    held = {False: common_users(versions, lambda version: version.results),
            True: common_users(versions, lambda version: version.judged)}
    if len(versions) > 1:
        print(f"\nIn every version: {len(held[False])} users, {len(held[True])} of them judged. "
              f"The diff columns are each version minus {names[0]}, user by user, with the 95% "
              "interval over the users; shares in percentage points.")

    found = []
    for title, judgments, compute in SECTIONS:
        users = held[judgments]
        if not users:
            print(f"\n{title}: no user " + ("judged in every version: launch "
                                           "halumem_adapter.py evaluate first."
                                           if judgments else "in every version."))
            continue
        table = section_table(versions, users, judgments, compute)
        print_section(title, len(users), names, table)
        found += csv_rows(title, names, table)

    if arguments.csv:
        with open(arguments.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["section", "kpi", "version", "users", "mean", "difference",
                             "difference_low", "difference_high"])
            writer.writerows(found)
        print(f"\nEvery row, version by version: {arguments.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
