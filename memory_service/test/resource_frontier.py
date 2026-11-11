#!/usr/bin/env python3
"""What the memory costs the robot at each archive limit, and which rules are worth it.

From a trace of halumem_adapter.py (a run with --archive-limit 100000) and its replay
(eviction_replay.py simulate), after both:

    space       archive_bytes against active_archive at the end of every session: the
                least-squares line, its slope the bytes of one more memory, its intercept
                the empty folder; next to it the bytes of one vector alone, from the
                replay's embeddings
    search      vector_search_ms of every question against the archive of its session:
                the line, and the median by archive size
    limits      every limit of the replay in memories, MB and search ms, from the lines
    frontier    retention and of oracle of every rule at every limit; a point is on the
                frontier when no rule keeps more with an archive no larger. The oracle
                knows the questions to come: it is shown, never on the frontier

The disk size is the folder as Chroma leaves it (see halumem_adapter.py), so the space of
a limit is that of an archive grown to it, not of one kept there by prunes.

    python3 memory_service/test/resource_frontier.py --halumem PATH --version NAME [--csv FILE]

The CSV, one row per rule and limit, is for the plots: by default replay/frontier.csv in
the run's folder.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import halumem_adapter as adapter

SIZE_BUCKETS = ((0, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, None))
FLOAT_BYTES = 4
MEBIBYTE = 2 ** 20

Point = Tuple[float, float]
Key = Tuple[str, int]


@dataclass(frozen=True)
class Line:
    """y = intercept + slope * x, fitted on `points` points."""

    intercept: float
    slope: float
    points: int

    def at(self, x: float) -> float:
        return self.intercept + self.slope * x


def fit(points: Sequence[Point]) -> Optional[Line]:
    """The least-squares line through the points; None with fewer than two distinct x."""
    if len({x for x, _ in points}) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in points)
             / sum((x - mean_x) ** 2 for x, _ in points))
    return Line(mean_y - slope * mean_x, slope, len(points))


def space_points(users: Sequence[Dict[str, Any]]) -> List[Point]:
    """(active memories, bytes on disk) at the end of every session that recorded them."""
    return [(session["active_archive"], session["archive_bytes"])
            for user in users for session in user["sessions"]
            if "archive_bytes" in session and "active_archive" in session]


def search_points(users: Sequence[Dict[str, Any]]) -> List[Point]:
    """(active memories, ms of the archive search) of every question that recorded it."""
    return [(session["active_archive"], question["vector_search_ms"])
            for user in users for session in user["sessions"] if "active_archive" in session
            for question in session.get("questions", [])
            if question.get("vector_search_ms") is not None]


def by_size(points: Sequence[Point]) -> List[Tuple[str, int, float]]:
    """(bucket of archive size, points, median y) of the buckets that have points."""
    found = []
    for low, high in SIZE_BUCKETS:
        values = [y for x, y in points if x >= low and (high is None or x <= high)]
        if values:
            found.append((f"{low}-{high}" if high is not None else f">{low - 1}", len(values),
                          statistics.median(values)))
    return found


def vector_dimensions(run_dir: str) -> Optional[int]:
    """The size of the replay's vectors, from the first user's file; None before embed."""
    paths = sorted(glob.glob(os.path.join(run_dir, "replay", "embeddings", "*.npz")))
    if not paths:
        return None
    import numpy

    with numpy.load(paths[0]) as stored:
        return int(stored["memory_vectors"].shape[1])


def quality(summary: Dict[str, Any]) -> Tuple[List[str], List[int], Dict[Key, Dict[str, Any]]]:
    """The rules in the order of the report, the limits, and (rule, limit) -> its values."""
    rules: List[str] = []
    limits: Set[int] = set()
    values: Dict[Key, Dict[str, Any]] = {}
    for name, value in summary.items():
        if "@" not in name or not isinstance(value, dict):
            continue
        rule, limit = name.rsplit("@", 1)
        if rule not in rules:
            rules.append(rule)
        limits.add(int(limit))
        values[(rule, int(limit))] = value
    return rules, sorted(limits), values


def frontier(kept: Dict[Key, Optional[float]]) -> Set[Key]:
    """The (rule, limit) that no other rule beats: as much kept with an archive no larger,
    and more kept or a smaller archive. The oracle is not a rule a robot can run."""
    feasible = {key: value for key, value in kept.items()
                if key[0] != "oracle" and value is not None}
    return {(rule, limit) for (rule, limit), value in feasible.items()
            if not any(other >= value and other_limit <= limit
                       and (other > value or other_limit < limit)
                       for (_, other_limit), other in feasible.items())}


def percent(value: Optional[float]) -> str:
    return "  -  " if value is None else f"{100 * value:5.1f}"


def resource_lines(users: Sequence[Dict[str, Any]], run_dir: str
                   ) -> Tuple[Optional[Line], Optional[Line], List[str]]:
    """The space and search lines, and what to print about them."""
    space, search = fit(space_points(users)), fit(search_points(users))
    lines = []
    if space is None:
        lines.append("Space: no archive_bytes in the trace, or one archive size only.")
    else:
        lines.append(f"Space: {space.slope / 1024:.1f} KB per memory, the empty folder "
                     f"{space.intercept / MEBIBYTE:.2f} MB ({space.points} sessions)")
    dimensions = vector_dimensions(run_dir)
    if dimensions:
        lines.append(f"  the vector alone: {dimensions} dimensions, "
                     f"{dimensions * FLOAT_BYTES / 1024:.1f} KB")
    if search is None:
        lines.append("Search: no vector_search_ms in the trace, or one archive size only.")
    else:
        lines.append(f"Search: {search.intercept:.1f} ms + {100 * search.slope:.2f} ms per "
                     f"100 memories ({search.points} questions); median by archive size:")
        lines += [f"  {bucket:>10} memories  {count:>5} questions  {median:7.1f} ms"
                  for bucket, count, median in by_size(search_points(users))]
    return space, search, lines


def write_csv(path: str, rules: Sequence[str], limits: Sequence[int],
              values: Dict[Key, Dict[str, Any]], on: Set[Key], space: Optional[Line],
              search: Optional[Line]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rule", "limit", "megabytes", "search_ms", "retention",
                         "retention_low", "retention_high", "of_oracle", "old_retention",
                         "old_of_oracle", "retrievable", "nps", "same_session", "frontier"])
        for rule in rules:
            for limit in limits:
                value = values.get((rule, limit))
                if value is None:
                    continue
                interval = value.get("retention interval") or (None, None)
                writer.writerow([
                    rule, limit,
                    "" if space is None else round(space.at(limit) / MEBIBYTE, 4),
                    "" if search is None else round(search.at(limit), 2),
                    value.get("retention"), interval[0], interval[1], value.get("of oracle"),
                    value.get("old retention"), value.get("old of oracle"),
                    value.get("retrievable"), value.get("NPS"), value.get("same session"),
                    int((rule, limit) in on)])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Resources per limit and the frontier of the rules, from a trace and its replay."""
    parser = argparse.ArgumentParser(description="What the memory costs at each archive "
                                                 "limit, and which rules are worth it.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--version", required=True, type=adapter.version_name,
                        help="the trace, already replayed by eviction_replay.py simulate")
    parser.add_argument("--csv", help="where to write the table, instead of the run's replay/")
    arguments = parser.parse_args(argv)

    run_dir = adapter.run_folder(arguments)
    report_path = os.path.join(run_dir, "replay", "report.json")
    if not os.path.exists(report_path):
        raise SystemExit(f"No replay report in {run_dir}: launch eviction_replay.py "
                         "simulate first.")
    with open(report_path, encoding="utf-8") as handle:
        summary = json.load(handle)["summary"]
    users = adapter.load_users(os.path.join(run_dir, f"{adapter.FRAME}_eval_results.jsonl"))

    space, search, lines = resource_lines(users, run_dir)
    for line in lines:
        print(line)
    rules, limits, values = quality(summary)
    print(f"\n{'limit (memories)':>16}{'MB':>9}{'search ms':>11}")
    for limit in limits:
        megabytes = "-" if space is None else f"{space.at(limit) / MEBIBYTE:.2f}"
        milliseconds = "-" if search is None else f"{search.at(limit):.1f}"
        print(f"{limit:>16}{megabytes:>9}{milliseconds:>11}")

    on = frontier({key: value.get("retention") for key, value in values.items()})
    width = max(len(rule) for rule in rules) + 2 if rules else 10
    print("\nKept % (of oracle %) at each limit; * on the frontier:")
    print(f"{'rule':<{width}}" + "".join(f"{limit:>14}" for limit in limits))
    for rule in rules:
        cells = []
        for limit in limits:
            value = values.get((rule, limit), {})
            mark = "*" if (rule, limit) in on else " "
            cells.append(f"{percent(value.get('retention'))} ({percent(value.get('of oracle'))})"
                         f"{mark}")
        print(f"{rule:<{width}}" + "".join(f"{cell:>14}" for cell in cells))

    path = arguments.csv or os.path.join(run_dir, "replay", "frontier.csv")
    write_csv(path, rules, limits, values, on, space, search)
    print(f"\nThe table, one row per rule and limit: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
