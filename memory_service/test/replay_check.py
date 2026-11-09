#!/usr/bin/env python3
"""Whether the replay of a trace foresaw what a real eviction did, user by user.

A run with a real archive limit (halumem_adapter.py run --archive-limit L) prunes as it
goes, and HaluMem judges its answers. The replay of the trace (eviction_replay.py
simulate) has already said, for the same rule at the same limit, how much of what the
questions need would be kept and left findable. This script sets the two side by side
for each run:

    retention, replay           the share of the needed memories the run's rule keeps at
                                the run's limit, replayed on the trace
    retrievable, replay         the share left in the core or among the memories a
                                question's search returns
    answers correct, the run    HaluMem's verdicts on the run's answers
    answers correct, vs trace   the same minus the trace's, which has no limit: what the
                                real eviction cost

The rule is the trace's own score with the run's mode, combination and prune fraction:
the calibrated one with --calibrated, when the runs used the tables calibrate printed,
pasted into eviction.py. A run without utility is the score without utility, on the
trace's memories. The diff columns are each run minus the first, user by user, with the
95% interval over the users, and for every run after the first the users on which the
real difference has the sign of the replayed one. The replay keeps the trace's writing,
so a real run that wrote other memories can still differ: that is what this measures.

    python3 memory_service/test/replay_check.py --halumem PATH --trace TRACE
                                                --runs RUN [RUN ...] [--calibrated]
                                                [--csv FILE]

The runs must be evaluated, and the trace evaluated and replayed.
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import consolidation_report as kpi
import eviction_replay as replay
import halumem_adapter as adapter
import replay_compare
from memory_service.config import MemoryConfig

# What the run may change and the replay still stands for it.
SAME = ("core_memory_limit", "eviction_frequency_function", "eviction_time_decay_field",
        "eviction_time_decay_function", "eviction_time_decay_scale_hours",
        "maximum_historical_messages", "retrieve_k")
MEASURES = (("retention, replay", "kept"), ("retrievable, replay", "found"))
CORRECT, LOST = "answers correct, the run", "answers correct, vs trace"


def run_config(version: kpi.Version) -> MemoryConfig:
    return MemoryConfig(**version.manifest["config"])


def differences(trace: MemoryConfig, run: MemoryConfig) -> List[str]:
    """The settings in which the run is not the trace with another eviction."""
    found = [name for name in SAME if getattr(trace, name) != getattr(run, name)]
    if trace.llm_config.get("model_name") != run.llm_config.get("model_name"):
        found.append("model")
    if run.consolidation_utility not in ("none", trace.consolidation_utility):
        found.append("consolidation_utility")
    return found


def rule_name(trace: MemoryConfig, run: MemoryConfig, calibrated: bool) -> str:
    """The replay's rule for the run: expected_gain reads no calibrated table."""
    table = (() if calibrated and run.consolidation_utility != "none"
             and run.eviction_combination != "expected_gain" else None)
    return replay.ScoreRule(trace, table, consolidation_utility=run.consolidation_utility,
                            eviction_combination=run.eviction_combination,
                            prune_fraction=run.prune_fraction).name


def correct(version: kpi.Version, uuid: str) -> Optional[float]:
    return kpi.answered(kpi.judged_records(version.judged[uuid]).answers, "Correct")


Table = Dict[str, Dict[str, Optional[float]]]


def run_values(tallies: replay_compare.Tallies, rule: str, limit: int,
               run: Optional[float], trace: Optional[float]) -> List[kpi.Value]:
    values = [(label, kpi.SHARE, replay_compare.measure(tallies, rule, limit, field))
              for label, field in MEASURES]
    return values + [(CORRECT, kpi.SHARE, run),
                     (LOST, kpi.SHARE, None if run is None or trace is None else run - trace)]


def same_sign(first: Table, other: Table, label: str) -> Tuple[int, int, int]:
    """Other minus first, user by user, in `label` and in the correct answers: (users with
    the same sign, users with a difference in both, users with both values)."""
    pairs = []
    for uuid, before in first[label].items():
        after, real_before = other[label].get(uuid), first[CORRECT].get(uuid)
        real_after = other[CORRECT].get(uuid)
        if None not in (before, after, real_before, real_after):
            pairs.append((after - before, real_after - real_before))
    both = [(predicted, real) for predicted, real in pairs if predicted and real]
    agreed = sum(1 for predicted, real in both if (predicted > 0) == (real > 0))
    return agreed, len(both), len(pairs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Whether the replay of a trace foresaw what "
                                                 "a real eviction did.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--trace", required=True, type=adapter.version_name,
                        help="the trace, evaluated and replayed by eviction_replay.py simulate")
    parser.add_argument("--runs", nargs="+", required=True, type=adapter.version_name,
                        help="the runs with a real archive limit, evaluated")
    parser.add_argument("--calibrated", action="store_true",
                        help="the runs used the calibrated tables pasted into eviction.py")
    parser.add_argument("--csv", help="also write every row, run by run, to this file")
    arguments = parser.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    trace = kpi.load_version(root, arguments.trace)
    trace_config, report = replay_compare.load_report(root, arguments.trace)
    runs = [kpi.load_version(root, name) for name in arguments.runs]
    rules = []
    for run in runs:
        config = run_config(run)
        wrong = differences(trace_config, config)
        if wrong:
            raise SystemExit(f"{run.name} is not {trace.name} with another eviction: "
                             f"{', '.join(wrong)} differ.")
        if config.archive_memory_limit not in replay_compare.limits(report):
            raise SystemExit(f"{run.name} has limit {config.archive_memory_limit}, which the "
                             f"replay of {trace.name} never tried.")
        rule = rule_name(trace_config, config, arguments.calibrated)
        if not any(f"{rule}@{config.archive_memory_limit}" in tallies
                   for tallies in report.values()):
            raise SystemExit(f"The replay of {trace.name} has no rule {rule!r}: with "
                             "--calibrated, launch simulate with --calibration-trace.")
        rules.append((rule, config.archive_memory_limit))
        print(f"{run.name}: {rule} at limit {config.archive_memory_limit}")

    users = [uuid for uuid in report if uuid in trace.judged
             and all(uuid in run.judged for run in runs)]
    if not users:
        raise SystemExit("No user is in the replay and judged in every run: launch "
                         "halumem_adapter.py evaluate on the trace and on the runs first.")
    tables = []
    for run, (rule, limit) in zip(runs, rules):
        table: Table = {}
        for uuid in users:
            for label, _, value in run_values(report[uuid], rule, limit, correct(run, uuid),
                                              correct(trace, uuid)):
                table.setdefault(label, {})[uuid] = value
        tables.append(table)
    rows = [(label, kpi.SHARE) for label, _ in MEASURES] + [(CORRECT, kpi.SHARE),
                                                            (LOST, kpi.SHARE)]
    names = [run.name for run in runs]
    print(f"\nThe diff columns are each run minus {names[0]}, user by user, with the 95% "
          "interval over the users, in percentage points.")
    kpi.print_section(f"replay of {trace.name} against the real eviction", len(users),
                      names, (rows, tables))

    for name, table in zip(names[1:], tables[1:]):
        for label, _ in MEASURES:
            agreed, both, count = same_sign(tables[0], table, label)
            print(f"{name} minus {names[0]}: the real difference in correct answers has the "
                  f"sign of the {label.split(',')[0]} of the replay for {agreed} of the {both} "
                  f"users with a difference in both ({count} users)")

    if arguments.csv:
        with open(arguments.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["section", "kpi", "version", "users", "mean", "difference",
                             "difference_low", "difference_high"])
            writer.writerows(kpi.csv_rows("replay check", names, (rows, tables)))
        print(f"\nEvery row, run by run: {arguments.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
