#!/usr/bin/env python3
"""The eviction rules of two traces or more side by side, user by user, on their replays.

From the report of eviction_replay.py simulate of each trace: at every limit, how much a
few rules keep (retention) and leave findable (retrievable) of what the questions need,
user by user, and what the judgment gives:

    own score              the trace's own score: its utility, combination and prune fraction
    own score, calibrated  the same with the calibrated table, when the replay had one
    own score, one vote    the same with the term of one vote without context, and with
    own score, mean vote   the mean of several (simulate --rerated), when the replay had them
    score none             the score without utility, at the same fraction
    lru, random, oracle    the references
    judgment gain          own score minus score none: what the ratings add to recency and
                           frequency; and the same for the calibrated score and the votes

The rows are named by role, so traces of different modes line up: the rule each role
stands for, per trace, is printed first. Each version after the first is set against it
on the users all of them hold: the mean difference user by user, with its 95% bootstrap
interval over the users. It compares what the judgment gives the eviction across models
(4B, 9B), sampling or core limit, where each trace has memories of its own.

    python3 memory_service/test/replay_compare.py --halumem PATH --versions A B [C ...]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import consolidation_report as kpi
import eviction_replay as replay
import halumem_adapter as adapter

MEASURES = (("retention", "kept"), ("retrievable", "found"))
REFERENCES = ("lru", "random", "oracle")
VARIANTS = (("own score, one vote", "rerated"), ("own score, mean vote", "mean-vote"))
GAINS = (("judgment gain", "own score"), ("judgment gain, calibrated", "own score, calibrated"),
         ("judgment gain, one vote", "own score, one vote"),
         ("judgment gain, mean vote", "own score, mean vote"))

Tallies = Dict[str, Dict[str, float]]


def load_report(root: str, name: str) -> Tuple[Any, Dict[str, Tallies]]:
    """The trace's configuration and, by uuid, the tallies of every rule at every limit."""
    run_dir = os.path.join(root, "eval", "results", f"{adapter.FRAME}-{name}")
    path = os.path.join(run_dir, replay.FOLDER, "report.json")
    if not os.path.exists(path):
        raise SystemExit(f"No replay report in {run_dir}: launch eviction_replay.py "
                         "simulate first.")
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    return (replay.trace_config(run_dir),
            {user["uuid"]: user["tallies"] for user in report["users"]})


def roles(config: Any) -> Dict[str, str]:
    """Role -> the name of the rule that plays it in this trace's replay."""
    named = {"own score": replay.ScoreRule(config).name,
             "own score, calibrated": replay.ScoreRule(config, table=()).name,
             **{role: replay.ScoreRule(config, variant=variant).name
                for role, variant in VARIANTS},
             "score none": replay.ScoreRule(config, consolidation_utility="none").name}
    if config.consolidation_utility == "none":
        named = {"score none": named["score none"]}
    named.update({name: name for name in REFERENCES})
    return named


def limits(users: Dict[str, Tallies]) -> List[int]:
    return sorted({int(name.rsplit("@", 1)[1]) for tallies in users.values()
                   for name in tallies if "@" in name})


def measure(tallies: Tallies, rule: str, limit: int, field: str) -> Optional[float]:
    counts = tallies.get(f"{rule}@{limit}")
    if not counts:
        return None
    return kpi.ratio(counts.get(field, 0), counts.get("needed", 0))


def user_values(tallies: Tallies, named: Dict[str, str], limit: int, field: str
                ) -> Dict[str, Optional[float]]:
    """Role -> value, the gains included, for one user at one limit."""
    found = {role: measure(tallies, rule, limit, field) for role, rule in named.items()}
    none = found.get("score none")
    for gain, role in GAINS:
        if found.get(role) is not None and none is not None:
            found[gain] = found[role] - none
    return found


ROWS = (("own score", "own score, calibrated") + tuple(role for role, _ in VARIANTS)
        + ("score none",) + REFERENCES + tuple(gain for gain, _ in GAINS))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The eviction rules of several traces side "
                                                 "by side, user by user.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--versions", nargs="+", required=True, type=adapter.version_name,
                        help="traces already replayed by eviction_replay.py simulate")
    arguments = parser.parse_args(argv)
    if len(arguments.versions) < 2:
        parser.error("--versions needs two traces or more")

    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    loaded = [load_report(root, name) for name in arguments.versions]
    named = [roles(config) for config, _ in loaded]
    for name, rules in zip(arguments.versions, named):
        print(f"{name}: " + "; ".join(f"{role} = {rule}" for role, rule in rules.items()
                                      if role not in REFERENCES))
    users = [uuid for uuid in loaded[0][1] if all(uuid in found for _, found in loaded[1:])]
    shared = sorted(set(limits(loaded[0][1])).intersection(
        *(limits(found) for _, found in loaded[1:])))
    print(f"\n{len(users)} users in every version. The diff columns are each version minus "
          f"{arguments.versions[0]}, user by user, with the 95% interval over the users; "
          "shares and gains in percentage points.")

    for limit in shared:
        for title, field in MEASURES:
            tables = []
            for (_, found), rules in zip(loaded, named):
                table: Dict[str, Dict[str, Optional[float]]] = {}
                for uuid in users:
                    for role, value in user_values(found[uuid], rules, limit, field).items():
                        table.setdefault(role, {})[uuid] = value
                tables.append(table)
            # A rule no replay of these traces had is no row.
            rows = [(role, kpi.SHARE) for role in ROWS
                    if any(kpi.present(table.get(role, {})) for table in tables)]
            kpi.print_section(f"{title} at limit {limit}", len(users), arguments.versions,
                              (rows, tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
