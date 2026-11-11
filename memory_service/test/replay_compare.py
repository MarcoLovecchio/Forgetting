#!/usr/bin/env python3
"""The eviction rules of two traces or more side by side, user by user, on their replays.

From the report of eviction_replay.py simulate of each trace: at every limit, how much a
few rules keep (retention) and leave findable (retrievable) of what the questions need,
and how much they keep of the needed memories written two sessions or more before their
question (old, the primary measure), user by user, and what the judgment gives:

    own score              the trace's own score: its utility, combination and prune fraction
    own score, calibrated  the same with the table calibrated in context, when the replay
                           had one (simulate --calibration-trace)
    own score, calibrated alone
                           the same with the table calibrated without context, when the
                           replay had one (simulate --ratings)
    own score, one vote    the same with the term of one vote without context, and with
    own score, mean vote   the mean of several (simulate --rerated), when the replay had them
    score none             the score without utility, at the same fraction
    lru, random, oracle    the references
    judgment gain          own score minus score none: what the ratings add to recency and
                           frequency; and the same for the calibrated scores and the votes

The rows are named by role, so traces of different modes line up: the rule each role
stands for, per trace, is printed first. Each version after the first is set against it
on the users all of them hold: the mean difference user by user, with its 95% bootstrap
interval over the users and how many users it is above, below and at zero for. It compares
what the judgment gives the eviction across models
(4B, 9B), sampling or core limit, where each trace has memories of its own.

With --repeat-at-end it reads the reports of simulate --repeat-at-end instead, and adds
the retention and the retrievable share when every question is asked again at the end of
the story.

    python3 memory_service/test/replay_compare.py --halumem PATH --versions A B [C ...]
                                                  [--repeat-at-end]
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import consolidation_report as kpi
import eviction_replay as replay
import halumem_adapter as adapter

MEASURES = (("retention", "kept", "needed"), ("retrievable", "found", "needed"),
            ("retention of the memories two sessions old or more", "old kept", "old needed"))
END_MEASURES = (("retention when asked again at the end", "end kept", "end needed"),
                ("retrievable when asked again at the end", "end found", "end needed"))
REFERENCES = ("lru", "random", "oracle")
VARIANTS = (("own score, one vote", "rerated"), ("own score, mean vote", "mean-vote"))
GAINS = (("judgment gain", "own score"), ("judgment gain, calibrated", "own score, calibrated"),
         ("judgment gain, calibrated alone", "own score, calibrated alone"),
         ("judgment gain, one vote", "own score, one vote"),
         ("judgment gain, mean vote", "own score, mean vote"))

Tallies = Dict[str, Dict[str, float]]


def load_report(root: str, name: str, report: str = "report.json"
                ) -> Tuple[Any, Dict[str, Tallies]]:
    """The trace's configuration and, by uuid, the tallies of every rule at every limit."""
    run_dir = os.path.join(root, "eval", "results", f"{adapter.FRAME}-{name}")
    path = os.path.join(run_dir, replay.FOLDER, report)
    if not os.path.exists(path):
        raise SystemExit(f"No replay report {report} in {run_dir}: launch eviction_replay.py "
                         "simulate first.")
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    return (replay.trace_config(run_dir),
            {user["uuid"]: user["tallies"] for user in report["users"]})


def roles(config: Any) -> Dict[str, str]:
    """Role -> the name of the rule that plays it in this trace's replay."""
    named = {"own score": replay.ScoreRule(config).name,
             "own score, calibrated": replay.ScoreRule(config, table=()).name,
             "own score, calibrated alone": replay.ScoreRule(config, table=(),
                                                             variant=replay.ALONE).name,
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


def measure(tallies: Tallies, rule: str, limit: int, part: str,
            whole: str = "needed") -> Optional[float]:
    counts = tallies.get(f"{rule}@{limit}")
    if not counts:
        return None
    return kpi.ratio(counts.get(part, 0), counts.get(whole, 0))


def user_values(tallies: Tallies, named: Dict[str, str], limit: int, part: str,
                whole: str = "needed") -> Dict[str, Optional[float]]:
    """Role -> value, the gains included, for one user at one limit."""
    found = {role: measure(tallies, rule, limit, part, whole) for role, rule in named.items()}
    none = found.get("score none")
    for gain, role in GAINS:
        if found.get(role) is not None and none is not None:
            found[gain] = found[role] - none
    return found


ROWS = (("own score", "own score, calibrated", "own score, calibrated alone")
        + tuple(role for role, _ in VARIANTS)
        + ("score none",) + REFERENCES + tuple(gain for gain, _ in GAINS))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The eviction rules of several traces side "
                                                 "by side, user by user.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--versions", nargs="+", required=True, type=adapter.version_name,
                        help="traces already replayed by eviction_replay.py simulate")
    parser.add_argument("--repeat-at-end", action="store_true", dest="repeat_at_end",
                        help="the reports of simulate --repeat-at-end, with the end measures")
    arguments = parser.parse_args(argv)
    if len(arguments.versions) < 2:
        parser.error("--versions needs two traces or more")

    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    report = replay.REPEAT_REPORT if arguments.repeat_at_end else "report.json"
    loaded = [load_report(root, name, report) for name in arguments.versions]
    measures = MEASURES + (END_MEASURES if arguments.repeat_at_end else ())
    named = [roles(config) for config, _ in loaded]
    for name, rules in zip(arguments.versions, named):
        print(f"{name}: " + "; ".join(f"{role} = {rule}" for role, rule in rules.items()
                                      if role not in REFERENCES))
    users = [uuid for uuid in loaded[0][1] if all(uuid in found for _, found in loaded[1:])]
    shared = sorted(set(limits(loaded[0][1])).intersection(
        *(limits(found) for _, found in loaded[1:])))
    print(f"\n{len(users)} users in every version. The diff columns are each version minus "
          f"{arguments.versions[0]}, user by user, with the 95% interval over the users "
          "and the users above/below/at zero; shares and gains in percentage points.")

    for limit in shared:
        for title, part, whole in measures:
            tables = []
            for (_, found), rules in zip(loaded, named):
                table: Dict[str, Dict[str, Optional[float]]] = {}
                for uuid in users:
                    for role, value in user_values(found[uuid], rules, limit, part,
                                                   whole).items():
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
