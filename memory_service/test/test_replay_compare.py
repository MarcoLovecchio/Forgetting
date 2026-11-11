"""The rules of two invented replays side by side, user by user.

Run: python memory_service/run_tests.py -v
"""

import contextlib
import dataclasses
import io
import json
import os
import sys
import tempfile
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import replay_compare as rc  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402

CONFIG = MemoryConfig(
    node_name="memory_agent",
    consolidation_utility="need_gain",
    eviction_combination="weighted_sum",
    prune_fraction=0.10,
    eviction=True,
    count_candidate_search=False,
    query_rewrite=False,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)
OWN, NONE = "score need_gain weighted_sum 0.10", "score none 0.10"


def tallies(own, none, limit=50):
    """Out of 10 needed memories, how many each rule kept; found is half of it, and of the
    4 old ones each rule kept one less than its share."""
    kept = {OWN: own, NONE: none, "lru": 4, "random": 3, "oracle": 9}
    return {f"{rule}@{limit}": {"needed": 10, "kept": count, "found": count / 2,
                                "old needed": 4, "old kept": min(count, 5) - 1}
            for rule, count in kept.items()}


class ReplayCompareTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name

    def write(self, name, users, config=CONFIG):
        run_dir = os.path.join(self.root, "eval", "results", f"memory_service-{name}")
        os.makedirs(os.path.join(run_dir, "replay"))
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump({"config": dataclasses.asdict(config)}, handle)
        with open(os.path.join(run_dir, "replay", "report.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"summary": {}, "users": [{"uuid": uuid, "tallies": found}
                                                for uuid, found in users.items()]}, handle)

    def test_the_roles_name_the_rules_of_the_trace(self):
        named = rc.roles(CONFIG)

        self.assertEqual((named["own score"], named["own score, calibrated"],
                          named["own score, mean vote"], named["score none"]),
                         (OWN, "score need_gain calibrated weighted_sum 0.10",
                          "score need_gain mean-vote weighted_sum 0.10", NONE))
        self.assertEqual(named["own score, calibrated alone"],
                         "score need_gain calibrated-alone weighted_sum 0.10")
        self.assertNotIn("own score", rc.roles(dataclasses.replace(
            CONFIG, consolidation_utility="none")))

    def test_the_gain_is_the_own_score_minus_the_score_without_utility(self):
        found = rc.user_values(tallies(own=7, none=5), rc.roles(CONFIG), 50, "kept")

        self.assertAlmostEqual(found["judgment gain"], 0.2)
        self.assertIsNone(found["own score, calibrated"], "the replay had no calibrated table")
        self.assertNotIn("judgment gain, calibrated", found)

    def test_the_old_memories_are_measured_on_their_own_count(self):
        found = rc.user_values(tallies(own=7, none=3), rc.roles(CONFIG), 50, "old kept",
                               "old needed")

        self.assertAlmostEqual(found["own score"], 4 / 4)
        self.assertAlmostEqual(found["judgment gain"], (4 - 2) / 4)

    def test_two_traces_are_compared_on_the_users_both_hold(self):
        self.write("small", {"u1": tallies(6, 5), "u2": tallies(8, 6)})
        self.write("large", {"u1": tallies(7, 5), "u2": tallies(9, 6), "u3": tallies(9, 1)})
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            self.assertEqual(rc.main(["--halumem", self.root, "--versions", "small",
                                      "large"]), 0)

        printed = output.getvalue()
        self.assertIn("2 users in every version", printed)
        self.assertNotIn("calibrated", printed.split("users in every version")[1],
                         "no row for a rule the replays did not have")
        gain = next(line for line in printed.splitlines()
                    if line.strip().startswith("judgment gain ") and "+" in line)
        self.assertIn("+10.0 [+10.0, +10.0]", gain, "both users gain ten points more")


class FractionsTest(unittest.TestCase):
    """The prune fractions of one trace, each against the trace's own."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name

    def test_the_traces_fraction_comes_first(self):
        self.assertEqual(rc.fractions(CONFIG), [0.10, 0.05, 0.20])
        self.assertEqual(rc.fraction_roles(CONFIG, 0.05)["own score"],
                         "score need_gain weighted_sum 0.05")

    def test_one_trace_is_enough_to_compare_its_fractions(self):
        def kept(count):
            return {"needed": 10, "kept": count, "found": count}
        tallies = {f"score need_gain weighted_sum {fraction}@50": kept(own)
                   for fraction, own in (("0.05", 8), ("0.10", 7), ("0.20", 5))}
        tallies.update({f"score none {fraction}@50": kept(5) for fraction in ("0.05", "0.10",
                                                                              "0.20")})
        ReplayCompareTest.write(self, "only", {"u1": tallies, "u2": tallies})
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            self.assertEqual(rc.main(["--halumem", self.root, "--versions", "only",
                                      "--fractions"]), 0)

        own = next(line for line in output.getvalue().splitlines()
                   if line.strip().startswith("own score ") and "+" in line)
        self.assertIn("+10.0 [+10.0, +10.0] 2/0/0", own, "5% keeps one more of ten")


if __name__ == "__main__":
    unittest.main()
