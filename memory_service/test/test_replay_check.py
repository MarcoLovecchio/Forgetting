"""The replay of an invented trace against two invented runs with a real eviction.

Run: python memory_service/run_tests.py -v
"""

import contextlib
import csv
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

import replay_check as rc  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402

TRACE = MemoryConfig(
    node_name="memory_agent",
    consolidation_utility="need_gain",
    eviction_combination="weighted_sum",
    prune_fraction=0.10,
    archive_memory_limit=100000,
    core_memory_limit=400,
    eviction=True,
    count_candidate_search=False,
    query_rewrite=False,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)
NG = dataclasses.replace(TRACE, archive_memory_limit=50)
NONE = dataclasses.replace(NG, consolidation_utility="none")
OWN, WITHOUT = "score need_gain weighted_sum 0.10", "score none 0.10"
USERS = ("u1", "u2", "u3")


class ReplayCheckTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name

    def write(self, name, config, correct, kept=None):
        """A run judged with `correct` answers out of 4 per user; a trace with `kept`, the
        needed memories out of 10 each rule keeps at limit 50."""
        run_dir = os.path.join(self.root, "eval", "results", f"memory_service-{name}")
        os.makedirs(os.path.join(run_dir, "tmp2"))
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump({"config": dataclasses.asdict(config)}, handle)
        with open(os.path.join(run_dir, "memory_service_eval_results.jsonl"), "w",
                  encoding="utf-8") as handle:
            for uuid in USERS:
                handle.write(json.dumps({"uuid": uuid, "sessions": []}) + "\n")
        for uuid, right in zip(USERS, correct):
            answers = [{"result_type": "Correct" if index < right else "Omission",
                        "question_type": "Basic Fact Recall"} for index in range(4)]
            with open(os.path.join(run_dir, "tmp2", f"{uuid}.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"question_answering_records": answers}, handle)
        if kept:
            os.makedirs(os.path.join(run_dir, "replay"))
            users = [{"uuid": uuid, "tallies": {
                f"{rule}@50": {"needed": 10, "kept": count, "found": count / 2}
                for rule, count in zip((OWN, WITHOUT), counts)}}
                for uuid, counts in zip(USERS, kept)]
            with open(os.path.join(run_dir, "replay", "report.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"summary": {}, "users": users}, handle)

    def check(self, *runs, calibrated=False):
        path = os.path.join(self.root, "check.csv")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            rc.main(["--halumem", self.root, "--trace", "trace", "--runs", *runs, "--csv", path]
                    + (["--calibrated"] if calibrated else []))
        with open(path, encoding="utf-8") as handle:
            rows = {(row["kpi"], row["version"]): row for row in csv.DictReader(handle)}
        return rows, printed.getvalue()

    def test_the_rule_is_the_trace_score_with_the_eviction_of_the_run(self):
        product = dataclasses.replace(NG, eviction_combination="product", prune_fraction=0.05)
        expected = dataclasses.replace(NG, eviction_combination="expected_gain")

        self.assertEqual(rc.rule_name(TRACE, product, calibrated=True),
                         "score need_gain calibrated product 0.05")
        self.assertEqual(rc.rule_name(TRACE, expected, calibrated=True),
                         "score need_gain expected_gain 0.10")
        self.assertEqual(rc.rule_name(TRACE, NONE, calibrated=True), WITHOUT)

    def test_a_run_that_changed_more_than_the_eviction_is_not_the_trace(self):
        other = dataclasses.replace(NG, core_memory_limit=800,
                                    llm_config={**NG.llm_config, "model_name": "bigger"})
        single = dataclasses.replace(NG, consolidation_utility="single")

        self.assertEqual(rc.differences(TRACE, other), ["core_memory_limit", "model"])
        self.assertEqual(rc.differences(TRACE, single), ["consolidation_utility"])
        self.assertEqual(rc.differences(TRACE, NONE), [])

    def test_the_replayed_and_the_real_differences_user_by_user(self):
        self.write("trace", TRACE, (3, 3, 3), kept=((8, 6), (6, 7), (7, 5)))
        self.write("ng-a50", NG, (3, 2, 3))
        self.write("none-a50", NONE, (2, 3, 3))

        rows, printed = self.check("ng-a50", "none-a50")

        self.assertAlmostEqual(float(rows[("retention, replay", "ng-a50")]["mean"]), 0.7)
        self.assertAlmostEqual(float(rows[("retention, replay", "none-a50")]["difference"]),
                               -0.1)
        self.assertAlmostEqual(float(rows[("answers correct, vs trace", "ng-a50")]["mean"]),
                               -0.25 / 3)
        self.assertAlmostEqual(float(rows[("answers correct, the run", "none-a50")]
                                     ["difference"]), 0.0)
        self.assertIn("sign of the retention of the replay for 2 of the 2 users with a "
                      "difference in both (3 users)", printed)

    def test_a_limit_the_replay_never_tried_is_refused(self):
        self.write("trace", TRACE, (3, 3, 3), kept=((8, 6), (6, 7), (7, 5)))
        self.write("ng-a75", dataclasses.replace(NG, archive_memory_limit=75), (3, 3, 3))

        with self.assertRaises(SystemExit) as raised:
            self.check("ng-a75")
        self.assertIn("never tried", str(raised.exception))

    def test_a_calibrated_rule_the_replay_did_not_have_is_refused(self):
        self.write("trace", TRACE, (3, 3, 3), kept=((8, 6), (6, 7), (7, 5)))
        self.write("ng-a50", NG, (3, 3, 3))

        with self.assertRaises(SystemExit) as raised:
            self.check("ng-a50", calibrated=True)
        self.assertIn("--calibration-trace", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
