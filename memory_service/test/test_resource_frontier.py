"""What the memory costs at each archive limit, and the frontier of the rules, without a model.

Run: python memory_service/run_tests.py -v
"""

import contextlib
import csv
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

import halumem_adapter as adapter  # noqa: E402
import resource_frontier as rf  # noqa: E402


def session(active, size=None, *search_ms):
    found = {"active_archive": active,
             "questions": [{"question": "Dove vive Mario?", "vector_search_ms": ms}
                           for ms in search_ms]}
    if size is not None:
        found["archive_bytes"] = size
    return found


class LinesTest(unittest.TestCase):
    """Bytes and milliseconds of one more memory, from the sessions of the trace."""

    def test_the_line_has_the_cost_of_one_memory_and_of_the_empty_folder(self):
        users = [{"sessions": [session(n, 100000 + 12000 * n) for n in (0, 10, 40)]},
                 {"sessions": [session(n, 100000 + 12000 * n) for n in (5, 80)]}]

        line = rf.fit(rf.space_points(users))

        self.assertAlmostEqual(line.slope, 12000)
        self.assertAlmostEqual(line.intercept, 100000)
        self.assertEqual(line.points, 5)

    def test_every_question_is_a_point_of_the_search(self):
        users = [{"sessions": [session(0, None, 50.0, 50.0), session(100, None, 60.0)]}]

        line = rf.fit(rf.search_points(users))

        self.assertAlmostEqual(line.at(200), 70.0)

    def test_one_archive_size_gives_no_line(self):
        self.assertIsNone(rf.fit([(10, 1.0), (10, 2.0)]))
        self.assertIsNone(rf.fit([]))

    def test_a_trace_from_before_the_measures_has_no_points(self):
        users = [{"sessions": [{"active_archive": 10, "questions": [{"question": "?"}]}]}]

        self.assertEqual((rf.space_points(users), rf.search_points(users)), ([], []))

    def test_the_median_by_archive_size(self):
        points = [(10, 1.0), (20, 3.0), (60, 5.0), (1500, 9.0)]

        self.assertEqual(rf.by_size(points), [("0-50", 2, 2.0), ("51-100", 1, 5.0),
                                              (">1000", 1, 9.0)])


class FrontierTest(unittest.TestCase):
    """A point is worth it when no rule keeps more with an archive no larger."""

    def test_only_the_points_nobody_beats(self):
        kept = {("lru", 25): 0.50, ("lru", 50): 0.55,
                ("score", 25): 0.40, ("score", 50): 0.60, ("score", 100): 0.58}

        self.assertEqual(rf.frontier(kept), {("lru", 25), ("score", 50)})

    def test_the_oracle_is_never_on_the_frontier_and_beats_nobody(self):
        kept = {("oracle", 25): 0.90, ("lru", 25): 0.50}

        self.assertEqual(rf.frontier(kept), {("lru", 25)})

    def test_two_rules_that_keep_the_same_are_both_on_it(self):
        kept = {("lru", 25): 0.50, ("fifo", 25): 0.50}

        self.assertEqual(rf.frontier(kept), {("lru", 25), ("fifo", 25)})

    def test_a_rule_with_nothing_needed_is_left_out(self):
        self.assertEqual(rf.frontier({("lru", 25): None, ("fifo", 25): 0.1}), {("fifo", 25)})


class MainTest(unittest.TestCase):
    """From the run's folder to the table and the CSV."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name
        self.run_dir = os.path.join(self.root, "eval", "results", f"{adapter.FRAME}-trace")
        os.makedirs(os.path.join(self.run_dir, "replay"))
        users = [{"uuid": "u1",
                  "sessions": [session(0, 100000, 40.0), session(100, 1300000, 50.0)]}]
        with open(os.path.join(self.run_dir, f"{adapter.FRAME}_eval_results.jsonl"), "w",
                  encoding="utf-8") as handle:
            for user in users:
                handle.write(json.dumps(user) + "\n")

    def report(self, summary):
        with open(os.path.join(self.run_dir, "replay", "report.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"summary": summary, "users": []}, handle)

    def run_main(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return rf.main(["--halumem", self.root, "--version", "trace"])

    def test_the_csv_has_a_row_per_rule_and_limit_with_its_resources(self):
        self.report({"lru@25": {"retention": 0.5, "retention interval": [0.4, 0.6],
                                "of oracle": 0.3, "retrievable": 0.45},
                     "oracle@25": {"retention": 0.9, "of oracle": 1.0},
                     "rating AUC": 0.6, "main rule": "score need_gain weighted_sum 0.10"})

        self.assertEqual(self.run_main(), 0)

        with open(os.path.join(self.run_dir, "replay", "frontier.csv"),
                  encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(row["rule"], row["limit"], row["frontier"]) for row in rows],
                         [("lru", "25", "1"), ("oracle", "25", "0")])
        self.assertAlmostEqual(float(rows[0]["megabytes"]), (100000 + 12000 * 25) / 2 ** 20,
                               places=4)
        self.assertAlmostEqual(float(rows[0]["search_ms"]), 42.5)
        self.assertEqual(rows[0]["retention_low"], "0.4")


if __name__ == "__main__":
    unittest.main()
