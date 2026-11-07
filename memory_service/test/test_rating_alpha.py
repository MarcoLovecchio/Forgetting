"""Krippendorff's alpha between calibration series, without a model.

Run: python memory_service/run_tests.py -v
"""

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import rating_alpha as ra  # noqa: E402

# Krippendorff (2011), Computing Krippendorff's Alpha-Reliability: four observers, twelve
# units, None where an observer gave no value.
KRIPPENDORFF = (
    (1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None),
    (1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3),
    (None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None),
    (1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None),
)
SCALE = (1, 2, 3, 4, 5)


def columns(rows):
    return [[value for value in column if value is not None] for column in zip(*rows)]


def alpha_of(found, metric, categories=SCALE):
    return ra.alpha(ra.coincidences(found, categories).sum(axis=0), metric)


def record(mode, ratings, commit="abc1234", model="qwen3.5-4b", uuid="u1", session=0):
    return {"uuid": uuid, "session": session, "mode": mode, "model": model, "commit": commit,
            "ratings": {str(point): judgment for point, judgment in ratings.items()}}


class AlphaTest(unittest.TestCase):
    """The published values of Krippendorff's example, and the limit cases."""

    def test_the_example_gives_the_published_nominal_and_ordinal_alpha(self):
        found = columns(KRIPPENDORFF)
        self.assertAlmostEqual(alpha_of(found, "nominal"), 0.743, places=3)
        self.assertAlmostEqual(alpha_of(found, "ordinal"), 0.815, places=3)

    def test_the_same_values_in_every_series_give_one(self):
        self.assertAlmostEqual(alpha_of([[1, 1], [3, 3], [5, 5, 5]], "ordinal"), 1.0)

    def test_a_point_with_one_value_changes_nothing(self):
        found = columns(KRIPPENDORFF)
        self.assertAlmostEqual(alpha_of(found + [[4]], "ordinal"), alpha_of(found, "ordinal"))

    def test_one_category_only_has_no_alpha(self):
        self.assertTrue(math.isnan(alpha_of([[2, 2], [2, 2]], "ordinal")))

    def test_the_ordinal_alpha_weighs_a_far_disagreement_more(self):
        # The same values in both: only the distance of the disagreements changes.
        near = [[1, 2], [1, 2], [5, 5]]
        far = [[1, 5], [1, 5], [2, 2]]
        self.assertGreater(alpha_of(near, "ordinal"), alpha_of(far, "ordinal"))
        self.assertAlmostEqual(alpha_of(near, "nominal"), alpha_of(far, "nominal"))

    def test_the_interval_holds_the_alpha_of_all_the_points(self):
        matrices = ra.coincidences(columns(KRIPPENDORFF), SCALE)
        lower, upper = ra.interval(matrices, "ordinal", draws=200, seed=1)
        self.assertLessEqual(lower, ra.alpha(matrices.sum(axis=0), "ordinal"))
        self.assertGreaterEqual(upper, ra.alpha(matrices.sum(axis=0), "ordinal"))


class SeriesTest(unittest.TestCase):
    """The files of calibration_ratings.py, read point by point."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def write(self, name, *records):
        path = os.path.join(self.folder.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            for line in records:
                handle.write(json.dumps(line) + "\n")
        return path

    def rating(self, name, mode):
        return next(rating for rating in ra.RATINGS if rating.name == name and rating.mode == mode)

    def test_a_safety_point_has_no_need_but_keeps_its_flag(self):
        first = self.write("a.jsonl", record("need_gain", {
            0: {"safety": True, "need": None, "gain": None},
            1: {"safety": False, "need": 3, "gain": 2}}))
        second = self.write("b.jsonl", record("need_gain", {
            0: {"safety": False, "need": 2, "gain": 2},
            1: {"safety": False, "need": 3, "gain": 1}}))
        series = [ra.read_series(path)["need_gain"][1] for path in (first, second)]
        self.assertEqual(ra.units(series, self.rating("need", "need_gain")), [[2], [3, 3]])
        self.assertEqual(ra.units(series, self.rating("safety", "need_gain")),
                         [[True, False], [False, False]])
        self.assertEqual(ra.units(series, self.rating("need x gain", "need_gain")),
                         [[4], [6, 3]])

    def test_a_point_never_judged_is_missing(self):
        first = self.write("a.jsonl", record("single", {0: None, 1: {"safety": False,
                                                                     "utility": 4}}))
        second = self.write("b.jsonl", record("single", {0: {"safety": False, "utility": 2},
                                                         1: {"safety": False, "utility": 4}}))
        series = [ra.read_series(path)["single"][1] for path in (first, second)]
        self.assertEqual(ra.units(series, self.rating("utility", "single")), [[2], [4, 4]])

    def test_each_mode_reads_its_own_last_commit(self):
        path = self.write(
            "a.jsonl",
            record("need_gain", {0: {"safety": False, "need": 1, "gain": 1}}, commit="old"),
            record("need_gain", {0: {"safety": False, "need": 3, "gain": 3}}, commit="new"),
            record("single", {0: {"safety": False, "utility": 5}}, commit="other"))
        series = ra.read_series(path)
        self.assertEqual(series["need_gain"][1], {("u1", 0, 0): {"safety": False, "need": 3,
                                                                 "gain": 3}})
        self.assertIn("single", series)

    def test_model_reads_only_that_models_series(self):
        path = self.write(
            "a.jsonl",
            record("need_gain", {0: {"safety": False, "need": 1, "gain": 1}}, model="qwen3.5-9b"),
            record("need_gain", {0: {"safety": False, "need": 3, "gain": 3}}, model="qwen3.5-4b"))

        self.assertEqual(ra.read_series(path, "qwen3.5-9b")["need_gain"][1][("u1", 0, 0)]["need"],
                         1)
        self.assertEqual(ra.read_series(path)["need_gain"][1][("u1", 0, 0)]["need"], 3)

    def test_the_report_runs_on_two_series(self):
        judged = {0: {"safety": False, "need": 3, "gain": 2},
                  1: {"safety": False, "need": 1, "gain": 1}}
        paths = [self.write(name, record("need_gain", judged)) for name in ("a.jsonl", "b.jsonl")]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ra.main(["--ratings", *paths]), 0)

    def test_one_series_is_not_enough(self):
        path = self.write("a.jsonl", record("single", {0: {"safety": False, "utility": 3}}))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ra.main(["--ratings", path])


if __name__ == "__main__":
    unittest.main()
