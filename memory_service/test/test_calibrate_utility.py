"""The calibration of UTILITY_TABLE and NEED_GAIN_TABLE, without a model.

Run: python memory_service/run_tests.py -v
"""

import contextlib
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

import calibrate_utility as cu  # noqa: E402
import calibration_ratings as cr  # noqa: E402
import halumem_adapter as adapter  # noqa: E402
from calibrate_utility import Pair  # noqa: E402

from fakes import ScriptedChatModel  # noqa: E402


DOG = "Rossi Mario ha un cane di nome Argo."
JAZZ = "A Rossi Mario piace il jazz."
CAT = "Rossi Mario ha un gatto."


def point(content, source="secondary"):
    return {"memory_content": content, "memory_source": source, "is_update": "False",
            "original_memories": []}


def question(*evidence):
    return {"question": "?", "answer": "-", "question_type": "Basic Fact Recall",
            "evidence": [{"memory_content": text} for text in evidence]}


def session(points=(), questions=()):
    return {"memory_points": list(points), "questions": list(questions)}


def user(uuid, *sessions):
    return {"uuid": uuid, "persona_info": "Name: Rossi Mario; Gender: Male; Age: 30",
            "sessions": list(sessions)}


def pair(served, user=0, **ratings):
    return Pair(user, {"safety": False, **ratings}, served)


def is_valid_utility(line):
    return all(0 < value <= 1 for value in line) and list(line) == sorted(line)


class PairsTest(unittest.TestCase):
    """A rated point and a question of a later session: served if the question cites it."""

    USER = user("u", session([point(DOG), point(JAZZ), point(CAT, "interference")],
                             [question(DOG)]),
                session([], [question(DOG)]))

    def test_only_the_questions_of_later_sessions_count(self):
        pairs = cu.pairs_of(self.USER, 0, {(0, 0): {"need": 3}, (0, 1): {"need": 2}})

        self.assertEqual(sorted((item.ratings["need"], item.served) for item in pairs),
                         [(2, False), (3, True)])

    def test_safety_and_points_never_judged_stay_out(self):
        self.assertEqual(cu.pairs_of(self.USER, 0, {(0, 0): {"safety": True}, (0, 1): None}),
                         [])

    def test_the_old_point_still_serves_after_an_update(self):
        updated = {**point("Rossi Mario ha due cani."), "is_update": "True",
                   "original_memories": [DOG]}
        history = user("u", session([point(DOG)]), session([updated]),
                       session([], [question(DOG)]))

        pairs = cu.pairs_of(history, 0, {(0, 0): {"need": 3}, (1, 0): {"need": 3}})

        self.assertEqual(sorted(item.served for item in pairs), [False, True])


class IsotonicTest(unittest.TestCase):
    def test_violators_are_pooled_by_weight(self):
        for values, weights, expected in (([0.5, 0.1], [1, 3], [0.2, 0.2]),
                                          ([0.1, 0.5, 0.3], [1, 1, 1], [0.1, 0.4, 0.4])):
            with self.subTest(values=values):
                for got, want in zip(cu.isotonic(values, weights), expected):
                    self.assertAlmostEqual(got, want)


class ByLevelTest(unittest.TestCase):
    """The fraction served per level, pulled toward all the levels together, then monotone."""

    def test_the_fraction_is_shrunk_toward_all_levels(self):
        # All levels: 1 of 4 served. The pair without the rating does not count.
        pairs = [pair(True, utility=5), pair(False, utility=5), pair(False, utility=1),
                 pair(False, utility=1), pair(True)]

        levels = cu.by_level(pairs, "utility")

        self.assertEqual((levels.pairs, levels.served), ((2, 0, 0, 0, 2), (0, 0, 0, 0, 1)))
        for got, want in zip(levels.rate, (0.5 / 4, 0.25, 0.25, 0.25, 1.5 / 4)):
            self.assertAlmostEqual(got, want)

    def test_a_level_against_the_order_is_pooled(self):
        # Level 1 always serves, level 3 rarely: the estimate cannot go down going up.
        pairs = [pair(True, need=1)] * 2 + [pair(False, need=3)] * 3 + [pair(True, need=3)]

        rate = cu.by_level(pairs, "need").rate

        for got in rate:
            self.assertAlmostEqual(got, 0.5)

    def test_without_a_served_pair_there_is_no_estimate(self):
        with self.assertRaises(ValueError):
            cu.by_level([pair(False, need=2)] * 3, "need")


class TablesTest(unittest.TestCase):
    PAIRS = ([pair(True, need=3, gain=3)] * 4 + [pair(False, need=3, gain=3)] * 4
             + [pair(True, need=3, gain=1)] + [pair(False, need=3, gain=1)] * 5
             + [pair(False, need=1, gain=3)] * 6
             + [pair(True, need=2, gain=2)] + [pair(False, need=2, gain=2)] * 3
             + [pair(False, need=2, gain=1)] * 4)

    def test_the_utility_table_is_the_fraction_over_the_highest(self):
        pairs = ([pair(True, utility=5)] * 2 + [pair(False, utility=5)]
                 + [pair(True, utility=3)] + [pair(False, utility=3)] * 3
                 + [pair(False, utility=1)] * 2)
        rate = cu.by_level(pairs, "utility").rate

        table = cu.utility_table(pairs)

        for got, want in zip(table, rate):
            self.assertAlmostEqual(got, want / max(rate))
        self.assertTrue(is_valid_utility(table))

    def test_the_product_is_the_row_times_the_column_over_all_the_pairs(self):
        need = cu.by_level(self.PAIRS, "need").rate
        gain = cu.by_level(self.PAIRS, "gain").rate
        overall = sum(item.served for item in self.PAIRS) / len(self.PAIRS)

        table = cu.product_table(self.PAIRS)

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(table[i][j], need[i] * gain[j] / overall)

    def test_the_free_table_is_monotone_even_when_the_data_are_not(self):
        # need 1 and gain 1 serves more than every other cell.
        pairs = self.PAIRS + [pair(True, need=1, gain=1)] * 20

        table = cu.normalized(cu.free_table(pairs))

        for line in (*table, *zip(*table)):
            self.assertTrue(is_valid_utility(line), table)

    def test_an_empty_cell_takes_the_value_of_the_product(self):
        # No pair with need 3 and gain 2, and its neighbours do not force the order on it.
        self.assertAlmostEqual(cu.free_table(self.PAIRS)[2][1],
                               cu.product_table(self.PAIRS)[2][1])

    def test_a_cell_with_many_pairs_follows_its_own_fraction(self):
        # One high rating is enough to serve half the time, two add nothing: need and gain
        # are not independent, and the product puts cell (3, 3) at two thirds.
        half = [True] * 100 + [False] * 100
        pairs = ([pair(False, need=1, gain=1)] * 200
                 + [pair(served, need=need, gain=gain) for need, gain in ((1, 3), (3, 1), (3, 3))
                    for served in half])

        self.assertGreater(cu.product_table(pairs)[2][2], 0.6)
        self.assertAlmostEqual(cu.free_table(pairs)[2][2], 0.5, delta=0.02)


class AucTest(unittest.TestCase):
    def test_a_served_pair_above_one_not_served_counts_one_and_a_tie_a_half(self):
        self.assertEqual(cu.auc([(0.9, True), (0.1, False)]), 1.0)
        self.assertEqual(cu.auc([(0.1, True), (0.9, False)]), 0.0)
        self.assertEqual(cu.auc([(0.5, True), (0.5, False), (0.5, False)]), 0.5)

    def test_without_both_kinds_of_pair_there_is_no_auc(self):
        self.assertIsNone(cu.auc([(0.5, True)]))
        self.assertIsNone(cu.auc([]))

    def test_the_user_left_out_is_scored_only_by_the_others(self):
        # A model that gives 1 to the need levels that served in the pairs it is estimated on.
        def model(train):
            served = {item.ratings["need"] for item in train if item.served}
            return lambda item: float(item.ratings["need"] in served)

        pairs = [pair(True, user=0, need=1), pair(False, user=0, need=2),
                 pair(True, user=1, need=2), pair(False, user=1, need=1)]

        self.assertEqual(cu.held_out_auc(pairs, model), (0.0, {0: 0.0, 1: 0.0}))

    def test_with_a_single_user_nothing_is_predicted(self):
        pairs = [pair(True, need=3, gain=3), pair(False, need=1, gain=1)]

        self.assertEqual(cu.held_out_auc(pairs, cu.table_model(cu.product_table)), (None, {}))


class MainTest(unittest.TestCase):
    """From the ratings file to the tables to paste into eviction.py."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name
        self.ratings = os.path.join(self.root, "ratings.jsonl")
        os.makedirs(os.path.join(self.root, "data"))
        with open(os.path.join(self.root, adapter.DATASET), "w", encoding="utf-8") as handle:
            for uuid in ("u0", "u1"):
                handle.write(json.dumps(user(uuid, session([point(DOG), point(JAZZ)]),
                                             session([], [question(DOG)]))) + "\n")

    def record(self, uuid, mode, commit="abc", model="qwen"):
        judged = {"utility": (5, 1)} if mode == "single" else {"need": (3, 1), "gain": (3, 1)}
        ratings = {str(index): {"safety": False, **{name: levels[index]
                                                    for name, levels in judged.items()}}
                   for index in (0, 1)}
        return {"uuid": uuid, "session": 0, "mode": mode, "model": model, "commit": commit,
                "ratings": ratings}

    def run_main(self, records, *options):
        with open(self.ratings, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cu.main(["--halumem", self.root, "--ratings", self.ratings, *options])
        return code, output.getvalue()

    def test_model_picks_the_ratings_of_that_model(self):
        small = [self.record(uuid, "need_gain", model="qwen3.5-4b") for uuid in ("u0", "u1")]
        large = [self.record(uuid, "single", model="qwen3.5-9b") for uuid in ("u0", "u1")]

        _, last = self.run_main(small + large)
        _, chosen = self.run_main(small + large, "--model", "qwen3.5-4b")

        self.assertIn("need_gain: no ratings", last)
        self.assertIn("single: no ratings", chosen)
        self.assertIn("NEED_GAIN_TABLE = (", chosen)

    def test_the_calibrated_table_is_the_one_main_prints(self):
        records = [self.record(uuid, "need_gain") for uuid in ("u0", "u1")]

        _, output = self.run_main(records)

        table = cu.calibrated_table(self.root, records, "need_gain")
        for line in cu.literal("NEED_GAIN_TABLE", table):
            self.assertIn(line, output)

    def test_series_keeps_the_last_commit_of_the_mode_and_model(self):
        records = [self.record("u0", "need_gain", commit="old", model="qwen3.5-9b"),
                   self.record("u0", "need_gain", commit="new", model="qwen3.5-9b"),
                   self.record("u0", "need_gain", commit="other", model="qwen3.5-4b"),
                   self.record("u0", "single", commit="last", model="qwen3.5-9b")]

        chosen = cu.series(records, "need_gain", "qwen3.5-9b")

        self.assertEqual([record["commit"] for record in chosen], ["new"])

    def test_only_the_ratings_of_the_last_commit_and_model_count(self):
        old = self.record("u0", "single", commit="vecchio")

        code, output = self.run_main([old, self.record("u0", "need_gain"),
                                      self.record("u1", "need_gain")])

        self.assertEqual(code, 0)
        self.assertIn("single: no ratings", output)
        self.assertIn("NEED_GAIN_TABLE = (", output)

    def test_without_ratings_there_is_nothing_to_calibrate(self):
        code, _ = self.run_main([])

        self.assertEqual(code, 1)


class RatingsTest(unittest.TestCase):
    """The judgment of the calibration users' points, with the model replaced."""

    STAMP = {"mode": "need_gain", "model": "qwen", "commit": "abc"}
    USER = user("u", session([point(DOG), point(CAT, "interference"), point(JAZZ)]))

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "ratings.jsonl")
        self.llm = ScriptedChatModel(tool_responses={})
        self.llm.invocations = []

    def script(self, *facts):
        self.llm.script({"JudgeFacts": {"facts": list(facts)}})

    def rate(self, stamp=STAMP):
        with contextlib.redirect_stdout(io.StringIO()):
            return cr.rate_users([self.USER], self.llm, stamp, self.path)

    def ratings(self):
        return [record["ratings"] for record in cu.load_ratings(self.path)]

    def sent(self):
        return " ".join(str(message.content) for invocation in self.llm.invocations
                        for message in invocation["messages"])

    def test_the_interference_points_are_left_out_and_the_person_is_the_user(self):
        self.script({"number": 1, "safety": False, "need": 3, "gain": 3},
                    {"number": 2, "safety": False, "need": 1, "gain": 2})

        self.rate()

        self.assertEqual(self.ratings(), [{"0": {"safety": False, "need": 3, "gain": 3},
                                           "2": {"safety": False, "need": 1, "gain": 2}}])
        self.assertNotIn("gatto", self.sent())
        self.assertNotIn("Rossi", self.sent())

    def test_a_fact_the_model_skipped_is_asked_again(self):
        # Every time the model judges only the first fact it is given.
        self.script({"number": 1, "safety": False, "need": 2, "gain": 2})

        self.rate()

        self.assertEqual(len(self.llm.invocations), 2)
        self.assertEqual(self.ratings(), [{"0": {"safety": False, "need": 2, "gain": 2},
                                           "2": {"safety": False, "need": 2, "gain": 2}}])

    def test_a_fact_never_judged_is_kept_as_missing(self):
        self.script()

        self.rate()

        self.assertEqual(len(self.llm.invocations), cr.ATTEMPTS)
        self.assertEqual(self.ratings(), [{"0": None, "2": None}])

    def test_a_safety_fact_keeps_no_rating_and_an_off_scale_one_is_not_given(self):
        self.script({"number": 1, "safety": True, "need": 3, "gain": 3},
                    {"number": 2, "safety": "false", "need": "2", "gain": 9})

        self.rate()

        self.assertEqual(self.ratings(), [{"0": {"safety": True, "need": None, "gain": None},
                                           "2": {"safety": False, "need": 2, "gain": None}}])

    def test_a_session_already_judged_with_the_same_stamp_is_skipped(self):
        self.script({"number": 1, "safety": False, "need": 2, "gain": 2},
                    {"number": 2, "safety": False, "need": 2, "gain": 2})

        self.assertEqual((self.rate(), self.rate()), (1, 0))
        self.assertEqual(self.rate({**self.STAMP, "mode": "single"}), 1)
        self.assertEqual(len(self.ratings()), 2)


if __name__ == "__main__":
    unittest.main()
