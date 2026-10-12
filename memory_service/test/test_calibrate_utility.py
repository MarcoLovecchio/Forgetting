"""La calibrazione di UTILITY_TABLE e NEED_GAIN_TABLE, senza modello.

Esecuzione: python memory_service/run_tests.py -v
"""

import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import calibrate_utility as cu  # noqa: E402
import fama  # noqa: E402
from calibrate_utility import Pair  # noqa: E402


GOAL = "The user's daily calorie goal is 2200."
DOG = "The user has a dog named Argo."
HOME = "The user lives in Mondello."
NUTS = "The user is allergic to peanuts."


# Due domande di comodo, staccate dalla conversazione del test lungo: GOAL_QUESTION
# vuole il 2200, HOME_QUESTION non vuole l'indirizzo cancellato.
GOAL_QUESTION, HOME_QUESTION = 1001, 1002
QUESTIONS = {
    GOAL_QUESTION: fama.EvaluationQuestion(GOAL_QUESTION, "prova", (
        fama.recalls("2200", "2200"),
        fama.supersedes("2000", "2000", replaced_by=("2200",)))),
    HOME_QUESTION: fama.EvaluationQuestion(HOME_QUESTION, "prova", (
        fama.forgets("indirizzo", "mondello", "palermo"),)),
}


def setUpModule():
    patcher = mock.patch.dict(fama.QUESTIONS, QUESTIONS)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def trace(mode, ratings):
    """Una run con le due domande di comodo."""
    def memory(name, content):
        return {"id": name, "content": content, **ratings.get(name, {})}

    return {"commit": "abc", "consolidation_utility": mode, "questions": [
        {"index": GOAL_QUESTION, "fama": 1.0, "memories": [
            memory("goal", GOAL), memory("dog", DOG),
            {"id": "nuts", "content": NUTS, "safety": True}]},
        {"index": HOME_QUESTION, "fama": 0.0,
         "memories": [memory("goal", GOAL), memory("home", HOME)]},
    ]}


def pair(value, run=0, **ratings):
    return Pair(run, GOAL_QUESTION, ratings, value)


def is_valid_utility(line):
    return all(0 < value <= 1 for value in line) and list(line) == sorted(line)


class PairsTest(unittest.TestCase):
    """Ogni coppia vale FAMA-memoria con la memoria meno FAMA-memoria senza."""

    RUN = trace("single", {"goal": {"utility": 4}, "dog": {"utility": 3},
                           "home": {"utility": 2}})

    def test_a_memory_is_worth_what_its_absence_costs_and_safety_stays_out(self):
        values = {(item.question, item.memory["id"]): item.value
                  for item in cu.pairs_of(self.RUN, 0)}

        self.assertEqual(values, {(GOAL_QUESTION, "goal"): 1.0, (GOAL_QUESTION, "dog"): 0.0,
                                  (HOME_QUESTION, "goal"): 0.0,
                                  (HOME_QUESTION, "home"): -1.0})

    def test_a_harmful_memory_counts_as_not_served(self):
        harmful = pair(-1.0)

        self.assertFalse(harmful.served)
        self.assertEqual(harmful.realized, 0.0)

    def test_a_snapshot_scored_with_other_criteria_is_reported(self):
        run = json.loads(json.dumps(self.RUN))
        run["questions"][1]["fama"] = 1.0

        self.assertEqual(cu.changed_snapshots(self.RUN), 0)
        self.assertEqual(cu.changed_snapshots(run), 1)


class IsotonicTest(unittest.TestCase):
    def test_a_monotone_sequence_is_left_alone(self):
        self.assertEqual(cu.isotonic([0.1, 0.2, 0.3], [1, 1, 1]), [0.1, 0.2, 0.3])

    def test_violators_are_pooled_by_weight(self):
        for values, weights, expected in (([0.5, 0.1], [1, 3], [0.2, 0.2]),
                                          ([0.1, 0.5, 0.3], [1, 1, 1], [0.1, 0.4, 0.4])):
            with self.subTest(values=values):
                for got, want in zip(cu.isotonic(values, weights), expected):
                    self.assertAlmostEqual(got, want)


class ByLevelTest(unittest.TestCase):
    """need e gain per livello, avvicinati a tutti i livelli insieme e poi monotoni."""

    PAIRS = [pair(1.0, utility=5), pair(0.5, utility=5), pair(0.0, utility=5),
             pair(0.0, utility=1), pair(0.0, utility=1), pair(1.0)]

    def test_need_and_gain_are_shrunk_toward_all_levels(self):
        # Tutti i livelli: servite 2 su 5, gain medio 0.75. La coppia senza voto non conta.
        levels = cu.by_level(self.PAIRS, "utility")

        self.assertEqual((levels.pairs, levels.served), ((2, 0, 0, 0, 3), (0, 0, 0, 0, 2)))
        expected_need = ((0 + 2 * 0.4) / 4, 0.4, 0.4, 0.4, (2 + 2 * 0.4) / 5)
        for got, want in zip(levels.need, expected_need):
            self.assertAlmostEqual(got, want)
        for got in levels.gain:
            self.assertAlmostEqual(got, 0.75)

    def test_a_level_against_the_order_is_pooled(self):
        # Il livello 1 serve sempre, il 3 quasi mai: la stima non puo' scendere salendo.
        pairs = [pair(1.0, need=1), pair(1.0, need=1)] + [pair(0.0, need=3)] * 3 + [
            pair(1.0, need=3)]

        need = cu.by_level(pairs, "need").need

        self.assertEqual(list(need), sorted(need))
        for got in need:
            self.assertAlmostEqual(got, 0.5)

    def test_a_gain_against_the_order_is_pooled_too(self):
        pairs = [pair(1.0, gain=1)] * 3 + [pair(0.2, gain=3)] * 3

        gain = cu.by_level(pairs, "gain").gain

        self.assertEqual(list(gain), sorted(gain))
        self.assertAlmostEqual(gain[0], gain[2])

    def test_without_a_served_pair_there_is_no_table(self):
        for pairs in ([pair(0.0, utility=3)], [pair(1.0)]):
            with self.subTest(pairs=pairs):
                self.assertRaises(ValueError, cu.by_level, pairs, "utility")


class TablesTest(unittest.TestCase):
    PAIRS = ([pair(1.0, need=3, gain=3)] * 4 + [pair(0.5, need=3, gain=1)] * 2
             + [pair(0.0, need=1, gain=3)] * 6 + [pair(0.25, need=2, gain=2)] * 2
             + [pair(0.0, need=2, gain=1)] * 4)

    def test_the_utility_table_is_need_times_gain_over_the_highest(self):
        pairs = [pair(1.0, utility=5)] * 2 + [pair(0.2, utility=3), pair(0.0, utility=3),
                                              pair(0.0, utility=1), pair(0.0, utility=1)]
        levels = cu.by_level(pairs, "utility")
        values = [n * g for n, g in zip(levels.need, levels.gain)]

        table = cu.utility_table(pairs)

        for got, want in zip(table, values):
            self.assertAlmostEqual(got, want / max(values))
        self.assertTrue(is_valid_utility(table))

    def test_the_product_table_is_need_of_the_row_times_gain_of_the_column(self):
        need = cu.by_level(self.PAIRS, "need").need
        gain = cu.by_level(self.PAIRS, "gain").gain

        table = cu.product_table(self.PAIRS)

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(table[i][j], need[i] * gain[j])

    def test_the_free_table_is_monotone_even_when_the_data_are_not(self):
        # need 1 e gain 1 vale piu' di tutte le altre celle.
        pairs = self.PAIRS + [pair(1.0, need=1, gain=1)] * 20

        table = cu.normalized(cu.free_table(pairs))

        for line in (*table, *zip(*table)):
            self.assertTrue(is_valid_utility(line), table)

    def test_a_harmful_pair_counts_as_zero_in_the_free_table(self):
        self.assertEqual(cu.free_table(self.PAIRS + [pair(-1.0, need=2, gain=2)] * 5),
                         cu.free_table(self.PAIRS + [pair(0.0, need=2, gain=2)] * 5))

    def test_an_empty_cell_takes_the_value_of_the_product(self):
        # Nessuna coppia con need 2 e gain 3.
        self.assertAlmostEqual(cu.free_table(self.PAIRS)[1][2],
                               cu.product_table(self.PAIRS)[1][2])

    def test_a_cell_with_many_pairs_follows_its_own_mean(self):
        pairs = self.PAIRS + [pair(0.4, need=3, gain=3)] * 200
        own = (4 * 1.0 + 200 * 0.4) / 204

        self.assertAlmostEqual(cu.free_table(pairs)[2][2], own, places=2)

    def test_the_held_out_error_needs_two_runs(self):
        self.assertIsNone(cu.held_out_error(self.PAIRS, cu.product_table))

        two_runs = self.PAIRS + [Pair(1, item.question, item.memory, item.value)
                                 for item in self.PAIRS]
        self.assertIsInstance(cu.held_out_error(two_runs, cu.product_table), float)

    def test_the_run_left_out_is_predicted_only_from_the_others(self):
        # Un modello che predice la media delle coppie su cui e' stimato.
        def mean_model(train):
            mean = sum(item.realized for item in train) / len(train)
            return ((mean,) * 3,) * 3

        runs = ([pair(1.0, run=0, need=1, gain=1)] * 3
                + [pair(0.0, run=1, need=1, gain=1)] * 3)

        self.assertEqual(cu.held_out_error(runs, mean_model), 1.0)


class MainTest(unittest.TestCase):
    """Dalle tracce alle tabelle da incollare in eviction.py."""

    def run_main(self, runs):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "trace.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                for run in runs:
                    handle.write(json.dumps(run) + "\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cu.main(["calibrate_utility.py", path])
        return code, output.getvalue()

    @staticmethod
    def printed(output, name):
        """Il valore della tabella stampata, letto come Python."""
        start = output.index(f"{name} = (")
        end = output.index(")\n", start) + 1
        return ast.literal_eval(output[start:end].split("=", 1)[1].strip())

    def test_each_mode_prints_its_table_ready_to_paste(self):
        code, output = self.run_main([
            trace("single", {"goal": {"utility": 4}, "dog": {"utility": 3},
                             "home": {"utility": 2}}),
            trace("need_gain", {"goal": {"need": 3, "gain": 3}, "dog": {"need": 1, "gain": 1},
                                "home": {"need": 2, "gain": 2}}),
        ])

        self.assertEqual(code, 0)
        self.assertTrue(is_valid_utility(self.printed(output, "UTILITY_TABLE")))
        table = self.printed(output, "NEED_GAIN_TABLE")
        for line in (*table, *zip(*table)):
            self.assertTrue(is_valid_utility(line), table)

    def test_only_the_runs_of_the_last_commit_count(self):
        old = trace("single", {"goal": {"utility": 4}})
        old["commit"] = "vecchio"

        _, output = self.run_main([old, trace("need_gain", {"goal": {"need": 3, "gain": 3}})])

        self.assertIn("single: nessuna run", output)

    def test_without_traces_it_says_so(self):
        code, output = self.run_main([])

        self.assertEqual(code, 1)
        self.assertIn("Nessuna traccia", output)


if __name__ == "__main__":
    unittest.main()
