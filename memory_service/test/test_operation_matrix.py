"""La matrice di confusione delle operazioni, verificata senza modello.

Esecuzione: python memory_service/run_tests.py -v
"""

import contextlib
import io
import os
import re
import sys
import tempfile
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import fama  # noqa: E402
import operation_matrix as om  # noqa: E402


LONGRUN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "test_long_term_interaction.py")

_TAG_TO_CLASS = {"": "new", "new": "new", "update": "update", "contradict": "update",
                 "redundant": "redundant", "delete": "delete", "domanda": "none",
                 "chiacchiere": "none"}


def conversation_tags():
    """Il commento accanto a ogni messaggio di CONVERSATION, nell'ordine."""
    with open(LONGRUN, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("CONVERSATION = ["))
    tags = []
    for line in lines[start + 1:]:
        if line.strip() == "]":
            break
        if line.strip().startswith('"'):
            match = re.search(r"#\s*([A-Za-z]+)", line)
            tags.append(match.group(1).lower() if match else "")
    return tags


class ExpectedTableTest(unittest.TestCase):
    """La tabella delle operazioni attese contro la conversazione vera."""

    def test_the_table_agrees_with_the_comments_of_the_conversation(self):
        tags = conversation_tags()
        self.assertEqual(len(tags), 117)
        for index, tag in enumerate(tags, 1):
            self.assertEqual(om.expected(index)[0], _TAG_TO_CLASS[tag],
                             "msg %d (%s)" % (index, tag or "presentazione"))

    def test_the_questions_are_the_ones_fama_evaluates(self):
        none = {index for index in range(1, 118) if om.expected(index)[0] == "none"}
        self.assertEqual(none - {26}, set(fama.QUESTIONS))

    def test_alternatives_are_real_classes_and_not_the_label_again(self):
        for index, also in om.ACCEPTED_ALSO.items():
            for combination in also:
                self.assertNotEqual(tuple(combination), (om.expected(index)[0],), "msg %d" % index)
                self.assertTrue(set(combination) <= set(om.CLASSES), "msg %d" % index)

    def test_a_changed_value_always_needs_an_update(self):
        # 2000 e 2200 attivi insieme sarebbero una contraddizione in memoria: un
        # new passa solo accanto all'update che toglie il valore vecchio.
        for index in (29, 42, 48, 54, 59, 94, 105):
            for combination in om.expected(index)[1]:
                self.assertIn("update", combination, "msg %d" % index)

    def test_expected_repeats_are_on_facts(self):
        for index in om.MULTIPLE_EXPECTED:
            self.assertNotEqual(om.expected(index)[0], "none", "msg %d" % index)


class ClassifyTest(unittest.TestCase):
    """Dalle voci del log di un messaggio alla sua etichetta."""

    def test_create_is_new_and_the_split_is_not_a_classification(self):
        self.assertEqual(om.classify(["create", "archive"]), ("new", ("new",), 1))
        self.assertEqual(om.classify(["evict"]), ("none", (), 0))

    def test_two_new_are_one_new_counted_twice(self):
        self.assertEqual(om.classify(["create", "create"]), ("new", ("new",), 2))

    def test_different_types_are_mixed(self):
        self.assertEqual(om.classify(["create", "redundant"])[0], om.MIXED)

    def test_nothing_is_none(self):
        self.assertEqual(om.classify([])[0], "none")


class AttributionTest(unittest.TestCase):
    """A quale messaggio appartengono le operazioni di un turno."""

    def test_the_first_turn_consolidates_nothing(self):
        self.assertLess(om.consolidated_message(1, 2), 1)

    def test_it_agrees_with_the_fama_snapshot(self):
        # Al turno in cui si fotografa la memoria di una domanda si consolida il
        # messaggio che la precede: se i due conti divergono, uno dei due mente.
        for window in (1, 2, 4, 6):
            for question in fama.QUESTIONS:
                turn = fama.memory_snapshot_turn(question, window)
                self.assertEqual(om.consolidated_message(turn, window), question - 1)


class AccuracyTest(unittest.TestCase):
    """Accuratezza stretta e tollerante, e la matrice."""

    def test_strict_and_lenient(self):
        exact = om.row(29, ["update"])
        tolerated = om.row(66, ["create"])            # il mercoledi' come new
        wrong = om.row(29, ["create"])                # 2200 accanto a 2000
        mixed = om.row(52, ["redundant", "create"])   # pianoforte e chitarra
        self.assertTrue(om.is_strict(exact) and om.is_lenient(exact))
        self.assertTrue(not om.is_strict(tolerated) and om.is_lenient(tolerated))
        self.assertFalse(om.is_strict(wrong) or om.is_lenient(wrong))
        self.assertTrue(not om.is_strict(mixed) and om.is_lenient(mixed))

    def test_the_types_must_match_a_combination_not_be_part_of_it(self):
        self.assertFalse(om.is_lenient(om.row(52, ["redundant"])))         # chitarra persa
        self.assertTrue(om.is_lenient(om.row(54, ["update", "create"])))   # corsa e ginocchio
        self.assertFalse(om.is_lenient(om.row(54, ["create"])))            # corre e non corre

    def test_repeats_are_fragmentation_only_where_not_expected(self):
        text = "\n".join(om.format_report([
            om.row(18, ["create", "create"]), om.row(19, ["create", "create"]),
            om.row(112, ["delete", "delete"]), om.row(33, ["update", "update"]),
            om.row(54, ["update", "create", "create"]), om.row(12, ["create"])]))
        self.assertIn("Frammentazione: 2 messaggi con piu' operazioni dello stesso tipo "
                      "(msg 33, 54), piu' 3 dove sono previste", text)

    def test_a_question_that_produced_nothing_is_right(self):
        self.assertTrue(om.is_strict(om.row(27, [])))
        self.assertFalse(om.is_lenient(om.row(27, ["create"])))

    def test_the_matrix_counts_expected_against_predicted(self):
        counts = om.matrix([om.row(29, ["update"]), om.row(33, ["create"]), om.row(44, [])])
        self.assertEqual(counts["update"]["update"], 1)
        self.assertEqual(counts["update"]["new"], 1)
        self.assertEqual(counts["delete"]["none"], 1)

    def test_the_report_names_every_class(self):
        text = "\n".join(om.format_report([om.row(29, ["update"]), om.row(27, [])]))
        for name in ("new", "redundant", "update", "delete", "nessuna", "misto", "precisione"):
            self.assertIn(name, text)
        self.assertIn("Accuratezza stretta 2/2", text)


class SavedRunsTest(unittest.TestCase):
    """Le run si salvano una per riga e si sommano per commit."""

    def test_runs_are_appended_and_read_back(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "runs.jsonl")
            om.save_run(path, {"commit": "a", "messages": [om.row(29, ["update"])]})
            om.save_run(path, {"commit": "b", "messages": [om.row(29, ["create"])]})
            runs = om.load_runs(path)
        self.assertEqual([run["commit"] for run in runs], ["a", "b"])
        self.assertEqual(runs[1]["messages"][0]["predicted"], "new")

    def test_a_missing_file_is_no_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(om.load_runs(os.path.join(folder, "none.jsonl")), [])

    def test_the_summary_sums_only_the_last_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "runs.jsonl")
            om.save_run(path, {"commit": "a", "messages": [om.row(29, ["update"])]})
            om.save_run(path, {"commit": "b", "messages": [om.row(44, [])]})
            om.save_run(path, {"commit": "b", "messages": [om.row(44, ["delete"])]})
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = om.main(["operation_matrix.py", path])
        self.assertEqual(code, 0)
        self.assertIn("2 run del commit b su 3 salvate", out.getvalue())
        self.assertIn("1/2", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
