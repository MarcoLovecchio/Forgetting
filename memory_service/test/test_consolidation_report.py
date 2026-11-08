"""What consolidation wrote and what it cost, from invented runs and judgments.

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

import consolidation_report as cr  # noqa: E402


def point(score, source="secondary", importance=0.75, memory_type="Persona Memory",
          session=0, content="Mario vive a Palermo"):
    return {"memory_content": content, "memory_type": memory_type, "memory_source": source,
            "importance": importance, "ssession_id": session, "memory_integrity_score": score}


def extracted(score, target=True, session=0):
    return {"memory_content": "Mario lavora come infermiere", "ssession_id": session,
            "memory_accuracy_score": score,
            "is_included_in_golden_memories": "true" if target else "false"}


def update(label, memory_type="Event Memory", session=0):
    return {"memory_content": "Mario si è trasferito a Catania", "memory_type": memory_type,
            "importance": 0.75, "ssession_id": session, "memory_update_type": label}


def judgments(points=(), memories=(), updates=(), answers=()):
    return {"memory_integrity_records": list(points), "memory_accuracy_records": list(memories),
            "memory_update_records": list(updates), "question_answering_records": list(answers)}


def operation(op_type, insert):
    return {"op_type": op_type, "insert": insert, "exchange": insert,
            "item_id": f"m{insert}", "content": "Mario ama il mare"}


def session(exchanges, operations, sources, milliseconds):
    dialogue = [{"role": role, "content": "Ciao", "timestamp": "Sep 04, 2025, 14:46:37"}
                for _ in range(exchanges) for role in ("user", "assistant")]
    return {"dialogue": dialogue, "operations": operations,
            "memory_points": [{"memory_source": source} for source in sources],
            "add_dialogue_duration_ms": milliseconds}


def result(uuid="u1"):
    """Two sessions of two exchanges: the second exchange of the first writes nothing."""
    return {"uuid": uuid, "user_name": "Mario Rossi",
            "sessions": [session(2, [operation("create", 1), operation("archive", 2)],
                                 ("secondary", "interference"), 10000),
                         session(2, [operation("update", 1), operation("redundant", 2),
                                     operation("prune", 2)], ("system",), 6000)],
            "node_stats": {"consolidation": {"calls": 4, "seconds": 8.0, "input_tokens": 4000,
                                             "output_tokens": 800}}}


def values(found):
    return {label: value for label, _, value in found}


class FidelityTest(unittest.TestCase):
    """HaluMem's metrics on one user, as its evaluation.py computes them."""

    def test_recall_counts_full_marks_and_interference_apart(self):
        judged = cr.judged_records(judgments(points=[
            point(2, importance=1.0), point(1), point(0),
            point(0, source="interference"), point(2, source="interference")]))

        found = values(cr.fidelity(judged))

        self.assertAlmostEqual(found["recall"], 1 / 3)
        self.assertAlmostEqual(found["weighted recall"], (1.0 + 0.375) / 2.5)
        self.assertAlmostEqual(found["interference kept out"], 0.5)

    def test_target_accuracy_reads_the_memories_of_gold_points(self):
        judged = cr.judged_records(judgments(
            points=[point(2), point(0)],
            memories=[extracted(2), extracted(1), extracted(0, target=False)]))

        found = values(cr.fidelity(judged))

        self.assertAlmostEqual(found["target accuracy"], 0.75)
        self.assertAlmostEqual(found["weighted accuracy"], 0.5)
        self.assertAlmostEqual(found["F1"], 2 * 0.75 * 0.5 / 1.25)

    def test_an_unusable_judgment_is_left_out(self):
        judged = cr.judged_records(judgments(
            points=[point(2), point(None)], memories=[extracted(2), extracted(None)],
            updates=[update("Correct"), update("Hallucination"), update(None),
                     update("Corretto")]))

        found = values(cr.fidelity(judged))

        self.assertEqual(found["recall"], 1.0)
        self.assertEqual(found["target accuracy"], 1.0)
        self.assertEqual(found["updates correct"], 0.5)
        self.assertEqual(found["updates hallucinated"], 0.5)

    def test_a_user_with_no_updates_has_no_update_values(self):
        found = values(cr.fidelity(cr.judged_records(judgments(points=[point(2)]))))

        self.assertIsNone(found["updates correct"])


class GroupTest(unittest.TestCase):
    """Each record counts in its type and in the bucket of its session."""

    def test_a_point_counts_in_its_own_groups(self):
        judged = cr.judged_records(judgments(
            points=[point(2, memory_type="Event Memory", session=0),
                    point(0, memory_type="Persona Memory", session=60)],
            memories=[extracted(2, session=0), extracted(0, session=60)],
            updates=[update("Omission", memory_type="Relationship Memory", session=12)]))

        kinds = values(cr.by_type(judged))
        sessions = values(cr.by_session(judged))

        self.assertEqual((kinds["recall, event"], kinds["recall, persona"]), (1.0, 0.0))
        self.assertIsNone(kinds["recall, relationship"])
        self.assertEqual(kinds["updates correct, relationship"], 0.0)
        self.assertEqual((sessions["recall, sessions 1-10"], sessions["recall, sessions 51+"]),
                         (1.0, 0.0))
        self.assertEqual(sessions["target accuracy, sessions 51+"], 0.0)
        self.assertEqual(sessions["updates correct, sessions 11-25"], 0.0)

    def test_the_buckets_follow_the_session_number(self):
        self.assertEqual([cr.session_bucket(index) for index in (0, 9, 10, 25, 50, 81)],
                         ["sessions 1-10", "sessions 1-10", "sessions 11-25", "sessions 26-50",
                          "sessions 51+", "sessions 51+"])


class OperationsTest(unittest.TestCase):
    """The operation log and the cost of consolidation, without a judge."""

    def test_an_exchange_that_wrote_nothing_counts_session_by_session(self):
        found = values(cr.operations(result()))

        self.assertEqual(found["exchanges with no operation"], 0.25)
        self.assertEqual((found["create per exchange"], found["update per exchange"],
                          found["redundant per exchange"], found["delete per exchange"]),
                         (0.25, 0.25, 0.25, 0.0))

    def test_a_memory_brought_back_counts_once_until_it_is_archived_again(self):
        def entry(kind, item_id):
            return {"op_type": kind, "item_id": item_id, "insert": 1}

        result = {"sessions": [
            {"operations": [entry("create", "a"), entry("archive", "a")]},
            {"operations": [entry("redundant", "a"), entry("redundant", "a"),
                            entry("archive", "a"), entry("redundant", "a")]}]}

        self.assertEqual(cr.from_archive(result), (2, 3))

    def test_written_per_point_leaves_interference_out(self):
        self.assertEqual(values(cr.operations(result()))["written per memory point"], 1.0)

    def test_the_cost_is_per_call_and_per_exchange(self):
        found = values(cr.cost(result()))

        self.assertEqual((found["calls per exchange"], found["seconds per call"],
                          found["input tokens per call"], found["output tokens per call"],
                          found["seconds per exchange"]), (1.0, 2.0, 1000, 200, 2.0))
        self.assertEqual(found["share of the dialogue time"], 0.5)


class AnswersTest(unittest.TestCase):
    """The answers HaluMem judged, and what they cost the user in time."""

    def test_the_three_verdicts_with_the_unusable_one_left_out(self):
        records = [{"result_type": verdict, "question_type": kind} for verdict, kind in (
            ("Correct", "Memory Boundary"), ("Omission", "Basic Fact Recall"),
            ("Hallucination", "Basic Fact Recall"), ("Correct", "Basic Fact Recall"),
            (None, "Basic Fact Recall"))]

        found = values(cr.answers(cr.judged_records(judgments(answers=records))))

        self.assertEqual((found["correct"], found["hallucinated"], found["omitted"]),
                         (0.5, 0.25, 0.25))
        self.assertAlmostEqual(found["correct, Basic Fact Recall"], 1 / 3)
        self.assertEqual(found["correct, Memory Boundary"], 1.0)
        self.assertIsNone(found["correct, Dynamic Update"])

    def test_the_answer_costs_per_call_and_per_question(self):
        user = {"sessions": [{"questions": [
            {"question": "Dove vive Mario?", "vector_search_ms": 40.0,
             "search_duration_ms": 900.0},
            {"question": "Cosa beve Mario?", "vector_search_ms": 60.0,
             "search_duration_ms": 1100.0}]}],
            "node_stats": {"generate_answer": {"calls": 2, "seconds": 1.2, "input_tokens": 3000,
                                               "output_tokens": 100}}}

        found = values(cr.answer_cost(user))

        self.assertEqual((found["seconds per answer"], found["input tokens per answer"],
                          found["output tokens per answer"]), (0.6, 1500, 50))
        self.assertEqual((found["archive search per question"],
                          found["retrieve branch per question"]), (50.0, 1000.0))


class ComparisonTest(unittest.TestCase):
    """Versions side by side, on the users all of them hold."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def write_version(self, name, scores):
        """A run of the users in `scores`, judged when the score is not None."""
        run_dir = os.path.join(self.folder.name, "eval", "results", f"memory_service-{name}")
        os.makedirs(os.path.join(run_dir, "tmp2"))
        with open(os.path.join(run_dir, "memory_service_eval_results.jsonl"), "w",
                  encoding="utf-8") as handle:
            for uuid in scores:
                handle.write(json.dumps(result(uuid)) + "\n")
        for uuid, score in scores.items():
            if score is not None:
                with open(os.path.join(run_dir, "tmp2", f"{uuid}.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump(judgments(points=[point(score), point(2)]), handle)

    def report(self, *names):
        path = os.path.join(self.folder.name, "report.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cr.main(["--halumem", self.folder.name, "--versions", *names,
                                      "--csv", path]), 0)
        with open(path, encoding="utf-8") as handle:
            return {(row["kpi"], row["version"]): row for row in csv.DictReader(handle)}

    def test_the_difference_is_paired_user_by_user(self):
        self.assertEqual([round(value, 6) for value in
                          cr.paired({"a": 0.5, "b": 0.7, "c": None},
                                    {"a": 0.6, "b": 0.6, "d": 0.9})], [0.1, -0.1])

    def test_the_interval_needs_two_users(self):
        low, high = cr.bootstrap([0.1, 0.2, 0.3])
        self.assertTrue(0.1 <= low < 0.2 < high <= 0.3)
        self.assertIsNone(cr.bootstrap([0.2]))

    def test_only_the_users_of_every_version_are_compared(self):
        self.write_version("none", {"u1": 0, "u2": 0})
        self.write_version("need_gain", {"u1": 2})

        rows = self.report("none", "need_gain")

        self.assertEqual(rows[("recall", "none")]["users"], "1")
        self.assertAlmostEqual(float(rows[("recall", "none")]["mean"]), 0.5)
        self.assertAlmostEqual(float(rows[("recall", "need_gain")]["difference"]), 0.5)

    def test_a_version_not_yet_judged_still_has_operations_and_cost(self):
        self.write_version("judged", {"u1": 2})
        self.write_version("replayed", {"u1": None})

        rows = self.report("judged", "replayed")

        self.assertNotIn(("recall", "judged"), rows)
        self.assertEqual(rows[("seconds per call", "replayed")]["mean"], "2.0")


if __name__ == "__main__":
    unittest.main()
