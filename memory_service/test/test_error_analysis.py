"""Where each answer failed, on an invented trace with one question per case.

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

import error_analysis as ea  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402

CONFIG = MemoryConfig(
    node_name="memory_agent",
    maximum_historical_messages=2,
    core_memory_limit=25,
    retrieval_mode="always",
    query_rewrite=False,
    eviction=True,
    consolidation_utility="none",
    count_candidate_search=False,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)
EVIDENCE = [{"memory_content": "Mario beve te' nero"}]
# (question, evidence, context, needed, verdict): None for a question never judged, and
# needed None for one never labelled.
QUESTIONS = (
    ("Cosa beve Mario?", EVIDENCE, "ID: a, Date: 2030-01-01, Content: x", ["a"], "Correct"),
    ("Mario beve te'?", EVIDENCE, "", ["a"], "Omission"),
    ("Dove vive Mario?", EVIDENCE, "", [], "Hallucination"),
    ("Che lavoro fa Mario?", EVIDENCE, "Date: 2030-01-01, Content: y", ["b"], "Correct"),
    ("Cosa beve e dove lavora Mario?", EVIDENCE, "", ["a", "b"], "Omission"),
    ("Mario ha un drago?", [], "", [], "Hallucination"),
    ("Mario ha un cane?", EVIDENCE, "", ["a"], None),
    ("Mario ha un gatto?", EVIDENCE, "", None, "Correct"),
)


def operation(kind, item_id, insert, hour):
    return {"op_type": kind, "item_id": item_id, "content": ("x" if item_id == "a" else "y") * 20,
            "insert": insert, "exchange": 0, "timestamp": f"2030-01-01T{hour:02}:00:00",
            "ratings": None}


class ErrorAnalysisTest(unittest.TestCase):
    """a is archived by the split at the flush, b stays in core."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name
        os.makedirs(os.path.join(self.root, "data"))
        user = {"uuid": "u1", "persona_info": "Name: Mario Rossi; Gender: Male",
                "sessions": [{"dialogue": [{"timestamp": "Jan 01, 2030, 09:00:00"},
                                           {"timestamp": "x"}],
                              "end_time": "Jan 01, 2030, 10:00:00"}]}
        with open(os.path.join(self.root, "data", "HaluMem-Medium.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(user) + "\n")
        run_dir = os.path.join(self.root, "eval", "results", "memory_service-trace")
        os.makedirs(os.path.join(run_dir, "tmp2"))
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump({"config": dataclasses.asdict(CONFIG), "commit": "abc"}, handle)
        questions = [{"question": text, "evidence": evidence, "context": context,
                      "question_type": "Basic Fact Recall", "answer": "-"}
                     for text, evidence, context, _, _ in QUESTIONS]
        result = {"uuid": "u1", "user_name": "Mario Rossi", "sessions": [{
            "operations": [operation("create", "a", 0, 9), operation("create", "b", 1, 10),
                           operation("archive", "a", 1, 10)],
            "questions": questions}]}
        with open(os.path.join(run_dir, "memory_service_eval_results.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        with open(os.path.join(run_dir, "tmp2", "u1.json"), "w", encoding="utf-8") as handle:
            json.dump({"question_answering_records": [
                {"ssession_id": 0, "question": text, "result_type": verdict}
                for text, _, _, _, verdict in QUESTIONS if verdict]}, handle)
        ea.replay.save_json(ea.replay.labels_path(run_dir, "u1"), {"labels": {
            f"0:{number}": {"needed": needed}
            for number, (_, _, _, needed, _) in enumerate(QUESTIONS) if needed is not None}})

    def test_each_question_is_placed_where_its_memory_failed(self):
        rows, _, agreements, read = ea.load_version(self.root, "trace")

        self.assertEqual({row["question"]: row["category"] for row in rows},
                         {"0:0": "found", "0:1": "not found", "0:2": "not stored",
                          "0:3": "found", "0:4": "found in part", "0:5": ea.NO_EVIDENCE})
        self.assertEqual((agreements, read), ([1.0], 1), "the replay archives what the trace did")

    def test_what_was_not_judged_or_labelled_is_left_out(self):
        _, left_out, _, _ = ea.load_version(self.root, "trace")

        self.assertEqual(dict(left_out), {"not judged": 1, "not labelled": 1})

    def test_the_wrong_answers_split_by_layer(self):
        rows, _, _, _ = ea.load_version(self.root, "trace")

        self.assertEqual(ea.layers(rows), {"wrong": 3, "not stored": 1, "not found": 1,
                                           "in context": 1, "in part": 1})

    def test_the_report_writes_the_counts(self):
        path = os.path.join(self.root, "errors.csv")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ea.main(["--halumem", self.root, "--versions", "trace",
                                      "--csv", path]), 0)

        with open(path, encoding="utf-8") as handle:
            counts = {(row["category"], row["verdict"]): row["count"]
                      for row in csv.DictReader(handle)}
        self.assertEqual(counts[("found", "Correct")], "2")
        self.assertEqual(counts[(ea.NO_EVIDENCE, "Hallucination")], "1")


class CategoryTest(unittest.TestCase):
    def test_a_question_is_found_only_when_every_needed_memory_is_in_context(self):
        self.assertEqual([ea.category(needed, {"a"}) for needed in ([], ["b"], ["a", "b"], ["a"])],
                         ["not stored", "not found", "found in part", "found"])


if __name__ == "__main__":
    unittest.main()
