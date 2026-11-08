"""Follow-up questions with and without the model's query, on an invented trace.

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
from types import SimpleNamespace
from unittest import mock

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

import followup_questions as fq  # noqa: E402
from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402

from fakes import ScriptedChatModel  # noqa: E402

CONFIG = MemoryConfig(
    node_name="memory_agent",
    maximum_historical_messages=2,
    core_memory_limit=10,
    retrieval_mode="always",
    query_rewrite=False,
    retrieve_k=1,
    answer_with_dates=True,
    generate_answer=True,
    track_used=False,
    eviction=False,
    consolidation_utility="none",
    count_candidate_search=False,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)
TEA = "Mario beve tè nero il pomeriggio"
HOME = "Mario vive a Palermo"
EXCHANGE = {"question": "Cosa bevo il pomeriggio?", "said": "Parliamo di cosa bevo.",
            "reply": "Certo, chiedi pure.", "follow_up": "E a quell'ora?"}
REWRITTEN = "pomeriggio bevanda"
# The follow-up alone lands on the wrong memory; the question, the rewrite and the
# sentence with it land on the tea.
VECTORS = {EXCHANGE["question"]: [1.0, 0.2, 0.0], REWRITTEN: [1.0, 0.0, 0.0],
           EXCHANGE["follow_up"]: [0.0, 1.0, 0.0],
           f"{EXCHANGE['said']} {EXCHANGE['follow_up']}": [0.8, 0.6, 0.0]}


class ReadingModel(ScriptedChatModel):
    """Answers with the tea only when the memory about it reached the prompt."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop, run_manager, **kwargs)
        if kwargs.get("tools"):
            return result
        seen = "tè nero" in " ".join(str(message.content) for message in messages)
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="Bevi tè nero." if seen else "Non lo so."))])


def create(item_id, content):
    return {"op_type": "create", "item_id": item_id, "content": content, "insert": 0,
            "exchange": 0, "timestamp": "2030-01-01T09:00:00", "ratings": None}


class FollowupTest(unittest.TestCase):
    """The four steps on one user with two archived memories and one question."""

    def setUp(self):
        import numpy

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name
        os.makedirs(os.path.join(self.root, "data"))
        user = {"uuid": "u1", "persona_info": "Name: Mario Rossi; Gender: Male",
                "sessions": [{"dialogue": [{"timestamp": "Jan 01, 2030, 09:00:00"},
                                           {"timestamp": "Jan 01, 2030, 09:00:30"}],
                              "end_time": "Jan 01, 2030, 10:00:00"}]}
        with open(os.path.join(self.root, "data", "HaluMem-Medium.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(user) + "\n")
        self.run_dir = os.path.join(self.root, "eval", "results", "memory_service-trace")
        os.makedirs(self.run_dir)
        with open(os.path.join(self.run_dir, "run.json"), "w", encoding="utf-8") as handle:
            json.dump({"config": dataclasses.asdict(CONFIG), "node_sampling": fq.NODE_SAMPLING,
                       "commit": "abc", "users": "evaluated"}, handle)
        question = {"question": "Cosa beve Mario Rossi il pomeriggio?",
                    "asked": "Cosa beve the user il pomeriggio?", "answer": "Tè nero",
                    "evidence": [{"memory_content": "Mario Rossi beve tè nero"}],
                    "question_type": "Basic Fact Recall", "context": ""}
        self.result = {"uuid": "u1", "user_name": "Mario Rossi", "sessions": [
            {"operations": [create("a", TEA), create("b", HOME)], "questions": [question]}]}
        with open(os.path.join(self.run_dir, "memory_service_eval_results.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(self.result) + "\n")
        numpy.savez_compressed(
            fq.replay.vectors_path(self.run_dir, "u1"),
            memory_keys=numpy.array([fq.replay.text_key(TEA), fq.replay.text_key(HOME)]),
            memory_vectors=numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32"),
            question_keys=numpy.array(["0:0"]),
            question_vectors=numpy.array([[1.0, 0.0, 0.0]], dtype="float32"))
        fq.replay.save_json(fq.replay.labels_path(self.run_dir, "u1"),
                            {"labels": {"0:0": {"needed": ["a"]}}})
        self.llm = ReadingModel(tool_responses={"retrieve_memory": {"query": REWRITTEN, "k": 1}})
        backends.reset()
        backends.configure(llm=self.llm, config=CONFIG)
        self.addCleanup(backends.reset)
        self.verdicts = []

    def step(self, name, module=None):
        arguments = fq.parse_arguments([name, "--halumem", self.root, "--version", "trace"])
        output = io.StringIO()
        with mock.patch.object(fq, "halumem_module", return_value=module), \
                mock.patch.object(fq, "query_embedder", return_value=VECTORS.__getitem__), \
                contextlib.redirect_stdout(output):
            self.assertEqual(arguments.step_function(arguments), 0)
        return output.getvalue()

    def judge(self, question, answer, evidence, response):
        self.verdicts.append(question)
        return {"evaluation_result": "Correct" if "tè nero" in response else "Omission"}

    def stored(self, part):
        return fq.load_json(fq.user_path(self.run_dir, part, "u1"))

    def run_steps(self):
        self.step("generate", SimpleNamespace(llm_request_for_json=lambda prompt: EXCHANGE))
        self.step("answer")
        self.step("judge", SimpleNamespace(evaluation_for_question=self.judge))

    def test_each_search_finds_its_memory_and_equal_inputs_share_the_answer(self):
        self.run_steps()

        searches = self.stored("answers")["questions"]["0:0"]["searches"]
        self.assertEqual({name: search["found"] for name, search in searches.items()},
                         {"Q": ["a"], "Q-rw": ["a"], "Qw": ["a"], "Qw-rw": ["a"], "F": ["b"],
                          "F-rw": ["a"], "F+said": ["a"]})
        self.assertEqual(searches["F-rw"]["query"], REWRITTEN)
        # Q with Q-rw, Qw with Qw-rw, F-rw with F+said: the same words, window and memories.
        self.assertEqual(len(self.stored("answers")["answers"]), 4)
        self.assertEqual(len(self.llm.invocations), 3 + 4, "three rewrites, four answers")

    def test_the_judge_reads_the_original_question_once_per_answer(self):
        self.run_steps()

        self.assertEqual(self.verdicts, ["Cosa beve Mario Rossi il pomeriggio?"] * 4)

    def test_the_report_sets_each_search_against_its_reference(self):
        self.run_steps()

        found = fq.user_report(self.stored("answers"), self.stored("judgments"),
                               {"0:0": ["a"]})["values"]

        self.assertEqual([found[name]["found"] for name in ("Qw", "F", "F-rw", "F+said")],
                         [1.0, 0.0, 1.0, 1.0])
        self.assertEqual([found[name]["correct"] for name in ("Qw", "F", "F-rw")],
                         [1.0, 0.0, 1.0])
        self.assertEqual(found["F"]["omitted"], 1.0)
        self.assertIn("F - Qw", self.step("report"))

    def test_launched_again_every_step_asks_nothing(self):
        self.run_steps()
        calls = len(self.llm.invocations)

        self.run_steps()

        self.assertEqual(len(self.llm.invocations), calls)
        self.assertEqual(len(self.verdicts), 4)


class PiecesTest(unittest.TestCase):
    """The parts that need no trace."""

    def test_an_exchange_without_a_field_is_refused(self):
        self.assertEqual(fq.parse_exchange(EXCHANGE), EXCHANGE)
        with self.assertRaises(ValueError):
            fq.parse_exchange({**EXCHANGE, "follow_up": " "})

    def test_a_failed_exchange_is_asked_again(self):
        result = {"sessions": [{"questions": [{"asked": "Cosa bevo?"}]}]}
        stored = {}
        replies = iter([ValueError("no JSON block"), EXCHANGE])

        def ask(prompt):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        fq.generate_user(result, stored, ask, 1)
        self.assertIn("error", stored["0:0"])
        self.assertEqual(fq.generate_user(result, stored, ask, 1), 1)
        self.assertEqual(stored["0:0"], EXCHANGE)

    def test_an_empty_rewrite_searches_the_words_of_the_user(self):
        search = next(search for search in fq.SEARCHES if search.name == "F-rw")

        self.assertEqual(fq.search_query(search, EXCHANGE, {"follow-up": {"written": ""}}),
                         EXCHANGE["follow_up"])
        self.assertIsNone(fq.search_query(search, EXCHANGE, {}))

    def test_the_question_after_the_exchange_is_rewritten_apart(self):
        kinds = {search.name: fq.rewrite_kind(search) for search in fq.SEARCHES
                 if search.rewritten}

        self.assertEqual(kinds, {"Q-rw": "question", "Qw-rw": "question+window",
                                 "F-rw": "follow-up"})

    def test_the_follow_up_has_the_exchange_in_the_window(self):
        state = fq.node_state(fq.Moment([], []), EXCHANGE["follow_up"],
                              [EXCHANGE["said"], EXCHANGE["reply"]])

        self.assertEqual([type(message) for message in state["messages"]],
                         [HumanMessage, AIMessage])
        self.assertEqual(state["messages"][0].content, EXCHANGE["said"])
        self.assertEqual(state["current_query"], EXCHANGE["follow_up"])


if __name__ == "__main__":
    unittest.main()
