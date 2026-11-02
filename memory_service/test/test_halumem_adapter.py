"""The HaluMem adapter, on a small invented user, without a model.

Run: python memory_service/run_tests.py -v
"""

import dataclasses
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import halumem_adapter as adapter  # noqa: E402
from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.memory_manager_llm import MemoryAgent  # noqa: E402

from fakes import FakeVectorStore, ScriptedChatModel  # noqa: E402


CONFIG = MemoryConfig(
    node_name="memory_agent",
    maximum_historical_messages=2,
    core_memory_limit=2000,
    archive_memory_limit=50,
    retrieval_mode="always",
    consolidation_utility="none",
    count_candidate_search=True,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)


def turn(role, content, minute):
    return {"role": role, "content": content, "timestamp": f"Dec 08, 2033, 10:{minute:02}:00"}


def session(*exchanges, **extra):
    dialogue = []
    for minute, (said, replied) in enumerate(exchanges):
        dialogue += [turn("user", said, minute), turn("assistant", replied, minute)]
    return {"dialogue": dialogue, "memory_points": [], "start_time": "Dec 08, 2033, 10:00:00",
            "end_time": "Dec 08, 2033, 11:00:00", **extra}


def point(content, is_update="False", original=()):
    return {"memory_content": content, "is_update": is_update, "original_memories": list(original)}


USER = {"uuid": "uuid-1", "persona_info": "Name: Rossi Mario; Gender: Male; Age: 30"}


class TextTest(unittest.TestCase):
    """Between the names of the dataset and the user of the service."""

    def test_every_form_of_the_name_becomes_the_user(self):
        for text, expected in (
                ("Che cosa ha comprato Rossi Mario?", "Che cosa ha comprato the user?"),
                ("Il cane di Mario's sister?", "Il cane di the user's sister?"),
                ("Rossi vive a Mondello?", "the user vive a Mondello?"),
                ("Mariolina vive a Mondello?", "Mariolina vive a Mondello?")):
            with self.subTest(text=text):
                self.assertEqual(adapter.neutral(text, "Rossi Mario"), expected)

    def test_the_user_becomes_the_person(self):
        self.assertEqual(adapter.personalized("The user has a dog, the user's dog", "Rossi Mario"),
                         "Rossi Mario has a dog, Rossi Mario's dog")

    def test_only_what_was_stored_or_confirmed_is_extracted_once(self):
        operations = [{"op_type": op_type, "content": content} for op_type, content in (
            ("create", "The user has a dog"), ("redundant", "The user has a dog"),
            ("update", "The user has two dogs"), ("delete", "The user has a cat"),
            ("archive", "The user runs"), ("prune", "The user swims"))]

        self.assertEqual(adapter.extracted_memories(operations, "Mario"),
                         ["Mario has a dog", "Mario has two dogs"])


class ReplayTest(unittest.TestCase):
    """A session through the real graph, with the model and the archive replaced."""

    def setUp(self):
        self.store = FakeVectorStore()
        self.llm = ScriptedChatModel(tool_responses={}, default_content="Il cane si chiama Argo")
        self.llm.invocations = []
        backends.reset()
        backends.configure(llm=self.llm, vector_store=self.store, config=CONFIG)
        MemoryAgent.reset_instance()
        self.agent = MemoryAgent(config=CONFIG)
        self.addCleanup(backends.reset)
        self.addCleanup(MemoryAgent.reset_instance)
        self.script("The user has a dog named Argo")

    def script(self, fact):
        self.llm.script({
            "InsertCoreMemories": {"memories": [{"fact": fact, "operation": "new"}]},
            "retrieve_memory": {"query": "dog", "k": 3},
        })

    def consolidations(self):
        return self.llm.bound_tool_names().count("InsertCoreMemories")

    def replay(self, replayed_session):
        return adapter.replay_session(self.agent, replayed_session, "Rossi Mario")

    def test_the_last_exchange_is_consolidated_and_the_next_session_starts_empty(self):
        self.replay(session(("a", "b"), ("c", "d"), ("e", "f")))

        self.assertEqual(self.consolidations(), 3, "two by the window, one by the flush")
        self.assertEqual(self.agent.state["messages"], [])
        self.assertEqual(self.agent.config, CONFIG, "the window of the flush does not stay")

    def test_the_memories_carry_the_time_of_the_dataset(self):
        self.replay(session(("a", "b"), ("c", "d")))

        created = [item.updated_at for item in self.agent.state["core_memory"]]
        self.assertEqual(len(created), 2)
        self.assertEqual(created[0].replace(microsecond=0), datetime(2033, 12, 8, 10, 1))
        self.assertEqual(created[1].replace(microsecond=0), datetime(2033, 12, 8, 11, 0))

    def test_what_the_evaluation_reads(self):
        replayed = self.replay(session(("a", "b"), ("c", "d")))

        self.assertEqual(replayed["extracted_memories"], ["Rossi Mario has a dog named Argo"])
        self.assertEqual([entry["exchange"] for entry in replayed["operations"]], [1, 1])
        self.assertGreater(replayed["add_dialogue_duration_ms"], 0)
        self.assertEqual(replayed["active_core"], 2)
        self.assertNotIn("questions", replayed)

    def test_an_update_point_gets_the_memories_of_the_system(self):
        replayed = self.replay(session(("a", "b"), memory_points=[
            point("Mario has a dog"), point("Mario has two dogs", "True", ["Mario has a dog"])]))

        plain, update = replayed["memory_points"]
        self.assertNotIn("memories_from_system", plain)
        self.assertEqual(update["memories_from_system"], ["Rossi Mario has a dog named Argo"])

    def test_the_evaluation_of_an_update_is_not_a_retrieval(self):
        self.script("The user has a dog named Argo")
        self.agent.state["core_memory"] = []
        self.store.add_texts(["The user has two dogs"], ids=["kept"], metadatas=[
            {"status": "active", "n_retrieve": 0}])

        self.assertEqual(adapter.memories_about(self.agent, "Mario has two dogs", "Rossi Mario"),
                         ["Rossi Mario has two dogs"])
        self.assertEqual(self.store.metadatas["kept"]["n_retrieve"], 0)

    def test_the_question_is_answered_and_leaves_no_trace(self):
        question = {"question": "Come si chiama il cane di Mario?", "answer": "Argo",
                    "evidence": [], "question_type": "Basic Fact Recall"}

        replayed = self.replay(session(("a", "b"), questions=[question]))

        answered = replayed["questions"][0]
        self.assertEqual(answered["system_response"], "Il cane si chiama Argo")
        self.assertEqual(answered["asked"], "Come si chiama il cane di the user?")
        self.assertEqual(answered["question"], question["question"], "the judge sees the original")
        self.assertIn("The user has a dog named Argo", answered["context"])
        self.assertEqual(self.agent.state["messages"], [])
        self.assertNotIn("search_duration_ms", question, "the dataset stays as it is")

    def test_the_same_question_twice_is_asked_twice(self):
        question = {"question": "Dove vive Mario?", "answer": "Mondello", "evidence": []}

        replayed = self.replay(session(("a", "b"), questions=[question, question]))

        self.assertEqual([answered["system_response"] for answered in replayed["questions"]],
                         ["Il cane si chiama Argo"] * 2)

    def test_the_sessions_of_a_user_share_the_memory(self):
        user = {**USER, "sessions": [session(("a", "b")), session(("c", "d"))]}

        replayed = adapter.replay_user(self.agent, user)

        self.assertEqual(replayed["user_name"], "Rossi Mario")
        self.assertEqual([one["active_core"] for one in replayed["sessions"]], [1, 2])

    def test_a_failed_call_is_tried_again(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise ConnectionError("cluster")
            return "ok"

        with mock.patch.object(adapter.time, "sleep"):
            self.assertEqual(adapter.with_retries(flaky), "ok")

    def test_a_call_that_keeps_failing_is_not_hidden(self):
        with mock.patch.object(adapter.time, "sleep"), self.assertRaises(ConnectionError):
            adapter.with_retries(mock.Mock(side_effect=ConnectionError("cluster")))


class RunTest(unittest.TestCase):
    """The configuration and the files of a run."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = folder.name
        os.makedirs(os.path.join(self.run_dir, "tmp"))

    def arguments(self, *extra):
        return adapter.parse_arguments(["run", "--halumem", "x", "--version", "v", *extra])

    def test_what_the_experiments_fix_does_not_depend_on_the_defaults(self):
        with mock.patch.object(adapter.MemoryConfig, "from_environment",
                               return_value=MemoryConfig(retrieval_mode="decide",
                                                         eviction_time_decay_function="weibull",
                                                         maximum_historical_messages=5)):
            config = adapter.run_config(self.arguments())

        self.assertEqual(config.retrieval_mode, "always")
        self.assertEqual(config.eviction_time_decay_function, "power")
        self.assertEqual(config.maximum_historical_messages, 2)
        self.assertTrue(config.count_candidate_search)

    def test_what_the_experiments_vary_comes_from_the_command_line(self):
        environment = MemoryConfig(llm_config={"model_name": "piccolo", "temperature": 1.0})
        with mock.patch.object(adapter.MemoryConfig, "from_environment",
                               return_value=environment):
            config = adapter.run_config(self.arguments(
                "--utility", "single", "--combination", "product", "--core-limit", "300",
                "--archive-limit", "20", "--prune-fraction", "0.25", "--model", "grande"))

        self.assertEqual((config.consolidation_utility, config.eviction_combination,
                          config.core_memory_limit, config.archive_memory_limit,
                          config.prune_fraction),
                         ("single", "product", 300, 20, 0.25))
        self.assertEqual(config.llm_config, {"model_name": "grande", "temperature": 1.0})

    def test_a_run_is_not_mixed_with_a_different_one(self):
        with mock.patch.object(adapter, "git_commit", return_value="abc"):
            adapter.check_manifest(self.run_dir, CONFIG)
            adapter.check_manifest(self.run_dir, CONFIG)
            with self.assertRaises(SystemExit):
                adapter.check_manifest(self.run_dir,
                                       dataclasses.replace(CONFIG, prune_fraction=0.5))

    def test_the_results_file_keeps_the_order_and_only_who_finished(self):
        for uuid in ("b", "a"):
            with open(adapter.result_path(self.run_dir, uuid), "w", encoding="utf-8") as handle:
                json.dump({"uuid": uuid}, handle)

        merged = adapter.merge_results([{"uuid": name} for name in "acb"], self.run_dir)

        self.assertEqual(merged, 2)
        path = os.path.join(self.run_dir, f"{adapter.FRAME}_eval_results.jsonl")
        self.assertEqual([user["uuid"] for user in adapter.load_users(path)], ["a", "b"])

    def test_launched_again_the_results_file_does_not_double(self):
        with open(adapter.result_path(self.run_dir, "a"), "w", encoding="utf-8") as handle:
            json.dump({"uuid": "a"}, handle)

        adapter.merge_results([{"uuid": "a"}], self.run_dir)
        adapter.merge_results([{"uuid": "a"}], self.run_dir)

        path = os.path.join(self.run_dir, f"{adapter.FRAME}_eval_results.jsonl")
        self.assertEqual(len(adapter.load_users(path)), 1)

    def test_models_that_do_not_answer_stop_the_run_before_it_starts(self):
        arguments = adapter.parse_arguments(["run", "--halumem", self.run_dir, "--version", "v"])
        self.addCleanup(backends.reset)
        with mock.patch("live_model.live_stack_unavailable", return_value="LLM_CONFIG missing"), \
                mock.patch.object(adapter, "run_config", return_value=CONFIG), \
                self.assertRaises(SystemExit) as stopped:
            adapter.run(arguments)

        self.assertIn("LLM_CONFIG missing", str(stopped.exception))
        self.assertFalse(os.path.exists(adapter.run_folder(arguments)))

    def test_a_user_that_fails_leaves_the_traceback_and_no_result(self):
        os.makedirs(os.path.join(self.run_dir, "logs"))
        user = {**USER, "sessions": []}
        with mock.patch.object(adapter, "replay_user", side_effect=ConnectionError("cluster")), \
                mock.patch.object(adapter.backends, "configure"):
            outcome = adapter.process_user(user, CONFIG, self.run_dir)

        self.assertEqual(outcome, ("uuid-1", "failed"))
        self.assertFalse(os.path.exists(adapter.result_path(self.run_dir, "uuid-1")))
        with open(os.path.join(self.run_dir, "tmp", "uuid-1_error.log"), encoding="utf-8") as log:
            self.assertIn("ConnectionError", log.read())


if __name__ == "__main__":
    unittest.main()
