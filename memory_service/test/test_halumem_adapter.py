"""The HaluMem adapter, on a small invented user, without a model.

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
    query_rewrite=False,
    consolidation_utility="none",
    count_candidate_search=False,
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
            "retrieve_memory": {"k": 3},
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
        self.assertEqual([entry["insert"] for entry in replayed["operations"]], [1, 2],
                         "the flush is an insert of its own")
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

    def test_the_question_gets_the_memories_and_the_days_of_the_benchmark(self):
        self.agent.config = dataclasses.replace(CONFIG, retrieve_k=20, answer_with_dates=True)
        question = {"question": "Come si chiama il cane di Mario?", "answer": "Argo",
                    "evidence": []}

        replayed = self.replay(session(("a", "b"), questions=[question]))

        self.assertEqual(self.store.searches[-1]["k"], 20)
        self.assertIn("Date: 2033-12-08, Content: The user has a dog named Argo",
                      replayed["questions"][0]["context"])

    def test_the_search_of_a_question_is_timed_apart_from_its_answer(self):
        question = {"question": "Come si chiama il cane di Mario?", "answer": "Argo",
                    "evidence": []}
        self.replay(session(("a", "b")))
        searches = len(self.store.searches)

        answered = adapter.answer_question(self.agent, question, "Rossi Mario")

        self.assertEqual(len(self.store.searches) - searches, 1)
        self.assertGreaterEqual(answered["vector_search_ms"], 0)
        self.assertLessEqual(answered["vector_search_ms"], answered["search_duration_ms"])

    def test_every_session_keeps_the_size_of_the_archive_on_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "chroma.sqlite3"), "wb") as handle:
                handle.write(b"\0" * 1234)
            self.agent.config = dataclasses.replace(CONFIG, chroma_path=folder)

            replayed = self.replay(session(("a", "b")))

        self.assertEqual(replayed["archive_bytes"], 1234)

    def test_the_searches_of_a_user_are_counted_with_the_model_calls(self):
        user = {**USER, "sessions": [session(("a", "b"))]}
        adapter.reset_node_stats()

        replayed = adapter.replay_user(self.agent, user)

        self.assertEqual(replayed["node_stats"]["archive_search"]["calls"],
                         len(self.store.searches))

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
        self.assertNotIn("unrated", replayed, "a mode that rates nothing has nothing to count")

    def test_a_user_keeps_how_many_facts_came_without_ratings(self):
        self.agent.config = dataclasses.replace(CONFIG, consolidation_utility="need_gain")
        user = {**USER, "sessions": [session(("a", "b"), ("c", "d"))]}

        replayed = adapter.replay_user(self.agent, user)

        self.assertEqual(replayed["unrated"], {"without_ratings": 2, "of": 2},
                         "the scripted model gives no rating")

    def test_a_user_keeps_how_many_times_each_level_was_given(self):
        self.agent.config = dataclasses.replace(CONFIG, consolidation_utility="need_gain")
        self.llm.script({"InsertCoreMemories": {"memories": [
            {"fact": "The user has a dog named Argo", "operation": "new", "safety": False,
             "need": 3, "gain": 2}]}})
        user = {**USER, "sessions": [session(("a", "b"), ("c", "d"))]}

        replayed = adapter.replay_user(self.agent, user)

        self.assertEqual(replayed["unrated"], {"without_ratings": 0, "of": 2})
        self.assertEqual(replayed["ratings"], {"need": {"1": 0, "2": 0, "3": 2},
                                               "gain": {"1": 0, "2": 2, "3": 0}})

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
                                                         maximum_historical_messages=5,
                                                         count_candidate_search=True,
                                                         query_rewrite=True)):
            config = adapter.run_config(self.arguments())

        self.assertEqual(config.retrieval_mode, "always")
        self.assertFalse(config.query_rewrite)
        self.assertEqual(config.eviction_time_decay_function, "power")
        self.assertEqual(config.maximum_historical_messages, 2)
        self.assertFalse(config.count_candidate_search)
        self.assertEqual(config.retrieve_k, 20)
        self.assertTrue(config.answer_with_dates)

    def test_what_the_experiments_vary_comes_from_the_command_line(self):
        environment = MemoryConfig(llm_config={"model_name": "piccolo", "temperature": 1.0})
        with mock.patch.object(adapter.MemoryConfig, "from_environment",
                               return_value=environment):
            config = adapter.run_config(self.arguments(
                "--utility", "single", "--combination", "product", "--frequency", "rate",
                "--core-limit", "300", "--archive-limit", "20", "--prune-fraction", "0.25",
                "--model", "grande"))

        self.assertEqual((config.consolidation_utility, config.eviction_combination,
                          config.eviction_frequency_function, config.core_memory_limit,
                          config.archive_memory_limit, config.prune_fraction),
                         ("single", "product", "rate", 300, 20, 0.25))
        self.assertEqual(config.llm_config, {"model_name": "grande", "temperature": 1.0})

    def test_a_run_is_not_mixed_with_a_different_one(self):
        with mock.patch.object(adapter, "git_commit", return_value="abc"):
            adapter.check_manifest(self.run_dir, CONFIG, "evaluated")
            adapter.check_manifest(self.run_dir, CONFIG, "evaluated")
            with self.assertRaises(SystemExit):
                adapter.check_manifest(self.run_dir,
                                       dataclasses.replace(CONFIG, prune_fraction=0.5),
                                       "evaluated")

    def test_the_calibration_users_never_share_a_run_with_the_evaluated_ones(self):
        with mock.patch.object(adapter, "git_commit", return_value="abc"):
            adapter.check_manifest(self.run_dir, CONFIG, "calibration")
            with self.assertRaises(SystemExit):
                adapter.check_manifest(self.run_dir, CONFIG, "evaluated")

    def test_the_results_file_keeps_the_order_and_only_who_finished(self):
        for uuid in ("b", "a"):
            with open(adapter.result_path(self.run_dir, uuid), "w", encoding="utf-8") as handle:
                json.dump({"uuid": uuid}, handle)

        merged = adapter.merge_results([{"uuid": name} for name in "acb"], self.run_dir)

        self.assertEqual(merged, 2)
        path = os.path.join(self.run_dir, f"{adapter.FRAME}_eval_results.jsonl")
        self.assertEqual([user["uuid"] for user in adapter.load_users(path)], ["a", "b"])

    def test_the_judgment_is_summed_over_the_users_that_finished(self):
        for uuid, counter, need in (("a", {"without_ratings": 1, "of": 40}, [5, 30, 4]),
                                    ("b", {"without_ratings": 0, "of": 35}, [2, 25, 8])):
            ratings = {"need": dict(zip("123", need))}
            with open(adapter.result_path(self.run_dir, uuid), "w", encoding="utf-8") as handle:
                json.dump({"uuid": uuid, "unrated": counter, "ratings": ratings}, handle)

        self.assertEqual(adapter.finished_judgment([{"uuid": name} for name in "abc"],
                                                   self.run_dir),
                         ", 1 of 75 new and updated facts without ratings (safety facts apart); "
                         "need 1: 7, 2: 55, 3: 12")

    def test_the_judgments_the_evaluation_cannot_use_are_found_user_by_user(self):
        os.makedirs(os.path.join(self.run_dir, "tmp2"))
        judged = {
            "a": {"memory_integrity_records": [{"memory_integrity_score": None},
                                               {"memory_integrity_score": 0}],
                  "memory_accuracy_records": [{"memory_accuracy_score": 2}],
                  "memory_update_records": [{"memory_update_type": "correct"},
                                            {"memory_update_type": "Other"}],
                  "question_answering_records": [{"result_type": "Other"},
                                                 {"result_type": None},
                                                 {"result_type": "Omission"}]},
            "b": {"memory_integrity_records": [{"memory_integrity_score": 2}],
                  "question_answering_records": [{"result_type": "Correct"}]},
        }
        for uuid, records in judged.items():
            with open(os.path.join(self.run_dir, "tmp2", f"{uuid}.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(records, handle)

        found = adapter.unusable_judgments(self.run_dir)

        self.assertEqual(found, {"a": {"integrity": 1, "accuracy": 0, "update": 1, "question": 2}})
        self.assertTrue(adapter.unusable_text(found).startswith(
            "a: 1 integrity, 1 update, 2 question\n4 judgments empty or invalid in 1 users"))

    def test_before_the_judge_there_is_nothing_to_report(self):
        self.assertEqual(adapter.unusable_judgments(self.run_dir), {})

    def clean(self, *extra):
        os.makedirs(os.path.join(self.run_dir, "eval", "results"), exist_ok=True)
        run_dir = os.path.join(self.run_dir, "eval", "results", f"{adapter.FRAME}-v")
        os.makedirs(os.path.join(run_dir, "archives", "uuid-1"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "tmp"), exist_ok=True)
        with open(os.path.join(run_dir, "tmp", "uuid-1.json"), "w", encoding="utf-8") as handle:
            json.dump({"uuid": "uuid-1"}, handle)
        arguments = adapter.parse_arguments(
            ["clean", "--halumem", self.run_dir, "--version", "v", "-y", *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(adapter.clean(arguments), 0)
        return run_dir

    def test_clean_deletes_the_archives_and_keeps_the_results(self):
        run_dir = self.clean()

        self.assertFalse(os.path.exists(os.path.join(run_dir, "archives")))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "tmp", "uuid-1.json")),
                        "launched again, the users already done are still skipped")

    def test_clean_all_deletes_the_whole_run(self):
        run_dir = self.clean("--all")

        self.assertFalse(os.path.exists(run_dir))

    def test_a_version_cannot_lead_out_of_the_results_folder(self):
        for version in ("../x", "..", "a/b", ".hidden", "-v"):
            with self.subTest(version=version), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                adapter.parse_arguments(["clean", "--halumem", "x", f"--version={version}"])

    def test_the_calibration_users_are_replayed_only_when_asked_and_only_them(self):
        uuids = [f"u{index}" for index in range(adapter.CALIBRATION_USERS + 3)]
        os.makedirs(os.path.join(self.run_dir, "data"))
        with open(os.path.join(self.run_dir, adapter.DATASET), "w", encoding="utf-8") as handle:
            for uuid in uuids:
                handle.write(json.dumps({"uuid": uuid}) + "\n\n")

        def replayed(*extra):
            arguments = adapter.parse_arguments(["run", "--halumem", self.run_dir,
                                                 "--version", "v", *extra])
            return [user["uuid"] for user in adapter.replayed_users(self.run_dir, arguments)]

        self.assertEqual(replayed(), uuids[adapter.CALIBRATION_USERS:])
        self.assertEqual(replayed("--users", "1"), [uuids[adapter.CALIBRATION_USERS]],
                         "the pilot is the first user after the calibration ones")
        self.assertEqual(replayed("--calibration-users"), uuids[:adapter.CALIBRATION_USERS])
        self.assertEqual(replayed("--calibration-users", "--users", "2"), uuids[:2])

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
