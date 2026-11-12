"""The load probe on an invented dialogue, with a model that answers by itself.

Run: python memory_service/run_tests.py -v
"""

import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from types import SimpleNamespace

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import AIMessage  # noqa: E402

import load_probe as probe  # noqa: E402

USERS = [{"sessions": [{"dialogue": [
    {"role": "user", "content": "Ho adottato un cane che si chiama Argo"},
    {"role": "assistant", "content": "Che bello, di che razza è"},
    {"role": "user", "content": "Un golden retriever di tre anni"},
    {"role": "assistant", "content": "Allora avrà bisogno di lunghe passeggiate"}]}]}]


def point(text, hour):
    return {"memory_content": text, "timestamp": f"Sep 04, 2025, {hour:02d}:00:00"}


ASKED = [{"sessions": [
    {"dialogue": [{"role": "user", "content": "Ciao"}, {"role": "assistant", "content": "Ciao"}],
     "memory_points": [point("L'utente vive a Palermo", 9)],
     "questions": [{"question": "Dove vive l'utente?"}]},
    {"dialogue": [{"role": "user", "content": "Ho cambiato lavoro"},
                  {"role": "assistant", "content": "Racconta"},
                  {"role": "user", "content": "Ora insegno matematica"},
                  {"role": "assistant", "content": "Bello, in che scuola"}],
     "memory_points": [point(f"L'utente ha un ricordo numero {number:02d}", 10)
                       for number in range(40)],
     "questions": [{"question": "Che lavoro fa l'utente?"}]}]}]


class CountingModel:
    """Answers 40 tokens after a short wait, and records how many calls overlapped."""

    def __init__(self, fail_every=0):
        self.lock = threading.Lock()
        self.running = self.most = self.calls = 0
        self.fail_every = fail_every

    def invoke(self, prompt):
        with self.lock:
            self.calls += 1
            self.running += 1
            self.most = max(self.most, self.running)
            failing = self.fail_every and self.calls % self.fail_every == 0
        time.sleep(0.02)
        with self.lock:
            self.running -= 1
        if failing:
            raise TimeoutError("the server did not answer")
        return AIMessage(content="The user has a dog.", usage_metadata={
            "input_tokens": 300, "output_tokens": 40, "total_tokens": 340})


class LoadProbeTest(unittest.TestCase):
    def test_each_prompt_is_a_new_stretch_of_dialogue(self):
        prompts = probe.dialogue_prompts(USERS, words=12, count=2)

        self.assertEqual(len(prompts), 2)
        self.assertNotEqual(prompts[0], prompts[1])
        self.assertTrue(prompts[0].startswith("user: Ho adottato un cane"))
        self.assertTrue(all(prompt.endswith(probe.INSTRUCTION) for prompt in prompts))

    def test_a_small_dialogue_is_reused(self):
        prompts = probe.dialogue_prompts(USERS, words=12, count=5)

        self.assertEqual(len(prompts), 5)
        self.assertEqual(prompts[2], prompts[0])

    def test_a_level_never_has_more_calls_at_once_than_its_number(self):
        model = CountingModel()

        found = probe.probe_level(model, ["a"] * 9, level=3)

        self.assertEqual((model.calls, model.most, found["calls"]), (9, 3, 9))
        self.assertEqual(found["tokens per call"], 40)
        self.assertAlmostEqual(found["tokens/s in all"], 9 * 40 / found["wall s"])

    def test_a_failed_call_is_counted_apart(self):
        found = probe.probe_level(CountingModel(fail_every=2), ["a"] * 4, level=2)

        self.assertEqual(found["calls"], 2)
        self.assertEqual(len(found["errors"]), 2)
        self.assertIn("TimeoutError", found["errors"][0])

    def test_each_level_sends_requests_of_its_own(self):
        model = CountingModel()

        found = probe.probe_levels(model, ["a"] * 6, levels=[1, 2], rounds=2)

        self.assertEqual([level["calls"] for level in found], [2, 4])
        self.assertEqual(model.calls, 6)

    def test_an_answer_has_the_latest_memories_as_core_and_the_older_as_recalled(self):
        asked = probe.answer_inputs(ASKED, count=2, dates="")[1]

        contents = [line.split("Content: ")[1] for line in asked["core_memory"]]
        recalled = asked["retrieved_memory"].split("\n")
        self.assertEqual(asked["input"], "Che lavoro fa l'utente?")
        self.assertTrue(contents[-1].endswith("39"))
        self.assertLessEqual(len("\n".join(contents)), probe.CORE_CHARS)
        self.assertEqual(len(recalled), 20)
        self.assertTrue(all(line.startswith("ID: ") and "Date: 2025-09-04" in line
                            for line in recalled))
        self.assertTrue(recalled[-1].endswith(f"{39 - len(contents):02d}"))
        self.assertEqual(asked["messages"], "Human: Ora insegno matematica\n"
                                            "AI: Bello, in che scuola")

    def test_the_first_question_recalls_nothing_and_questions_are_reused(self):
        asked = probe.answer_inputs(ASKED, count=3, dates="")

        self.assertEqual(asked[0]["retrieved_memory"], probe.NO_ARCHIVAL_RESULTS)
        self.assertEqual(len(asked[0]["core_memory"]), 1)
        self.assertEqual(asked[2], asked[0])

    def test_a_csv_row_per_level_with_the_ratios_to_the_first(self):
        levels = [probe.probe_level(CountingModel(), ["a"] * 2, level=1),
                  probe.probe_level(CountingModel(), ["a"] * 4, level=2),
                  probe.probe_level(CountingModel(fail_every=1), ["a"] * 2, level=3)]

        rows = probe.csv_rows("2026-11-09T11:00:00", "answer", "qwen3.5-9b", levels)

        named = [dict(zip(probe.CSV_HEADER, row)) for row in rows]
        self.assertEqual([row["level"] for row in named], [1, 2, 3])
        self.assertEqual((named[0]["all vs first"], named[0]["s vs first"]), (1.0, 1.0))
        self.assertEqual(named[1]["tokens per call"], 40)
        self.assertEqual((named[2]["calls"], named[2]["errors"], named[2]["median s"]),
                         (0, 2, ""))
        self.assertEqual(len(rows[2]), len(probe.CSV_HEADER))

    def test_the_stats_of_a_level(self):
        found = probe.level_stats([1.0, 2.0, 4.0], [10, 20, 40], wall=5.0)

        self.assertEqual((found["median s"], found["p90 s"]), (2.0, 4.0))
        self.assertAlmostEqual(found["tokens/s per call"], 10.0)
        self.assertAlmostEqual(found["tokens/s in all"], 14.0)


def said(role, content):
    return {"role": role, "content": content, "timestamp": "Sep 04, 2025, 09:00:00"}


def golden(text, source="primary", **fields):
    return {"memory_content": text, "is_update": "False", "memory_source": source,
            "original_memories": [], **fields}


RUN = [{"user_name": "Mario Rossi", "sessions": [
    {"dialogue": [said("user", "Vivo a Palermo e ora lavoro a Catania"),
                  said("assistant", "Bello")],
     "memory_points": [
         golden("Mario vive a Palermo"),
         golden("Mario lavora a Catania", is_update="True",
                memories_from_system=["Mario lavora a Catania"],
                original_memories=["Mario lavora a Messina"]),
         golden("Mario ama il blu", source="interference")],
     "extracted_memories": ["Mario vive a Palermo", "Mario lavora a Catania"],
     "questions": [
         {"question": "Dove vive Mario?", "answer": "A Palermo",
          "evidence": [{"memory_content": "Mario vive a Palermo"}], "system_response": "A Palermo"},
         {"question": "Come si chiama il gatto?", "answer": "Non si sa", "evidence": [],
          "system_response": "Non lo so"}]},
    {"is_generated_qa_session": True, "dialogue": [], "memory_points": [golden("Domanda")],
     "extracted_memories": ["Domanda"]},
    {"dialogue": [said("user", "Ciao")], "memory_points": [golden("Mario saluta")],
     "extracted_memories": []}]}]


def reply(content, tokens, finish="stop"):
    return SimpleNamespace(usage=SimpleNamespace(completion_tokens=tokens), choices=[
        SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)])


class FakeLlms:
    """eval/llms.py: a client that answers 42 tokens with a JSON block, and keeps what it
    was sent and the retries it was given."""

    MODEL = "qwen3.8-27b"
    common_params = {"temperature": 0.0}

    def __init__(self, content='```json\n{"score": "2"}\n```'):
        self.sent, self.retries, self.content = [], [], content
        self.client = SimpleNamespace(with_options=self.with_options)

    def with_options(self, max_retries):
        self.retries.append(max_retries)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=self.create)))

    def create(self, **request):
        self.sent.append(request)
        return reply(self.content, 42, "stop" if "json" in self.content else "length")


class JudgeRequestsTest(unittest.TestCase):
    def test_evaluate_sends_one_call_per_point_written_memory_and_question(self):
        found = probe.evaluate_requests(RUN)

        self.assertEqual(Counter(kind for kind, _ in found),
                         {"integrity": 2, "update": 1, "accuracy": 2, "question": 2})
        accuracy = next(fields for kind, fields in found if kind == "accuracy")
        self.assertEqual(accuracy["golden_memories"],
                         "Mario vive a Palermo\nMario lavora a Catania", "interference left out")

    def test_the_dialogue_as_halumem_writes_it(self):
        self.assertEqual(probe.evaluation_dialogue(RUN[0]["sessions"][0]["dialogue"]),
                         "[Sep 04, 2025, 09:00:00]user: Vivo a Palermo e ora lavoro a Catania\n"
                         "[Sep 04, 2025, 09:00:00]assistant: Bello\n")

    def test_label_asks_the_questions_with_evidence_on_the_latest_memories(self):
        written = [f"Mario ha un ricordo numero {number:02d}" for number in range(60)]
        run = [{"user_name": "Mario Rossi", "sessions": [
            dict(RUN[0]["sessions"][0], extracted_memories=written[:30]),
            {"extracted_memories": written[30:], "questions": RUN[0]["sessions"][0]["questions"]}]}]

        found = probe.label_requests(run)

        self.assertEqual([question["question"] for question, _, _ in found],
                         ["Dove vive Mario?", "Dove vive Mario?"])
        self.assertEqual(found[0][1], written[:30])
        self.assertEqual(len(found[1][1]), probe.LABEL_SHOWN)
        self.assertEqual(found[1][1][-1], written[-1])
        self.assertEqual(found[1][2], "Mario Rossi")

    def test_spread_keeps_the_shares_and_reuses_a_short_list(self):
        self.assertEqual(probe.spread(list(range(10)), 5, "empty"), [0, 2, 4, 6, 8])
        self.assertEqual(probe.spread([1, 2], 3, "empty"), [1, 2, 1])
        with self.assertRaises(ValueError):
            probe.spread([], 3, "empty")

    def test_the_judge_has_the_settings_of_eval_env(self):
        llms = FakeLlms()

        found = probe.probe_level(probe.Judge(llms), ["a", "b"], level=2)

        self.assertEqual(found["tokens per call"], 42)
        self.assertEqual({(request["model"], request["temperature"]) for request in llms.sent},
                         {("qwen3.8-27b", 0.0)})
        self.assertEqual(llms.retries, [0, 0], "not even the client's own retries")

    def test_a_reply_without_the_json_block_is_a_failed_call(self):
        found = probe.probe_level(probe.Judge(FakeLlms("Let me think about the memories")),
                                  ["a"], level=1)

        self.assertEqual(found["calls"], 0)
        self.assertIn("no JSON block after 42 tokens, finish reason length", found["errors"][0])

    def test_with_progress_every_call_is_printed_as_it_ends(self):
        printed = io.StringIO()

        with contextlib.redirect_stdout(printed):
            probe.probe_level(CountingModel(fail_every=2), ["a"] * 4, level=2, progress=True)

        lines = printed.getvalue().splitlines()
        self.assertEqual([line.split(" after ")[0].strip() for line in lines],
                         ["1/4", "2/4", "3/4", "4/4"])
        self.assertEqual(sum("40 tokens" in line for line in lines), 2)
        self.assertEqual(sum("TimeoutError" in line for line in lines), 2)


FAKE_LLMS = '''
from types import SimpleNamespace
MODEL = "qwen3.8-27b"
common_params = {}
def create(**request):
    return SimpleNamespace(usage=SimpleNamespace(completion_tokens=7), choices=[SimpleNamespace(
        message=SimpleNamespace(content='```json {"needed": [1]} ```'), finish_reason="stop")])
def with_options(max_retries):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
client = SimpleNamespace(with_options=with_options)
'''
FAKE_TOOLS = "".join(f'EVALUATION_PROMPT_FOR_{name} = "{name} {{{field}}}"\n' for name, field in (
    ("MEMORY_INTEGRITY", "memories"), ("MEMORY_ACCURACY", "dialogue"),
    ("UPDATE_MEMORY", "memories"), ("QUESTION", "question")))


class JudgePartTest(unittest.TestCase):
    """--part judge on a finished run, with eval/ of HaluMem replaced by fakes."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name
        os.makedirs(os.path.join(self.root, "eval"))
        for name, text in (("llms.py", FAKE_LLMS), ("eval_tools.py", FAKE_TOOLS)):
            with open(os.path.join(self.root, "eval", name), "w", encoding="utf-8") as handle:
                handle.write(text)
        run_dir = os.path.join(self.root, "eval", "results", "memory_service-trace-ng-4b")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "memory_service_eval_results.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(RUN[0]) + "\n")
        self.addCleanup(os.chdir, os.getcwd())
        self.addCleanup(self.forget_eval)
        self.forget_eval()

    def forget_eval(self):
        eval_dir = os.path.join(self.root, "eval")
        while eval_dir in sys.path:
            sys.path.remove(eval_dir)
        for name in ("llms", "eval_tools"):
            sys.modules.pop(name, None)

    def test_evaluate_and_label_at_every_level_and_the_csv_where_it_was_asked(self):
        target = tempfile.TemporaryDirectory()
        self.addCleanup(target.cleanup)
        os.chdir(target.name)

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(["--halumem", self.root, "--part", "judge", "--version",
                                         "trace-ng-4b", "--levels", "1", "2", "--judge-rounds",
                                         "1", "--csv", "probe.csv"]), 0)

        with open(os.path.join(target.name, "probe.csv"), encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(row["part"], row["level"], row["calls"]) for row in rows],
                         [("evaluate", "1", "1"), ("evaluate", "2", "2"),
                          ("label", "1", "1"), ("label", "2", "2")])
        self.assertEqual({(row["model"], row["tokens per call"]) for row in rows},
                         {("qwen3.8-27b", "7.0")})

    def test_label_alone(self):
        printed = io.StringIO()

        with contextlib.redirect_stdout(printed):
            probe.main(["--halumem", self.root, "--part", "label", "--version", "trace-ng-4b",
                        "--levels", "1", "--judge-rounds", "1"])

        self.assertIn("\nLabel\n", printed.getvalue())
        self.assertNotIn("\nEvaluate\n", printed.getvalue())

    def test_judge_needs_a_run(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.main(["--halumem", self.root, "--part", "judge"])


if __name__ == "__main__":
    unittest.main()
