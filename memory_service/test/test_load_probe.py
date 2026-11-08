"""The load probe on an invented dialogue, with a model that answers by itself.

Run: python memory_service/run_tests.py -v
"""

import os
import sys
import threading
import time
import unittest

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

    def test_the_stats_of_a_level(self):
        found = probe.level_stats([1.0, 2.0, 4.0], [10, 20, 40], wall=5.0)

        self.assertEqual((found["median s"], found["p90 s"]), (2.0, 4.0))
        self.assertAlmostEqual(found["tokens/s per call"], 10.0)
        self.assertAlmostEqual(found["tokens/s in all"], 14.0)


if __name__ == "__main__":
    unittest.main()
