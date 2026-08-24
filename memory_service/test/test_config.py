"""Tests for the configuration layer of the memory service."""

import os
import sys
import tempfile
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig, load_environment  # noqa: E402


class EnvironmentTestCase(unittest.TestCase):
    """Runs with a controlled environment, never with the workspace one."""

    managed_variables = (
        "MEMORY_ENV_FILE",
        "LLM_CONFIG",
        "MEMORY_MAX_HISTORICAL_MESSAGES",
        "MEMORY_CORE_MEMORY_LIMIT",
        "MEMORY_CHROMA_PATH",
        "MEMORY_COLLECTION_NAME",
        "MEMORY_LLM_NODE",
        "MEMORY_API_KEY_ENV",
    )

    def setUp(self):
        self._saved = {name: os.environ.get(name) for name in self.managed_variables}
        for name in self.managed_variables:
            os.environ.pop(name, None)
        # An empty env file stops the automatic lookup from reaching the
        # workspace .env/.config, keeping the tests hermetic.
        self._env_file = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        self._env_file.close()
        os.environ["MEMORY_ENV_FILE"] = self._env_file.name
        backends.reset()

    def tearDown(self):
        backends.reset()
        os.unlink(self._env_file.name)
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ConfigTest(EnvironmentTestCase):
    def test_defaults_when_nothing_is_configured(self):
        config = MemoryConfig.from_environment()

        self.assertEqual(config.node_name, "memory_agent")
        self.assertEqual(config.maximum_historical_messages, 5)
        self.assertEqual(config.core_memory_limit, 150)
        self.assertEqual(config.collection_name, "memory_archive")
        self.assertEqual(config.llm_config, {})

    def test_values_are_read_from_the_environment(self):
        os.environ["MEMORY_MAX_HISTORICAL_MESSAGES"] = "9"
        os.environ["MEMORY_CORE_MEMORY_LIMIT"] = "400"
        os.environ["MEMORY_COLLECTION_NAME"] = "other_archive"

        config = MemoryConfig.from_environment()

        self.assertEqual(config.maximum_historical_messages, 9)
        self.assertEqual(config.core_memory_limit, 400)
        self.assertEqual(config.collection_name, "other_archive")

    def test_invalid_numbers_fall_back_to_the_defaults(self):
        os.environ["MEMORY_MAX_HISTORICAL_MESSAGES"] = "not a number"

        self.assertEqual(MemoryConfig.from_environment().maximum_historical_messages, 5)

    def test_chroma_path_is_absolute(self):
        os.environ["MEMORY_CHROMA_PATH"] = "./relative_db"

        self.assertTrue(os.path.isabs(MemoryConfig.from_environment().chroma_path))

    def test_llm_config_entry_of_the_node_is_selected(self):
        os.environ["LLM_CONFIG"] = str(
            {
                "memory_agent": {"model_name": "m1", "model_provider": "groq", "temperature": 0.0},
                "other_node": {"model_name": "m2", "model_provider": "groq", "temperature": 1.0},
            }
        )

        config = MemoryConfig.from_environment()

        self.assertEqual(config.llm_config["model_name"], "m1")

    def test_malformed_llm_config_does_not_raise(self):
        os.environ["LLM_CONFIG"] = "{ this is not python"

        self.assertEqual(MemoryConfig.from_environment().llm_config, {})

    def test_missing_node_entry_does_not_raise(self):
        os.environ["LLM_CONFIG"] = str({"other_node": {"model_name": "m2"}})

        self.assertEqual(MemoryConfig.from_environment().llm_config, {})

    def test_explicit_env_file_is_loaded(self):
        with open(self._env_file.name, "w", encoding="utf-8") as handle:
            handle.write('MEMORY_COLLECTION_NAME = "from_file"\n')

        loaded = load_environment(override=True)

        self.assertEqual(loaded, [os.path.abspath(self._env_file.name)])
        self.assertEqual(os.environ["MEMORY_COLLECTION_NAME"], "from_file")


class BackendsTest(EnvironmentTestCase):
    def test_missing_llm_configuration_raises_a_clear_error(self):
        with self.assertRaises(RuntimeError) as raised:
            backends.get_llm()

        self.assertIn("LLM_CONFIG", str(raised.exception))

    def test_incomplete_llm_configuration_is_reported(self):
        os.environ["LLM_CONFIG"] = str({"memory_agent": {"model_name": "m1"}})

        with self.assertRaises(RuntimeError) as raised:
            backends.get_llm()

        self.assertIn("model_provider", str(raised.exception))

    def test_injected_backends_are_returned_as_is(self):
        sentinel_llm, sentinel_store = object(), object()

        backends.configure(llm=sentinel_llm, vector_store=sentinel_store)

        self.assertIs(backends.get_llm(), sentinel_llm)
        self.assertIs(backends.get_vector_store(), sentinel_store)

    def test_reset_forgets_the_injected_backends(self):
        backends.configure(llm=object(), vector_store=object())
        backends.reset()

        self.assertIsNone(backends._llm)
        self.assertIsNone(backends._vector_store)


if __name__ == "__main__":
    unittest.main()
