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
        "EMBEDDING_CONFIG",
        "MEMORY_MAX_HISTORICAL_MESSAGES",
        "MEMORY_CORE_MEMORY_LIMIT",
        "MEMORY_ARCHIVE_LIMIT",
        "MEMORY_GENERATE_ANSWER",
        "MEMORY_CHROMA_PATH",
        "MEMORY_COLLECTION_NAME",
        "MEMORY_LLM_NODE",
        "MEMORY_API_KEY_ENV",
        "GROQ_API_KEY",
        "MEMORY_LLM_BASE_URL",
        "MEMORY_EMBEDDING_BASE_URL",
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

    def test_values_are_read_from_the_environment(self):
        os.environ["MEMORY_MAX_HISTORICAL_MESSAGES"] = "9"
        # A value that does not match the default, otherwise the test would
        # pass even if the variable were ignored.
        os.environ["MEMORY_CORE_MEMORY_LIMIT"] = "777"
        os.environ["MEMORY_ARCHIVE_LIMIT"] = "123"
        os.environ["MEMORY_COLLECTION_NAME"] = "other_archive"

        config = MemoryConfig.from_environment()

        self.assertEqual(config.maximum_historical_messages, 9)
        self.assertEqual(config.core_memory_limit, 777)
        self.assertEqual(config.archive_memory_limit, 123)
        self.assertEqual(config.collection_name, "other_archive")

    def test_the_answer_switch_is_read_from_the_environment(self):
        os.environ["MEMORY_GENERATE_ANSWER"] = "false"

        self.assertFalse(MemoryConfig.from_environment().generate_answer)

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

    def test_base_urls_are_read_from_the_environment(self):
        os.environ["MEMORY_LLM_BASE_URL"] = "http://192.168.1.50:11434"
        os.environ["MEMORY_EMBEDDING_BASE_URL"] = "http://192.168.1.50:11434"

        config = MemoryConfig.from_environment()

        self.assertEqual(config.base_url, "http://192.168.1.50:11434")
        self.assertEqual(config.embedding_base_url, "http://192.168.1.50:11434")

    def test_an_empty_base_url_counts_as_absent(self):
        os.environ["MEMORY_LLM_BASE_URL"] = ""

        self.assertIsNone(MemoryConfig.from_environment().base_url)

    def test_embedding_config_is_read_for_the_node(self):
        os.environ["EMBEDDING_CONFIG"] = str({
            "memory_agent": {"model_name": "qwen3-embedding",
                             "model_provider": "openai"},
            "other_node": {"model_name": "altro", "model_provider": "openai"},
        })

        config = MemoryConfig.from_environment()

        self.assertEqual(config.embedding_config["model_name"], "qwen3-embedding")

    def test_llm_and_embedding_configs_are_independent(self):
        # They are two different models, potentially on different runtimes: one
        # variable must not read the other.
        os.environ["LLM_CONFIG"] = str(
            {"memory_agent": {"model_name": "qwen3.5-4b", "model_provider": "openai"}})

        config = MemoryConfig.from_environment()

        self.assertEqual(config.llm_config["model_name"], "qwen3.5-4b")
        self.assertEqual(config.embedding_config, {}, "EMBEDDING_CONFIG is not set")

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

    def test_the_injected_model_serves_every_node(self):
        # The double sits outside the per-node cache on purpose: if it were inside,
        # the first call with a name not yet present would build a real model in
        # the middle of the offline suite.
        sentinel = object()
        backends.configure(llm=sentinel)

        for node in ("", "retrieval", "generate_answer", "consolidation", "mai visto"):
            self.assertIs(backends.get_llm(node), sentinel, node)


class LocalModelParametersTest(EnvironmentTestCase):
    """What really reaches the chat model's constructor.

    init_chat_model is replaced: this way the parameters are checked without the
    endpoint having to be reachable, and without installing the provider.
    """

    def setUp(self):
        super().setUp()
        import langchain.chat_models as chat_models

        self.calls = []
        self._original = chat_models.init_chat_model

        def spy(**parameters):
            self.calls.append(parameters)
            return object()

        chat_models.init_chat_model = spy
        self.addCleanup(setattr, chat_models, "init_chat_model", self._original)

    def _build(self, llm_config, node_sampling=None, **overrides):
        from memory_service.backends import _build_llm

        config = MemoryConfig(node_name="memory_agent", llm_config=llm_config, **overrides)
        _build_llm(config, node_sampling)
        return self.calls[-1]

    def test_a_new_configuration_invalidates_the_cached_models(self):
        # The cached models derive from the configuration: keeping them, after a
        # configure(config=...) instances tuned on the previous one would still be
        # served, and with four slots instead of one it would be noticed even
        # less.
        self.addCleanup(backends.reset)
        backends.reset()

        backends.configure(config=MemoryConfig(
            node_name="memory_agent",
            llm_config={"model_name": "primo", "model_provider": "openai"}))
        backends.get_llm("retrieval")

        backends.configure(config=MemoryConfig(
            node_name="memory_agent",
            llm_config={"model_name": "secondo", "model_provider": "openai"}))
        backends.get_llm("retrieval")

        self.assertEqual(self.calls[-1]["model"], "secondo")

    def test_switching_one_node_off_leaves_the_others_untouched(self):
        # The practical question: turning reasoning off for a single node, do the
        # others stay as they are? Every build starts from a new merge, so the
        # override table and the configuration are not touched.
        base = {"model_name": "qwen3.5-4b", "model_provider": "openai",
             "temperature": 1.0, "top_p": 0.95, "top_k": 20, "enable_thinking": True}

        off = self._build(base, node_sampling={"enable_thinking": False,
                                               "temperature": 0.7})
        still_on = self._build(base, node_sampling={})

        self.assertIs(
            off["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(off["temperature"], 0.7)
        self.assertIs(
            still_on["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(still_on["temperature"], 1.0)

    def test_building_does_not_modify_the_override_table(self):
        # NODE_SAMPLING.get(node) returns the real dictionary: if _build_llm
        # mutated it, a node's knob would change on its own.
        override = {"enable_thinking": False}

        self._build({"model_name": "qwen3.5-4b", "model_provider": "openai",
             "temperature": 1.0, "top_p": 0.95, "top_k": 20, "enable_thinking": True}, node_sampling=override)

        self.assertEqual(override, {"enable_thinking": False})

    def test_none_in_an_override_falls_back_to_the_server_default(self):
        # Removing a parameter for a single node: passing it as None must omit
        # it, not send temperature=None to the constructor.
        parameters = self._build(base := {"model_name": "qwen3.5-4b", "model_provider": "openai",
             "temperature": 1.0, "top_p": 0.95, "top_k": 20, "enable_thinking": True},
                                 node_sampling={"temperature": None, "top_k": None})

        self.assertNotIn("temperature", parameters)
        self.assertNotIn("top_k", parameters.get("extra_body", {}))
        self.assertEqual(parameters["top_p"], base["top_p"], "the rest stays")

    def test_what_the_override_does_not_mention_is_inherited(self):
        # It is the reason the merge happens before the split between top-level
        # parameters and extra_body: binding extra_body to the call would have
        # replaced the whole dictionary, and top_k - which sits in there together
        # with enable_thinking - would have silently disappeared.
        parameters = self._build(
            {"model_name": "qwen3.5-4b", "model_provider": "openai",
             "temperature": 1.0, "top_p": 0.95, "top_k": 20, "enable_thinking": True},
            node_sampling={"enable_thinking": False})

        self.assertEqual(parameters["extra_body"]["top_k"], 20, "top_k survives")
        self.assertEqual(parameters["top_p"], 0.95)
        self.assertEqual(parameters["temperature"], 1.0)

    def test_the_endpoint_address_is_passed_through(self):
        parameters = self._build(
            {"model_name": "qwen3.5-4b", "model_provider": "openai", "temperature": 0.0},
            base_url="http://modelli.cluster.local:8000/v1")

        self.assertEqual(parameters["model"], "qwen3.5-4b")
        self.assertEqual(parameters["model_provider"], "openai")
        self.assertEqual(parameters["base_url"], "http://modelli.cluster.local:8000/v1")

    def test_top_k_travels_in_extra_body(self):
        # top_k is not a parameter of the OpenAI API: passing it as a direct
        # argument would not reach the server, which accepts it in the body instead.
        parameters = self._build({
            "model_name": "Qwen/Qwen3.5-4B", "model_provider": "openai", "top_k": 20})

        self.assertEqual(parameters["extra_body"], {"top_k": 20})

    def test_thinking_is_switched_through_the_chat_template(self):
        parameters = self._build({
            "model_name": "Qwen/Qwen3.5-4B", "model_provider": "openai",
            "enable_thinking": False})

        self.assertEqual(parameters["extra_body"],
                         {"chat_template_kwargs": {"enable_thinking": False}})

    def test_what_is_not_configured_is_left_to_the_server(self):
        parameters = self._build(
            {"model_name": "Qwen/Qwen3.5-4B", "model_provider": "openai"})

        for name in ("temperature", "top_p", "presence_penalty", "extra_body"):
            self.assertNotIn(name, parameters)

    def test_base_url_is_omitted_when_not_configured(self):
        parameters = self._build(
            {"model_name": "un-modello", "model_provider": "groq"})

        self.assertNotIn("base_url", parameters,
                         "without an address the provider's default holds")


class OpenAIEmbeddingParametersTest(EnvironmentTestCase):
    """What reaches the embeddings constructor.

    langchain_openai is not importable everywhere, so the module is replaced:
    what matters here are the parameters, not the real client.
    """

    def setUp(self):
        super().setUp()
        import sys
        import types

        self.calls = []

        module = types.ModuleType("langchain_openai")

        def spy(**parameters):
            self.calls.append(parameters)
            return object()

        module.OpenAIEmbeddings = spy
        # Not self._saved: the base class already uses it for the environment variables.
        self._saved_module = sys.modules.get("langchain_openai")
        sys.modules["langchain_openai"] = module
        self.addCleanup(self._restore)

    def _restore(self):
        import sys

        if self._saved_module is None:
            sys.modules.pop("langchain_openai", None)
        else:
            sys.modules["langchain_openai"] = self._saved_module

    def _build(self, **overrides):
        from memory_service.backends import _build_embeddings

        config = MemoryConfig(
            node_name="memory_agent",
            embedding_config={"model_name": "Qwen/Qwen3-Embedding-0.6B",
                              "model_provider": "openai"},
            **overrides)
        _build_embeddings(config)
        return self.calls[-1]

    def test_the_text_is_sent_not_openai_tokens(self):
        # With the default, OpenAIEmbeddings tokenizes before sending and, for a
        # model tiktoken does not know, falls back on cl100k_base, OpenAI's
        # vocabulary. The ids of one vocabulary are different words in another:
        # the server would embed noise and return it without complaining.
        self.assertIs(self._build()["check_embedding_ctx_length"], False)

    def test_the_endpoint_address_is_passed_through(self):
        parameters = self._build(embedding_base_url="http://modelli.local:8000/v1")

        self.assertEqual(parameters["base_url"], "http://modelli.local:8000/v1")
        self.assertEqual(parameters["model"], "Qwen/Qwen3-Embedding-0.6B")


class EmbeddingBackendTest(EnvironmentTestCase):
    """The embedding configuration errors, checkable without the server."""

    def test_an_unknown_provider_is_reported(self):
        from memory_service.backends import _build_embeddings

        config = MemoryConfig(
            node_name="memory_agent",
            embedding_config={"model_name": "x", "model_provider": "inventato"})

        with self.assertRaises(RuntimeError) as raised:
            _build_embeddings(config)

        self.assertIn("inventato", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
