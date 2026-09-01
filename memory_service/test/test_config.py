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
        "MEMORY_CHROMA_PATH",
        "MEMORY_COLLECTION_NAME",
        "MEMORY_LLM_NODE",
        "MEMORY_API_KEY_ENV",
        "GROQ_API_KEY",
        "MEMORY_LLM_BASE_URL",
        "MEMORY_EMBEDDING_BASE_URL",
        "MEMORY_NUM_CTX",
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

    def test_local_model_defaults(self):
        config = MemoryConfig.from_environment()

        self.assertIsNone(config.base_url, "senza indirizzo vale il default del provider")
        self.assertIsNone(config.embedding_base_url)
        self.assertEqual(config.embedding_config, {})
        self.assertEqual(config.num_ctx, 8192,
                         "il default di Ollama e' 2048 e troncherebbe i prompt")

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
            "memory_agent": {"model_name": "qwen3-embedding:0.6b",
                             "model_provider": "ollama"},
            "other_node": {"model_name": "altro", "model_provider": "ollama"},
        })

        config = MemoryConfig.from_environment()

        self.assertEqual(config.embedding_config["model_name"], "qwen3-embedding:0.6b")

    def test_malformed_embedding_config_does_not_raise(self):
        os.environ["EMBEDDING_CONFIG"] = "{ questo non e' python"

        self.assertEqual(MemoryConfig.from_environment().embedding_config, {})

    def test_num_ctx_is_read_from_the_environment(self):
        os.environ["MEMORY_NUM_CTX"] = "16384"

        self.assertEqual(MemoryConfig.from_environment().num_ctx, 16384)

    def test_llm_and_embedding_configs_are_independent(self):
        # Sono due modelli diversi, potenzialmente su runtime diversi: una
        # variabile non deve leggere l'altra.
        os.environ["LLM_CONFIG"] = str(
            {"memory_agent": {"model_name": "qwen3.5:4b", "model_provider": "ollama"}})

        config = MemoryConfig.from_environment()

        self.assertEqual(config.llm_config["model_name"], "qwen3.5:4b")
        self.assertEqual(config.embedding_config, {}, "EMBEDDING_CONFIG non e' impostata")

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


class LocalModelParametersTest(EnvironmentTestCase):
    """Che cosa arriva davvero al costruttore del chat model.

    init_chat_model viene sostituito: cosi' si verificano i parametri senza che
    Ollama debba essere raggiungibile, e senza installare il provider.
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

    def _build(self, llm_config, **overrides):
        from memory_service.backends import _build_llm

        config = MemoryConfig(node_name="memory_agent", llm_config=llm_config, **overrides)
        _build_llm(config)
        return self.calls[-1]

    def test_ollama_receives_base_url_and_num_ctx(self):
        parameters = self._build(
            {"model_name": "qwen3.5:4b", "model_provider": "ollama", "temperature": 0.0},
            base_url="http://192.168.1.50:11434", num_ctx=8192)

        self.assertEqual(parameters["model"], "qwen3.5:4b")
        self.assertEqual(parameters["model_provider"], "ollama")
        self.assertEqual(parameters["base_url"], "http://192.168.1.50:11434")
        self.assertEqual(parameters["num_ctx"], 8192,
                         "senza, Ollama userebbe 2048 e troncherebbe i prompt")

    def test_no_api_key_is_sent_when_there_is_none(self):
        # Con un modello locale non c'e' nessuna chiave, e api_key=None non e'
        # accettato da tutti i provider.
        parameters = self._build(
            {"model_name": "qwen3.5:4b", "model_provider": "ollama"})

        self.assertNotIn("api_key", parameters)

    def test_no_key_is_sent_to_a_local_model_even_when_one_exists(self):
        # Il caso vero, che quello sopra non copre: GROQ_API_KEY *e'* impostata
        # nel .env, perche' gli altri cinque nodi dell'architettura parlano
        # ancora con Groq, e api_key_env vale GROQ_API_KEY per default. Senza
        # guardia la chiave di un provider hosted finirebbe a un server locale.
        # ChatOllama ignora il campo sconosciuto invece di sollevare, quindi
        # l'errore non si vedrebbe.
        os.environ["GROQ_API_KEY"] = "gsk_una_chiave_vera"

        parameters = self._build(
            {"model_name": "qwen3.5:4b", "model_provider": "ollama"})

        self.assertNotIn("api_key", parameters)

    def test_the_api_key_is_sent_when_the_variable_is_set(self):
        os.environ["A_TEST_KEY"] = "segreto"
        self.addCleanup(os.environ.pop, "A_TEST_KEY", None)

        parameters = self._build(
            {"model_name": "un-modello", "model_provider": "groq"},
            api_key_env="A_TEST_KEY")

        self.assertEqual(parameters["api_key"], "segreto")

    def test_num_ctx_is_not_sent_to_other_providers(self):
        # E' un parametro specifico di Ollama: mandarlo a groq sarebbe un errore.
        parameters = self._build(
            {"model_name": "un-modello", "model_provider": "groq"}, num_ctx=8192)

        self.assertNotIn("num_ctx", parameters)

    def test_base_url_is_omitted_when_not_configured(self):
        parameters = self._build(
            {"model_name": "un-modello", "model_provider": "groq"})

        self.assertNotIn("base_url", parameters,
                         "senza indirizzo vale il default del provider")


class EmbeddingBackendTest(EnvironmentTestCase):
    """Gli errori di configurazione dell'embedding, verificabili senza il server."""

    def test_a_missing_embedding_config_is_reported(self):
        from memory_service.backends import _build_embeddings

        with self.assertRaises(RuntimeError) as raised:
            _build_embeddings(MemoryConfig(node_name="memory_agent"))

        self.assertIn("EMBEDDING_CONFIG", str(raised.exception))

    def test_an_incomplete_embedding_config_is_reported(self):
        from memory_service.backends import _build_embeddings

        config = MemoryConfig(node_name="memory_agent",
                              embedding_config={"model_name": "qwen3-embedding:0.6b"})

        with self.assertRaises(RuntimeError) as raised:
            _build_embeddings(config)

        self.assertIn("model_provider", str(raised.exception))

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
