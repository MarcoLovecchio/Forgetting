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
        self.assertEqual(config.maximum_historical_messages, 4,
                         "pari, per non tagliare a meta' uno scambio")
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

        self.assertEqual(MemoryConfig.from_environment().maximum_historical_messages, 4)

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
        self.assertEqual(config.api_key_env, "",
                         "senza variabile configurata non si legge nessuna chiave")

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

    def test_malformed_embedding_config_does_not_raise(self):
        os.environ["EMBEDDING_CONFIG"] = "{ questo non e' python"

        self.assertEqual(MemoryConfig.from_environment().embedding_config, {})

    def test_llm_and_embedding_configs_are_independent(self):
        # Sono due modelli diversi, potenzialmente su runtime diversi: una
        # variabile non deve leggere l'altra.
        os.environ["LLM_CONFIG"] = str(
            {"memory_agent": {"model_name": "qwen3.5-4b", "model_provider": "openai"}})

        config = MemoryConfig.from_environment()

        self.assertEqual(config.llm_config["model_name"], "qwen3.5-4b")
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
    l'endpoint debba essere raggiungibile, e senza installare il provider.
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

    def test_the_endpoint_address_is_passed_through(self):
        parameters = self._build(
            {"model_name": "qwen3.5-4b", "model_provider": "openai", "temperature": 0.0},
            base_url="http://modelli.cluster.local:8000/v1")

        self.assertEqual(parameters["model"], "qwen3.5-4b")
        self.assertEqual(parameters["model_provider"], "openai")
        self.assertEqual(parameters["base_url"], "http://modelli.cluster.local:8000/v1")

    def test_the_sampling_parameters_come_from_the_configuration(self):
        # Qwen sconsiglia il greedy decoding: "can lead to performance
        # degradation and endless repetitions". I valori stanno in .config
        # proprio per poter essere tarati senza toccare il codice.
        parameters = self._build({
            "model_name": "Qwen/Qwen3.5-4B", "model_provider": "openai",
            "temperature": 1.0, "top_p": 0.95, "top_k": 20})

        self.assertEqual(parameters["temperature"], 1.0)
        self.assertEqual(parameters["top_p"], 0.95)

    def test_top_k_travels_in_extra_body(self):
        # top_k non e' un parametro dell'API OpenAI: passarlo come argomento
        # diretto non arriverebbe al server, che invece lo accetta nel corpo.
        parameters = self._build({
            "model_name": "Qwen/Qwen3.5-4B", "model_provider": "openai", "top_k": 20})

        self.assertEqual(parameters["extra_body"], {"top_k": 20})

    def test_thinking_is_switched_through_the_chat_template(self):
        parameters = self._build({
            "model_name": "Qwen/Qwen3.5-4B", "model_provider": "openai",
            "enable_thinking": False})

        self.assertEqual(parameters["extra_body"],
                         {"chat_template_kwargs": {"enable_thinking": False}})

    def test_thinking_left_alone_is_not_sent_at_all(self):
        # None non e' False: senza indicazione decide il server, e mandare
        # esplicitamente un valore sarebbe decidere al posto suo.
        parameters = self._build(
            {"model_name": "Qwen/Qwen3.5-4B", "model_provider": "openai"})

        self.assertNotIn("extra_body", parameters)

    def test_what_is_not_configured_is_left_to_the_server(self):
        parameters = self._build(
            {"model_name": "Qwen/Qwen3.5-4B", "model_provider": "openai"})

        for name in ("temperature", "top_p", "presence_penalty", "extra_body"):
            self.assertNotIn(name, parameters)

    def test_extra_body_is_not_sent_to_a_hosted_provider(self):
        # E' un campo di ChatOpenAI: altrove sarebbe un argomento sconosciuto.
        parameters = self._build(
            {"model_name": "un-modello", "model_provider": "groq", "top_k": 20})

        self.assertNotIn("extra_body", parameters)

    def test_a_placeholder_key_is_sent_when_the_endpoint_wants_none(self):
        # L'SDK di OpenAI solleva alla costruzione del client, non alla prima
        # chiamata, se non trova nessuna chiave - anche verso un server che non
        # la controlla. Senza segnaposto il servizio non parte proprio.
        parameters = self._build(
            {"model_name": "qwen3.5-4b", "model_provider": "openai"})

        self.assertEqual(parameters["api_key"], "EMPTY")

    def test_the_key_of_another_provider_never_leaves(self):
        # GROQ_API_KEY *e'* impostata nel .env, perche' gli altri cinque nodi
        # dell'architettura parlano ancora con Groq. Con un endpoint OpenAI
        # compatible quella chiave finirebbe nell'header Authorization verso il
        # cluster: non scartata in silenzio, proprio spedita. La protezione e'
        # che api_key_env parte vuota e va scelta apposta.
        os.environ["GROQ_API_KEY"] = "gsk_una_chiave_vera"

        parameters = self._build(
            {"model_name": "qwen3.5-4b", "model_provider": "openai"})

        self.assertEqual(parameters["api_key"], "EMPTY")

    def test_a_hosted_provider_keeps_looking_up_its_own_key(self):
        # Passare api_key=None non e' neutro: sopprimerebbe il lookup che il
        # provider fa da solo sulla propria variabile.
        parameters = self._build(
            {"model_name": "un-modello", "model_provider": "groq"})

        self.assertNotIn("api_key", parameters)

    def test_the_api_key_is_sent_when_the_variable_is_set(self):
        os.environ["A_TEST_KEY"] = "segreto"
        self.addCleanup(os.environ.pop, "A_TEST_KEY", None)

        parameters = self._build(
            {"model_name": "un-modello", "model_provider": "groq"},
            api_key_env="A_TEST_KEY")

        self.assertEqual(parameters["api_key"], "segreto")

    def test_base_url_is_omitted_when_not_configured(self):
        parameters = self._build(
            {"model_name": "un-modello", "model_provider": "groq"})

        self.assertNotIn("base_url", parameters,
                         "senza indirizzo vale il default del provider")


class OpenAIEmbeddingParametersTest(EnvironmentTestCase):
    """Che cosa arriva al costruttore degli embedding.

    langchain_openai non e' importabile ovunque, quindi il modulo viene
    sostituito: qui interessano i parametri, non il client vero.
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
        # Non self._saved: la classe base lo usa gia' per le variabili d'ambiente.
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
        # Con il default, OpenAIEmbeddings tokenizza prima di spedire e per un
        # modello che tiktoken non conosce ripiega su cl100k_base, il vocabolario
        # di OpenAI. Gli id di un vocabolario sono parole diverse in un altro:
        # il server incorporerebbe rumore e lo restituirebbe senza protestare.
        self.assertIs(self._build()["check_embedding_ctx_length"], False)

    def test_the_endpoint_address_is_passed_through(self):
        parameters = self._build(embedding_base_url="http://modelli.local:8000/v1")

        self.assertEqual(parameters["base_url"], "http://modelli.local:8000/v1")
        self.assertEqual(parameters["model"], "Qwen/Qwen3-Embedding-0.6B")

    def test_the_placeholder_key_reaches_the_embeddings_too(self):
        self.assertEqual(self._build()["api_key"], "EMPTY")


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
