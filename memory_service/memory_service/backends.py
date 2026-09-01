"""Lazy, replaceable backends (chat model and vector store).

Both backends used to be created while importing ``memory_manager_llm``, which
meant that merely importing the module opened a ChromaDB directory, built the
Mistral embeddings and instantiated a chat model. That made the package
impossible to import - let alone test - without API keys, network access and the
whole architecture installed.

They are now created on first use and can be replaced by test doubles through
:func:`configure`.
"""

from typing import Any, Optional

from memory_service.config import MemoryConfig

_config: Optional[MemoryConfig] = None
_llm: Any = None
_vector_store: Any = None


def get_config() -> MemoryConfig:
    """Return the active configuration, reading the environment on first use."""
    global _config
    if _config is None:
        _config = MemoryConfig.from_environment()
    return _config


def configure(llm: Any = None, vector_store: Any = None,
              config: Optional[MemoryConfig] = None) -> None:
    """Inject explicit backends.

    Used by the tests to run the whole graph offline, and available to any
    application that wants to provide its own chat model or vector store.
    """
    global _config, _llm, _vector_store
    if config is not None:
        _config = config
    if llm is not None:
        _llm = llm
    if vector_store is not None:
        _vector_store = vector_store


def reset() -> None:
    """Forget the cached backends (mostly useful between tests)."""
    global _config, _llm, _vector_store
    _config = None
    _llm = None
    _vector_store = None


def get_llm() -> Any:
    """Return the chat model, building it from the configuration if needed."""
    global _llm
    if _llm is None:
        _llm = _build_llm(get_config())
    return _llm


def get_vector_store() -> Any:
    """Return the archival vector store, building it if needed."""
    global _vector_store
    if _vector_store is None:
        _vector_store = _build_vector_store(get_config())
    return _vector_store


def _require_model_config(model_config: dict, variable: str, node_name: str) -> str:
    """Validate a per-node model entry and return its provider."""
    if not model_config:
        raise RuntimeError(
            f"No {variable} entry found for node '{node_name}'. "
            f"Set {variable} in the .env/.config file, or inject the backend "
            "with memory_service.backends.configure(...)."
        )

    missing = [key for key in ("model_name", "model_provider") if key not in model_config]
    if missing:
        raise RuntimeError(
            f"{variable} entry for node '{node_name}' is missing: {', '.join(missing)}"
        )
    return model_config["model_provider"]


def _build_llm(config: MemoryConfig) -> Any:
    # Imported here so that the package can be imported without langchain's
    # provider extras installed.
    import os

    from langchain.chat_models import init_chat_model

    provider = _require_model_config(config.llm_config, "LLM_CONFIG", config.node_name)

    parameters = {
        "model": config.llm_config["model_name"],
        "model_provider": provider,
        "temperature": config.llm_config.get("temperature", 0),
    }

    # Endpoint of a locally served model. Passed only when configured, so the
    # hosted providers keep using their own default.
    if config.base_url:
        parameters["base_url"] = config.base_url

    # Ollama defaults to a 2048 token context and truncates longer prompts
    # without a word: the consolidation prompt (core memories plus archival
    # candidates) and the core/archival split both go past that. The parameter
    # is Ollama specific, so it is only passed to Ollama.
    if provider == "ollama":
        parameters["num_ctx"] = config.num_ctx

    # A locally served model has no API key. The default api_key_env is
    # GROQ_API_KEY, which in this workspace *is* set - the other five nodes of
    # the architecture still talk to Groq - so without this guard the key of a
    # hosted provider would be handed to a local server. ChatOllama ignores the
    # unknown field instead of raising, which makes the mistake invisible.
    if provider != "ollama":
        api_key = os.getenv(config.api_key_env) if config.api_key_env else None
        if api_key:
            parameters["api_key"] = api_key

    return init_chat_model(**parameters)


def _build_embeddings(config: MemoryConfig) -> Any:
    """Embedding model used to index and search the archive.

    Changing this model invalidates an existing archive: the stored vectors live
    in the space of whatever model wrote them, so queries embedded by a
    different model come back as noise. Two models can even share the same
    dimensionality and still be incompatible, in which case nothing raises and
    the retrieval quietly degrades.
    """
    import os

    provider = _require_model_config(
        config.embedding_config, "EMBEDDING_CONFIG", config.node_name)
    model_name = config.embedding_config["model_name"]

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        parameters = {"model": model_name}
        if config.embedding_base_url:
            parameters["base_url"] = config.embedding_base_url
        return OllamaEmbeddings(**parameters)

    if provider == "mistralai":
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model=model_name)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        parameters = {"model": model_name}
        if config.embedding_base_url:
            parameters["base_url"] = config.embedding_base_url
        api_key = os.getenv(config.api_key_env) if config.api_key_env else None
        if api_key:
            parameters["api_key"] = api_key
        return OpenAIEmbeddings(**parameters)

    raise RuntimeError(
        f"Unsupported embedding provider '{provider}' for node "
        f"'{config.node_name}'. Known providers: ollama, mistralai, openai."
    )


def _build_vector_store(config: MemoryConfig) -> Any:
    # Same reasoning as above: chromadb and the embedding provider are only
    # needed when the archival memory is actually used.
    import chromadb
    from chromadb.errors import NotFoundError
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(path=config.chroma_path)

    # Create or get the collection
    try:
        client.get_collection(name=config.collection_name)
    except NotFoundError:
        client.create_collection(name=config.collection_name)

    return Chroma(
        client=client,
        collection_name=config.collection_name,
        embedding_function=_build_embeddings(config),
    )
