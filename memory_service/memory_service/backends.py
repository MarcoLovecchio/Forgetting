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


def _build_llm(config: MemoryConfig) -> Any:
    # Imported here so that the package can be imported without langchain's
    # provider extras installed.
    import os

    from langchain.chat_models import init_chat_model

    if not config.llm_config:
        raise RuntimeError(
            f"No LLM configuration found for node '{config.node_name}'. "
            "Set LLM_CONFIG in the .env/.config file, or inject a chat model "
            "with memory_service.backends.configure(llm=...)."
        )

    missing = [key for key in ("model_name", "model_provider") if key not in config.llm_config]
    if missing:
        raise RuntimeError(
            f"LLM configuration for node '{config.node_name}' is missing: {', '.join(missing)}"
        )

    return init_chat_model(
        model=config.llm_config["model_name"],
        model_provider=config.llm_config["model_provider"],
        temperature=config.llm_config.get("temperature", 0),
        api_key=os.getenv(config.api_key_env),
    )


def _build_vector_store(config: MemoryConfig) -> Any:
    # Same reasoning as above: chromadb and the embedding provider are only
    # needed when the archival memory is actually used.
    import chromadb
    from chromadb.errors import NotFoundError
    from langchain_chroma import Chroma
    from langchain_mistralai import MistralAIEmbeddings

    client = chromadb.PersistentClient(path=config.chroma_path)

    # Create or get the collection
    try:
        client.get_collection(name=config.collection_name)
    except NotFoundError:
        client.create_collection(name=config.collection_name)

    return Chroma(
        client=client,
        collection_name=config.collection_name,
        embedding_function=MistralAIEmbeddings(),
    )
