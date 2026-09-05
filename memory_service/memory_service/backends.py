"""Lazy, replaceable backends (chat model and vector store).

Both backends used to be created while importing ``memory_manager_llm``, which
meant that merely importing the module opened a ChromaDB directory, built the
Mistral embeddings and instantiated a chat model. That made the package
impossible to import - let alone test - without API keys, network access and the
whole architecture installed.

They are now created on first use and can be replaced by test doubles through
:func:`configure`.
"""

import os
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


# The OpenAI SDK refuses to build a client without a key - it raises at
# construction, not at the first call - even against a server that never checks
# one. A self-hosted OpenAI compatible endpoint is exactly that case, so a
# placeholder stands in for the key that does not exist.
PLACEHOLDER_API_KEY = "EMPTY"


def _api_key(config: MemoryConfig, provider: str) -> Optional[str]:
    """Key to hand to the provider, or None when it must not receive one.

    api_key_env is empty by default: the hosted providers find their own key
    through their own variable, and passing None explicitly would suppress that
    lookup instead of falling back to it.
    """
    api_key = os.getenv(config.api_key_env) if config.api_key_env else None
    if api_key:
        return api_key
    return PLACEHOLDER_API_KEY if provider == "openai" else None


# Sampling parameters the OpenAI API understands directly. They are read from
# the LLM_CONFIG entry, so tuning them never requires touching the code; what is
# absent is left to the server default.
_SAMPLING_PARAMETERS = ("temperature", "top_p", "presence_penalty",
                        "frequency_penalty", "max_tokens")


def _extra_body(config: MemoryConfig) -> dict:
    """Options an OpenAI compatible server accepts but the OpenAI API does not."""
    extra: dict = {}
    if "top_k" in config.llm_config:
        # Not an OpenAI parameter, but vLLM and sglang honour it, and Qwen's
        # recommended settings rely on it to cut the tail of the distribution.
        extra["top_k"] = config.llm_config["top_k"]

    # Absent means "leave it to the server": None is not False, and deciding
    # on the server's behalf is not the same as not deciding.
    thinking = config.llm_config.get("enable_thinking")
    if thinking is not None:
        # Qwen switches reasoning through its chat template, not through a
        # sampling parameter.
        extra["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    return extra


def _build_llm(config: MemoryConfig) -> Any:
    # Imported here so that the package can be imported without langchain's
    # provider extras installed.
    from langchain.chat_models import init_chat_model

    provider = _require_model_config(config.llm_config, "LLM_CONFIG", config.node_name)

    parameters = {
        "model": config.llm_config["model_name"],
        "model_provider": provider,
    }
    for name in _SAMPLING_PARAMETERS:
        if name in config.llm_config:
            parameters[name] = config.llm_config[name]

    # extra_body exists on ChatOpenAI and would be an unknown argument elsewhere.
    if provider == "openai":
        extra_body = _extra_body(config)
        if extra_body:
            parameters["extra_body"] = extra_body

    # Endpoint of the model server. Passed only when configured, so the hosted
    # providers keep using their own default.
    if config.base_url:
        parameters["base_url"] = config.base_url

    api_key = _api_key(config, provider)
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
    provider = _require_model_config(
        config.embedding_config, "EMBEDDING_CONFIG", config.node_name)
    model_name = config.embedding_config["model_name"]

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        parameters = {
            "model": model_name,
            "api_key": _api_key(config, provider),
            # By default OpenAIEmbeddings does not send the text: it tokenizes
            # it first, and for a model tiktoken does not know it silently falls
            # back to cl100k_base - OpenAI's vocabulary. The token ids of one
            # vocabulary read as different words in another, so the server would
            # embed noise and return it without complaining. Sending the strings
            # and letting the server tokenize is the only correct option outside
            # OpenAI's own models.
            "check_embedding_ctx_length": False,
        }
        if config.embedding_base_url:
            parameters["base_url"] = config.embedding_base_url
        return OpenAIEmbeddings(**parameters)

    if provider == "mistralai":
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model=model_name)

    raise RuntimeError(
        f"Unsupported embedding provider '{provider}' for node "
        f"'{config.node_name}'. Known providers: openai, mistralai."
    )


ARCHIVE_DISTANCE = "cosine"
ARCHIVE_COLLECTION_METADATA = {"hnsw:space": ARCHIVE_DISTANCE}


def distance_mismatch(collection_metadata: Optional[dict]) -> Optional[str]:
    """The metric of an existing collection, when it is not the one we ask for.

    Worth a warning rather than a failure: the archive still works and the
    ranking is unchanged, but every distance in the reports is on a different
    scale, and a threshold tuned on one is wrong on the other.
    """
    space = (collection_metadata or {}).get("hnsw:space", "l2")
    return None if space == ARCHIVE_DISTANCE else space


def _build_vector_store(config: MemoryConfig) -> Any:
    # Same reasoning as above: chromadb and the embedding provider are only
    # needed when the archival memory is actually used.
    import chromadb
    from chromadb.errors import NotFoundError
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(path=config.chroma_path)

    # Create or get the collection
    try:
        existing = client.get_collection(name=config.collection_name)
    except NotFoundError:
        client.create_collection(name=config.collection_name,
                                 metadata=dict(ARCHIVE_COLLECTION_METADATA))
    else:
        found = distance_mismatch(existing.metadata)
        if found:
            print(f"	WARNING: collection '{config.collection_name}' measures "
                  f"'{found}' distance, not '{ARCHIVE_DISTANCE}'. It was created "
                  f"before this setting: the distances in the reports are on the "
                  f"'{found}' scale until the archive is rebuilt.")

    return Chroma(
        client=client,
        collection_name=config.collection_name,
        embedding_function=_build_embeddings(config),
    )
