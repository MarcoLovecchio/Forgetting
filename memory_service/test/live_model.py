"""Are the real models usable right now?

The tests that talk to the real stack must **skip** when the model is not
there, not fail: a server that is down is not a defect of the code. With the
hosted providers it was enough to check whether the key was set; with a model
served locally the key does not exist and the only way to know is to try.

The result is cached: the tests ask for it several times and there is no point
paying for a call each time.
"""

import os
from typing import Optional

from memory_service import backends
from memory_service.config import load_environment

# The .env/.config files are loaded as soon as this module is imported, and the
# import sits at the top of the tests that need it. Without it,
# os.getenv("LLM_CONFIG") would be None when pytest evaluates the skipif: the
# tests on the real stack would **always** be skipped, even with .config in
# place, saying that a configuration is missing when it is actually there.
load_environment()

_chat_reason: Optional[str] = None
_chat_checked = False
_embeddings_reason: Optional[str] = None
_embeddings_checked = False


def _describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def chat_model_unavailable() -> Optional[str]:
    """Why the chat model is not usable, or None if it is."""
    global _chat_reason, _chat_checked
    if _chat_checked:
        return _chat_reason

    _chat_checked = True
    if not os.getenv("LLM_CONFIG"):
        _chat_reason = "LLM_CONFIG is not set"
        return _chat_reason

    try:
        backends.get_llm().invoke("Answer with the word: ready.")
        _chat_reason = None
    except Exception as error:
        _chat_reason = f"chat model unreachable ({_describe(error)})"
    return _chat_reason


def embeddings_unavailable() -> Optional[str]:
    """Why the embedding model is not usable, or None if it is.

    It is needed by everything that touches the archive: consolidating a delete,
    archiving for the split, searching among the archived memories.
    """
    global _embeddings_reason, _embeddings_checked
    if _embeddings_checked:
        return _embeddings_reason

    _embeddings_checked = True
    try:
        # The model directly, without going through the vector store: checking an
        # embedding must not also require chromadb, nor depend on how Chroma
        # exposes its embedding function.
        backends._build_embeddings(backends.get_config()).embed_query("test")
        _embeddings_reason = None
    except Exception as error:
        _embeddings_reason = f"embedding unreachable ({_describe(error)})"
    return _embeddings_reason


def live_stack_unavailable() -> Optional[str]:
    """Why it cannot run on the full stack, or None if it can."""
    return chat_model_unavailable() or embeddings_unavailable()


def reset() -> None:
    """Forget the cached result (useful between different configurations)."""
    global _chat_reason, _chat_checked, _embeddings_reason, _embeddings_checked
    _chat_reason = _embeddings_reason = None
    _chat_checked = _embeddings_checked = False
