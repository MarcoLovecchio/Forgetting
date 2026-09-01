"""Standalone configuration for the memory service.

The memory agent used to get its chat model through
``shared_utils.llm_helpers.LLM_Initializer``, which also builds a database
adapter and loads the scenario customization of the whole architecture. None of
that is used by the memory agent: all it needs is a chat model and a couple of
limits. Keeping the configuration here makes the package buildable, runnable and
testable on its own.

Every value can be overridden through the environment, so the runtime behaviour
is unchanged with respect to the previous implementation (same ``LLM_CONFIG``
entry, same API key, same Chroma path).
"""

import ast
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Environment files are looked up walking the directory tree upwards, so the
# service behaves the same whether it is launched from the workspace root, from
# a colcon install space or from a bare test run.
_ENV_FILE_NAMES = (".env", ".config")


def _candidate_directories():
    """Directories to scan for the env files, nearest first."""
    seen = []
    for start in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        current = os.path.abspath(start)
        while True:
            if current not in seen:
                seen.append(current)
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    return seen


def load_environment(override: bool = False) -> list:
    """Load the ``.env`` / ``.config`` files of the workspace, if present.

    Returns the list of files that were actually loaded. An explicit
    ``MEMORY_ENV_FILE`` always wins over the automatic lookup.
    """
    loaded = []

    explicit = os.getenv("MEMORY_ENV_FILE")
    if explicit and os.path.isfile(explicit):
        load_dotenv(explicit, override=override)
        return [os.path.abspath(explicit)]

    for directory in _candidate_directories():
        for name in _ENV_FILE_NAMES:
            path = os.path.join(directory, name)
            if os.path.isfile(path) and path not in loaded:
                load_dotenv(path, override=override)
                loaded.append(path)
        if loaded:
            # Stop at the first directory that carries the configuration.
            break

    return loaded


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        print(f"\033[33mInvalid value for {name}: {raw!r}, falling back to {default}\033[0m")
        return default


@dataclass
class MemoryConfig:
    """Everything the memory service needs to run."""

    # Agent behaviour
    node_name: str = "memory_agent"
    maximum_historical_messages: int = 5  # Limit the number of historical messages to keep
    core_memory_limit: int = 150  # Character limit for core memory

    # Archival memory (ChromaDB)
    chroma_path: str = "./chroma_db"
    collection_name: str = "memory_archive"

    # Chat model, read from the LLM_CONFIG entry of this node
    llm_config: Dict[str, Any] = field(default_factory=dict)
    api_key_env: str = "GROQ_API_KEY"
    # Endpoint of a locally served model (Ollama and any OpenAI compatible
    # server). None means "the provider default", which is what the hosted
    # providers use.
    base_url: Optional[str] = None
    # Context window to ask the local runtime for. Ollama defaults to 2048 and
    # truncates longer prompts without saying anything: the consolidation and
    # the core/archival split both go past that.
    num_ctx: int = 8192

    # Embedding model, read from the EMBEDDING_CONFIG entry of this node.
    # Kept separate from llm_config because it is a different model, served
    # possibly by a different runtime.
    embedding_config: Dict[str, Any] = field(default_factory=dict)
    embedding_base_url: Optional[str] = None

    @classmethod
    def from_environment(cls, node_name: Optional[str] = None) -> "MemoryConfig":
        load_environment()
        node_name = node_name or os.getenv("MEMORY_LLM_NODE", "memory_agent")
        return cls(
            node_name=node_name,
            maximum_historical_messages=_env_int("MEMORY_MAX_HISTORICAL_MESSAGES", 5),
            core_memory_limit=_env_int("MEMORY_CORE_MEMORY_LIMIT", 150),
            chroma_path=os.path.abspath(os.getenv("MEMORY_CHROMA_PATH", "./chroma_db")),
            collection_name=os.getenv("MEMORY_COLLECTION_NAME", "memory_archive"),
            llm_config=_read_model_config("LLM_CONFIG", node_name),
            api_key_env=os.getenv("MEMORY_API_KEY_ENV", "GROQ_API_KEY"),
            base_url=os.getenv("MEMORY_LLM_BASE_URL") or None,
            num_ctx=_env_int("MEMORY_NUM_CTX", 8192),
            embedding_config=_read_model_config("EMBEDDING_CONFIG", node_name),
            embedding_base_url=os.getenv("MEMORY_EMBEDDING_BASE_URL") or None,
        )


def _read_model_config(variable: str, node_name: str) -> Dict[str, Any]:
    """Parse a per-node model configuration variable for the given node.

    ``LLM_CONFIG`` and ``EMBEDDING_CONFIG`` have the same shape: a dict keyed by
    node name, each entry carrying model_name, model_provider and whatever else
    the provider needs.

    Returns an empty dict when the variable is missing or malformed: the error
    is raised later, only if that model is actually needed, so that importing
    this package never requires a configured environment.
    """
    raw = os.getenv(variable)
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as error:
        print(f"\033[31mCannot parse {variable}: {error}\033[0m")
        return {}
    if not isinstance(parsed, dict):
        return {}
    node_config = parsed.get(node_name, {})
    return node_config if isinstance(node_config, dict) else {}
