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


# Sovrascritture per nodo dei parametri del modello. Quello che non e' elencato
# qui viene dalla voce LLM_CONFIG di .config, quindi un dizionario vuoto
# significa "come tutti gli altri" - ed e' lo stato attuale, cioe' nessun
# cambiamento di comportamento finche' non ci si scrive dentro.
#
# I quattro nodi hanno bisogni diversi: classificare fatti contro id e' il posto
# dove il ragionamento paga, scegliere fra due strumenti o scrivere due frasi da
# un testo gia' pronto molto meno. Accendere o spegnere il ragionamento per uno
# di loro e' una riga qui.
#
# Attenzione al campionamento: temperature e top_p in .config sono i valori che
# Qwen consiglia PER la modalita' con ragionamento. Spegnendolo per un nodo,
# vanno messi qui anche i suoi, presi dal model card - lasciare solo
# enable_thinking lo farebbe girare con il preset sbagliato.
NODE_SAMPLING: Dict[str, Dict[str, Any]] = {
    "retrieval": {},        # decide fra retrieve_memory e NoSearchNeeded
    "generate_answer": {},  # compone la risposta all'utente
    "consolidation": {},    # estrae i fatti e li classifica
    "core_split": {},       # decide cosa resta in core e cosa va in archivio
}


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    value = str(raw).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    print(f"\033[33mInvalid value for {name}: {raw!r}, falling back to {default}\033[0m")
    return default


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
    maximum_historical_messages: int = 4
    core_memory_limit: int = 400
    # Whether the retrieve branch also composes the reply to the user. On,
    # because this is the only node in the architecture that answers FROM the
    # memory: explainability answers from the results of a database query and
    # never reads what it remembers. It was off for a while, when the reply
    # reached no caller and only landed in last_messages as a turn nobody read;
    # retrieved_memories in the service response and this node appending the
    # question together with its answer closed both holes.
    # MEMORY_GENERATE_ANSWER=false turns it off again, for a caller that has its
    # own answering node and only wants the context.
    generate_answer: bool = True

    # Archival memory (ChromaDB)
    chroma_path: str = "./chroma_db"
    collection_name: str = "memory_archive"

    # Chat model, read from the LLM_CONFIG entry of this node
    llm_config: Dict[str, Any] = field(default_factory=dict)
    # Environment variable holding the API key, empty when the endpoint does not
    # want one. It is deliberately NOT defaulted to GROQ_API_KEY: the memory
    # agent talks to the cluster, and that key belongs to the other nodes - it
    # would travel to an endpoint that has no business seeing it.
    api_key_env: str = ""
    # Endpoint of the model server. None means "the provider default", which is
    # what the hosted providers use.
    base_url: Optional[str] = None

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
            maximum_historical_messages=_env_int("MEMORY_MAX_HISTORICAL_MESSAGES", 4),
            core_memory_limit=_env_int("MEMORY_CORE_MEMORY_LIMIT", 400),
            generate_answer=_env_bool("MEMORY_GENERATE_ANSWER", True),
            chroma_path=os.path.abspath(os.getenv("MEMORY_CHROMA_PATH", "./chroma_db")),
            collection_name=os.getenv("MEMORY_COLLECTION_NAME", "memory_archive"),
            llm_config=_read_model_config("LLM_CONFIG", node_name),
            api_key_env=os.getenv("MEMORY_API_KEY_ENV", ""),
            base_url=os.getenv("MEMORY_LLM_BASE_URL") or None,
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
