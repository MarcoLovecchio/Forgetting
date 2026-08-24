"""Readable dump of the whole memory state, printed at every turn.

Used by both the offline lifecycle test and the one that talks to the real LLM:
what makes those tests useful is being able to read, turn by turn, what ended up
in core memory, what moved to the archive and why.
"""

import sys

from langchain_core.messages import HumanMessage

from memory_service.consolidation import get_active_items

SEPARATOR = "=" * 78


def _write(line: str) -> None:
    """print() that survives a console unable to encode the emoji."""
    try:
        print(line)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(encoding, errors="replace").decode(encoding))


def print_turn_header(number, operation: str, description: str) -> None:
    """Announce a turn before running it."""
    _write("\n" + "#" * 78)
    _write(f"### TURNO {number} - operazione '{operation}': {description}")
    _write("#" * 78)


def print_memory_snapshot(title: str, state: dict, vector_store=None) -> None:
    """Everything worth looking at after a turn."""
    _write("\n" + SEPARATOR)
    _write(f"=== {title.upper()} ===")
    _write(SEPARATOR)

    _print_core_memory(state)
    _print_messages(state)
    _print_archive(vector_store)
    _print_retrieved(state)
    _print_operation_log(state)

    _write(SEPARATOR + "\n")


def _print_core_memory(state: dict) -> None:
    core_memory = state.get("core_memory", [])
    active = get_active_items(core_memory)
    limit = state.get("core_memory_limit")
    used = len("\n".join(item.content for item in active))

    header = f"\n[CORE MEMORY] {len(active)} elementi attivi"
    if limit is not None:
        header += f", {used}/{limit} caratteri"
    _write(f"\U0001F9E0 {header}:")

    if not active:
        _write("  (core memory vuota)")
    for item in active:
        _write(f"  - [{item.id}] (status: {item.status}) {item.content}")
        _write(f"      created_at: {item.created_at} | updated_at: {item.updated_at}"
               f" | supersedes: {item.supersedes or '-'}")


def _print_messages(state: dict) -> None:
    messages = state.get("messages", [])
    _write(f"\n\U0001F4AC [MAIN MEMORY / MESSAGES] {len(messages)} messaggi:")
    if not messages:
        _write("  (nessun messaggio in finestra)")
    for message in messages:
        role = "Utente" if isinstance(message, HumanMessage) else "Bot"
        _write(f"  [{role}]: {message.content}")


def _print_archive(vector_store) -> None:
    _write("\n\U0001F4E6 [ARCHIVE MEMORY / VECTOR STORE]:")
    if vector_store is None or not hasattr(vector_store, "get"):
        _write("  (nessun archivio da ispezionare)")
        return

    try:
        stored = vector_store.get()
    except Exception as error:  # a real Chroma may refuse an unfiltered get
        _write(f"  (archivio non ispezionabile: {error})")
        return

    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    if not ids:
        _write("  (archivio vettoriale attualmente vuoto)")
        return

    for doc_id, content, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        status = metadata.get("status", "N/A")
        _write(f"  - [{doc_id}] (status: {status}) {content}")
        if metadata:
            _write(f"      metadata: {metadata}")


def _print_retrieved(state: dict) -> None:
    retrieved = state.get("retrieved_memory", "")
    _write("\n\U0001F4DA [ARCHIVE MEMORY RETRIEVED (ultima query)]:")
    if not retrieved:
        _write("  (nessun recupero attivo)")
        return
    for line in str(retrieved).splitlines():
        _write(f"  {line}")


def _print_operation_log(state: dict) -> None:
    log = state.get("operation_log", [])
    _write(f"\n\U0001F4CB [OPERATION LOG] {len(log)} operazioni:")
    if not log:
        _write("  (nessuna operazione registrata)")
    for entry in log:
        _write(f"  - op: {entry.op_type:<10} | item: {entry.item_id}"
               f" | related: {entry.related_item_id or '-'}")
        _write(f"      time: {entry.timestamp} | content: {entry.content}")
