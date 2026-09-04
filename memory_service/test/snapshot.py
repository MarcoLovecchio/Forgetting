"""Readable dump of the whole memory state, printed at every turn.

Used by both the offline lifecycle test and the one that talks to the real LLM:
what makes those tests useful is being able to read, turn by turn, what ended up
in core memory, what moved to the archive and why.
"""

import sys

from langchain_core.messages import HumanMessage

from memory_service.consolidation import get_active_items

SEPARATOR = "=" * 78


def safe_print(line: str) -> None:
    """print() that survives a console unable to encode the emoji."""
    try:
        print(line)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(encoding, errors="replace").decode(encoding))


def print_turn_header(number, operation: str, description: str,
                      consolidates: str = "") -> None:
    """Announce a turn before running it.

    `consolidates` says what the turn is about to digest, which with an even
    window is the PREVIOUS exchange: without it the trace looks wrong when it
    is only late.
    """
    safe_print("\n" + "#" * 78)
    safe_print(f"### TURNO {number} - operazione '{operation}': {description}")
    if consolidates:
        safe_print(f"###   consolida: {consolidates}")
    safe_print("#" * 78)


def print_memory_snapshot(title: str, state: dict, vector_store=None) -> None:
    """Everything worth looking at after a turn."""
    safe_print("\n" + SEPARATOR)
    safe_print(f"=== {title.upper()} ===")
    safe_print(SEPARATOR)

    _print_core_memory(state)
    _print_messages(state)
    _print_archive(vector_store)
    _print_retrieved(state)
    _print_operation_log(state)

    safe_print(SEPARATOR + "\n")


def _print_core_memory(state: dict) -> None:
    core_memory = state.get("core_memory", [])
    active = get_active_items(core_memory)
    limit = state.get("core_memory_limit")
    used = len("\n".join(item.content for item in active))

    header = f"\n[CORE MEMORY] {len(active)} elementi attivi"
    if limit is not None:
        header += f", {used}/{limit} caratteri"
    safe_print(f"\U0001F9E0 {header}:")

    if not active:
        safe_print("  (core memory vuota)")
    for item in active:
        safe_print(f"  - [{item.id}] (status: {item.status}) {item.content}")
        safe_print(f"      created_at: {item.created_at} | updated_at: {item.updated_at}")


def _print_messages(state: dict) -> None:
    messages = state.get("messages", [])
    safe_print(f"\n\U0001F4AC [MAIN MEMORY / MESSAGES] {len(messages)} messaggi:")
    if not messages:
        safe_print("  (nessun messaggio in finestra)")
    for message in messages:
        role = "Utente" if isinstance(message, HumanMessage) else "Bot"
        safe_print(f"  [{role}]: {message.content}")


def _print_archive(vector_store) -> None:
    safe_print("\n\U0001F4E6 [ARCHIVE MEMORY / VECTOR STORE]:")
    if vector_store is None or not hasattr(vector_store, "get"):
        safe_print("  (nessun archivio da ispezionare)")
        return

    try:
        stored = vector_store.get()
    except Exception as error:  # a real Chroma may refuse an unfiltered get
        safe_print(f"  (archivio non ispezionabile: {error})")
        return

    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    if not ids:
        safe_print("  (archivio vettoriale attualmente vuoto)")
        return

    for doc_id, content, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        status = metadata["status"]
        safe_print(f"  - [{doc_id}] (status: {status}) {content}")
        if metadata:
            safe_print(f"      metadata: {metadata}")


def _print_retrieved(state: dict) -> None:
    retrieved = state.get("retrieved_memory", "")
    safe_print("\n\U0001F4DA [ARCHIVE MEMORY RETRIEVED (ultima query)]:")
    if not retrieved:
        safe_print("  (nessun recupero attivo)")
        return
    for line in str(retrieved).splitlines():
        safe_print(f"  {line}")


def _print_operation_log(state: dict) -> None:
    log = state.get("operation_log", [])
    safe_print(f"\n\U0001F4CB [OPERATION LOG] {len(log)} operazioni:")
    if not log:
        safe_print("  (nessuna operazione registrata)")
    for entry in log:
        safe_print(f"  - op: {entry.op_type:<10} | item: {entry.item_id}"
                   f" | related: {entry.related_item_id or '-'}")
        safe_print(f"      time: {entry.timestamp} | content: {entry.content}")
