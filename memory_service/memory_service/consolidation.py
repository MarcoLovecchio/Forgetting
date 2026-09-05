"""Memory consolidation.

Core memory stops being a flat list of strings and becomes a small auditable
data model: every fact is a :class:`CoreMemoryItem` with a status and a lineage,
and every change is recorded in an operation log.

When the summarizer extracts a fact from the conversation, the LLM classifies it
against the memories that are already known - both the core ones and the active
archival ones - and the classification decides what happens:

===========  =============================================  =======================
operation    meaning                                        effect
===========  =============================================  =======================
new          nothing similar is in memory                   a new item is created
redundant    the same information is already stored         the existing item is
                                                            reinforced (updated_at)
update       it adds detail to a fact already stored        the old item becomes
                                                            "superseded", a new item
                                                            takes its place, and the
                                                            operation log links them
contradict   it contradicts a fact already stored           same as update
delete       the user explicitly asked to forget a fact     the item becomes
                                                            "deleted"
===========  =============================================  =======================

Items that stop being active (superseded or deleted) are written to the archive
as tombstones and dropped from core memory: ``core_memory`` in the agent state
only ever holds active items, while the history survives a restart. The same is
true when an item is moved to the archive by the core/archival split - the whole
item is archived, so status, lineage and timestamps are never lost.

Eviction - deciding what to move out of core memory when the character budget is
exceeded - is deliberately not handled here.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from memory_service import backends

# How many archival memories are offered to the classifier as possible targets.
ARCHIVE_CANDIDATES_K = 5


MemoryStatus = Literal["active", "superseded", "deleted"]
MemoryOperationType = Literal["new", "redundant", "update", "contradict", "delete"]

ITEM_ID_LENGTH = 8


def new_item_id() -> str:
    """Identifier of a memory item, unique across core memory and archive."""
    return uuid.uuid4().hex[:ITEM_ID_LENGTH]


class CoreMemoryItem(BaseModel):
    """A single fact about the user, with its status and its lineage."""

    id: str = Field(default_factory=new_item_id)
    content: str
    status: MemoryStatus = "active"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class OperationLogEntry(BaseModel):
    """One consolidation decision, kept for inspection and evaluation."""

    op_type: Literal["create", "redundant", "update", "contradict", "delete", "archive"]
    item_id: str
    related_item_id: Optional[str] = None
    content: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# --------------------------------------------------------------------------- #
# Tools exposed to the LLM
# --------------------------------------------------------------------------- #

class MemoryOperation(BaseModel):
    """A fact extracted from the conversation, classified against what is known."""

    fact: str
    operation: MemoryOperationType
    target_item_id: Optional[str] = None


class InsertCoreMemories(BaseModel):
    """These are the core memories of our assistant. They store relevant facts about the
    user in a concise and synthetic manner. This information is given as context to every
    interaction, so only important facts belong here.

    For each fact you find in the messages, classify it against the memories you are given
    as `id: content` pairs:
    - new: a genuinely new fact, unrelated to any memory you were given (no target_item_id).
    - redundant: it confirms a memory that is already stored, without changing it
      (target_item_id required).
    - update: it refines or adds detail to a memory already stored, without contradicting
      it (target_item_id required).
    - contradict: it replaces a memory that is now wrong or outdated (target_item_id
      required).
    - delete: use ONLY when the user explicitly asks to forget, remove or stop storing a
      specific fact (target_item_id required).

    Do not use delete for facts that merely became less relevant or less interesting: use
    update or contradict for those."""

    memories: List[MemoryOperation]


class MemorySplitDecision(BaseModel):
    """Where a single core memory should live from now on."""

    item_id: str
    destination: Literal["core", "archive"]


class SplitCoreAndArchivalMemory(BaseModel):
    """This tool decides, for each active core memory, whether it stays in core memory or
    moves to archival (long term, less frequently accessed) memory.

    The core memory character limit is a HARD constraint, not a suggestion: after your
    decision, the memories you keep in core MUST fit within it. Exceeding it is never an
    acceptable answer, and neither is keeping everything in core.

    Archiving is not deleting. An archived memory keeps all its content and stays
    searchable: it is retrieved and brought back into the conversation whenever it becomes
    relevant again. The only thing it loses is the seat in the always-on context. So when
    you are unsure about a memory, archive it - the cost of archiving something useful is
    far lower than the cost of breaking the limit.

    Every memory you are given comes as `id: content (N characters)`. Add up the lengths of
    the ones you decide to keep in core and check the total against the limit BEFORE
    answering. Return one decision for every memory in the list, referring to it by id."""

    decisions: List[MemorySplitDecision]


# The classifier improvises on the labels more often than one would like.
_OPERATION_ALIASES = {
    "new": "new", "create": "new", "add": "new", "insert": "new",
    "redundant": "redundant", "reinforce": "redundant", "duplicate": "redundant",
    "confirm": "redundant",
    "update": "update", "refine": "update", "extend": "update",
    "contradict": "contradict", "contradiction": "contradict", "replace": "contradict",
    "delete": "delete", "remove": "delete", "forget": "delete",
}


def normalize_operation(operation: Any) -> Optional[str]:
    """Map what the LLM produced onto one of the five known operations."""
    if not isinstance(operation, str):
        return None
    return _OPERATION_ALIASES.get(operation.strip().lower())


# --------------------------------------------------------------------------- #
# Reading core memory
# --------------------------------------------------------------------------- #

def get_active_items(items: Iterable[CoreMemoryItem]) -> List[CoreMemoryItem]:
    """Only the items that still count: superseded and deleted ones are ignored."""
    return [item for item in items if item.status == "active"]


def serialize_core_memory_for_prompt(items: Iterable[CoreMemoryItem]) -> List[str]:
    """Core memory as the plain list of strings that prompts and ROS fields expect."""
    return [item.content for item in get_active_items(items)]


def serialize_core_memory_ids(items: Iterable[CoreMemoryItem]) -> List[str]:
    """Ids of the active core memories.

    Same order as :func:`serialize_core_memory_for_prompt`, so the two lists can
    be zipped: that is how the ROS response keeps contents and ids aligned.
    """
    return [item.id for item in get_active_items(items)]


def serialize_core_memory_with_ids(items: Iterable[CoreMemoryItem]) -> Dict[str, str]:
    """Core memory as `id: content` pairs, the form the classifier reasons on."""
    return {item.id: item.content for item in get_active_items(items)}


def core_memory_length(items: Iterable[CoreMemoryItem]) -> int:
    """Characters currently occupied by the active core memories."""
    return len("\n".join(serialize_core_memory_for_prompt(items)))


def serialize_operation_log_for_response(log: Iterable[OperationLogEntry]) -> List[str]:
    """Flatten the operation log into JSON strings, for a ROS string[] field."""
    return [entry.model_dump_json() for entry in log]


# --------------------------------------------------------------------------- #
# Archive access
# --------------------------------------------------------------------------- #

def archive_metadata(item: CoreMemoryItem) -> dict:
    """Every CoreMemoryItem field except the content, which is the document itself.

    Chroma only accepts scalars in the metadata, hence the empty string instead of
    None and the ISO timestamps.
    """
    return {
        "status": item.status,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
    }


def archive_items(items: Iterable[CoreMemoryItem]) -> str:
    """Write items to the archive, preserving all their fields.

    Ids are stable across the two stores, so re-archiving an item updates the
    existing entry instead of duplicating it.
    """
    items = list(items)
    if not items:
        return "No items archived."

    ids = [item.id for item in items]
    print(f"\tArchiving items: {ids}")
    backends.get_vector_store().add_texts(
        texts=[item.content for item in items],
        ids=ids,
        metadatas=[archive_metadata(item) for item in items],
    )
    return f"Archived items with IDs: {', '.join(ids)}"


def get_archive_item(item_id: str) -> Optional[dict]:
    """Content and metadata of an archived item, or None if it is not there."""
    if not item_id:
        return None
    try:
        result = backends.get_vector_store().get(ids=[item_id])
    except Exception as error:
        print(f"\tArchive lookup failed for {item_id}: {error}")
        return None
    if not result or not result.get("ids"):
        return None
    documents = result.get("documents") or [""]
    metadatas = result.get("metadatas") or [{}]
    return {"content": documents[0], "metadata": dict(metadatas[0] or {})}


def _rewrite_archive_metadata(item_id: str, status: Optional[str] = None) -> Optional[dict]:
    """Refresh updated_at of an archived item, optionally changing its status.

    The item is updated in place: it is not pulled back into core memory, and
    its document is not rewritten. Passing the text through add_texts would make
    the store recompute the embedding - a call to the embedding model, over the
    network, to change a scalar field. Chroma only recomputes when `documents`
    are supplied, so sending metadata alone leaves the vector where it is.
    """
    archived = get_archive_item(item_id)
    if archived is None:
        print(f"\tArchived item not found: {item_id}")
        return None

    metadata = dict(archived["metadata"])
    if status is not None:
        metadata["status"] = status
    metadata["updated_at"] = datetime.now().isoformat()
    backends.get_vector_store()._collection.update(ids=[item_id], metadatas=[metadata])
    return archived


def search_archive(query: str,
                   k: int = ARCHIVE_CANDIDATES_K) -> List[Tuple[str, str, float]]:
    """The k active archival memories closest to a text.

    Returns (id, content, distance) triples. The distance is the store's own,
    raw: lower means closer. It is not normalised on purpose - the mapping to a
    0-1 "relevance" depends on the collection's metric and on whether the
    embeddings are normalised, so a converted number would look meaningful while
    being wrong. What it is for is choosing a threshold by looking at the values
    a real run produces.
    """
    try:
        k = max(1, int(k))
    except (TypeError, ValueError):
        k = ARCHIVE_CANDIDATES_K

    try:
        found = backends.get_vector_store().similarity_search_with_score(
            query, k=k, filter={"status": "active"})
    except Exception as error:
        print(f"\tArchive search failed: {error}")
        return []

    return [(doc.id, doc.page_content, float(distance)) for doc, distance in found]


NO_ARCHIVAL_RESULTS = "No relevant active memories found."


def serialize_retrieved_for_response(retrieved) -> List[str]:
    """Retrieved archival memories as a list, empty when nothing came back."""
    text = str(retrieved or "").strip()
    if not text or text == NO_ARCHIVAL_RESULTS:
        return []
    return [line for line in text.splitlines() if line.strip()]


def retrieve_active_archival_memories(query: str, k: int = 3) -> str:
    """Archive lookup for the retrieval path, tombstones excluded."""
    results = search_archive(query, k=k)
    if not results:
        return NO_ARCHIVAL_RESULTS
    # La distanza esce insieme al contenuto perche' questa stringa e' l'unico
    # canale verso il resoconto: chi legge deve poter distinguere una memoria
    # centrata da una raschiata dal fondo per riempire k.
    return "\n".join(f"ID: {doc_id}, Content: {content}, Distance: {distance:.3f}"
                      for doc_id, content, distance in results)


def build_candidate_memories(
    core_memory: Iterable[CoreMemoryItem],
    query: str,
    k: int = ARCHIVE_CANDIDATES_K,
) -> Dict[str, str]:
    """Memories the classifier may point at: the active core ones, plus the archival
    ones that look related to the text being consolidated."""
    candidates = serialize_core_memory_with_ids(core_memory)
    # La distanza qui non serve: al classificatore interessa a cosa puntare,
    # non quanto il vettoriale ci abbia creduto.
    for doc_id, content, _ in search_archive(query, k=k):
        if doc_id not in candidates:
            candidates[doc_id] = content
    return candidates


# --------------------------------------------------------------------------- #
# Operations on a core memory item
# --------------------------------------------------------------------------- #

def create_item(content: str, log: List[OperationLogEntry]) -> CoreMemoryItem:
    """A fact nothing in memory covers yet."""
    item = CoreMemoryItem(content=content)
    log.append(OperationLogEntry(op_type="create", item_id=item.id, content=content))
    return item


def reinforce_item(
    item: CoreMemoryItem,
    log: List[OperationLogEntry],
    content: Optional[str] = None,
) -> CoreMemoryItem:
    """A fact already stored: refresh it instead of storing it twice."""
    item.updated_at = datetime.now()
    log.append(OperationLogEntry(
        op_type="redundant", item_id=item.id, content=content or item.content))
    return item


def _retire_item(item: CoreMemoryItem, status: MemoryStatus) -> None:
    """Take an item out of active duty and keep it in the archive as a tombstone."""
    item.status = status
    item.updated_at = datetime.now()
    archive_items([item])


def supersede_item(
    old_item: CoreMemoryItem,
    new_content: str,
    log: List[OperationLogEntry],
    op_type: Literal["update", "contradict"] = "update",
) -> CoreMemoryItem:
    """Replace a core item with a newer version that points back at it."""
    _retire_item(old_item, "superseded")
    new_item = CoreMemoryItem(content=new_content)
    log.append(OperationLogEntry(
        op_type=op_type, item_id=new_item.id,
        related_item_id=old_item.id, content=new_content))
    return new_item


def delete_item(item: CoreMemoryItem, log: List[OperationLogEntry]) -> CoreMemoryItem:
    """The user asked to forget this fact."""
    _retire_item(item, "deleted")
    log.append(OperationLogEntry(op_type="delete", item_id=item.id, content=item.content))
    return item


# --------------------------------------------------------------------------- #
# Operations on an item that already lives in the archive
# --------------------------------------------------------------------------- #

def reinforce_archived_item(
    item_id: str,
    log: List[OperationLogEntry],
    content: Optional[str] = None,
) -> bool:
    """Same as reinforce_item, for a memory that already moved to the archive.

    The item stays archived: being mentioned again is not a reason to bring it
    back into the core memory budget.
    """
    archived = _rewrite_archive_metadata(item_id)
    if archived is None:
        return False
    log.append(OperationLogEntry(
        op_type="redundant", item_id=item_id, content=content or archived["content"]))
    return True


def supersede_archived_item(
    item_id: str,
    new_content: str,
    log: List[OperationLogEntry],
    op_type: Literal["update", "contradict"] = "update",
) -> Optional[CoreMemoryItem]:
    """Flag an archived memory as superseded; the newer version starts in core memory."""
    if _rewrite_archive_metadata(item_id, status="superseded") is None:
        return None
    new_item = CoreMemoryItem(content=new_content)
    log.append(OperationLogEntry(
        op_type=op_type, item_id=new_item.id,
        related_item_id=item_id, content=new_content))
    return new_item


def delete_archived_item(item_id: str, log: List[OperationLogEntry]) -> bool:
    """Flag an archived memory as deleted, so retrieval stops returning it."""
    archived = _rewrite_archive_metadata(item_id, status="deleted")
    if archived is None:
        return False
    log.append(OperationLogEntry(
        op_type="delete", item_id=item_id, content=archived["content"]))
    return True


# --------------------------------------------------------------------------- #
# Applying what the LLM decided
# --------------------------------------------------------------------------- #

def _guess_target(
    fact: str,
    candidates: Iterable[CoreMemoryItem],
) -> Optional[CoreMemoryItem]:
    """Best effort match when the classifier gives no usable target id.

    Only core memories are matched here: an archival target without a valid id is
    left to the "no target" path below.
    """
    text = fact.strip().lower()
    if not text:
        return None
    for item in candidates:
        content = item.content.strip().lower()
        if content and (text in content or content in text):
            return item
    return None


def apply_memory_operations(
    operations: Iterable[Any],
    core_memory: Iterable[CoreMemoryItem],
    log: List[OperationLogEntry],
) -> Tuple[List[CoreMemoryItem], str]:
    """Apply the classified facts to core memory.

    Returns the new core memory - active items only, since superseded and deleted
    ones move to the archive - and a short summary used as the tool result.
    """
    items = list(core_memory)
    counters = {"new": 0, "redundant": 0, "update": 0, "contradict": 0, "delete": 0}

    for raw_operation in operations or []:
        if not isinstance(raw_operation, dict):
            print(f"\tSkipping malformed operation: {raw_operation!r}")
            continue

        fact = str(raw_operation.get("fact", "") or "").strip()
        operation = normalize_operation(raw_operation.get("operation"))
        target_id = raw_operation.get("target_item_id") or None

        if operation is None:
            print(f"\tSkipping operation with unknown type: {raw_operation!r}")
            continue
        if not fact and operation != "delete":
            print(f"\tSkipping operation without a fact: {raw_operation!r}")
            continue

        active_by_id = {item.id: item for item in items if item.status == "active"}

        target_item = active_by_id.get(target_id) if target_id else None
        archived_target = None
        if operation != "new" and target_item is None:
            archived_target = get_archive_item(target_id) if target_id else None
            if archived_target is None:
                # The classifier forgot the id, or invented one: try the text.
                target_item = _guess_target(fact, active_by_id.values())
                if target_item is not None:
                    print(f"\t[{operation}] target matched by content: {target_item.id}")
                    target_id = target_item.id

        if operation != "new" and target_item is None and archived_target is None:
            if operation == "delete":
                print(f"\tNothing to delete for: {fact!r}")
                continue
            # The fact refers to something we do not actually have: keep it anyway.
            print(f"\tNo target for '{operation}', storing {fact!r} as a new memory")
            operation = "new"

        if operation == "new":
            items.append(create_item(fact, log))
            counters["new"] += 1

        elif target_item is not None:
            # The target is a live core memory.
            if operation == "redundant":
                reinforce_item(target_item, log, fact)
            elif operation == "delete":
                delete_item(target_item, log)
            else:  # update | contradict
                items.append(supersede_item(target_item, fact, log, op_type=operation))
            counters[operation] += 1

        else:
            # The target already lives in the archive and stays there.
            if operation == "redundant":
                reinforce_archived_item(target_id, log, fact)
            elif operation == "delete":
                delete_archived_item(target_id, log)
            else:  # update | contradict
                new_item = supersede_archived_item(target_id, fact, log, op_type=operation)
                if new_item is not None:
                    items.append(new_item)
            counters[operation] += 1

    summary = ", ".join(f"{count} {name}" for name, count in counters.items() if count)
    result = f"Core memories updated ({summary})." if summary else "Core memories unchanged."
    return [item for item in items if item.status == "active"], result


def apply_split_decisions(
    decisions: Iterable[Any],
    core_memory: Iterable[CoreMemoryItem],
    log: List[OperationLogEntry],
) -> Tuple[List[CoreMemoryItem], str]:
    """Move to the archive the items the LLM decided not to keep in core memory.

    The whole item is archived - status, lineage and timestamps included - so
    nothing is lost in the transition.
    """
    items = list(core_memory)
    items_by_id = {item.id: item for item in items}
    to_archive: List[CoreMemoryItem] = []

    for raw_decision in decisions or []:
        if not isinstance(raw_decision, dict):
            print(f"\tSkipping malformed split decision: {raw_decision!r}")
            continue
        item = items_by_id.get(raw_decision.get("item_id"))
        if item is None:
            print(f"\tSkipping split decision with unknown item_id: {raw_decision!r}")
            continue
        if raw_decision.get("destination") == "archive" and item not in to_archive:
            to_archive.append(item)

    if not to_archive:
        return [item for item in items if item.status == "active"], "No items archived."

    result = archive_items(to_archive)
    archived_ids = {item.id for item in to_archive}
    for item in to_archive:
        log.append(OperationLogEntry(op_type="archive", item_id=item.id, content=item.content))

    remaining = [
        item for item in items
        if item.status == "active" and item.id not in archived_ids
    ]
    return remaining, result
