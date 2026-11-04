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
update       it changes a fact already stored: more         the old item becomes
             detail, a new value, or it was wrong           "superseded", a new item
                                                            takes its place, and the
                                                            operation log links them
delete       the user explicitly asked to forget a fact     the item becomes
                                                            "deleted"
===========  =============================================  =======================

With MemoryConfig.consolidation_utility the classifier also judges every new and
every updated fact, in the same call: a safety flag and, unless it is set, either
a single utility rating from 1 to 5 (single) or need and gain from 1 to 3
(need_gain). An update is judged again; a redundant keeps the ratings it had.

Items that stop being active (superseded or deleted) are written to the archive
as tombstones and dropped from core memory: ``core_memory`` in the agent state
only ever holds active items, while the history survives a restart. The same is
true when an item is moved to the archive by the core/archival split - the whole
item is archived, so status, lineage and timestamps are never lost. The one
exception is MemoryConfig.eviction: when it is on, the tombstones leave the
archive at the end of every insert, and so do the active memories with the
lowest score when the archive is over its limit (see eviction.py).

Deciding what moves out of core memory when the character budget is exceeded is
the core split's job, in memory_manager_llm.py: this module only applies its
decisions (apply_split_decisions).
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service import backends

ARCHIVE_CANDIDATES_K = 10
CONTENT_MATCH_K = 5

MemoryStatus = Literal["active", "superseded", "deleted"]
MemoryOperationType = Literal["new", "redundant", "update", "delete"]

ITEM_ID_LENGTH = 8


def new_item_id() -> str:
    """Identifier of a memory item, unique across core memory and archive."""
    return uuid.uuid4().hex[:ITEM_ID_LENGTH]


class CoreMemoryItem(BaseModel):
    """A single fact about the user, with its status and its lineage."""

    id: str = Field(default_factory=new_item_id)
    content: str
    status: MemoryStatus = "active"
    updated_at: datetime = Field(default_factory=backends.now)

    n_retrieve: int = 0
    retrieved_at: datetime = Field(default_factory=backends.now)
    n_used: int = 0

    # None when not given: consolidation_utility off, or a safety fact.
    safety: Optional[bool] = None
    utility: Optional[int] = None
    need: Optional[int] = None
    gain: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _one_creation_instant(cls, data: Any) -> Any:
        """updated_at and retrieved_at start from the same instant."""
        if isinstance(data, dict):
            data = dict(data)
            created = data.setdefault("updated_at", backends.now())
            data.setdefault("retrieved_at", created)
        return data


class OperationLogEntry(BaseModel):
    """One consolidation decision, kept for inspection and evaluation."""

    op_type: Literal["create", "redundant", "update", "delete", "archive", "evict", "prune"]
    item_id: str
    related_item_id: Optional[str] = None
    content: Optional[str] = None
    score: Optional[float] = None
    score_terms: Optional[Dict[str, float]] = None
    # create and update only: see logged_ratings.
    ratings: Optional[Dict[str, Union[bool, int]]] = None
    timestamp: datetime = Field(default_factory=backends.now)


# --------------------------------------------------------------------------- #
# Tools exposed to the LLM
# --------------------------------------------------------------------------- #

class MemoryOperation(BaseModel):
    """A fact extracted from the conversation, classified against what is known.
    """

    fact: str = Field(description=(
        "The fact itself, in English whatever language the user spoke. Write it about the "
        "user, in the third person, and call them \"the user\", never by name: the name is a "
        "fact of its own. Never write it as a copy of the sentence the user wrote. State "
        "what is true now, not what changed. For an update, write the whole memory again: "
        "copy the old memory and change only the parts that what the user said "
        "contradicts. A part the user says has changed, ended or is no longer true is "
        "contradicted; every other part stays, even if the user did not mention it."))
    operation: MemoryOperationType
    target_item_id: Optional[str] = Field(
        default=None,
        description=("Id of the memory this fact concerns, required for every operation "
                     "except new. The link belongs here and never in the text of the "
                     "fact."))


class InsertCoreMemories(BaseModel):
    """These are the core memories of our assistant. They store relevant facts about the
    user in a concise and synthetic manner. This information is given as context to every
    interaction, so only important facts belong here.

    For each fact you find in the messages, classify it against the memories you are given
    as `id: content` pairs:
    - new: a genuinely new fact, unrelated to any memory you were given (no target_item_id).
    - redundant: the stored memory already says everything the user said: its text would not
      change (target_item_id required).
    - update: the stored memory has to change to include what the user said - a new value, a
      correction, or one more detail (target_item_id required). The fact REPLACES that memory,
      so write it complete: keep every part of the old memory that what the user said does
      not contradict.
    - delete: use ONLY when the user explicitly asks to delete, forget, remove or stop storing
      a specific fact (target_item_id required). Write as fact the text of the memory being
      deleted, not the request.

    Do not use delete for facts that merely became less relevant or less interesting: use
    update for those."""

    memories: List[MemoryOperation] = Field(description=(
        "Empty for a question, small talk, or a request to forget something that is not "
        "among the known memories."))


SAFETY_DESCRIPTION = (
    "True only if forgetting this fact could put the user at risk: without it the "
    "assistant could harm the user or put them in danger, as with an allergy, a medical "
    "condition or treatment, a physical limitation, or something the user must avoid for "
    "their health or safety. False for everything else, however important it is to the "
    "user.")

# need, gain and utility judge with the same words: the two modes differ only in how the
# judgments are combined.
ONE_OFF = (
    "it is already over or holds only for now: something that happened once, such as a "
    "purchase, an errand, a mishap, or a passing mood")
PART_OF_LIFE = (
    "it stays true and names one of these: who the user is, their home, their work or "
    "study, a person or an animal close to them, their diet, health or limits, a habit or "
    "routine, an ongoing activity; or when it is a plan or an appointment that has not "
    "come yet. What the user does not, cannot or never does is a limit. A change to one "
    "of these, such as a habit the user stopped, counts too")
TOPICAL = (
    "it stays true, but only one specific topic would bring it up, such as a detail of a "
    "hobby, a taste, an opinion, an old experience")
DETAIL = "a detail that would change no answer, such as a color or a passing remark"
EXACT_VALUE = "an exact value that cannot be guessed: a name, a place, a date, a time, a number"


class SafetyJudgedOperation(MemoryOperation):
    """A classified fact, flagged when forgetting it could put the user at risk."""

    safety: Optional[bool] = Field(default=None, description=(
        SAFETY_DESCRIPTION + " Only for new and update: leave it out for redundant and "
        "delete."))


class UtilityJudgedOperation(SafetyJudgedOperation):
    """A classified fact, with its safety flag and one utility rating."""

    utility: Optional[Literal[1, 2, 3, 4, 5]] = Field(default=None, description=(
        "How valuable it will be to keep this fact and find it again in future "
        "conversations, from what kind of fact it is and from what it holds that an answer "
        "could not guess. "
        "Judge the fact itself, as you wrote it: not how often or how recently the user "
        "said it, not how new or surprising it is, not how long it is. Do not invent "
        "a goal of the user that the messages do not show. "
        "Decide in this order and stop at the first that applies. "
        "First, is it 1? It is 1 when " + ONE_OFF + ". "
        "If not, is it part of the user's life? It is when " + PART_OF_LIFE + ". Then it "
        "is 5 if it holds a limit or " + EXACT_VALUE + "; otherwise it is 4. "
        "Otherwise " + TOPICAL + ": it is 2 if it is only " + DETAIL + ", otherwise 3. "
        "Only for new and update, and only when safety is false: leave it out for redundant "
        "and delete, and when safety is true."))


class NeedGainJudgedOperation(MemoryOperation):
    """A new or updated fact that is not about safety, with need and gain, both required."""

    operation: Literal["new", "update"]
    safety: Literal[False] = Field(description=SAFETY_DESCRIPTION)
    need: Literal[1, 2, 3] = Field(description=(
        "NEED: what kind of fact is this? Only the kind: how much the fact helps when it "
        "is used is gain, rated separately. "
        "Judge the fact itself, as you wrote it: not how often or how recently the user "
        "said it, not how new or surprising it is. Do not invent a goal of the user that "
        "the messages do not show. "
        "Decide in this order and stop at the first that applies. "
        "First, is it 1, one-off? It is 1 when " + ONE_OFF + ". "
        "If not, is it 3, part of the user's life? It is 3 when " + PART_OF_LIFE + ". "
        "Otherwise it is 2, topical: " + TOPICAL + "."))
    gain: Literal[1, 2, 3] = Field(description=(
        "GAIN: what does the fact hold that an answer could not guess? Rate what the fact "
        "says, not how often it would come up: that is need, rated separately. "
        "1: " + DETAIL + ". "
        "2: a preference, a habit or a circumstance that makes an answer personal, with no "
        "exact value in it. "
        "3: a limit, something the user cannot do, will not do or must avoid, as is any "
        "fact that says what the user does not or never does; or " + EXACT_VALUE + "."))


class SafetyFactOperation(MemoryOperation):
    """A new or updated fact whose loss could put the user at risk: protected, not rated."""

    operation: Literal["new", "update"]
    safety: Literal[True] = Field(description=SAFETY_DESCRIPTION)


class UnjudgedOperation(MemoryOperation):
    """A redundant or deleted fact: it carries no judgment."""

    operation: Literal["redundant", "delete"]


# The model sees the same tool name in every mode: only the schema changes.
class InsertCoreMemoriesWithUtility(InsertCoreMemories):
    __doc__ = InsertCoreMemories.__doc__ + """

    Every new and every update also carries a judgment of the fact you wrote, given after
    classifying it: safety, true only if forgetting the fact could put the user at risk,
    and, when safety is false, utility, how useful the fact will be in future
    conversations. Redundant and delete carry neither."""

    model_config = ConfigDict(title="InsertCoreMemories")

    memories: List[UtilityJudgedOperation] = Field(
        description=InsertCoreMemories.model_fields["memories"].description)


class InsertCoreMemoriesWithNeedGain(InsertCoreMemories):
    __doc__ = InsertCoreMemories.__doc__ + """

    Every new and every update also carries a judgment of the fact you wrote, given after
    classifying it: safety, true only if forgetting the fact could put the user at risk,
    and, when safety is false, need and gain, rated separately. Redundant and delete carry
    neither."""

    model_config = ConfigDict(title="InsertCoreMemories")

    memories: List[Union[NeedGainJudgedOperation, SafetyFactOperation,
                         UnjudgedOperation]] = Field(
        description=InsertCoreMemories.model_fields["memories"].description)


CONSOLIDATION_TOOLS = {
    "none": InsertCoreMemories,
    "single": InsertCoreMemoriesWithUtility,
    "need_gain": InsertCoreMemoriesWithNeedGain,
}


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


_OPERATION_ALIASES = {
    "new": "new", "create": "new", "add": "new", "insert": "new",
    "redundant": "redundant", "reinforce": "redundant", "duplicate": "redundant",
    "confirm": "redundant",
    "update": "update", "refine": "update", "extend": "update",
    "contradict": "update", "contradiction": "update", "replace": "update",
    "delete": "delete", "remove": "delete", "forget": "delete",
}


def normalize_operation(operation: Any) -> Optional[str]:
    """Map what the LLM produced onto one of the four known operations."""
    if not isinstance(operation, str):
        return None
    return _OPERATION_ALIASES.get(operation.strip().lower())


RATING_SCALES = {"utility": 5, "need": 3, "gain": 3}
# The ratings each mode asks of a new or updated fact that is not about safety.
MODE_RATINGS = {"none": (), "single": ("utility",), "need_gain": ("need", "gain")}


def count_unrated(entries: Iterable[Tuple[str, Optional[Dict[str, Any]]]],
                  mode: str) -> Optional[Tuple[int, int]]:
    """How many new and updated facts lack the ratings of the mode, out of how many.

    `entries` are (op_type, ratings) pairs of the operation log. Safety facts are left
    out, since they are never rated; None when the mode rates nothing.
    """
    names = MODE_RATINGS[mode]
    if not names:
        return None
    judged = [ratings or {} for op_type, ratings in entries
              if op_type in ("create", "update") and not (ratings or {}).get("safety")]
    unrated = sum(1 for ratings in judged if any(ratings.get(name) is None for name in names))
    return unrated, len(judged)


def count_ratings(entries: Iterable[Tuple[str, Optional[Dict[str, Any]]]],
                  mode: str) -> Optional[Dict[str, Dict[int, int]]]:
    """How many new and updated facts got each level of each rating of the mode.

    Every level is there, also the ones never given; None when the mode rates nothing.
    """
    names = MODE_RATINGS[mode]
    if not names:
        return None
    counts = {name: dict.fromkeys(range(1, RATING_SCALES[name] + 1), 0) for name in names}
    for op_type, ratings in entries:
        if op_type in ("create", "update"):
            for name in names:
                level = (ratings or {}).get(name)
                if level in counts[name]:
                    counts[name][level] += 1
    return counts


def describe_ratings(counts: Dict[str, Dict[Any, int]]) -> str:
    """The levels given, as the reports print them: "need 1: 4, 2: 30, 3: 9; gain ..."."""
    return "; ".join(
        f"{name} " + ", ".join(f"{level}: {count}" for level, count
                               in sorted(levels.items(), key=lambda item: int(item[0])))
        for name, levels in counts.items())


def _flag(value: Any) -> Optional[bool]:
    """A boolean as the model wrote it, also as the string true or false."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _rating(value: Any, top: int) -> Optional[int]:
    """An integer from 1 to `top`, also written as a string; anything else is not given."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number.is_integer() or not 1 <= number <= top:
        return None
    return int(number)


def read_ratings(raw_operation: dict) -> Dict[str, Any]:
    """safety, utility, need and gain of an operation, as CoreMemoryItem fields.

    A safety fact is protected whatever its usefulness: its ratings are dropped
    even when the model gave them.
    """
    safety = _flag(raw_operation.get("safety"))
    ratings = {name: None if safety else _rating(raw_operation.get(name), top)
               for name, top in RATING_SCALES.items()}
    return {"safety": safety, **ratings}


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


def memory_line(content: str, item_id: Optional[str] = None,
                updated_at: Union[datetime, str, None] = None) -> str:
    """A memory as the prompts list it: the id, the day it was last written, the content."""
    fields = [] if item_id is None else [f"ID: {item_id}"]
    if updated_at is not None:
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        fields.append(f"Date: {updated_at.date().isoformat()}")
    return ", ".join(fields + [f"Content: {content}"])


def serialize_core_memory_with_dates(items: Iterable[CoreMemoryItem]) -> List[str]:
    """Core memory with the day each fact was last written, for answer_with_dates."""
    return [memory_line(item.content, updated_at=item.updated_at)
            for item in get_active_items(items)]


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

    Chroma only accepts scalars in the metadata, hence the ISO timestamps, and a
    rating that was never given is left out instead of being None.
    """
    metadata = {
        "status": item.status,
        "updated_at": item.updated_at.isoformat(),
        "n_retrieve": item.n_retrieve,
        "n_used": item.n_used,
        "retrieved_at": item.retrieved_at.isoformat(),
    }
    for name in ("safety", *RATING_SCALES):
        if getattr(item, name) is not None:
            metadata[name] = getattr(item, name)
    return metadata


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
    metadata["updated_at"] = backends.now().isoformat()
    backends.get_vector_store()._collection.update(ids=[item_id], metadatas=[metadata])
    return archived


def _search_active(query: str, k: int) -> List[Any]:
    """The k active archival memories closest to a text, as documents with their metadata."""
    try:
        k = max(1, int(k))
    except (TypeError, ValueError):
        k = ARCHIVE_CANDIDATES_K

    try:
        return backends.get_vector_store().similarity_search(
            query, k=k, filter={"status": "active"})
    except Exception as error:
        print(f"\tArchive search failed: {error}")
        return []


def search_archive(query: str, k: int = ARCHIVE_CANDIDATES_K) -> List[Tuple[str, str]]:
    """The k active archival memories closest to a text, as (id, content) pairs."""
    return [(doc.id, doc.page_content) for doc in _search_active(query, k)]


NO_ARCHIVAL_RESULTS = "No relevant active memories found."


def serialize_retrieved_for_response(retrieved: str) -> List[str]:
    """Retrieved archival memories as a list, empty when nothing came back."""
    if not retrieved or retrieved == NO_ARCHIVAL_RESULTS:
        return []
    return [line for line in retrieved.splitlines() if line.strip()]


def count_archived(item_ids: Iterable[str], counter: str, stamp: bool) -> None:
    """`counter` + 1 on archived memories, and retrieved_at now when `stamp`.

    Metadata only, like _rewrite_archive_metadata, so the vector stays where it
    is; updated_at is left alone, it belongs to the writes. An id that is not in
    the archive changes nothing. Never raises: a missed count must not cost the
    user an answer.
    """
    ids = list(dict.fromkeys(item_ids))
    if not ids:
        return
    try:
        store = backends.get_vector_store()
        found = store.get(ids=ids) or {}
        found_ids = found.get("ids") or []
        metadatas = found.get("metadatas") or [{}] * len(found_ids)
        now = backends.now().isoformat()
        updated = []
        for metadata in metadatas:
            metadata = dict(metadata)
            metadata[counter] += 1
            if stamp:
                metadata["retrieved_at"] = now
            updated.append(metadata)
        if found_ids:
            store._collection.update(ids=found_ids, metadatas=updated)
    except Exception as error:
        print(f"\t{counter} not recorded: {error}")


def record_retrievals(item_ids: Iterable[str], stamp: bool = True) -> None:
    """Count a retrieval on archived memories: n_retrieve + 1, retrieved_at now.

    Without `stamp` retrieved_at is left to record_uses: when uses are counted
    (track_used, with generate_answer) it is the time of the last use.
    """
    count_archived(item_ids, "n_retrieve", stamp)


def record_uses(item_ids: Iterable[str], core_memory: Iterable[CoreMemoryItem]) -> None:
    """Count a use on the memories an answer was built on: n_used + 1, retrieved_at now.

    Core items change in place; every other id is looked for in the archive.
    """
    ids = list(dict.fromkeys(item_ids))
    now = backends.now()
    in_core = set()
    for item in get_active_items(core_memory):
        if item.id in ids:
            item.n_used += 1
            item.retrieved_at = now
            in_core.add(item.id)
    count_archived([item_id for item_id in ids if item_id not in in_core], "n_used", stamp=True)


def retrieve_active_archival_memories(query: str, k: int = 5, stamp: bool = True,
                                      dated: bool = False) -> str:
    """Archive lookup for the retrieval path, tombstones excluded.

    The search that counts as a retrieval: the candidates the classifier sees during
    consolidation go through search_archive, recorded only with count_candidate_search.
    With `dated` every memory also carries the day it was last written.
    """
    found = _search_active(query, k)
    if not found:
        return NO_ARCHIVAL_RESULTS
    record_retrievals((doc.id for doc in found), stamp)
    return "\n".join(
        memory_line(doc.page_content, doc.id, doc.metadata.get("updated_at") if dated else None)
        for doc in found)


def build_candidate_memories(
    core_memory: Iterable[CoreMemoryItem],
    query: str,
    k: int = ARCHIVE_CANDIDATES_K,
    count: bool = False,
    stamp: bool = True,
) -> Dict[str, str]:
    """Memories the classifier may point at: the active core ones, plus the archival
    ones that look related to the text being consolidated.

    With `count` the search is recorded as a retrieval, `stamp` as in record_retrievals.
    """
    candidates = serialize_core_memory_with_ids(core_memory)
    found = search_archive(query, k=k)
    if count:
        record_retrievals((doc_id for doc_id, _ in found), stamp)
    for doc_id, content in found:
        if doc_id not in candidates:
            candidates[doc_id] = content
    return candidates


# --------------------------------------------------------------------------- #
# Operations on a core memory item
# --------------------------------------------------------------------------- #

def logged_ratings(item: CoreMemoryItem) -> Optional[Dict[str, Union[bool, int]]]:
    """The judgment of an item as the log shows it, None when there is nothing to show.

    Safety only when true; then the ratings that were given.
    """
    shown = {"safety": True} if item.safety else {}
    shown.update({name: getattr(item, name) for name in RATING_SCALES
                  if getattr(item, name) is not None})
    return shown or None


def create_item(
    content: str,
    log: List[OperationLogEntry],
    ratings: Optional[Dict[str, Any]] = None,
) -> CoreMemoryItem:
    """A fact nothing in memory covers yet, with the ratings from read_ratings."""
    item = CoreMemoryItem(content=content, **(ratings or {}))
    log.append(OperationLogEntry(op_type="create", item_id=item.id, content=content,
                                 ratings=logged_ratings(item)))
    return item


def reinforce_item(
    item: CoreMemoryItem,
    log: List[OperationLogEntry],
    content: Optional[str] = None,
) -> CoreMemoryItem:
    """A fact already stored: refresh it instead of storing it twice."""
    item.updated_at = backends.now()
    log.append(OperationLogEntry(
        op_type="redundant", item_id=item.id, content=content or item.content))
    return item


def _retire_item(item: CoreMemoryItem, status: MemoryStatus) -> None:
    """Take an item out of active duty and keep it in the archive as a tombstone."""
    item.status = status
    item.updated_at = backends.now()
    archive_items([item])


def supersede_item(
    old_item: CoreMemoryItem,
    new_content: str,
    log: List[OperationLogEntry],
    ratings: Optional[Dict[str, Any]] = None,
) -> CoreMemoryItem:
    """Replace a core item with a newer version that points back at it.

    The newer version inherits n_retrieve and n_used: it is the same fact, evolved.
    The ratings are not inherited: an update is judged again (see _target_judgment).
    """
    _retire_item(old_item, "superseded")
    new_item = CoreMemoryItem(content=new_content, n_retrieve=old_item.n_retrieve,
                              n_used=old_item.n_used, **(ratings or {}))
    log.append(OperationLogEntry(
        op_type="update", item_id=new_item.id,
        related_item_id=old_item.id, content=new_content, ratings=logged_ratings(new_item)))
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
    ratings: Optional[Dict[str, Any]] = None,
) -> Optional[CoreMemoryItem]:
    """Flag an archived memory as superseded; the newer version starts in core memory.

    The newer version inherits n_retrieve and n_used and is judged again, as in
    supersede_item.
    """
    archived = _rewrite_archive_metadata(item_id, status="superseded")
    if archived is None:
        return None
    new_item = CoreMemoryItem(content=new_content,
                              n_retrieve=archived["metadata"]["n_retrieve"],
                              n_used=archived["metadata"]["n_used"], **(ratings or {}))
    log.append(OperationLogEntry(
        op_type="update", item_id=new_item.id,
        related_item_id=item_id, content=new_content, ratings=logged_ratings(new_item)))
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

def _comparable(text: str) -> str:
    """A text as the content match reads it: lowercase, without the closing punctuation."""
    return text.strip().lower().rstrip(" .!?;:")


def _same_fact(fact: str, content: str) -> bool:
    """One text contains the other, the best effort match when the target id is unusable."""
    fact, content = _comparable(fact), _comparable(content)
    return bool(fact and content) and (fact in content or content in fact)


def _match_by_content(
    fact: str,
    core_items: Iterable[CoreMemoryItem],
) -> Union[CoreMemoryItem, str, None]:
    """The one memory the fact matches by text, when the classifier gives no usable id.

    Looked for in core memory and among the archived memories closest to the fact: a
    core item, an archived id, or None. Different texts that all match are no match,
    since picking one could change the wrong memory; identical texts are one memory
    stored twice, and the core copy wins. search_archive, because looking for a target
    is not a retrieval.
    """
    if not _comparable(fact):
        return None
    candidates = [(item, item.content) for item in core_items]
    candidates += search_archive(fact, k=CONTENT_MATCH_K)
    matches = [(target, _comparable(content)) for target, content in candidates
               if _same_fact(fact, content)]
    if len({text for _, text in matches}) > 1:
        print(f"\tSeveral memories match {fact!r} by content: none is chosen")
        return None
    return matches[0][0] if matches else None


def _target_judgment(target_item: Optional[CoreMemoryItem],
                     archived_target: Optional[dict]) -> Dict[str, Any]:
    """safety and ratings of an update's target, in the form read_ratings returns."""
    names = ("safety", *RATING_SCALES)
    if target_item is not None:
        return {name: getattr(target_item, name) for name in names}
    return {name: archived_target["metadata"].get(name) for name in names}


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
    counters = {"new": 0, "redundant": 0, "update": 0, "delete": 0}

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
                match = _match_by_content(fact, active_by_id.values())
                if isinstance(match, CoreMemoryItem):
                    target_item, target_id = match, match.id
                    print(f"\t[{operation}] target matched by content: {target_id}")
                elif match is not None:
                    archived_target, target_id = get_archive_item(match), match
                    print(f"\t[{operation}] archived target matched by content: {target_id}")

        if operation != "new" and target_item is None and archived_target is None:
            if operation == "delete":
                print(f"\tNothing to delete for: {fact!r}")
                continue
            # The fact refers to something we do not actually have: keep it anyway.
            print(f"\tNo target for '{operation}', storing {fact!r} as a new memory")
            operation = "new"

        # Only new and update are judged: a redundant keeps the ratings it had.
        ratings = read_ratings(raw_operation)
        if operation == "update" and all(value is None for value in ratings.values()):
            # The model left the judgment out: the target's is the closest one there is.
            inherited = _target_judgment(target_item, archived_target)
            if any(value is not None for value in inherited.values()):
                print(f"\t[update] no judgment given, inherited from {target_id}")
                ratings = inherited

        if operation == "new":
            items.append(create_item(fact, log, ratings))
            counters["new"] += 1

        elif target_item is not None:
            # The target is a live core memory.
            if operation == "redundant":
                reinforce_item(target_item, log, fact)
            elif operation == "delete":
                delete_item(target_item, log)
            else:  # update
                items.append(supersede_item(target_item, fact, log, ratings))
            counters[operation] += 1

        else:
            # The target already lives in the archive and stays there.
            if operation == "redundant":
                reinforce_archived_item(target_id, log, fact)
            elif operation == "delete":
                delete_archived_item(target_id, log)
            else:  # update
                new_item = supersede_archived_item(target_id, fact, log, ratings)
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
