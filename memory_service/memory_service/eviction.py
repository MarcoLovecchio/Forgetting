"""Eviction: what leaves the archive for good.

Consolidation never deletes anything. A superseded or forgotten memory stops
being active and stays in the archive as a tombstone, with the status that says
so: retrieval and classification ignore it because they filter `status: active`
inside the store, but document, vector and text are still there.

This module really removes them, with two criteria.

The first is the status, with no exceptions: everything that is `superseded` or
`deleted` leaves the collection.

The second is the archive limit, archive_memory_limit
(MEMORY_ARCHIVE_LIMIT): a number of active memories that blocks no insertion
and is checked at the end of the turn, after the tombstones. If it is
exceeded, the prune_fraction of the active memories - rounded up - with the
lowest score leaves. If it is still exceeded, another share leaves on the next
turn.

Memories with safety true never leave: they are taken out of the candidates
before any computation, without a score. The share is still computed on all the
active ones, though, because that is what it takes to get back within the limit.

The score
---------
Each term is a separate function, with values in [0, 1], high meaning keep.
eviction_terms computes the terms, by name, and combine_terms combines them.
time_decay and frequency are always there; utility follows
consolidation_utility. A new term is added in eviction_terms and in the
weights, and from there it also reaches the log.

    utility           the classifier's judgment, according to
                      consolidation_utility; with `none` it is not there:
                        utility_term    `single`: the entry of UTILITY_TABLE
                                        for the rating from 1 to 5;
                        need_gain_term  `need_gain`: the cell of
                                        NEED_GAIN_TABLE, row need and column
                                        gain.
                      The two tables are calibrated with
                      test/calibrate_utility.py. A memory without a rating - the
                      model did not give it or it was off the scale - takes the
                      middle one, NEUTRAL_RATINGS.

    need, gain        in place of utility, with `need_gain` and the
                      expected_gain combination only, which keeps the two apart:
                        need_term   the entry of NEED_TABLE, how likely the
                                    memory is to be needed;
                        gain_term   the entry of GAIN_TABLE, what it is worth
                                    then: it multiplies, so its range is narrow.
                      Unrated, they take NEUTRAL_RATINGS too.

    time_decay        the time since a timestamp, with one of the two curves
                      according to eviction_time_decay_function; both are 1 at
                      the moment of the timestamp, share the scale
                      eviction_time_decay_scale_hours (91 days by default) and
                      take the same slot in the mean:
                        weibull_decay_term  exp(-(t/scale)^shape), shape 0.5:
                                            1/e at the scale;
                        power_decay_term    (1 + t/scale)^-d, d 0.5: 0.71 at the
                                            scale, 0.5 at three times it, with
                                            the long tail of the power law.
                      On their own they rank the memories the same way, because
                      they depend on the same timestamp: the choice matters when
                      the decay is averaged with other terms. The timestamp is
                      updated_at or retrieved_at, according to
                      eviction_time_decay_field, and only one is ever used.
                      When uses are counted, retrieved_at is the last use.

    frequency_term    a count relative to the highest among the candidates, the
                      active memories without safety, like LFU and N_visit of
                      MemoryOS, but on a log scale: log(1 + count) /
                      log(1 + highest). On a linear one the few memories
                      retrieved by almost every search push all the others
                      near 0. The count is n_used when uses are counted
                      (track_used, with generate_answer), n_retrieve otherwise:
                      as for the time decay, only one is ever used.
    rate_term         in its place with eviction_frequency_function `rate`: the
                      same count per search run while the memory was in the
                      archive, n_search, with Laplace's rule: (count + 1) /
                      (n_search + 2). A memory never searched is at 0.5 instead
                      of 0, and one searched often and never found falls toward
                      0; it needs no highest among the candidates.

combine_terms follows eviction_combination:

    weighted_sum      0.4 utility + 0.3 time_decay + 0.3 frequency
    product           utility * (0.5 time_decay + 0.5 frequency)
    expected_gain     gain * (0.4 need + 0.3 time_decay + 0.3 frequency): the
                      probability of being needed, from the rating and the
                      history, times the gain, as in Anderson's rational
                      analysis of memory

The weights are rescaled over the terms present: without utility the three
formulas are the same mean of time_decay and frequency. With `single` there is
no gain to multiply by, and expected_gain is the weighted sum.

The memories leave all together: the scores are computed only once, on the
candidates of that moment.

What it touches and what not
----------------------------
Only the archive. The core memory never holds tombstones - it is an invariant:
an item that stops being active is archived at the same moment - and the
operation log stays as it is: the entries naming a removed id stay, with
related_item_id values that from here on point to documents that no longer
exist.

Each removed document adds an entry to the log, with the text of the memory like
all the others: `evict` for a tombstone, `prune` for an active memory, with the
score in `score` and the terms that make it up in `score_terms`. It has a cost,
and it is worth knowing: the text of a fact the user asked to forget gains a copy
in the log, which lives in RAM for the whole life of the node and travels in the
service's response.

When it runs
------------
Only if MemoryConfig.eviction is on: it is changed only in the code, no
environment variable reads it.

Once per insert, in the evict_archive node of the graph, where all the exits of
the insert branch converge: after consolidation, which is where superseded and
deleted are born. So the tombstones of a turn disappear in the same turn, and
there is no concurrency with the next call on the same collection. From there
the exceed_archive_limit router sends to the prune_archive node only if the
active ones exceed the limit, like the other two limits of the branch. The
retrieve branch does not run it: it produces no tombstones and adds no
memories. Nodes and routers read the switches from the runtime context, the
MemoryConfig that run_memory_agent passes to invoke.

The evict and prune entries fall after the offset fixed at the start of the run,
so last_operations() publishes them in the same response as update_memory.

With eviction on
----------------
Tombstones stop staying in the archive, and whatever assumed it adjusts like
this:

    test_long_term_interaction
        every item that left the core memory is in the archive or has an
        evict or prune entry in the log
    test_consolidation
        runs with eviction off: step 5 looks at the tombstone, step 7 checks
        that retrieval excludes it

Not to be confused with the core split, which moves memories from the core
memory to the archive when the core exceeds its budget: here eviction means
leaving the archive.
"""

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from memory_service import backends
from memory_service.config import MemoryConfig
from memory_service.consolidation import OperationLogEntry

EVICTED_STATUSES = ("superseded", "deleted")

DECAY_SHAPE = 0.5
POWER_EXPONENT = 0.5
WEIGHTED_SUM_WEIGHTS = {"utility": 0.4, "time_decay": 0.3, "frequency": 0.3}
PRODUCT_WEIGHTS = {"time_decay": 0.5, "frequency": 0.5}
EXPECTED_GAIN_WEIGHTS = {"need": 0.4, "time_decay": 0.3, "frequency": 0.3}

# Provisional until they are calibrated: rating / 5 and (need / 3) * (gain / 3).
# A calibrated table must stay in (0, 1], never going down as the level goes up.
UTILITY_TABLE = (0.2, 0.4, 0.6, 0.8, 1.0)
# The middle rating of each scale, for a memory the classifier did not rate.
NEUTRAL_RATINGS = {"utility": 3, "need": 2, "gain": 2}
NEED_GAIN_TABLE = (
    # gain 1  gain 2  gain 3
    (1 / 9, 2 / 9, 3 / 9),  # need 1
    (2 / 9, 4 / 9, 6 / 9),  # need 2
    (3 / 9, 6 / 9, 9 / 9),  # need 3
)
# expected_gain: need / 3 and (6 + gain) / 9.
NEED_TABLE = (1 / 3, 2 / 3, 3 / 3)
GAIN_TABLE = (7 / 9, 8 / 9, 9 / 9)


def _find_tombstones(store) -> List[Tuple[str, str]]:
    """(id, text) of the documents to remove, in the order of EVICTED_STATUSES.

    One get per status with an equality filter, instead of a single one with
    `$in`: it costs one more local read and works with any version of chromadb,
    while the operator has not always been there.
    """
    found = []
    for status in EVICTED_STATUSES:
        result = store.get(where={"status": status}) or {}
        ids = result.get("ids") or []
        documents = result.get("documents") or [""] * len(ids)
        found.extend(zip(ids, documents))
    return found


def evict_archived_tombstones(log: List[OperationLogEntry]) -> List[str]:
    """Remove the superseded and deleted memories from the archive.

    Return the removed ids, and for each one add an `evict` entry to the log.

    It never raises: it runs in the last node of the insert branch, inside the
    update_memory callback, and an exception would end up in its except, which
    empties the response even when consolidation succeeded. If the store does
    not answer, it prints and removes nothing.

    The log records only what really happened: the entries are added after the
    deletion succeeded, never before.
    """
    try:
        store = backends.get_vector_store()
        tombstones = _find_tombstones(store)
    except Exception as error:
        print(f"\tEviction skipped, archive lookup failed: {error}")
        return []

    if not tombstones:
        return []

    ids = [doc_id for doc_id, _ in tombstones]
    try:
        store.delete(ids=ids)
    except Exception as error:
        print(f"\tEviction failed, nothing removed: {error}")
        return []

    print(f"\tEvicted from the archive: {ids}")
    for doc_id, content in tombstones:
        log.append(OperationLogEntry(op_type="evict", item_id=doc_id, content=content))
    return ids


def archive_over_limit(limit: int) -> int:
    """By how many active memories the archive exceeds the limit: 0 if it respects it.

    It counts only the active ones, because the tombstones leave anyway by status.
    It removes nothing. It never raises: if the store does not answer it prints
    and returns 0, so a missed count does not make memories leave.
    """
    try:
        result = backends.get_vector_store().get(where={"status": "active"}) or {}
    except Exception as error:
        print(f"\tArchive limit not checked, archive lookup failed: {error}")
        return 0

    active = len(result.get("ids") or [])
    excess = max(0, active - limit)
    if excess:
        print(f"\tArchive over its limit: {active} active memories, limit {limit}")
    return excess


def hours_since(timestamp: str, now: datetime) -> float:
    """Hours elapsed since timestamp, never negative.

    A timestamp in the future counts 0: it really happens, because the timestamps
    are local time and at the end of daylight saving time the clock goes back.
    """
    elapsed = now - datetime.fromisoformat(timestamp)
    return max(0.0, elapsed.total_seconds() / 3600)


def weibull_decay_term(timestamp: str, now: datetime, scale_hours: float,
                       shape: float = DECAY_SHAPE) -> float:
    """Weibull on the time elapsed since timestamp: 1 at the moment, 1/e at scale_hours."""
    return math.exp(-((hours_since(timestamp, now) / scale_hours) ** shape))


def power_decay_term(timestamp: str, now: datetime, scale_hours: float,
                     exponent: float = POWER_EXPONENT) -> float:
    """Power law on the time since timestamp: (1 + t/scale_hours)^-exponent, 1 at the moment."""
    return (1 + hours_since(timestamp, now) / scale_hours) ** -exponent


DECAY_FUNCTIONS = {"weibull": weibull_decay_term, "power": power_decay_term}


def frequency_field(config: MemoryConfig) -> str:
    """The count of the frequency term: n_used when uses are counted."""
    return "n_used" if config.counts_used else "n_retrieve"


def frequency_term(count: int, count_max: int) -> float:
    """The count of a memory relative to the highest one, on a log scale.

    If the highest is 0, none has ever been counted: it is 0 for all of them.
    """
    if count_max == 0:
        return 0.0
    return math.log1p(count) / math.log1p(count_max)


def rate_term(count: int, searches: int) -> float:
    """The count per search with Laplace's rule: (count + 1) / (searches + 2)."""
    # n_used also grows in the core, where nothing is searched: never above 1.
    return (count + 1) / (max(searches, count) + 2)


def utility_term(utility: int) -> float:
    """The entry of UTILITY_TABLE for the classifier's 1-5 rating."""
    return UTILITY_TABLE[utility - 1]


def need_gain_term(need: int, gain: int) -> float:
    """The cell of NEED_GAIN_TABLE: row need, column gain."""
    return NEED_GAIN_TABLE[need - 1][gain - 1]


def need_term(need: int) -> float:
    """The entry of NEED_TABLE for the 1-3 need."""
    return NEED_TABLE[need - 1]


def gain_term(gain: int) -> float:
    """The entry of GAIN_TABLE for the 1-3 gain."""
    return GAIN_TABLE[gain - 1]


def eviction_terms(metadata: dict, config: MemoryConfig, now: datetime,
                   frequency_max: int) -> Dict[str, float]:
    """The terms, by name: they are also the sub-scores the log reports.

    `frequency_max`, the highest count among the candidates on the field of
    frequency_field, depends on the other memories, so whoever has them all
    computes it.
    """
    terms = {}
    if config.consolidation_utility == "single":
        terms["utility"] = utility_term(metadata.get("utility", NEUTRAL_RATINGS["utility"]))
    elif config.consolidation_utility == "need_gain":
        need = metadata.get("need", NEUTRAL_RATINGS["need"])
        gain = metadata.get("gain", NEUTRAL_RATINGS["gain"])
        if config.eviction_combination == "expected_gain":
            terms["need"], terms["gain"] = need_term(need), gain_term(gain)
        else:
            terms["utility"] = need_gain_term(need, gain)
    decay = DECAY_FUNCTIONS[config.eviction_time_decay_function]
    terms["time_decay"] = decay(metadata[config.eviction_time_decay_field], now,
                                config.eviction_time_decay_scale_hours)
    count = metadata[frequency_field(config)]
    if config.eviction_frequency_function == "rate":
        terms["frequency"] = rate_term(count, metadata.get("n_search", 0))
    else:
        terms["frequency"] = frequency_term(count, frequency_max)
    return terms


def weighted_mean(terms: Dict[str, float], weights: Dict[str, float]) -> float:
    """Mean of the terms with their weights, rescaled over the ones present."""
    total = sum(weights[name] for name in terms)
    return sum(weights[name] * value for name, value in terms.items()) / total


def combine_terms(terms: Dict[str, float], combination: str) -> float:
    """The score from the terms, in [0, 1]: the lowest leaves first.

    Without utility the three combinations are the same mean.
    """
    if combination == "product" and "utility" in terms:
        rest = {name: value for name, value in terms.items() if name != "utility"}
        return terms["utility"] * weighted_mean(rest, PRODUCT_WEIGHTS)
    if combination == "expected_gain" and "gain" in terms:
        rest = {name: value for name, value in terms.items() if name != "gain"}
        return terms["gain"] * weighted_mean(rest, EXPECTED_GAIN_WEIGHTS)
    return weighted_mean(terms, WEIGHTED_SUM_WEIGHTS)


def prune_count(active: int, fraction: float) -> int:
    """How many memories leave out of `active`: the fraction, rounded up.
    """
    return math.ceil(active * fraction)


def choose_prune_targets(active: List[Tuple[str, str, dict]], config: MemoryConfig,
                         now: datetime) -> List[Tuple[float, str, str, Dict[str, float]]]:
    """(score, id, content, terms) of what leaves, lowest score first.

    `active` holds (id, content, metadata) of every active memory, the protected ones
    included: they count in the share but are never candidates. It removes nothing:
    test/eviction_replay.py calls it too.
    """
    candidates = [(doc_id, content, metadata) for doc_id, content, metadata in active
                  if not metadata.get("safety")]
    frequency_max = max((metadata[frequency_field(config)] for _, _, metadata in candidates),
                        default=0)
    scored = []
    for doc_id, content, metadata in candidates:
        terms = eviction_terms(metadata, config, now, frequency_max)
        scored.append((combine_terms(terms, config.eviction_combination), doc_id, content, terms))
    leaving = prune_count(len(active), config.prune_fraction)
    return sorted(scored, key=lambda target: target[:2])[:leaving]


def prune_archive(log: List[OperationLogEntry], config: MemoryConfig,
                  now: Optional[datetime] = None) -> List[str]:
    """Remove the active memories with the lowest score.

    It removes prune_count of the active ones, all together, choosing them among
    those without safety: the protected ones stay out before any computation,
    even from the highest count of the frequency. Return the removed ids, and
    for each one add a `prune` entry to the log with the score and the terms.
    Like evict_archived_tombstones it does not raise if the store fails, and the
    log records only the successful removals.

    It does not look at the limit: whether the archive exceeds it is decided by
    the graph, with archive_over_limit, before getting here. The `status: active`
    filter is the same set on which that one counts the memories.
    """
    try:
        store = backends.get_vector_store()
        result = store.get(where={"status": "active"}) or {}
    except Exception as error:
        print(f"\tPruning skipped, archive lookup failed: {error}")
        return []

    ids = result.get("ids") or []
    documents = result.get("documents") or [""] * len(ids)
    metadatas = result.get("metadatas") or [{}] * len(ids)
    targets = choose_prune_targets(list(zip(ids, documents, metadatas)), config,
                                   now or backends.now())
    if not targets:
        return []

    try:
        store.delete(ids=[doc_id for _, doc_id, _, _ in targets])
    except Exception as error:
        print(f"\tPruning failed, nothing removed: {error}")
        return []

    print(f"\tPruned from the archive: "
          f"{[(doc_id, round(score, 3)) for score, doc_id, _, _ in targets]}")
    for score, doc_id, content, terms in targets:
        log.append(OperationLogEntry(op_type="prune", item_id=doc_id, content=content,
                                     score=score, score_terms=terms))
    return [doc_id for _, doc_id, _, _ in targets]
