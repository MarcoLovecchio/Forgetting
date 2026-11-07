#!/usr/bin/env python3
"""The eviction rules replayed offline on the memories of one HaluMem trace.

A trace is a run of halumem_adapter.py whose archive never reaches its limit
(--archive-limit 100000): what the model wrote, insert by insert, and what every
question found. The replay walks it again with a limit and a rule, with no model, and
counts how many of the memories each question needs would still be there. Every rule
sees the same memories, so a difference between two rules comes from the rules only.

    embed     the vectors of every memory text and question of the trace, with the
              service's embedding model: the search of a smaller archive is computed
              again from them
    label     which memories each question needs, asked to HaluMem's judge (eval/.env)
              among the core, the LABEL_CANDIDATES archived memories closest to the
              question and the EVIDENCE_CANDIDATES closest to each sentence of its
              evidence; it prints where the chosen ones were in the question's ranking,
              to tell whether LABEL_CANDIDATES is enough
    simulate  every rule at every limit of LIMITS, and the indicators

What the replay does with the trace:

    create          a memory in the core
    update          the old one leaves, the new one enters the core with its counts
    redundant       updated_at of the target is now; it stays where it is
    delete          the target leaves
    end of insert   choose_core_split with the trace's core limit, then, over the archive
                    limit, choose_prune_targets or the reference rule
    question        the QUESTION_K archived memories closest to it get n_retrieve + 1

The archive, evict and prune entries of the trace are left out: the replay decides
them. When an operation targets a memory the rule already removed, the real system
would have gone another way. Each case is counted, and handled as the system would: an
update keeps its new memory, which starts from zero counts, a redundant stores its fact
again as a new memory, a delete finds nothing.

The rules: the service's score, with utility none or the trace's, every combination and
PRUNE_FRACTIONS; FIFO, LRU, LFU, utility alone and random (RANDOM_SEEDS) as references,
with the trace's prune fraction; the oracle, which knows the questions to come and
removes first what no question will need, then what is needed latest.

The indicators, summed over the users:

    retention       needed memories still in core or archive when their question comes
    retrievable     needed memories in core or among the QUESTION_K the search returns
    of oracle       (retention - random) / (oracle - random), at the same limit
    by type, by age the retention per question type and per sessions since the memory
                    was written
    divergence      updates and redundants on a memory the rule had removed
    prunes          per session, and how many of them a later question needed
    rating AUC      how often a memory a later session needs has a higher need x gain
                    than one no later session needs, on the trace alone: a question of
                    the same session is kept by recency anyway, as in calibrate_utility.py

Before the rules, simulate checks the replay against the trace: with no limit its core
split must archive what the trace archived, and its search must find what the
questions found.

The trace must keep count_candidate_search off, as halumem_adapter.py does: the
candidate searches of consolidation are not in the log, so their counts could not be
replayed.

    python3 memory_service/test/eviction_replay.py embed --halumem PATH --version NAME
    python3 memory_service/test/eviction_replay.py label --halumem PATH --version NAME
    python3 memory_service/test/eviction_replay.py simulate --halumem PATH --version NAME

Everything lands in PATH/eval/results/memory_service-NAME/replay/.
"""

import argparse
import ast
import bisect
import hashlib
import json
import multiprocessing
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import halumem_adapter as adapter  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    CoreMemoryItem,
    archive_metadata,
    choose_core_split,
)
from memory_service.eviction import (  # noqa: E402
    NEUTRAL_RATINGS,
    choose_prune_targets,
    need_gain_term,
    prune_count,
)

LIMITS = (25, 50, 100, 150, 200)
COMBINATIONS = ("weighted_sum", "product", "expected_gain")
PRUNE_FRACTIONS = (0.05, 0.10, 0.20)
RANDOM_SEEDS = 20
QUESTION_K = adapter.QUESTION_K
LABEL_CANDIDATES = 40
EVIDENCE_CANDIDATES = 10
RANK_BUCKETS = ((0, 0, "core"), (1, 20, "1-20"), (21, 30, "21-30"), (31, 40, "31-40"),
                (41, None, ">40"))
AGE_BUCKETS = ((0, 0, "0"), (1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, None, ">20"))
BOOTSTRAP_SAMPLES = 1000
EMBED_BATCH = 64
FOLDER = "replay"
RATING_FIELDS = ("safety", "utility", "need", "gain")
CONTEXT_ID = re.compile(r"ID: (\w+),")


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text)


# --------------------------------------------------------------------------- #
# The trace
# --------------------------------------------------------------------------- #

@dataclass
class Insert:
    session: int
    time: datetime
    operations: List[Dict[str, Any]]
    index: int = 0


@dataclass
class Question:
    session: int
    key: str
    time: datetime
    record: Dict[str, Any]


def question_key(session: int, index: int) -> str:
    return f"{session}:{index}"


def trace_events(result: Dict[str, Any], user: Dict[str, Any]) -> List[Any]:
    """Inserts and questions of a replayed user, in the order the run met them.

    Every insert of the session is there, also the ones that consolidated nothing: the
    archive limit is checked at the end of each. A run before the `insert` field merged
    the flush into the last exchange.
    """
    events: List[Any] = []
    for index, (replayed, session) in enumerate(zip(result["sessions"], user["sessions"])):
        dialogue = session["dialogue"]
        times = [adapter.dataset_time(said["timestamp"]) for said in dialogue[::2]]
        times.append(adapter.dataset_time(session["end_time"]))
        grouped = defaultdict(list)
        for operation in replayed["operations"]:
            grouped[operation.get("insert", operation["exchange"])].append(operation)
        for insert_index, time in enumerate(times):
            events.append(Insert(index, time, grouped.get(insert_index, []), insert_index))
        for number, question in enumerate(replayed.get("questions", [])):
            events.append(Question(index, question_key(index, number), times[-1], question))
    return events


def trace_config(run_dir: str) -> MemoryConfig:
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as handle:
        return MemoryConfig(**json.load(handle)["config"])


def trace_results(run_dir: str) -> List[Dict[str, Any]]:
    return adapter.load_users(os.path.join(run_dir, f"{adapter.FRAME}_eval_results.jsonl"))


def replay_folder(run_dir: str, *parts: str) -> str:
    folder = os.path.join(run_dir, FOLDER, *parts)
    os.makedirs(folder, exist_ok=True)
    return folder


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #

class Rule:
    """What leaves the archive when it is over the limit."""

    name = ""

    def choose(self, replay: "Replay", now: datetime) -> List[str]:
        raise NotImplementedError


class ScoreRule(Rule):
    """The service's own choice, with the trace's configuration changed by `changes`."""

    def __init__(self, config: MemoryConfig, **changes: Any):
        self.config = replace(config, **changes)
        self.name = (f"score {self.config.consolidation_utility} "
                     f"{self.config.eviction_combination} {self.config.prune_fraction:.2f}")

    def choose(self, replay: "Replay", now: datetime) -> List[str]:
        active = [(item_id, replay.items[item_id].content,
                   archive_metadata(replay.items[item_id])) for item_id in sorted(replay.archive)]
        return [item_id for _, item_id, _, _ in choose_prune_targets(active, self.config, now)]


class KeyRule(Rule):
    """The lowest keys leave, as many as the score would remove; safety never."""

    def __init__(self, name: str, key: Callable[["Replay", CoreMemoryItem, datetime], Any],
                 fraction: float):
        self.name, self.key, self.fraction = name, key, fraction

    def choose(self, replay: "Replay", now: datetime) -> List[str]:
        candidates = [replay.items[item_id] for item_id in replay.archive
                      if not replay.items[item_id].safety]
        ranked = sorted(candidates, key=lambda item: (self.key(replay, item, now), item.id))
        return [item.id for item in ranked[:prune_count(len(replay.archive), self.fraction)]]


def utility_value(item: CoreMemoryItem) -> float:
    return need_gain_term(item.need or NEUTRAL_RATINGS["need"],
                          item.gain or NEUTRAL_RATINGS["gain"])


def oracle_key(replay: "Replay", item: CoreMemoryItem, now: datetime) -> Tuple[int, float]:
    """Never needed again first, then the one needed latest."""
    upcoming = replay.next_need(item.id, now)
    if upcoming is None:
        return 0, item.updated_at.timestamp()
    return 1, -upcoming.timestamp()


def reference_rules(fraction: float, with_ratings: bool) -> List[Rule]:
    rules = [
        KeyRule("fifo", lambda replay, item, now: replay.created[item.id], fraction),
        KeyRule("lru", lambda replay, item, now: item.updated_at, fraction),
        KeyRule("lfu", lambda replay, item, now: (item.n_retrieve, item.updated_at), fraction),
    ]
    if with_ratings:
        rules.append(KeyRule("utility", lambda replay, item, now:
                             (utility_value(item), item.updated_at), fraction))
    rules.append(KeyRule("oracle", oracle_key, fraction))
    return rules


def random_rule(seed: int, fraction: float) -> Rule:
    generator = random.Random(seed)
    return KeyRule("random", lambda replay, item, now: generator.random(), fraction)


def score_rules(config: MemoryConfig) -> List[Rule]:
    """Utility none or the trace's, every combination and fraction, the trace's included."""
    utilities = dict.fromkeys(("none", config.consolidation_utility))
    combinations = dict.fromkeys(COMBINATIONS + (config.eviction_combination,))
    fractions = dict.fromkeys(PRUNE_FRACTIONS + (config.prune_fraction,))
    return [ScoreRule(config, consolidation_utility=utility, eviction_combination=combination,
                      prune_fraction=fraction)
            for utility in utilities for combination in combinations for fraction in fractions]


# --------------------------------------------------------------------------- #
# The replay
# --------------------------------------------------------------------------- #

def ratings_of(item: CoreMemoryItem) -> Dict[str, Any]:
    return {name: getattr(item, name) for name in RATING_FIELDS
            if getattr(item, name) is not None}


@dataclass
class Asked:
    """What one question found in the replay."""

    key: str
    question_type: str
    needed: List[Tuple[int, bool, bool]] = field(default_factory=list)  # age, kept, found


class Replay:
    """One rule at one limit on one trace.

    `rank(question_key, [(id, content)])` orders archived memories by closeness to the
    question; `needs` maps a question key to the trace ids it needs.
    """

    def __init__(self, config: MemoryConfig, limit: Optional[int], rule: Optional[Rule],
                 rank: Callable[[str, List[Tuple[str, str]]], List[str]],
                 needs: Optional[Dict[str, List[str]]] = None,
                 on_question: Optional[Callable[["Replay", Question, List[str]], None]] = None):
        self.config, self.limit, self.rule = config, limit, rule
        self.rank, self.needs, self.on_question = rank, needs or {}, on_question
        self.items: Dict[str, CoreMemoryItem] = {}
        self.core: set = set()
        self.archive: set = set()
        self.created: Dict[str, datetime] = {}
        self.born: Dict[str, int] = {}
        self.removed: Dict[str, CoreMemoryItem] = {}
        self.alias: Dict[str, str] = {}
        self.origin: Dict[str, str] = {}
        self.counts: Counter = Counter()
        self.asked: List[Asked] = []
        self.split_archived: List[Tuple[int, int, str]] = []  # session, insert, id
        self.found: Dict[str, List[str]] = {}
        self.need_times: Dict[str, List[datetime]] = {}
        self.session = 0

    def run(self, events: Sequence[Any]) -> "Replay":
        times = {event.key: event.time for event in events if isinstance(event, Question)}
        for key, needed in self.needs.items():
            for item_id in needed if key in times else ():
                self.need_times.setdefault(item_id, []).append(times[key])
        for moments in self.need_times.values():
            moments.sort()
        for event in events:
            self.session = event.session
            if isinstance(event, Insert):
                self.insert(event)
            else:
                self.question(event)
        return self

    def resolve(self, item_id: str) -> str:
        return self.alias.get(item_id, item_id)

    def next_need(self, item_id: str, now: datetime) -> Optional[datetime]:
        moments = self.need_times.get(self.origin.get(item_id, item_id), [])
        position = bisect.bisect_right(moments, now)
        return moments[position] if position < len(moments) else None

    def _add(self, item_id: str, content: str, time: datetime, ratings: Optional[Dict[str, Any]],
             counts: Tuple[int, int] = (0, 0)) -> None:
        fields = {name: value for name, value in (ratings or {}).items() if name in RATING_FIELDS}
        self.items[item_id] = CoreMemoryItem(id=item_id, content=content, updated_at=time,
                                             retrieved_at=time, n_retrieve=counts[0],
                                             n_used=counts[1], **fields)
        self.core.add(item_id)
        self.created[item_id] = time
        self.born.setdefault(item_id, self.session)

    def _drop(self, item_id: str) -> CoreMemoryItem:
        self.core.discard(item_id)
        self.archive.discard(item_id)
        return self.items.pop(item_id)

    def insert(self, event: Insert) -> None:
        for operation in event.operations:
            self.apply(operation)
        now = max([event.time] + [parse_time(operation["timestamp"])
                                  for operation in event.operations])
        core = [self.items[item_id] for item_id in self.core]
        for item in choose_core_split(core, self.config.core_memory_limit)[1]:
            self.core.discard(item.id)
            self.archive.add(item.id)
            self.split_archived.append((event.session, event.index, item.id))
        if self.limit is not None and len(self.archive) > self.limit:
            for item_id in self.rule.choose(self, now):
                self.removed[item_id] = self._drop(item_id)
                self.counts["prunes"] += 1
                if self.next_need(item_id, now) is not None:
                    self.counts["prunes needed later"] += 1

    def apply(self, operation: Dict[str, Any]) -> None:
        kind = operation["op_type"]
        time = parse_time(operation["timestamp"])
        if kind == "create":
            self._add(operation["item_id"], operation["content"], time, operation.get("ratings"))
        elif kind == "update":
            self._update(operation, time)
        elif kind == "redundant":
            self._redundant(operation, time)
        elif kind == "delete":
            self.counts["deletes"] += 1
            target = self.resolve(operation["item_id"])
            if target in self.items:
                self._drop(target)
            elif target in self.removed:
                self.counts["deletes on removed"] += 1

    def _update(self, operation: Dict[str, Any], time: datetime) -> None:
        self.counts["updates"] += 1
        old = self.resolve(operation["related_item_id"])
        new = operation["item_id"]
        if old not in self.items and old in self.removed:
            self.counts["updates on removed"] += 1
        if new in self.items:
            # A twin of this insert merged into it: the higher counts stay.
            merged = self.items[new]
            if old in self.items:
                gone = self._drop(old)
                merged.n_retrieve = max(merged.n_retrieve, gone.n_retrieve)
                merged.n_used = max(merged.n_used, gone.n_used)
            for name, value in (operation.get("ratings") or {}).items():
                if name in RATING_FIELDS:
                    setattr(merged, name, value)
            return
        counts = (0, 0)
        if old in self.items:
            gone = self._drop(old)
            counts = (gone.n_retrieve, gone.n_used)
        self._add(new, operation["content"], time, operation.get("ratings"), counts)

    def _redundant(self, operation: Dict[str, Any], time: datetime) -> None:
        self.counts["redundants"] += 1
        written = operation["item_id"]
        target = self.resolve(written)
        if target in self.items:
            self.items[target].updated_at = time
        elif target in self.removed:
            self.counts["redundants on removed"] += 1
            fresh = f"{written}+{self.counts['redundants on removed']}"
            self._add(fresh, operation["content"], time, ratings_of(self.removed[target]))
            self.alias[written] = fresh
            self.origin[fresh] = self.origin.get(target, target)

    def question(self, event: Question) -> None:
        ranked = self.rank(event.key, [(item_id, self.items[item_id].content)
                                       for item_id in self.archive])
        found = ranked[:QUESTION_K]
        for item_id in found:
            self.items[item_id].n_retrieve += 1
            self.items[item_id].retrieved_at = event.time
        self.found[event.key] = found
        self.counts["questions"] += 1
        self.counts["archive at questions"] += len(self.archive)
        if self.on_question:
            self.on_question(self, event, ranked)
        needed = self.needs.get(event.key)
        if not needed:
            return
        asked = Asked(event.key, event.record.get("question_type", ""))
        for item_id in needed:
            current = self.resolve(item_id)
            kept = current in self.items
            asked.needed.append((event.session - self.born.get(item_id, event.session), kept,
                                 current in self.core or current in found))
        self.asked.append(asked)


# --------------------------------------------------------------------------- #
# The indicators
# --------------------------------------------------------------------------- #

def age_bucket(age: int) -> str:
    for low, high, name in AGE_BUCKETS:
        if age >= low and (high is None or age <= high):
            return name
    return AGE_BUCKETS[-1][2]


def tally(replay: Replay) -> Counter:
    """What a replay counted, as sums that add up over users and seeds."""
    counts = Counter(replay.counts)
    counts["sessions"] = replay.session + 1
    for asked in replay.asked:
        counts["labelled questions"] += 1
        counts["complete questions"] += all(kept for _, kept, _ in asked.needed)
        for age, kept, found in asked.needed:
            for prefix in ("", f"type {asked.question_type} ", f"age {age_bucket(age)} "):
                counts[prefix + "needed"] += 1
                counts[prefix + "kept"] += kept
                counts[prefix + "found"] += found
    return counts


def ratio(part: float, whole: float) -> Optional[float]:
    return part / whole if whole else None


def bootstrap_interval(per_user: Sequence[Counter], part: str, whole: str,
                       samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
                       ) -> Optional[Tuple[float, float]]:
    """The 95% interval of sum(part) / sum(whole) over users drawn with replacement."""
    if len(per_user) < 2:
        return None
    generator = random.Random(seed)
    values = []
    for _ in range(samples):
        drawn = [generator.choice(per_user) for _ in per_user]
        value = ratio(sum(counts[part] for counts in drawn),
                      sum(counts[whole] for counts in drawn))
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def auc(scored: Sequence[Tuple[float, bool]]) -> Optional[float]:
    """How often a positive scores above a negative, ties counting half."""
    positives = sum(1 for _, label in scored if label)
    negatives = len(scored) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(scored, key=lambda pair: pair[0])
    rank_sum, position = 0.0, 0
    while position < len(ordered):
        end = position
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        mean_rank = (position + 1 + end) / 2
        rank_sum += mean_rank * sum(1 for _, label in ordered[position:end] if label)
        position = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def rated_memories(result: Dict[str, Any], needs: Dict[str, List[str]]
                   ) -> List[Tuple[float, bool, str]]:
    """(need x gain, needed by a later session, level) of every rated memory but safety.

    A twin merged into a memory logs it twice: the last entry holds the merged ratings.
    """
    needed_in = defaultdict(set)
    for key, needed in needs.items():
        for item_id in needed:
            needed_in[item_id].add(int(key.split(":")[0]))
    written = {}
    for session_index, session in enumerate(result["sessions"]):
        for operation in session["operations"]:
            if operation["op_type"] in ("create", "update"):
                born = written.get(operation["item_id"], (session_index,))[0]
                written[operation["item_id"]] = (born, operation.get("ratings") or {})
    rated = []
    for item_id, (born, ratings) in written.items():
        if ratings.get("safety") or ratings.get("need") is None or ratings.get("gain") is None:
            continue
        pair = (ratings["need"], ratings["gain"])
        level = "2/2" if pair == (2, 2) else ("high" if pair[0] * pair[1] > 4 else "low")
        needed = any(session > born for session in needed_in[item_id])
        rated.append((need_gain_term(*pair), needed, level))
    return rated


# --------------------------------------------------------------------------- #
# The vectors
# --------------------------------------------------------------------------- #

def memory_texts(result: Dict[str, Any]) -> List[str]:
    texts = (operation["content"] for session in result["sessions"]
             for operation in session["operations"]
             if operation["op_type"] in ("create", "update", "redundant") and operation["content"])
    return list(dict.fromkeys(texts))


def question_texts(result: Dict[str, Any]) -> Dict[str, str]:
    return {question_key(index, number): question["asked"]
            for index, session in enumerate(result["sessions"])
            for number, question in enumerate(session.get("questions", []))}


def vectors_path(run_dir: str, uuid: str) -> str:
    return os.path.join(replay_folder(run_dir, "embeddings"), f"{uuid}.npz")


def evidence_texts(result: Dict[str, Any], name: str) -> List[str]:
    """Every evidence sentence of the questions, with the person as "the user"."""
    texts = (adapter.neutral(text, name) for session in result["sessions"]
             for question in session.get("questions", []) for text in evidence_of(question))
    return list(dict.fromkeys(texts))


class Closeness:
    """The cosine of every query with every memory text of one user, computed once.

    The queries are the questions, by question key, and the evidence sentences, by
    text_key.
    """

    def __init__(self, memories: Dict[str, Any], queries: Dict[str, Any]):
        import numpy

        keys = list(memories)
        matrix = numpy.array([memories[key] for key in keys], dtype="float32")
        matrix /= numpy.linalg.norm(matrix, axis=1, keepdims=True)
        self.column = {key: index for index, key in enumerate(keys)}
        self.row = {key: index for index, key in enumerate(queries)}
        asked = numpy.array([queries[key] for key in queries], dtype="float32")
        asked /= numpy.linalg.norm(asked, axis=1, keepdims=True)
        self.cosine = asked @ matrix.T

    def rank(self, key: str, memories: List[Tuple[str, str]]) -> List[str]:
        if key not in self.row:
            raise SystemExit(f"No vector for the query {key}: launch embed again.")
        row = self.cosine[self.row[key]]
        scored = []
        for item_id, content in memories:
            column = self.column.get(text_key(content))
            if column is None:
                raise SystemExit(f"No vector for {content!r}: launch embed again.")
            scored.append((-float(row[column]), item_id))
        return [item_id for _, item_id in sorted(scored)]


def load_closeness(run_dir: str, result: Dict[str, Any]) -> Closeness:
    import numpy

    path = vectors_path(run_dir, result["uuid"])
    if not os.path.exists(path):
        raise SystemExit(f"No vectors for {result['uuid']}: launch embed first.")
    stored = numpy.load(path)
    memories = dict(zip(stored["memory_keys"], stored["memory_vectors"]))
    questions = dict(zip(stored["question_keys"], stored["question_vectors"]))
    missing = [key for key in question_texts(result) if key not in questions]
    if missing:
        raise SystemExit(f"{len(missing)} questions of {result['uuid']} without a vector: "
                         "launch embed again.")
    queries = {key: questions[key] for key in question_texts(result)}
    if "evidence_keys" in stored.files:
        queries.update(zip(stored["evidence_keys"], stored["evidence_vectors"]))
    return Closeness(memories, queries)


def embed(arguments: argparse.Namespace) -> int:
    """The vectors of the memories, questions and evidence of every user, with the service's
    model: the memories as the archive embeds them, questions and evidence as a search."""
    import numpy

    from memory_service import backends

    run_dir = adapter.run_folder(arguments)
    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    names = {user["uuid"]: adapter.user_name(user["persona_info"])
             for user in adapter.evaluated_users(root)}
    config = MemoryConfig.from_environment()
    embeddings = backends._build_embeddings(config)

    def queries(texts: Iterable[str]) -> List[List[float]]:
        return [adapter.with_retries(lambda: embeddings.embed_query(text)) for text in texts]

    for result in trace_results(run_dir)[:arguments.users]:
        path = vectors_path(run_dir, result["uuid"])
        stored = {}
        if os.path.exists(path):
            with numpy.load(path) as loaded:
                stored = {name: loaded[name] for name in loaded.files}
        if "evidence_keys" in stored:
            print(f"{result['uuid']}: already embedded")
            continue
        if "memory_keys" not in stored:
            texts = memory_texts(result)
            vectors = []
            for start in range(0, len(texts), EMBED_BATCH):
                vectors += adapter.with_retries(
                    lambda: embeddings.embed_documents(texts[start:start + EMBED_BATCH]))
            asked = question_texts(result)
            stored.update(memory_keys=numpy.array([text_key(text) for text in texts]),
                          memory_vectors=numpy.array(vectors, dtype="float32"),
                          question_keys=numpy.array(list(asked)),
                          question_vectors=numpy.array(queries(asked.values()), dtype="float32"))
        evidence = evidence_texts(result, names[result["uuid"]])
        dimensions = stored["memory_vectors"].shape[1]
        stored.update(evidence_keys=numpy.array([text_key(text) for text in evidence]),
                      evidence_vectors=numpy.array(queries(evidence), dtype="float32").reshape(
                          len(evidence), dimensions))
        numpy.savez_compressed(path, **stored)
        print(f"{result['uuid']}: {len(stored['memory_keys'])} memory texts, "
              f"{len(stored['question_keys'])} questions, {len(evidence)} evidence sentences",
              flush=True)
    return 0


# --------------------------------------------------------------------------- #
# The labels
# --------------------------------------------------------------------------- #

JUDGE_PROMPT = """You check which stored memories a question needs.

Question: {question}
Correct answer: {answer}
What the correct answer is based on:
{evidence}

Memories stored about {name}:
{memories}

Which memories hold information the correct answer depends on? A memory counts if it
states at least part of that information, even in other words or together with other
facts. A memory that is only on the same topic, or that contradicts the correct answer,
does not count.

Reply with a JSON block and nothing else:
```json
{{"needed": [the numbers of the memories, or an empty list]}}
```"""


def evidence_of(question: Dict[str, Any]) -> List[str]:
    evidence = question.get("evidence") or []
    if isinstance(evidence, str):
        try:
            evidence = ast.literal_eval(evidence)
        except (ValueError, SyntaxError):
            return []
    return [entry["memory_content"] for entry in evidence
            if isinstance(entry, dict) and entry.get("memory_content")]


def judge_candidates(replay: Replay, ranked: List[str], by_evidence: Sequence[List[str]],
                     shown: int = LABEL_CANDIDATES) -> List[Tuple[str, str, int]]:
    """(id, content, place) of the core, the `shown` archived memories closest to the
    question and those closest to each evidence sentence, once each.

    The place is the one in the question's ranking, 0 for the core.
    """
    place = {item_id: number for number, item_id in enumerate(ranked, 1)}
    chosen = sorted(replay.core, key=lambda item_id: replay.items[item_id].updated_at)
    chosen += ranked[:shown]
    for closest in by_evidence:
        chosen += closest
    return [(item_id, replay.items[item_id].content, place.get(item_id, 0))
            for item_id in dict.fromkeys(chosen)]


def rank_bucket(place: int) -> str:
    for low, high, name in RANK_BUCKETS:
        if place >= low and (high is None or place <= high):
            return name
    return RANK_BUCKETS[-1][2]


def places_text(labels: Iterable[Dict[str, Any]]) -> str:
    counts = Counter(rank_bucket(place) for outcome in labels
                     for place in outcome.get("places", []))
    return ", ".join(f"{name} {counts[name]}" for _, _, name in RANK_BUCKETS)


def judge_prompt(question: Dict[str, Any], candidates: List[Tuple[str, str, int]],
                 name: str) -> str:
    evidence = "\n".join(f"- {text}" for text in evidence_of(question))
    memories = "\n".join(f"[{number}] {adapter.personalized(content, name)}"
                         for number, (_, content, _) in enumerate(candidates, 1))
    return JUDGE_PROMPT.format(question=question["question"], answer=question["answer"],
                               evidence=evidence, name=name, memories=memories)


def needed_ids(reply: Any, candidates: Sequence[Tuple[Any, ...]]) -> List[str]:
    """The ids behind the numbers the judge gave; numbers out of the list are ignored."""
    numbers = reply.get("needed") if isinstance(reply, dict) else None
    if not isinstance(numbers, list):
        raise ValueError(f"no list of numbers in {reply!r}")
    chosen = []
    for number in numbers:
        if isinstance(number, str) and number.strip().isdigit():
            number = int(number)
        if isinstance(number, int) and 1 <= number <= len(candidates):
            chosen.append(candidates[number - 1][0])
    return list(dict.fromkeys(chosen))


def labels_path(run_dir: str, uuid: str) -> str:
    return os.path.join(replay_folder(run_dir, "labels"), f"{uuid}.json")


def load_labels(run_dir: str, uuid: str) -> Dict[str, Any]:
    path = labels_path(run_dir, uuid)
    if not os.path.exists(path):
        return {"labels": {}}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, content: Any) -> None:
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=1)
    os.replace(partial, path)


def label(arguments: argparse.Namespace) -> int:
    """Which memories each question needs, by HaluMem's judge, with a cache per user."""
    run_dir = adapter.run_folder(arguments)
    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    config = trace_config(run_dir)
    users = {user["uuid"]: user for user in adapter.evaluated_users(root)}
    results = trace_results(run_dir)[:arguments.users]

    # From HaluMem's folder, as evaluate does: llms.py reads eval/.env from there.
    eval_dir = os.path.join(root, "eval")
    os.chdir(eval_dir)
    sys.path.insert(0, eval_dir)
    import llms

    everyone: List[Dict[str, Any]] = []
    for result in results:
        user = users[result["uuid"]]
        name = adapter.user_name(user["persona_info"])
        stored = load_labels(run_dir, result["uuid"])
        stored["model"] = os.getenv("OPENAI_MODEL")
        labels = stored["labels"]
        pending = []

        def collect(replay: Replay, event: Question, ranked: List[str]) -> None:
            if "needed" in labels.get(event.key, {}):
                return
            evidence = evidence_of(event.record)
            if not evidence:
                labels[event.key] = {"needed": [], "skipped": "no evidence"}
                return
            archive = [(item_id, replay.items[item_id].content) for item_id in replay.archive]
            by_evidence = [closeness.rank(text_key(adapter.neutral(text, name)),
                                          archive)[:EVIDENCE_CANDIDATES] for text in evidence]
            pending.append((event.key, event.record,
                            judge_candidates(replay, ranked, by_evidence)))

        closeness = load_closeness(run_dir, result)
        unlimited = Replay(config, None, None, closeness.rank, on_question=collect).run(
            trace_events(result, user))
        checks = trace_checks(unlimited, result)
        print(f"{result['uuid']}: split agreement {percent(checks['split agreement'])}%, "
              f"search agreement {percent(checks['search agreement'])}%", flush=True)

        def ask(job: Tuple[str, Dict[str, Any], List[Tuple[str, str, int]]]
                ) -> Tuple[str, Dict[str, Any]]:
            key, question, candidates = job
            try:
                reply = llms.llm_request_for_json(judge_prompt(question, candidates, name))
                needed = needed_ids(reply, candidates)
            except Exception as error:
                return key, {"error": f"{type(error).__name__}: {error}"}
            place = {item_id: number for item_id, _, number in candidates}
            return key, {"needed": needed, "places": [place[item_id] for item_id in needed],
                         "shown": len(candidates)}

        with ThreadPoolExecutor(max_workers=arguments.judge_workers) as pool:
            for key, outcome in pool.map(ask, pending):
                labels[key] = outcome
        save_json(labels_path(run_dir, result["uuid"]), stored)
        failed = sum(1 for outcome in labels.values() if "error" in outcome)
        print(f"{result['uuid']}: {len(pending)} questions judged, {failed} failed "
              "(launch label again to retry them); the needed memories by place in the "
              f"question's ranking: {places_text(labels.values())}", flush=True)
        everyone.extend(labels.values())
    print(f"In all, the needed memories by place in the question's ranking: "
          f"{places_text(everyone)}. Many beyond {LABEL_CANDIDATES} mean the question's "
          "ranking alone would miss them.")
    return 0


def needs_of(stored: Dict[str, Any]) -> Dict[str, List[str]]:
    return {key: outcome["needed"] for key, outcome in stored["labels"].items()
            if outcome.get("needed")}


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #

def agreement(first: Iterable[Any], second: Iterable[Any]) -> Optional[float]:
    first, second = set(first), set(second)
    union = first | second
    return len(first & second) / len(union) if union else None


def trace_checks(replay: Replay, result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """The replay with no limit against the trace: what its split archived at each insert,
    and what its search found."""
    archived = [(index, operation.get("insert", operation["exchange"]), operation["item_id"])
                for index, session in enumerate(result["sessions"])
                for operation in session["operations"] if operation["op_type"] == "archive"]
    overlaps = []
    for index, session in enumerate(result["sessions"]):
        for number, question in enumerate(session.get("questions", [])):
            trace_found = CONTEXT_ID.findall(question.get("context", ""))
            if trace_found:
                found = set(replay.found.get(question_key(index, number), []))
                overlaps.append(len(found & set(trace_found)) / len(trace_found))
    return {"split agreement": agreement(replay.split_archived, archived),
            "search agreement": sum(overlaps) / len(overlaps) if overlaps else None}


def simulate_user(run_dir: str, result: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Every rule at every limit on one user: the tallies by (rule, limit)."""
    config = trace_config(run_dir)
    uuid = result["uuid"]
    events = trace_events(result, user)
    closeness = load_closeness(run_dir, result)
    stored = load_labels(run_dir, uuid)
    needs = needs_of(stored)

    unlimited = Replay(config, None, None, closeness.rank, needs).run(events)
    tallies = {"unlimited": tally(unlimited)}
    with_ratings = config.consolidation_utility != "none"
    rules = score_rules(config) + reference_rules(config.prune_fraction, with_ratings)
    for limit in LIMITS:
        for rule in rules:
            replay = Replay(config, limit, rule, closeness.rank, needs).run(events)
            tallies[f"{rule.name}@{limit}"] = tally(replay)
        seeds = Counter()
        for seed in range(RANDOM_SEEDS):
            seeds += tally(Replay(config, limit, random_rule(seed, config.prune_fraction),
                                  closeness.rank, needs).run(events))
        tallies[f"random@{limit}"] = Counter({name: value / RANDOM_SEEDS
                                              for name, value in seeds.items()})
    return {"uuid": uuid, "tallies": tallies, "checks": trace_checks(unlimited, result),
            "rated": rated_memories(result, needs),
            "unlabelled": sum(1 for outcome in stored["labels"].values() if "error" in outcome),
            "labels": len(stored["labels"]), "model": stored.get("model")}


def percent(value: Optional[float]) -> str:
    return "  -  " if value is None else f"{100 * value:5.1f}"


def summarize(users: List[Dict[str, Any]], config: MemoryConfig) -> Dict[str, Any]:
    """The indicators of every (rule, limit) summed over users, with the intervals."""
    names = list(users[0]["tallies"])
    summary = {}
    for name in names:
        per_user = [Counter(user["tallies"][name]) for user in users]
        total = sum(per_user, Counter())
        summary[name] = {
            "retention": ratio(total["kept"], total["needed"]),
            "retention interval": bootstrap_interval(per_user, "kept", "needed"),
            "retrievable": ratio(total["found"], total["needed"]),
            "complete questions": ratio(total["complete questions"], total["labelled questions"]),
            "updates on removed": ratio(total["updates on removed"], total["updates"]),
            "divergence": ratio(total["updates on removed"] + total["redundants on removed"],
                                total["updates"] + total["redundants"]),
            "prunes per session": ratio(total["prunes"], total["sessions"]),
            "prunes needed later": ratio(total["prunes needed later"], total["prunes"]),
            "archive at questions": ratio(total["archive at questions"], total["questions"]),
            "by type": {key[5:-7]: ratio(total[key[:-7] + " kept"], total[key])
                        for key in total if key.startswith("type ") and key.endswith(" needed")},
            "by age": {bucket: ratio(total[f"age {bucket} kept"], total[f"age {bucket} needed"])
                       for _, _, bucket in AGE_BUCKETS},
            "needed": total["needed"],
        }
    for name, values in summary.items():
        if "@" not in name:
            continue
        limit = name.split("@")[1]
        low = summary[f"random@{limit}"]["retention"]
        high = summary[f"oracle@{limit}"]["retention"]
        values["of oracle"] = (None if None in (low, high, values["retention"]) or high == low
                               else (values["retention"] - low) / (high - low))
    rated = [(score, needed) for user in users for score, needed, _ in user["rated"]]
    levels = defaultdict(lambda: [0, 0])
    for user in users:
        for _, needed, level in user["rated"]:
            levels[level][0] += needed
            levels[level][1] += 1
    summary["rating AUC"] = auc(rated)
    summary["needed by level"] = {level: ratio(*levels[level]) for level in levels}
    summary["main rule"] = ScoreRule(config).name
    return summary


def print_summary(summary: Dict[str, Any], users: List[Dict[str, Any]]) -> None:
    for user in users:
        checks = user["checks"]
        print(f"{user['uuid']}: split agreement {percent(checks['split agreement'])}%, "
              f"search agreement {percent(checks['search agreement'])}%, "
              f"{user['labels']} labels, {user['unlabelled']} failed")
    print(f"\nWith no limit: {percent(summary['unlimited']['retrievable'])}% of the needed "
          f"memories found, {summary['unlimited']['needed']} needed in all")
    main = summary["main rule"]
    header = (f"{'rule':<37}{'kept':>6}{'95% interval':>16}{'found':>7}{'of oracle':>10}"
              f"{'diverge':>9}{'prunes':>8}{'later':>7}")
    for limit in LIMITS:
        print(f"\nArchive limit {limit}\n{header}")
        rows = [name for name in summary if name.endswith(f"@{limit}")]
        for name in rows:
            values = summary[name]
            interval = values["retention interval"]
            shown = "" if interval is None else f"[{percent(interval[0])},{percent(interval[1])}]"
            rule = name.split("@")[0] + (" *" if name.split("@")[0] == main else "")
            print(f"{rule:<37}{percent(values['retention']):>6}{shown:>16}"
                  f"{percent(values['retrievable']):>7}{percent(values['of oracle']):>10}"
                  f"{percent(values['divergence']):>9}"
                  f"{values['prunes per session'] or 0:>8.1f}"
                  f"{percent(values['prunes needed later']):>7}")
        values = summary[f"{main}@{limit}"]
        print("  * by type: " + ", ".join(f"{name} {percent(value)}"
                                          for name, value in sorted(values["by type"].items())))
        print("  * by age:  " + ", ".join(f"{name} {percent(value)}"
                                          for name, value in values["by age"].items()))
    print(f"\nRating AUC (need x gain, needed by a later session): {summary['rating AUC']}")
    levels = sorted(summary["needed by level"].items())
    print("Needed by a later session, by level: " + ", ".join(
        f"{level} {percent(value)}%" for level, value in levels))
    print("\nkept: needed memories still in core or archive; found: in core or among the "
          f"{QUESTION_K} the search returns;\nof oracle: (kept - random) / (oracle - random); "
          "diverge: updates and redundants on a removed memory;\nprunes: per session; later: "
          "prunes a later question needed. * is the trace's rule.")


def simulate(arguments: argparse.Namespace) -> int:
    run_dir = adapter.run_folder(arguments)
    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    config = trace_config(run_dir)
    if config.count_candidate_search:
        raise SystemExit("The trace counts the candidate searches, which the log does not "
                         "hold: it cannot be replayed.")
    results = trace_results(run_dir)[:arguments.users]
    unlabelled = [result["uuid"] for result in results
                  if not os.path.exists(labels_path(run_dir, result["uuid"]))]
    if unlabelled:
        raise SystemExit(f"{len(unlabelled)} users without labels: launch label first.")
    dataset = {user["uuid"]: user for user in adapter.evaluated_users(root)}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=arguments.workers, mp_context=context) as pool:
        users = list(pool.map(simulate_user, [run_dir] * len(results), results,
                              [dataset[result["uuid"]] for result in results]))
    summary = summarize(users, config)
    save_json(os.path.join(replay_folder(run_dir), "report.json"),
              {"summary": summary, "users": users})
    print_summary(summary, users)
    return 0


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The eviction rules replayed on a trace.")
    steps = parser.add_subparsers(dest="step", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    common.add_argument("--version", required=True, type=adapter.version_name,
                        help="the trace: a run with no practical archive limit")
    common.add_argument("--users", type=int, help="only the first users of the trace")

    steps.add_parser("embed", parents=[common], help="the vectors of the trace").set_defaults(
        step_function=embed)
    judged = steps.add_parser("label", parents=[common], help="what each question needs")
    judged.add_argument("--judge-workers", type=int, default=10, dest="judge_workers",
                        help="judge calls at the same time")
    judged.set_defaults(step_function=label)
    replayed = steps.add_parser("simulate", parents=[common], help="every rule, every limit")
    replayed.add_argument("--workers", type=int, default=1, help="users at the same time")
    replayed.set_defaults(step_function=simulate)
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_arguments()
    sys.exit(parsed.step_function(parsed))
