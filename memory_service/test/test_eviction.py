"""Eviction: le memorie superate e cancellate escono davvero dall'archivio, e
oltre il limite escono le attive con lo score piu' basso.

Offline, sullo stesso doppio di Chroma degli altri test. I tombstone sono
prodotti dalle funzioni vere di consolidation - supersede_item, delete_item,
supersede_archived_item, delete_archived_item - e non scritti a mano: se un
giorno cambia il modo in cui vengono marcati, questi test se ne accorgono.

Esecuzione: python memory_service/run_tests.py -v
"""

import dataclasses
import json
import math
import os
import sys
import unittest
from datetime import datetime, timedelta

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    CoreMemoryItem,
    archive_items,
    delete_archived_item,
    delete_item,
    supersede_archived_item,
    supersede_item,
)
from memory_service.eviction import (  # noqa: E402
    archive_over_limit,
    combine_terms,
    evict_archived_tombstones,
    eviction_score,
    eviction_terms,
    frequency_term,
    novelty_term,
    novelty_terms,
    prune_archive,
    prune_count,
    similarity_matrix,
    time_decay_term,
)
from memory_service.memory_manager_llm import MemoryAgent  # noqa: E402

from fakes import FakeVectorStore, ScriptedChatModel  # noqa: E402


EVICTION_CONFIG = MemoryConfig(
    node_name="memory_agent",
    generate_answer=False,
    maximum_historical_messages=1,
    core_memory_limit=2000,
    eviction_time_decay=True,
    eviction_time_decay_field="updated_at",
    eviction_novelty=False,
    eviction_novelty_mode="nearest",
    eviction_frequency=False,
    track_used=False,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)

NOW = datetime(2026, 9, 17, 12, 0)


class EvictionTestCase(unittest.TestCase):
    """Archivio finto, log vuoto, e una scorciatoia per archiviare."""

    store_class = FakeVectorStore

    def setUp(self):
        self.store = self.store_class()
        backends.reset()
        backends.configure(vector_store=self.store, config=EVICTION_CONFIG)
        self.log = []

    def tearDown(self):
        backends.reset()

    def archive(self, content, status="active", days_old=None, now=NOW, retrieved_days_old=None):
        """Un item in archivio, come lo lascia il core split."""
        item = CoreMemoryItem(content=content, status=status)
        if days_old is not None:
            item.updated_at = now - timedelta(days=days_old)
        if retrieved_days_old is not None:
            item.retrieved_at = now - timedelta(days=retrieved_days_old)
        archive_items([item])
        return item.id


class WhatLeavesTest(EvictionTestCase):
    """Cosa esce dall'archivio e cosa resta."""

    def test_superseded_and_deleted_leave_the_archive(self):
        old = CoreMemoryItem(content="obiettivo 2000 calorie")
        supersede_item(old, "obiettivo 2200 calorie", self.log)
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        kept = self.archive("ha un cane di nome Argo")

        evicted = evict_archived_tombstones(self.log)

        self.assertCountEqual(evicted, [old.id, gone.id])
        self.assertIsNone(self.store.status_of(old.id))
        self.assertIsNone(self.store.status_of(gone.id))
        self.assertEqual(self.store.status_of(kept), "active",
                         "archiviato dal core split ma ancora valido: resta")

    def test_items_retired_while_already_archived_leave_too(self):
        """L'altra strada per diventare tombstone: cambiare status in archivio."""
        novels = self.archive("legge romanzi storici")
        piano = self.archive("suona il pianoforte")
        supersede_archived_item(novels, "legge solo gialli", self.log)
        delete_archived_item(piano, self.log)

        self.assertCountEqual(evict_archived_tombstones(self.log), [novels, piano])
        self.assertEqual(self.store.documents, {})


class WhatTheLogSaysTest(EvictionTestCase):
    """Una voce evict per documento."""

    def test_each_eviction_is_logged_with_the_memory_text(self):
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        before = len(self.log)

        evict_archived_tombstones(self.log)

        added = self.log[before:]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].op_type, "evict")
        self.assertEqual(added[0].item_id, gone.id)
        self.assertEqual(added[0].content, "vive a Mondello")
        self.assertIsNone(added[0].related_item_id)


class NothingToDoTest(EvictionTestCase):
    """Senza tombstone non succede niente, e soprattutto nessuna delete a vuoto."""

    def test_an_archive_without_tombstones_is_left_as_it_is(self):
        kept = self.archive("ha un cane di nome Argo")

        self.assertEqual(evict_archived_tombstones(self.log), [])
        self.assertEqual(self.store.status_of(kept), "active")
        self.assertEqual(self.store.deletes, [],
                         "una delete senza id, a seconda della versione, e' tutto")
        self.assertEqual(self.log, [])


class UnreadableStore(FakeVectorStore):
    """Lo store che non risponde alla ricerca dei tombstone."""

    def get(self, ids=None, where=None, **kwargs):
        if where is not None:
            raise RuntimeError("chroma non risponde")
        return super().get(ids=ids, **kwargs)


class StuckStore(FakeVectorStore):
    """Lo store che trova i tombstone ma non riesce a cancellarli."""

    def delete(self, ids=None, **kwargs):
        raise RuntimeError("collezione bloccata")


class WhenTheStoreFailsTest(EvictionTestCase):
    """Non solleva mai, e il log non racconta cancellazioni mai avvenute.

    Gira in coda alla callback di update_memory: un'eccezione finirebbe nel suo
    except e svuoterebbe la risposta di un consolidamento riuscito.
    """

    def use(self, store_class):
        self.store = store_class()
        backends.configure(vector_store=self.store)

    def test_a_store_that_cannot_be_searched_removes_nothing(self):
        self.use(UnreadableStore)
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        before = len(self.log)

        self.assertEqual(evict_archived_tombstones(self.log), [])
        self.assertIn(gone.id, self.store.documents)
        self.assertEqual(len(self.log), before)

    def test_a_failed_delete_logs_nothing(self):
        self.use(StuckStore)
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        before = len(self.log)

        self.assertEqual(evict_archived_tombstones(self.log), [])
        self.assertIn(gone.id, self.store.documents)
        self.assertEqual(len(self.log), before,
                         "il log registra solo cio' che e' successo davvero")

    def test_an_unreadable_store_is_never_over_the_limit(self):
        # Senza un conteggio non si deve togliere niente.
        self.use(UnreadableStore)
        self.archive("ha un cane di nome Argo")

        self.assertEqual(archive_over_limit(0), 0)

    def test_a_failed_prune_logs_nothing(self):
        self.use(StuckStore)
        old = self.archive("ha un cane di nome Argo", days_old=30)

        self.assertEqual(prune_archive(self.log, 0, EVICTION_CONFIG, now=NOW), [])
        self.assertIn(old, self.store.documents)
        self.assertEqual(self.log, [])


class ArchiveLimitTest(EvictionTestCase):
    """Il limite conta le memorie attive e dice di quanto e' superato."""

    def test_within_the_limit_there_is_no_excess(self):
        for fact in ("ha un cane di nome Argo", "ha un gatto di nome Milo"):
            self.archive(fact)

        self.assertEqual(archive_over_limit(2), 0)

    def test_the_excess_is_how_many_active_memories_are_over(self):
        for fact in ("ha un cane", "ha un gatto", "corre la mattina", "nuota"):
            self.archive(fact)

        self.assertEqual(archive_over_limit(1), 3)

    def test_tombstones_do_not_count(self):
        # Escono comunque per status: contarli farebbe scattare il limite per
        # memorie che non valgono piu'.
        self.archive("ha un cane di nome Argo")
        delete_item(CoreMemoryItem(content="vive a Mondello"), self.log)
        supersede_item(CoreMemoryItem(content="obiettivo 2000 calorie"),
                       "obiettivo 2200 calorie", self.log)

        self.assertEqual(archive_over_limit(1), 0)


class TimeDecayTermTest(unittest.TestCase):
    """Il termine di decadimento: una Weibull sul tempo da updated_at."""

    def worth(self, **age):
        return time_decay_term((NOW - timedelta(**age)).isoformat(), NOW)

    def test_the_decided_curve_shape_0_5_scale_7_days(self):
        self.assertAlmostEqual(self.worth(days=1), 0.6853, places=4)
        self.assertAlmostEqual(self.worth(days=7), 1 / math.e, places=4)
        self.assertAlmostEqual(self.worth(days=30), 0.1262, places=4)

    def test_a_timestamp_in_the_future_is_worth_one(self):
        # Un orologio spostato non deve dare piu' di 1.
        self.assertEqual(self.worth(hours=-5), 1.0)


class EvictionScoreTest(unittest.TestCase):
    """La combinazione dei termini accesi."""

    METADATA = {"updated_at": (NOW - timedelta(days=3)).isoformat(),
                "retrieved_at": (NOW - timedelta(days=20)).isoformat()}

    def test_the_timestamps_share_one_slot(self):
        # Mai una media fra timestamp: il termine e' uno, il selettore dice quale.
        for field in ("updated_at", "retrieved_at"):
            with self.subTest(field=field):
                config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay_field=field)
                self.assertEqual(eviction_score(self.METADATA, config, NOW),
                                 time_decay_term(self.METADATA[field], NOW))

    def test_with_every_term_off_there_is_no_score(self):
        config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay=False)

        self.assertEqual(eviction_terms({"updated_at": NOW.isoformat()}, config, NOW), {})
        self.assertIsNone(eviction_score({"updated_at": NOW.isoformat()}, config, NOW))

    def test_the_score_is_the_mean_of_the_terms(self):
        self.assertAlmostEqual(combine_terms({"a": 0.2, "b": 0.6}), 0.4)
        self.assertIsNone(combine_terms({}))

    def test_the_novelty_joins_the_terms_when_switched_on(self):
        config = dataclasses.replace(EVICTION_CONFIG, eviction_novelty=True)

        self.assertEqual(eviction_terms(self.METADATA, config, NOW, novelty=0.4),
                         {"time_decay": time_decay_term(self.METADATA["updated_at"], NOW),
                          "novelty": 0.4})


class NoveltyTermTest(unittest.TestCase):
    """La novelty grezza di una memoria, dalle sue similarita' con le altre."""

    def test_nearest_is_one_minus_the_closest(self):
        self.assertAlmostEqual(novelty_term([0.2, 0.9, 0.5], "nearest"), 0.1)

    def test_k_nearest_averages_the_k_closest(self):
        self.assertAlmostEqual(novelty_term([0.1, 0.9, 0.5], "k_nearest", k=2), 0.3)

    def test_with_fewer_than_k_it_uses_what_there_is(self):
        self.assertAlmostEqual(novelty_term([0.1, 0.9, 0.5], "k_nearest", k=5), 0.5)

    def test_alone_it_is_worth_one(self):
        for mode in ("nearest", "k_nearest"):
            with self.subTest(mode=mode):
                self.assertEqual(novelty_term([], mode), 1.0)


class SimilarityMatrixTest(unittest.TestCase):
    def test_it_is_the_cosine_whatever_the_length(self):
        similarity = similarity_matrix({"a": [3, 4], "b": [6, 8], "c": [1, 0]})

        self.assertAlmostEqual(similarity["a"]["b"], 1.0)
        self.assertAlmostEqual(similarity["a"]["c"], 0.6)
        self.assertAlmostEqual(similarity["c"]["b"], 0.6)
        self.assertNotIn("a", similarity["a"], "una memoria non e' vicina di se' stessa")


class NoveltyTermsTest(unittest.TestCase):
    """La novelty di ogni memoria fra quelle date, relativa alla piu' distinta."""

    def test_a_duplicate_is_worth_zero_and_the_most_distinct_one(self):
        similarity = similarity_matrix({"a": [1, 0], "b": [1, 0], "c": [0, 1]})

        self.assertEqual(novelty_terms(["a", "b", "c"], similarity, "nearest"),
                         {"a": 0.0, "b": 0.0, "c": 1.0})

    def test_the_values_are_relative_to_the_most_distinct(self):
        # a-b 0.8, b-c 0.6, a-c 0: grezze 0.2, 0.2, 0.4.
        similarity = similarity_matrix({"a": [1, 0], "b": [0.8, 0.6], "c": [0, 1]})

        novelty = novelty_terms(["a", "b", "c"], similarity, "nearest")

        for doc_id, expected in (("a", 0.5), ("b", 0.5), ("c", 1.0)):
            self.assertAlmostEqual(novelty[doc_id], expected)

    def test_only_the_listed_memories_are_compared(self):
        # Senza b, il suo doppione a non ha piu' nessuno vicino.
        similarity = similarity_matrix({"a": [1, 0], "b": [1, 0], "c": [0, 1]})

        self.assertEqual(novelty_terms(["a", "c"], similarity, "nearest"),
                         {"a": 1.0, "c": 1.0})

    def test_all_identical_are_worth_zero(self):
        similarity = similarity_matrix({"a": [1, 0], "b": [2, 0]})

        self.assertEqual(novelty_terms(["a", "b"], similarity, "nearest"),
                         {"a": 0.0, "b": 0.0})


class FrequencyTermTest(EvictionTestCase):
    """Il termine di frequenza: un conteggio relativo al piu' alto, su n_retrieve o n_used."""

    FREQUENCY_ONLY = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay=False,
                                         eviction_frequency=True)
    USES_COUNTED = dataclasses.replace(FREQUENCY_ONLY, track_used=True, generate_answer=True)
    METADATA = {"updated_at": (NOW - timedelta(days=3)).isoformat(),
                "n_retrieve": 3, "n_used": 1}

    def archive_counts(self, content, n_retrieve, n_used):
        item = CoreMemoryItem(content=content, n_retrieve=n_retrieve, n_used=n_used)
        archive_items([item])
        return item.id

    def test_the_count_over_the_highest(self):
        for count, expected in ((6, 1.0), (3, 0.5), (1, 1 / 6), (0, 0.0)):
            with self.subTest(count=count):
                self.assertAlmostEqual(frequency_term(count, 6), expected)

    def test_an_archive_never_counted_is_worth_zero_everywhere(self):
        self.assertEqual(frequency_term(0, 0), 0.0)

    def test_with_the_switch_off_it_stays_out_of_the_score(self):
        self.assertEqual(eviction_score({**self.METADATA, "n_retrieve": 0}, EVICTION_CONFIG, NOW),
                         eviction_score({**self.METADATA, "n_retrieve": 50}, EVICTION_CONFIG, NOW))

    def test_with_the_switch_on_it_joins_the_terms(self):
        config = dataclasses.replace(EVICTION_CONFIG, eviction_frequency=True)

        self.assertEqual(eviction_terms(self.METADATA, config, NOW, frequency_max=6),
                         {"time_decay": time_decay_term(self.METADATA["updated_at"], NOW),
                          "frequency": 0.5})

    def test_the_two_counts_share_one_slot(self):
        for field, config in (("n_retrieve", self.FREQUENCY_ONLY), ("n_used", self.USES_COUNTED)):
            with self.subTest(field=field):
                self.assertEqual(eviction_terms(self.METADATA, config, NOW, frequency_max=6),
                                 {"frequency": self.METADATA[field] / 6})

    def test_in_the_prune_the_value_is_relative_to_the_highest(self):
        ids = {n: self.archive_counts(f"fatto {n}", n_retrieve=n, n_used=0) for n in (4, 2, 3)}

        removed = prune_archive(self.log, 2, self.FREQUENCY_ONLY, now=NOW)

        self.assertEqual(removed, [ids[2]])
        self.assertEqual(self.log[0].score_terms, {"frequency": 0.5}, "2 su un massimo di 4")

    def test_the_highest_is_taken_on_the_chosen_count(self):
        # Recuperate tutte 9 volte: il massimo dei recuperi non c'entra con gli usi.
        ids = {n: self.archive_counts(f"fatto {n}", n_retrieve=9, n_used=n) for n in (4, 2, 3)}

        removed = prune_archive(self.log, 2, self.USES_COUNTED, now=NOW)

        self.assertEqual(removed, [ids[2]])
        self.assertEqual(self.log[0].score_terms, {"frequency": 0.5}, "2 usi su un massimo di 4")

    def retrieved_and_used(self):
        """Recuperata spesso ma mai usata, usata spesso ma mai recuperata, e una a meta'."""
        return (self.archive_counts("ha un cane", n_retrieve=4, n_used=0),
                self.archive_counts("ha un gatto", n_retrieve=0, n_used=4),
                self.archive_counts("corre la mattina", n_retrieve=2, n_used=2))

    def test_without_an_answer_the_uses_are_not_counted_and_n_retrieve_stays(self):
        _, never_retrieved, _ = self.retrieved_and_used()
        config = dataclasses.replace(self.USES_COUNTED, generate_answer=False)

        self.assertEqual(prune_archive(self.log, 2, config, now=NOW), [never_retrieved])


class PruneCountTest(unittest.TestCase):
    """Quante ne escono: il 10% delle attive, arrotondato per eccesso."""

    def test_ten_percent_rounded_up(self):
        for active, expected in ((1, 1), (5, 1), (50, 5), (51, 6)):
            with self.subTest(active=active):
                self.assertEqual(prune_count(active), expected)


class PruneArchiveTest(EvictionTestCase):
    """Oltre il limite escono le attive con lo score piu' basso, e solo quelle."""

    def prune(self, limit, config=EVICTION_CONFIG):
        return prune_archive(self.log, limit, config, now=NOW)

    def test_within_the_limit_nothing_leaves(self):
        for days in range(3):
            self.archive(f"fatto {days}", days_old=days)

        self.assertEqual(self.prune(limit=3), [])
        self.assertEqual(self.store.deletes, [])

    def test_over_the_limit_the_oldest_leave_first(self):
        ids = {days: self.archive(f"fatto {days}", days_old=days) for days in range(11)}

        removed = self.prune(limit=10)

        self.assertCountEqual(removed, [ids[10], ids[9]], "il 10% di 11, per eccesso, e' 2")
        self.assertEqual(len(self.store.documents), 9)

    def test_each_pruned_memory_is_logged_with_its_score(self):
        old = self.archive("ha un cane di nome Argo", days_old=30)
        self.archive("ha un gatto di nome Milo", days_old=0)

        self.prune(limit=1)

        self.assertEqual(len(self.log), 1)
        entry = self.log[0]
        self.assertEqual((entry.op_type, entry.item_id, entry.content),
                         ("prune", old, "ha un cane di nome Argo"))
        self.assertAlmostEqual(entry.score, 0.1262, places=4)
        self.assertEqual(list(entry.score_terms), ["time_decay"])
        self.assertAlmostEqual(entry.score_terms["time_decay"], 0.1262, places=4)

    def test_the_sub_scores_survive_the_trip_to_the_ros_response(self):
        from memory_service.consolidation import serialize_operation_log_for_response

        self.archive("ha un cane di nome Argo", days_old=30)
        self.archive("ha un gatto di nome Milo", days_old=0)

        self.prune(limit=1)

        published = json.loads(serialize_operation_log_for_response(self.log)[0])
        self.assertEqual(published["score_terms"], self.log[0].score_terms)

    def test_with_every_term_off_nothing_leaves(self):
        config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay=False)
        for days in range(3):
            self.archive(f"fatto {days}", days_old=days * 30)

        self.assertEqual(self.prune(limit=0, config=config), [])
        self.assertEqual(self.store.deletes, [])
        self.assertEqual(self.log, [])

    def reinforced_and_retrieved(self):
        """Rinforzata ieri ma mai piu' recuperata, e ferma da 10 giorni ma recuperata ieri."""
        return (self.archive("ha un cane di nome Argo", days_old=1, retrieved_days_old=30),
                self.archive("ha un gatto di nome Milo", days_old=10, retrieved_days_old=1))

    def test_on_updated_at_the_longest_untouched_leaves(self):
        _, untouched = self.reinforced_and_retrieved()

        self.assertEqual(self.prune(limit=1), [untouched])

    def test_on_retrieved_at_the_longest_unretrieved_leaves(self):
        unretrieved, _ = self.reinforced_and_retrieved()
        config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay_field="retrieved_at")

        self.assertEqual(self.prune(limit=1, config=config), [unretrieved])


class NoveltyPruneTest(EvictionTestCase):
    """Con la novelty accesa escono prima le memorie simili ad altre."""

    NOVELTY_ONLY = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay=False,
                                       eviction_novelty=True)

    def archive_with_vector(self, content, vector, **kwargs):
        doc_id = self.archive(content, **kwargs)
        self.store.embeddings[doc_id] = vector
        return doc_id

    def prune(self, limit, config):
        return prune_archive(self.log, limit, config, now=NOW)

    def test_a_duplicate_leaves_before_a_distinct_memory(self):
        twins = {self.archive_with_vector("ha un cane", [1, 0]),
                 self.archive_with_vector("ha un cane di nome Argo", [1, 0])}
        self.archive_with_vector("corre la mattina", [0, 1])

        removed = self.prune(limit=2, config=self.NOVELTY_ONLY)

        self.assertEqual(len(removed), 1)
        self.assertIn(removed[0], twins)
        self.assertEqual(self.log[0].score_terms, {"novelty": 0.0})

    def test_of_two_duplicates_only_one_leaves(self):
        # Quota 2 su 11. Tutte insieme uscirebbero i due gemelli, i piu' bassi;
        # una alla volta, dopo il primo l'altro non ha piu' un doppione.
        config = dataclasses.replace(self.NOVELTY_ONLY, eviction_time_decay=True)
        twins = {self.archive_with_vector(f"gemello {n}", [1] + [0] * 9, days_old=0)
                 for n in range(2)}
        for n in range(1, 10):
            self.archive_with_vector(f"fatto {n}", [0] * n + [1] + [0] * (9 - n), days_old=30)

        removed = self.prune(limit=10, config=config)

        self.assertEqual(len(removed), 2)
        self.assertEqual(len(twins & set(removed)), 1)
        self.assertEqual(set(self.log[0].score_terms), {"time_decay", "novelty"})

    def crowded_topic_and_twins(self):
        """Sei memorie sullo stesso argomento, simili 0.8 fra loro, e due gemelli isolati."""
        crowded = {self.archive_with_vector(
            f"argomento {n}", [1] + [0.5 if i == n else 0 for i in range(6)] + [0])
            for n in range(6)}
        twins = {self.archive_with_vector(f"gemello {n}", [0] * 7 + [1]) for n in range(2)}
        return crowded, twins

    def test_nearest_sees_the_twins(self):
        _, twins = self.crowded_topic_and_twins()

        removed = self.prune(limit=7, config=self.NOVELTY_ONLY)

        self.assertEqual(len(removed), 1)
        self.assertIn(removed[0], twins)

    def test_k_nearest_sees_the_crowded_topic(self):
        crowded, _ = self.crowded_topic_and_twins()
        config = dataclasses.replace(self.NOVELTY_ONLY, eviction_novelty_mode="k_nearest")

        removed = self.prune(limit=7, config=config)

        self.assertEqual(len(removed), 1)
        self.assertIn(removed[0], crowded)


class WiredIntoTheAgentTest(EvictionTestCase):
    """Nel nodo evict_archive, solo nel ramo insert e solo a switch acceso."""

    def setUp(self):
        super().setUp()
        self.llm = ScriptedChatModel(tool_responses={}, default_content="ok")
        backends.configure(llm=self.llm)
        self.use_agent(eviction=True)

    def tearDown(self):
        MemoryAgent.reset_instance()
        super().tearDown()

    def use_agent(self, **overrides):
        config = dataclasses.replace(EVICTION_CONFIG, **overrides)
        backends.configure(config=config)
        MemoryAgent.reset_instance()
        self.agent = MemoryAgent(config=config)

    def insert(self, memories):
        self.llm.script({"InsertCoreMemories": {"memories": memories}})
        self.agent.append_message("un messaggio qualsiasi", "user")
        self.agent.append_message("ok", "assistant")
        return self.agent.run_memory_agent("insert")

    def remember_then_forget(self):
        state = self.insert([{"fact": "vive a Mondello", "operation": "new"}])
        item = state["core_memory"][0]
        self.insert([{"fact": "vive a Mondello", "operation": "delete",
                      "target_item_id": item.id}])
        return item

    def test_the_tombstone_of_this_insert_leaves_in_the_same_insert(self):
        item = self.remember_then_forget()

        self.assertIsNone(self.store.status_of(item.id))
        operations = self.agent.last_operations()
        self.assertEqual([(entry.op_type, entry.item_id) for entry in operations],
                         [("delete", item.id), ("evict", item.id)],
                         "la voce evict esce nella stessa risposta di update_memory")

    def test_every_exit_of_the_insert_branch_goes_through_the_eviction(self):
        paths = {
            "finestra non superata": {"maximum_historical_messages": 10},
            "consolidamento senza split": {},
            "consolidamento con split": {"core_memory_limit": 5},
        }
        for path, overrides in paths.items():
            with self.subTest(path=path):
                self.use_agent(eviction=True, **overrides)
                gone = CoreMemoryItem(content="vive a Mondello")
                delete_item(gone, self.log)

                self.insert([{"fact": "ha un cane", "operation": "new"}])

                self.assertIsNone(self.store.status_of(gone.id))

    def test_with_the_switch_off_the_tombstone_stays(self):
        self.use_agent(eviction=False)

        item = self.remember_then_forget()

        self.assertEqual(self.store.status_of(item.id), "deleted")
        self.assertNotIn("evict", [entry.op_type for entry in self.agent.state["operation_log"]])

    def test_a_retrieve_removes_nothing(self):
        self.use_agent(eviction=True, archive_memory_limit=0)
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        old = self.archive("ha un cane di nome Argo", days_old=30, now=datetime.now())

        self.agent.run_memory_agent("retrieve", query="Dove abito?")

        self.assertEqual(self.store.status_of(gone.id), "deleted")
        self.assertEqual(self.store.status_of(old), "active")

    def test_over_the_limit_the_oldest_memory_is_pruned_in_the_same_insert(self):
        self.use_agent(eviction=True, archive_memory_limit=1)
        old = self.archive("ha un cane di nome Argo", days_old=30, now=datetime.now())
        recent = self.archive("ha un gatto di nome Milo", days_old=0, now=datetime.now())

        self.insert([])

        self.assertIsNone(self.store.status_of(old))
        self.assertEqual(self.store.status_of(recent), "active")
        operations = self.agent.last_operations()
        self.assertEqual([(entry.op_type, entry.item_id) for entry in operations],
                         [("prune", old)])

    def test_with_novelty_a_duplicate_is_pruned_in_the_same_insert(self):
        self.use_agent(eviction=True, eviction_time_decay=False, eviction_novelty=True,
                       archive_memory_limit=2)
        twins = {self.archive("ha un cane di nome Argo"), self.archive("ha un cane di nome Argo")}
        distinct = self.archive("corre ogni mattina al parco")

        self.insert([])

        gone = [entry.item_id for entry in self.agent.last_operations()
                if entry.op_type == "prune"]
        self.assertEqual(len(gone), 1)
        self.assertIn(gone[0], twins)
        self.assertEqual(self.store.status_of(distinct), "active")


if __name__ == "__main__":
    unittest.main(verbosity=2)
