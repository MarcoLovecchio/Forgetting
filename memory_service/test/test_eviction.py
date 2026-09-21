"""Eviction: le memorie superate e cancellate escono davvero dall'archivio, e
oltre il limite escono le attive con lo score piu' basso.

Offline, sullo stesso doppio di Chroma degli altri test. I tombstone sono
prodotti dalle funzioni vere di consolidation - supersede_item, delete_item,
supersede_archived_item, delete_archived_item - e non scritti a mano: se un
giorno cambia il modo in cui vengono marcati, questi test se ne accorgono.

Esecuzione: python memory_service/run_tests.py -v
"""

import contextlib
import dataclasses
import io
import math
import os
import sys
import unittest
from datetime import datetime, timedelta
from typing import get_args

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    CoreMemoryItem,
    MemoryStatus,
    archive_items,
    delete_archived_item,
    delete_item,
    supersede_archived_item,
    supersede_item,
)
from memory_service.eviction import (  # noqa: E402
    EVICTED_STATUSES,
    archive_over_limit,
    evict_archived_tombstones,
    eviction_score,
    prune_archive,
    prune_count,
    retrieval_count_term,
    time_decay_term,
)
from memory_service.memory_manager_llm import MemoryAgent  # noqa: E402

from fakes import FakeVectorStore, ScriptedChatModel  # noqa: E402


EVICTION_CONFIG = MemoryConfig(
    node_name="memory_agent",
    generate_answer=False,
    maximum_historical_messages=1,
    core_memory_limit=2000,
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

    def archive(self, content, status="active", days_old=None, now=NOW, created_days_old=None,
                retrieved_days_old=None):
        """Un item in archivio, come lo lascia il core split."""
        item = CoreMemoryItem(content=content, status=status)
        if days_old is not None:
            item.updated_at = now - timedelta(days=days_old)
        if created_days_old is not None:
            item.created_at = now - timedelta(days=created_days_old)
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

    def test_the_statuses_are_ones_consolidation_actually_writes(self):
        """Un nome sbagliato non rimuoverebbe niente, e senza dirlo.

        Lo status di una memoria dimenticata e' "deleted"; "delete" e' il nome
        dell'operazione. Scritto cosi', il filtro non troverebbe mai nulla. E
        "active" qui dentro svuoterebbe l'archivio.
        """
        self.assertNotIn("active", EVICTED_STATUSES)
        self.assertTrue(set(EVICTED_STATUSES) <= set(get_args(MemoryStatus)),
                        "%s non sono tutti status di MemoryStatus" % (EVICTED_STATUSES,))

    def test_a_second_pass_finds_nothing(self):
        delete_item(CoreMemoryItem(content="vive a Mondello"), self.log)
        self.assertEqual(len(evict_archived_tombstones(self.log)), 1)
        self.assertEqual(evict_archived_tombstones(self.log), [])


class WhatTheLogSaysTest(EvictionTestCase):
    """Una voce evict per documento, e il log di prima resta com'era."""

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

    def test_the_existing_log_is_left_alone(self):
        """Le voci che nominano un id rimosso restano, related_item_id compreso."""
        old = CoreMemoryItem(content="obiettivo 2000 calorie")
        supersede_item(old, "obiettivo 2200 calorie", self.log)
        before = [entry.model_dump() for entry in self.log]

        evict_archived_tombstones(self.log)

        self.assertEqual([entry.model_dump() for entry in self.log[:len(before)]], before)
        self.assertEqual(self.log[0].related_item_id, old.id,
                         "l'update continua a puntare alla versione rimossa")

    def test_an_evict_entry_survives_the_trip_to_the_ros_response(self):
        """Il log viaggia come JSON: op_type "evict" deve essere accettato."""
        from memory_service.consolidation import serialize_operation_log_for_response

        delete_item(CoreMemoryItem(content="vive a Mondello"), self.log)
        evict_archived_tombstones(self.log)
        self.assertIn('"op_type":"evict"', serialize_operation_log_for_response(self.log)[-1])


class NothingToDoTest(EvictionTestCase):
    """Senza tombstone non succede niente, e soprattutto nessuna delete a vuoto."""

    def test_an_archive_without_tombstones_is_left_as_it_is(self):
        kept = self.archive("ha un cane di nome Argo")

        self.assertEqual(evict_archived_tombstones(self.log), [])
        self.assertEqual(self.store.status_of(kept), "active")
        self.assertEqual(self.store.deletes, [],
                         "una delete senza id, a seconda della versione, e' tutto")
        self.assertEqual(self.log, [])

    def test_an_empty_archive_is_fine(self):
        self.assertEqual(evict_archived_tombstones(self.log), [])
        self.assertEqual(self.store.deletes, [])


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

    def test_an_unreadable_store_prunes_nothing(self):
        self.use(UnreadableStore)
        old = self.archive("ha un cane di nome Argo", days_old=30)

        self.assertEqual(prune_archive(self.log, 0, EVICTION_CONFIG, now=NOW), [])
        self.assertIn(old, self.store.documents)

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

    def test_checking_removes_nothing(self):
        kept = [self.archive(fact) for fact in ("ha un cane", "ha un gatto")]
        self.log = []

        archive_over_limit(0)

        self.assertCountEqual(self.store.documents, kept)
        self.assertEqual(self.store.deletes, [])
        self.assertEqual(self.log, [])


class TimeDecayTermTest(unittest.TestCase):
    """Il termine di decadimento: una Weibull sul tempo da updated_at."""

    def worth(self, **age):
        return time_decay_term((NOW - timedelta(**age)).isoformat(), NOW)

    def test_a_memory_touched_now_is_worth_one(self):
        self.assertEqual(self.worth(hours=0), 1.0)

    def test_the_decided_curve_shape_0_5_scale_7_days(self):
        self.assertAlmostEqual(self.worth(days=1), 0.6853, places=4)
        self.assertAlmostEqual(self.worth(days=7), 1 / math.e, places=4)
        self.assertAlmostEqual(self.worth(days=30), 0.1262, places=4)

    def test_older_is_always_worth_less(self):
        ages = [timedelta(0), timedelta(hours=1), timedelta(days=1), timedelta(days=7),
                timedelta(days=30), timedelta(days=365)]
        values = [time_decay_term((NOW - age).isoformat(), NOW) for age in ages]

        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(len(set(values)), len(values))

    def test_a_timestamp_in_the_future_is_worth_one(self):
        # Un orologio spostato non deve dare piu' di 1.
        self.assertEqual(self.worth(hours=-5), 1.0)


class EvictionScoreTest(unittest.TestCase):
    """La combinazione dei termini accesi."""

    METADATA = {"created_at": (NOW - timedelta(days=60)).isoformat(),
                "updated_at": (NOW - timedelta(days=3)).isoformat(),
                "retrieved_at": (NOW - timedelta(days=20)).isoformat()}

    def test_by_default_the_decay_runs_on_updated_at(self):
        self.assertEqual(eviction_score(self.METADATA, EVICTION_CONFIG, NOW),
                         time_decay_term(self.METADATA["updated_at"], NOW))

    def test_the_timestamps_share_one_slot(self):
        # Mai una media fra timestamp: il termine e' uno, il selettore dice quale.
        for field in ("updated_at", "created_at", "retrieved_at"):
            with self.subTest(field=field):
                config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay_field=field)
                self.assertEqual(eviction_score(self.METADATA, config, NOW),
                                 time_decay_term(self.METADATA[field], NOW))

    def test_with_every_term_off_there_is_no_score(self):
        config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay=False)

        self.assertIsNone(eviction_score({"updated_at": NOW.isoformat()}, config, NOW))


class RetrievalCountTermTest(unittest.TestCase):
    """Il termine dei recuperi: lineare, relativo alla memoria piu' recuperata."""

    def test_the_count_over_the_most_retrieved(self):
        for n_retrieve, expected in ((6, 1.0), (3, 0.5), (1, 1 / 6), (0, 0.0)):
            with self.subTest(n_retrieve=n_retrieve):
                self.assertAlmostEqual(retrieval_count_term(n_retrieve, 6), expected)

    def test_the_same_count_is_worth_less_in_a_busier_archive(self):
        self.assertEqual(retrieval_count_term(2, 4), 0.5)
        self.assertEqual(retrieval_count_term(2, 8), 0.25)

    def test_an_archive_never_retrieved_is_worth_zero_everywhere(self):
        self.assertEqual(retrieval_count_term(0, 0), 0.0)

    def test_it_does_not_enter_the_score_yet(self):
        base = {"updated_at": (NOW - timedelta(days=3)).isoformat()}

        self.assertEqual(eviction_score({**base, "n_retrieve": 0}, EVICTION_CONFIG, NOW),
                         eviction_score({**base, "n_retrieve": 50}, EVICTION_CONFIG, NOW))


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

    def test_the_quota_is_ten_percent_of_the_active_not_the_excess(self):
        ids = {days: self.archive(f"fatto {days}", days_old=days) for days in range(20)}

        removed = self.prune(limit=19)

        self.assertCountEqual(removed, [ids[19], ids[18]], "eccedenza 1, ma ne escono 2")

    def test_each_pruned_memory_is_logged_with_its_score(self):
        old = self.archive("ha un cane di nome Argo", days_old=30)
        self.archive("ha un gatto di nome Milo", days_old=0)

        self.prune(limit=1)

        self.assertEqual(len(self.log), 1)
        entry = self.log[0]
        self.assertEqual((entry.op_type, entry.item_id, entry.content),
                         ("prune", old, "ha un cane di nome Argo"))
        self.assertAlmostEqual(entry.score, 0.1262, places=4)

    def test_with_every_term_off_nothing_leaves(self):
        config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay=False)
        for days in range(3):
            self.archive(f"fatto {days}", days_old=days * 30)

        self.assertEqual(self.prune(limit=0, config=config), [])
        self.assertEqual(self.store.deletes, [])
        self.assertEqual(self.log, [])

    def old_but_reinforced_and_untouched(self):
        """Creata 60 giorni fa ma rinforzata ieri, e creata e toccata 10 giorni fa."""
        return (self.archive("ha un cane di nome Argo", created_days_old=60, days_old=1),
                self.archive("ha un gatto di nome Milo", created_days_old=10, days_old=10))

    def test_on_updated_at_the_longest_untouched_leaves(self):
        _, untouched = self.old_but_reinforced_and_untouched()

        self.assertEqual(self.prune(limit=1), [untouched])

    def test_on_created_at_the_oldest_leaves_even_if_reinforced(self):
        old_but_reinforced, _ = self.old_but_reinforced_and_untouched()
        config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay_field="created_at")

        self.assertEqual(self.prune(limit=1, config=config), [old_but_reinforced])

    def test_on_retrieved_at_the_longest_unretrieved_leaves(self):
        # Su updated_at e created_at uscirebbe la prima.
        self.archive("ha un cane di nome Argo", created_days_old=60, days_old=1,
                     retrieved_days_old=2)
        unretrieved = self.archive("ha un gatto di nome Milo", created_days_old=10, days_old=0,
                                   retrieved_days_old=30)
        config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay_field="retrieved_at")

        self.assertEqual(self.prune(limit=1, config=config), [unretrieved])


class WiredIntoTheAgentTest(EvictionTestCase):
    """In fondo a run_memory_agent, solo dopo un insert e solo a switch acceso."""

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

    def test_with_every_term_off_nothing_active_is_removed(self):
        self.use_agent(eviction=True, eviction_time_decay=False, archive_memory_limit=0)
        kept = self.archive("ha un cane di nome Argo", days_old=30, now=datetime.now())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.insert([])

        self.assertIn("Archive over its limit", output.getvalue(), "il limite viene controllato")
        self.assertEqual(self.store.status_of(kept), "active")
        self.assertEqual(self.store.deletes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
