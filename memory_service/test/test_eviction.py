"""Eviction: le memorie superate e cancellate escono davvero dall'archivio.

Offline, sullo stesso doppio di Chroma degli altri test. I tombstone sono
prodotti dalle funzioni vere di consolidation - supersede_item, delete_item,
supersede_archived_item, delete_archived_item - e non scritti a mano: se un
giorno cambia il modo in cui vengono marcati, questi test se ne accorgono.

Esecuzione: python memory_service/run_tests.py -v
"""

import os
import sys
import unittest
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
    evict_archived_tombstones,
)

from fakes import FakeVectorStore  # noqa: E402


EVICTION_CONFIG = MemoryConfig(
    node_name="memory_agent",
    generate_answer=False,
    maximum_historical_messages=1,
    core_memory_limit=2000,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)


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

    def archive(self, content, status="active"):
        """Un item in archivio, come lo lascia il core split."""
        item = CoreMemoryItem(content=content, status=status)
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

    def test_only_the_listed_statuses_are_evicted(self):
        """Un documento senza status, o con uno sconosciuto, non si tocca."""
        self.store.add_texts(texts=["senza status"], ids=["nostatus"], metadatas=[{}])
        self.store.add_texts(texts=["status ignoto"], ids=["pending"],
                             metadatas=[{"status": "pending"}])

        self.assertEqual(evict_archived_tombstones(self.log), [])
        self.assertIn("nostatus", self.store.documents)
        self.assertIn("pending", self.store.documents)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
