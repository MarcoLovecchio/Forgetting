"""Ciclo di vita della memoria, un turno per operazione, senza chiavi API.

Ogni turno l'utente dice qualcosa, il classificatore (qui scriptato) restituisce
una classificazione diversa, e dopo ogni turno viene stampato lo stato completo
della memoria: core, finestra dei messaggi, archivio vettoriale con i metadata,
ultimo recupero e operation log.

I turni sono in sequenza e ognuno parte dallo stato lasciato dal precedente:

    1. new         tre fatti nuovi entrano in core memory
    2. redundant   lo stesso fatto ripetuto rinforza l'item esistente
    3. update      un fatto viene raffinato: il vecchio diventa superseded
    4. contradict  un fatto viene smentito: stessa meccanica, op diversa
    5. delete      l'utente chiede di dimenticare: l'item diventa deleted
    6. archive     core memory oltre il limite: un item passa all'archivio
    7. retrieve    la risposta pesca dall'archivio, i tombstone restano fuori

Esecuzione: python memory_service/run_tests.py -v
"""

import os
import sys
import time
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import HumanMessage  # noqa: E402

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    NO_ARCHIVAL_RESULTS,
    CoreMemoryItem,
    archive_items,
    get_active_items,
    reinforce_archived_item,
    retrieve_active_archival_memories,
    search_archive,
    serialize_retrieved_for_response,
    supersede_archived_item,
)
from memory_service.memory_manager_llm import MemoryAgent  # noqa: E402

from fakes import FakeVectorStore, ScriptedChatModel  # noqa: E402
from snapshot import print_memory_snapshot, print_turn_header  # noqa: E402


# Un solo messaggio di storico: cosi' ogni turno fa scattare il consolidamento.
# Il limite di caratteri e' alto, il turno 6 lo abbassa apposta per lo split.
LIFECYCLE_CONFIG = MemoryConfig(
    node_name="memory_agent",
    generate_answer=True,  # il turno 7 guarda la risposta del ramo retrieve
    maximum_historical_messages=1,
    core_memory_limit=2000,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)


def find_item(items, needle):
    """Item attivo il cui contenuto contiene `needle`."""
    for item in items:
        if needle.lower() in item.content.lower():
            return item
    raise AssertionError(f"nessun item contiene {needle!r}: {[i.content for i in items]}")


class ConsolidationLifecycleTest(unittest.TestCase):
    """Un unico scenario: i turni si susseguono e costruiscono lo stato."""

    def setUp(self):
        self.llm = ScriptedChatModel(tool_responses={}, default_content="ok")
        self.store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=self.llm, vector_store=self.store, config=LIFECYCLE_CONFIG)
        MemoryAgent.reset_instance()
        self.agent = MemoryAgent(config=LIFECYCLE_CONFIG)

    def tearDown(self):
        MemoryAgent.reset_instance()
        backends.reset()

    # -- un turno ---------------------------------------------------------- #

    def consolidate(self, turn, operation, description, user_message, memories,
                    assistant_message="Va bene."):
        """L'utente dice qualcosa, il classificatore risponde `memories`."""
        print_turn_header(turn, operation, description)
        self.llm.script({"InsertCoreMemories": {"memories": memories}})
        self.agent.append_message(user_message, "user")
        self.agent.append_message(assistant_message, "assistant")

        state = self.agent.run_memory_agent("insert")

        print_memory_snapshot(f"turno {turn} - {operation}", state, self.store)
        return state

    def active(self, state):
        return get_active_items(state["core_memory"])

    # -- lo scenario -------------------------------------------------------- #

    def test_memory_lifecycle_turn_by_turn(self):
        # --- TURNO 1: new ------------------------------------------------- #
        state = self.consolidate(
            1, "new", "tre fatti mai visti prima entrano in core memory",
            "Mi chiamo Bianca, sono vegetariana e punto a 2000 calorie al giorno.",
            [
                {"fact": "L'utente si chiama Bianca", "operation": "new"},
                {"fact": "L'utente e' vegetariana", "operation": "new"},
                {"fact": "L'obiettivo giornaliero e' 2000 calorie", "operation": "new"},
            ],
        )

        items = self.active(state)
        self.assertEqual(len(items), 3, "i tre fatti diventano tre item distinti")
        self.assertEqual([entry.op_type for entry in state["operation_log"]], ["create"] * 3)
        self.assertEqual(self.store.documents, {}, "niente finisce in archivio")

        name_item = find_item(items, "Bianca")
        veg_item = find_item(items, "vegetariana")
        goal_item = find_item(items, "2000 calorie")

        # --- TURNO 2: redundant ------------------------------------------- #
        updated_at_before = name_item.updated_at
        time.sleep(0.02)  # updated_at deve risultare piu' recente

        state = self.consolidate(
            2, "redundant", "l'utente ripete un fatto gia' noto: si rinforza, non si duplica",
            "Ti ricordo che mi chiamo Bianca.",
            [{"fact": "L'utente si chiama Bianca", "operation": "redundant",
              "target_item_id": name_item.id}],
        )

        items = self.active(state)
        self.assertEqual(len(items), 3, "nessun item aggiunto")
        reinforced = find_item(items, "Bianca")
        self.assertEqual(reinforced.id, name_item.id, "e' lo stesso item di prima")
        self.assertGreater(reinforced.updated_at, updated_at_before, "updated_at rinfrescato")
        self.assertEqual(state["operation_log"][-1].op_type, "redundant")
        self.assertEqual(self.store.documents, {}, "un rinforzo non tocca l'archivio")

        # --- TURNO 3: update ---------------------------------------------- #
        state = self.consolidate(
            3, "update", "un fatto viene raffinato: il vecchio diventa superseded",
            "Ho alzato l'obiettivo a 2200 calorie al giorno.",
            [{"fact": "L'obiettivo giornaliero e' 2200 calorie", "operation": "update",
              "target_item_id": goal_item.id}],
        )

        items = self.active(state)
        self.assertEqual(len(items), 3, "il vecchio esce, il nuovo entra")
        find_item(items, "2200 calorie")
        self.assertNotIn(goal_item.id, [item.id for item in items], "il vecchio esce da core")
        self.assertEqual(self.store.status_of(goal_item.id), "superseded")
        self.assertEqual(self.store.documents[goal_item.id], goal_item.content,
                         "in archivio resta il contenuto originale")
        last = state["operation_log"][-1]
        self.assertEqual((last.op_type, last.related_item_id), ("update", goal_item.id))

        # --- TURNO 4: contradict ------------------------------------------ #
        state = self.consolidate(
            4, "contradict", "un fatto viene smentito: stessa meccanica dell'update",
            "In realta' non sono piu' vegetariana, ora mangio pesce.",
            [{"fact": "L'utente mangia pesce, non e' piu' vegetariana",
              "operation": "contradict", "target_item_id": veg_item.id}],
        )

        items = self.active(state)
        self.assertEqual(len(items), 3)
        find_item(items, "pesce")
        self.assertEqual(self.store.status_of(veg_item.id), "superseded")
        last = state["operation_log"][-1]
        self.assertEqual((last.op_type, last.related_item_id), ("contradict", veg_item.id))

        # --- TURNO 5: delete ---------------------------------------------- #
        state = self.consolidate(
            5, "delete", "l'utente chiede esplicitamente di dimenticare un fatto",
            "Dimentica il mio nome, non voglio che lo memorizzi.",
            [{"fact": "L'utente si chiama Bianca", "operation": "delete",
              "target_item_id": name_item.id}],
        )

        items = self.active(state)
        self.assertEqual(len(items), 2, "l'item cancellato esce da core memory")
        self.assertNotIn(name_item.id, [item.id for item in items])
        self.assertEqual(self.store.status_of(name_item.id), "deleted",
                         "resta in archivio come tombstone")
        self.assertEqual(state["operation_log"][-1].op_type, "delete")

        # --- TURNO 6: archive (split core -> archivio) --------------------- #
        survivor = find_item(items, "pesce")
        moved = find_item(items, "2200 calorie")

        print_turn_header(6, "archive", "core memory oltre il limite: un item passa all'archivio")
        self.agent.state["core_memory_limit"] = 40  # forza lo split al prossimo giro
        self.llm.script({
            "InsertCoreMemories": {"memories": []},
            "SplitCoreAndArchivalMemory": {
                "decisions": [
                    {"item_id": moved.id, "destination": "archive"},
                    {"item_id": survivor.id, "destination": "core"},
                ]
            },
        })
        self.agent.append_message("Parliamo d'altro.", "user")
        self.agent.append_message("Certo.", "assistant")

        state = self.agent.run_memory_agent("insert")
        print_memory_snapshot("turno 6 - archive", state, self.store)

        items = self.active(state)
        self.assertEqual([item.id for item in items], [survivor.id], "in core resta un item solo")
        self.assertEqual(self.store.documents[moved.id], moved.content)
        self.assertEqual(self.store.status_of(moved.id), "active",
                         "archiviato ma ancora valido, non e' un tombstone")
        archived_metadata = self.store.metadatas[moved.id]
        for field in ("status", "created_at", "updated_at"):
            self.assertIn(field, archived_metadata,
                          "i campi del CoreMemoryItem sopravvivono all'archiviazione")
        self.assertEqual(state["operation_log"][-1].op_type, "archive")

        # --- TURNO 7: retrieve --------------------------------------------- #
        print_turn_header(
            7, "retrieve", "la risposta pesca dall'archivio, i tombstone restano fuori")
        self.llm.script(
            {
                "retrieve_memory": {"query": "calorie obiettivo", "k": 10},
            },
            default_content="Il tuo obiettivo e' 2200 calorie al giorno.",
        )
        self.agent.state["messages"] = [
            HumanMessage(content="Qual e' il mio obiettivo calorico?")]
        self.agent.state["retrieved_memory"] = ""

        state = self.agent.run_memory_agent("retrieve")
        print_memory_snapshot("turno 7 - retrieve", state, self.store)

        retrieved = state["retrieved_memory"]
        self.assertIn("2200 calorie", retrieved, "l'item archiviato attivo viene recuperato")
        self.assertNotIn(name_item.content, retrieved, "il fatto cancellato non torna")
        self.assertNotIn(veg_item.content, retrieved, "il fatto smentito non torna")
        self.assertEqual(state["messages"][-1].content,
                         "Il tuo obiettivo e' 2200 calorie al giorno.")


class TargetResolutionTest(unittest.TestCase):
    """Cosa succede quando il classificatore non da' un target_item_id valido.

    Sono i casi che nella pratica si vedono piu' spesso: il modello dimentica
    l'id, oppure se lo inventa.
    """

    def setUp(self):
        self.llm = ScriptedChatModel(tool_responses={}, default_content="ok")
        self.store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=self.llm, vector_store=self.store, config=LIFECYCLE_CONFIG)
        MemoryAgent.reset_instance()
        self.agent = MemoryAgent(config=LIFECYCLE_CONFIG)

    def tearDown(self):
        MemoryAgent.reset_instance()
        backends.reset()

    def consolidate(self, memories):
        self.llm.script({"InsertCoreMemories": {"memories": memories}})
        self.agent.append_message("un messaggio qualsiasi", "user")
        self.agent.append_message("ok", "assistant")
        return self.agent.run_memory_agent("insert")

    def test_redundant_without_id_is_matched_by_content(self):
        state = self.consolidate([{"fact": "L'utente si chiama Bianca", "operation": "new"}])
        existing = state["core_memory"][0]

        state = self.consolidate(
            [{"fact": "L'utente si chiama Bianca", "operation": "redundant"}])

        self.assertEqual([item.id for item in state["core_memory"]], [existing.id],
                         "riconosciuto come lo stesso fatto, nessun duplicato")
        self.assertEqual(state["operation_log"][-1].op_type, "redundant")

    def test_update_on_an_unknown_id_is_kept_as_a_new_fact(self):
        state = self.consolidate(
            [{"fact": "L'utente vive a Palermo", "operation": "update",
              "target_item_id": "id-inventato-dal-modello"}])

        self.assertEqual([item.content for item in state["core_memory"]],
                         ["L'utente vive a Palermo"])
        self.assertEqual(state["operation_log"][-1].op_type, "create",
                         "meglio salvarlo come nuovo che perdere l'informazione")

    def test_delete_without_a_target_is_ignored(self):
        state = self.consolidate(
            [{"fact": "qualcosa che non e' in memoria", "operation": "delete"}])

        self.assertEqual(state["core_memory"], [], "non si cancella nulla a caso")
        self.assertEqual(state["operation_log"], [], "e non si registra nessuna operazione")


class MetadataRewriteTest(unittest.TestCase):
    """Cambiare lo status di un item archiviato non deve ricalcolare il vettore.

    Riscrivere il documento con add_texts farebbe ricalcolare l'embedding: una
    chiamata di rete al modello, sul cluster, per cambiare un campo scalare. E
    succede a ogni redundant, update, contradict o delete che colpisca un item
    gia' in archivio.
    """

    def setUp(self):
        self.store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=ScriptedChatModel(tool_responses={}),
                           vector_store=self.store, config=LIFECYCLE_CONFIG)
        self.item = CoreMemoryItem(content="L'utente e' vegetariana")
        archive_items([self.item])
        self.writes_after_archiving = len(self.store.writes)

    def tearDown(self):
        backends.reset()

    def test_a_status_change_does_not_rewrite_the_document(self):
        supersede_archived_item(self.item.id, "L'utente mangia pesce", [])

        self.assertEqual(len(self.store.writes), self.writes_after_archiving,
                         "nessuna scrittura in piu': niente embedding ricalcolato")
        self.assertIn(self.item.id, self.store._collection.updates,
                      "il cambio passa dall'aggiornamento dei soli metadata")

    def test_the_document_survives_the_status_change(self):
        supersede_archived_item(self.item.id, "L'utente mangia pesce", [])

        self.assertEqual(self.store.documents[self.item.id], "L'utente e' vegetariana")
        self.assertEqual(self.store.status_of(self.item.id), "superseded")

    def test_a_reinforcement_costs_no_embedding_either(self):
        # E' il caso piu' sprecato dei tre: si pagava un embedding per registrare
        # che un fatto e' stato ripetuto.
        reinforce_archived_item(self.item.id, [])

        self.assertEqual(len(self.store.writes), self.writes_after_archiving)
        self.assertEqual(self.store.status_of(self.item.id), "active")


class RetrievedSerializationTest(unittest.TestCase):
    """Il recupero esce dal servizio come lista, non come frase.

    retrieve_active_archival_memories restituisce testo perche' finisce dritto
    in un prompt: "No relevant active memories found." e' una frase, non un
    vuoto, e un chiamante che la ricevesse la tratterebbe come un risultato.
    """

    def test_the_lines_become_entries(self):
        retrieved = "ID: abc, Content: uno\nID: def, Content: due"

        self.assertEqual(
            serialize_retrieved_for_response(retrieved),
            ["ID: abc, Content: uno", "ID: def, Content: due"])

    def test_the_nothing_found_sentence_becomes_an_empty_list(self):
        self.assertEqual(serialize_retrieved_for_response(NO_ARCHIVAL_RESULTS), [])

    def test_no_retrieval_at_all_becomes_an_empty_list(self):
        for nothing in ("", "   ", None):
            with self.subTest(nothing=nothing):
                self.assertEqual(serialize_retrieved_for_response(nothing), [])


class ArchiveSearchTest(unittest.TestCase):
    """Il filtro sulle attive lo fa lo store, dentro l'indice.

    Chiedere k documenti e scartare dopo quelli non attivi ne restituisce meno
    di k, e il divario peggiora col tempo perche' i tombstone non vengono mai
    rimossi: il classificatore si ritroverebbe senza candidati proprio su un
    archivio molto usato, senza che niente segnali il problema.
    """

    def setUp(self):
        self.store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=ScriptedChatModel(tool_responses={}),
                           vector_store=self.store, config=LIFECYCLE_CONFIG)

    def tearDown(self):
        backends.reset()

    def fill(self, tombstones, active):
        """Prima i tombstone, poi le attive.

        Tutti i contenuti contengono la stessa parola della query, quindi hanno
        lo stesso punteggio: l'ordinamento e' stabile e le lapidi finiscono
        davanti. E' il caso peggiore, ed e' quello che succede davvero quando
        una memoria viene aggiornata piu' volte - la versione vecchia resta li',
        somigliantissima alla nuova.
        """
        for index in range(tombstones):
            self.store.add_texts(texts=[f"il gatto, versione vecchia {index}"],
                                 ids=[f"tomb_{index}"],
                                 metadatas=[{"status": "superseded"}])
        for index in range(active):
            self.store.add_texts(texts=[f"il gatto, versione buona {index}"],
                                 ids=[f"live_{index}"],
                                 metadatas=[{"status": "active"}])

    def test_tombstones_do_not_steal_the_places(self):
        # Con k=3 e 6 lapidi davanti, la ricerca stretta ne restituirebbe zero.
        self.fill(tombstones=6, active=4)

        results = search_archive("gatto", k=3)

        self.assertEqual(len(results), 3, "i candidati attivi devono essere k")
        self.assertTrue(all(doc_id.startswith("live_") for doc_id, _, _ in results))

    def test_every_result_carries_its_distance(self):
        self.fill(tombstones=0, active=3)

        results = search_archive("gatto", k=3)

        self.assertTrue(all(isinstance(distance, float) for _, _, distance in results))

    def test_the_distance_reaches_the_retrieval_string(self):
        """E' l'unico canale verso il resoconto.

        Senza il numero, tre risultati si leggono tutti uguali e non c'e' modo di
        distinguere una memoria centrata da una raschiata dal fondo per riempire
        k - che e' proprio l'informazione che serve per tarare una soglia.
        """
        self.fill(tombstones=0, active=2)

        retrieved = retrieve_active_archival_memories("gatto", k=2)

        for line in retrieved.splitlines():
            self.assertIn("Distance:", line)

    def test_the_search_delegates_the_filter_to_the_store(self):
        self.fill(tombstones=2, active=5)

        search_archive("gatto", k=3)

        search = self.store.searches[-1]
        self.assertEqual(search["filter"], {"status": "active"})
        self.assertEqual(search["k"], 3,
                         "niente documenti chiesti in piu': il filtro e' nell'indice")

    def test_the_result_is_never_longer_than_k(self):
        self.fill(tombstones=0, active=10)

        self.assertEqual(len(search_archive("gatto", k=3)), 3)

    def test_an_archive_of_only_tombstones_returns_nothing(self):
        # Meno candidati e' un prompt piu' povero, non un errore.
        self.fill(tombstones=8, active=0)

        self.assertEqual(search_archive("gatto", k=3), [])

    def test_fewer_active_than_k_returns_what_exists(self):
        self.fill(tombstones=5, active=2)

        results = search_archive("gatto", k=3)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(doc_id.startswith("live_") for doc_id, _, _ in results))


if __name__ == "__main__":
    unittest.main()
