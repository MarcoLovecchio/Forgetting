"""Eviction: cosa esce dall'archivio per sempre.

Consolidation non cancella mai niente. Una memoria superata o dimenticata smette
di essere attiva e resta in archivio come tombstone, con lo status che lo dice:
il recupero e la classificazione la ignorano perche' filtrano `status: active`
dentro lo store, ma documento, vettore e testo sono ancora li'.

Questo modulo li toglie davvero. Per ora con un solo criterio, lo status, e
nessuna eccezione: tutto cio' che e' `superseded` o `deleted` esce dalla
collezione. Altri criteri verranno aggiunti qui.

Cosa tocca e cosa no
--------------------
Solo l'archivio. La core memory non contiene mai tombstone - e' un invariante:
un item che smette di essere attivo viene archiviato nello stesso momento - e
l'operation log resta com'e': le voci che nominano un id rimosso restano, con i
related_item_id che da qui in poi puntano a documenti che non esistono piu'.

Ogni documento rimosso aggiunge al log una voce `evict`, con il testo della
memoria come tutte le altre voci. Ha un costo, ed e' bene saperlo: il testo di un
fatto che l'utente ha chiesto di dimenticare ne guadagna una copia nel log, che
vive in RAM per tutta la vita del nodo e viaggia nella risposta del servizio.

Quando gira
-----------
Una volta per turno, alla fine di update_memory: dopo il consolidamento, che e'
dove nascono superseded e deleted, e prima che la callback risponda. Cosi' i
tombstone di un turno spariscono nello stesso turno, e non c'e' concorrenza con
la chiamata successiva sulla stessa collezione.

Non e' ancora collegato. Quando lo sara', la chiamata va in
memory_server.update_memory_callback dopo run_memory_agent e **prima** di
_fill_operation_log, passando agent.state["operation_log"]: le voci evict cadono
dopo l'offset fissato all'inizio del run, quindi last_operations() le pubblica
nella stessa risposta.

Prima di collegarlo
-------------------
Tre cose oggi presuppongono che i tombstone restino in archivio, e smetteranno
di valere:

    test_long_term_interaction._assert_everything_that_left_core_is_in_the_archive
        pretende che ogni item uscito dalla core memory, delete compresi, si
        ritrovi in archivio
    test_consolidation, passo 5
        verifica che l'item cancellato "resta in archivio come tombstone"
    il docstring di consolidation.py
        promette che status, lineage e timestamp non vanno mai persi

Un'avvertenza sul nome: quel docstring usa "eviction" in un altro senso, per
cio' che esce dalla core memory quando supera il budget - che oggi fa il core
split. Qui eviction vuol dire uscire dall'archivio.
"""

from typing import List, Tuple

from memory_service import backends
from memory_service.consolidation import OperationLogEntry

EVICTED_STATUSES = ("superseded", "deleted")


def _find_tombstones(store) -> List[Tuple[str, str]]:
    """(id, testo) dei documenti da togliere, nell'ordine di EVICTED_STATUSES.

    Una get per status con un filtro di uguaglianza, invece di una sola con
    `$in`: costa una lettura locale in piu' e funziona con qualunque versione di
    chromadb, mentre l'operatore non c'e' sempre stato.
    """
    found = []
    for status in EVICTED_STATUSES:
        result = store.get(where={"status": status}) or {}
        ids = result.get("ids") or []
        documents = result.get("documents") or [""] * len(ids)
        found.extend(zip(ids, documents))
    return found


def evict_archived_tombstones(log: List[OperationLogEntry]) -> List[str]:
    """Toglie dall'archivio le memorie superate e cancellate.

    Restituisce gli id rimossi, e per ciascuno aggiunge al log una voce `evict`.

    Non solleva mai: gira in coda alla callback di update_memory, e un'eccezione
    finirebbe nel suo except, che svuota la risposta anche quando il
    consolidamento e' andato a buon fine. Se lo store non risponde, stampa e non
    rimuove niente.

    Il log registra solo cio' che e' successo davvero: le voci si aggiungono dopo
    che la cancellazione e' riuscita, mai prima.
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
