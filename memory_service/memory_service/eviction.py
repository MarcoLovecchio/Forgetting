"""Eviction: cosa esce dall'archivio per sempre.

Consolidation non cancella mai niente. Una memoria superata o dimenticata smette
di essere attiva e resta in archivio come tombstone, con lo status che lo dice:
il recupero e la classificazione la ignorano perche' filtrano `status: active`
dentro lo store, ma documento, vettore e testo sono ancora li'.

Questo modulo li toglie davvero, con due criteri.

Il primo e' lo status, senza eccezioni: tutto cio' che e' `superseded` o
`deleted` esce dalla collezione.

Il secondo e' il limite dell'archivio, archive_memory_limit
(MEMORY_ARCHIVE_LIMIT): un numero di memorie attive che non blocca nessun
inserimento e si controlla alla fine del turno, dopo i tombstone. Se e'
superato, esce il PRUNE_FRACTION delle memorie attive - arrotondato per eccesso -
con lo score piu' basso. Se resta superato, al turno dopo ne esce un'altra quota.

Lo score
--------
Ogni termine e' una funzione a parte, con valori in [0, 1], alto vuol dire da
tenere. eviction_score fa la media dei termini accesi; ogni termine ha il suo
switch in MemoryConfig, come eviction.

    time_decay_term   Weibull sul tempo da updated_at: exp(-(t/scala)^forma),
                      forma 0.5 e scala 7 giorni. Vale 1 appena toccata, 1/e a
                      7 giorni.

Una memoria di cui un termine acceso non si puo' calcolare - updated_at mancante
o illeggibile - non e' candidata: un dato mancante non fa mai togliere quella
memoria. Conta pero' fra le attive su cui si calcola la quota, che ricade sulle
altre. Con tutti i termini spenti non esce niente.

Cosa tocca e cosa no
--------------------
Solo l'archivio. La core memory non contiene mai tombstone - e' un invariante:
un item che smette di essere attivo viene archiviato nello stesso momento - e
l'operation log resta com'e': le voci che nominano un id rimosso restano, con i
related_item_id che da qui in poi puntano a documenti che non esistono piu'.

Ogni documento rimosso aggiunge al log una voce, con il testo della memoria come
tutte le altre: `evict` per un tombstone, `prune` con il suo score per una
memoria attiva. Ha un costo, ed e' bene saperlo: il testo di un fatto che
l'utente ha chiesto di dimenticare ne guadagna una copia nel log, che vive in RAM
per tutta la vita del nodo e viaggia nella risposta del servizio.

Quando gira
-----------
Solo se MemoryConfig.eviction e' acceso: e' spento di default e si cambia solo
nel codice, nessuna variabile d'ambiente lo legge.

Una volta per insert, in fondo a MemoryAgent.run_memory_agent, prima del return:
dopo il consolidamento, che e' dove nascono superseded e deleted. Cosi' i
tombstone di un turno spariscono nello stesso turno, e non c'e' concorrenza con
la chiamata successiva sulla stessa collezione. Il ramo retrieve non la esegue:
non produce tombstone e non aggiunge memorie.

Le voci evict e prune cadono dopo l'offset fissato all'inizio del run, quindi
last_operations() le pubblica nella stessa risposta di update_memory.

Con l'eviction accesa
---------------------
I tombstone smettono di restare in archivio, e chi lo presupponeva si regola
cosi':

    test_long_term_interaction
        ogni item uscito dalla core memory e' in archivio oppure ha una voce
        evict o prune nel log
    test_consolidation
        gira con l'eviction spenta: il passo 5 guarda il tombstone, il 7 che il
        recupero lo escluda

Un'avvertenza sul nome: il docstring di consolidation.py usa "eviction" in un
altro senso, per cio' che esce dalla core memory quando supera il budget - che
oggi fa il core split. Qui eviction vuol dire uscire dall'archivio.
"""

import math
from datetime import datetime
from typing import List, Optional, Tuple

from memory_service import backends
from memory_service.config import MemoryConfig
from memory_service.consolidation import OperationLogEntry

EVICTED_STATUSES = ("superseded", "deleted")

PRUNE_FRACTION = 0.10
DECAY_SHAPE = 0.5
DECAY_SCALE_HOURS = 7 * 24


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

    Non solleva mai: gira in coda a run_memory_agent, dentro la callback di
    update_memory, e un'eccezione finirebbe nel suo except, che svuota la
    risposta anche quando il consolidamento e' andato a buon fine. Se lo store
    non risponde, stampa e non rimuove niente.

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


def archive_over_limit(limit: int) -> int:
    """Di quante memorie attive l'archivio supera il limite: 0 se lo rispetta.

    Conta solo le attive, perche' i tombstone escono comunque per status. Non
    rimuove niente. Non solleva mai: se lo store non risponde stampa e
    restituisce 0, cosi' un conteggio mancato non fa togliere memorie.
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


def time_decay_term(updated_at: Optional[str], now: datetime,
                    scale_hours: float = DECAY_SCALE_HOURS,
                    shape: float = DECAY_SHAPE) -> Optional[float]:
    """Weibull sul tempo trascorso da updated_at: 1 appena toccata, 1/e a scale_hours.

    None se updated_at manca o non si legge, anche quando ha un fuso orario e now
    no: i due datetime non si sottraggono. Un updated_at nel futuro - un orologio
    spostato - vale 1, mai piu' di 1.
    """
    try:
        elapsed = now - datetime.fromisoformat(updated_at)
    except (TypeError, ValueError):
        return None
    hours = max(0.0, elapsed.total_seconds() / 3600)
    return math.exp(-((hours / scale_hours) ** shape))


def eviction_score(metadata: dict, config: MemoryConfig, now: datetime) -> Optional[float]:
    """Media dei termini accesi, in [0, 1]: lo score piu' basso esce per primo.

    None se nessun termine e' acceso, o se uno di quelli accesi non si calcola.
    """
    terms = []
    if config.eviction_time_decay:
        terms.append(time_decay_term(metadata.get("updated_at"), now))
    if not terms or None in terms:
        return None
    return sum(terms) / len(terms)


def prune_count(active: int, fraction: float = PRUNE_FRACTION) -> int:
    """Quante memorie escono su `active`: la frazione, arrotondata per eccesso.

    Il round prima del ceil serve se si cambia la frazione: 100 * 0.07 in virgola
    mobile fa 7.000000000000001, e senza ne uscirebbero 8.
    """
    return math.ceil(round(active * fraction, 9))


def prune_archive(log: List[OperationLogEntry], limit: int, config: MemoryConfig,
                  now: Optional[datetime] = None) -> List[str]:
    """Oltre il limite, toglie le memorie attive con lo score piu' basso.

    Ne toglie prune_count delle attive, scelte fra quelle che hanno uno score.
    Restituisce gli id rimossi, e per ciascuno aggiunge al log una voce `prune`
    con lo score. Come evict_archived_tombstones non solleva mai, e il log
    registra solo le rimozioni riuscite.

    Presuppone i tombstone gia' rimossi da evict_archived_tombstones, che nel
    memory manager gira subito prima. Il filtro `status: active` resta perche' e'
    lo stesso insieme su cui archive_over_limit conta le memorie.
    """
    if archive_over_limit(limit) == 0:
        return []

    try:
        store = backends.get_vector_store()
        result = store.get(where={"status": "active"}) or {}
    except Exception as error:
        print(f"\tPruning skipped, archive lookup failed: {error}")
        return []

    ids = result.get("ids") or []
    documents = result.get("documents") or [""] * len(ids)
    metadatas = result.get("metadatas") or [{}] * len(ids)
    now = now or datetime.now()

    scored = []
    for doc_id, content, metadata in zip(ids, documents, metadatas):
        score = eviction_score(metadata or {}, config, now)
        if score is not None:
            scored.append((score, doc_id, content))
    targets = sorted(scored)[:prune_count(len(ids))]
    if not targets:
        return []

    try:
        store.delete(ids=[doc_id for _, doc_id, _ in targets])
    except Exception as error:
        print(f"\tPruning failed, nothing removed: {error}")
        return []

    print(f"\tPruned from the archive: "
          f"{[(doc_id, round(score, 3)) for score, doc_id, _ in targets]}")
    for score, doc_id, content in targets:
        log.append(OperationLogEntry(op_type="prune", item_id=doc_id, content=content,
                                     score=score))
    return [doc_id for _, doc_id, _ in targets]
