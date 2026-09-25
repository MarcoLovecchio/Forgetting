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
tenere. eviction_terms calcola i termini accesi, per nome, e combine_terms ne fa
la media; ogni termine ha il suo switch in MemoryConfig, come eviction. Un
termine nuovo va aggiunto in eviction_terms, e da li' arriva anche nel log.

    time_decay_term   Weibull sul tempo da un timestamp: exp(-(t/scala)^forma),
                      forma 0.5 e scala 7 giorni. Vale 1 al momento del
                      timestamp, 1/e dopo 7 giorni. Il timestamp e'
                      updated_at o retrieved_at, secondo
                      eviction_time_decay_field: occupano lo stesso posto nella
                      media, e se ne usa sempre uno solo.

    retrieval_count_term
                      n_retrieve / n_max, con n_max il massimo fra le attive:
                      lineare e relativo all'archivio, come LFU e N_visit di
                      MemoryOS. Non ancora collegato a eviction_score.

Con tutti i termini spenti non esce niente.

Cosa tocca e cosa no
--------------------
Solo l'archivio. La core memory non contiene mai tombstone - e' un invariante:
un item che smette di essere attivo viene archiviato nello stesso momento - e
l'operation log resta com'e': le voci che nominano un id rimosso restano, con i
related_item_id che da qui in poi puntano a documenti che non esistono piu'.

Ogni documento rimosso aggiunge al log una voce, con il testo della memoria come
tutte le altre: `evict` per un tombstone, `prune` per una memoria attiva, con lo
score in `score` e i termini che lo compongono in `score_terms`. Ha un costo, ed e' bene saperlo: il testo di un fatto che
l'utente ha chiesto di dimenticare ne guadagna una copia nel log, che vive in RAM
per tutta la vita del nodo e viaggia nella risposta del servizio.

Quando gira
-----------
Solo se MemoryConfig.eviction e' acceso: si cambia solo nel codice, nessuna
variabile d'ambiente lo legge.

Una volta per insert, nel nodo evict_archive del grafo, in cui confluiscono tutte
le uscite del ramo insert: dopo il consolidamento, che e' dove nascono superseded
e deleted. Cosi' i tombstone di un turno spariscono nello stesso turno, e non c'e'
concorrenza con la chiamata successiva sulla stessa collezione. Il ramo retrieve
non la esegue: non produce tombstone e non aggiunge memorie. Il nodo legge gli
switch dal runtime context, la MemoryConfig che run_memory_agent passa a invoke.

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
from typing import Dict, List, Optional, Tuple

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

    Non solleva mai: gira nell'ultimo nodo del ramo insert, dentro la callback di
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


def time_decay_term(timestamp: str, now: datetime,
                    scale_hours: float = DECAY_SCALE_HOURS,
                    shape: float = DECAY_SHAPE) -> float:
    """Weibull sul tempo trascorso da timestamp: 1 al momento, 1/e a scale_hours.

    Un timestamp nel futuro vale 1, mai piu' di 1: succede davvero, perche' i
    timestamp sono ora locale e a fine ora legale l'orologio torna indietro.
    """
    elapsed = now - datetime.fromisoformat(timestamp)
    hours = max(0.0, elapsed.total_seconds() / 3600)
    return math.exp(-((hours / scale_hours) ** shape))


def retrieval_count_term(n_retrieve: int, n_max: int) -> float:
    """Recuperi di una memoria rispetto alla piu' recuperata: n_retrieve / n_max.

    Se nessuna e' mai stata recuperata (n_max = 0) vale 0 per tutte.
    """
    if n_max == 0:
        return 0.0
    return n_retrieve / n_max


def eviction_terms(metadata: dict, config: MemoryConfig, now: datetime) -> Dict[str, float]:
    """I termini accesi, per nome: sono anche i sotto-score che il log riporta."""
    terms = {}
    if config.eviction_time_decay:
        terms["time_decay"] = time_decay_term(metadata[config.eviction_time_decay_field], now)
    return terms


def combine_terms(terms: Dict[str, float]) -> Optional[float]:
    """Media dei termini, in [0, 1]: None se non ce n'e' nessuno."""
    if not terms:
        return None
    return sum(terms.values()) / len(terms)


def eviction_score(metadata: dict, config: MemoryConfig, now: datetime) -> Optional[float]:
    """Lo score di una memoria: il piu' basso esce per primo. None se nessun termine e' acceso."""
    return combine_terms(eviction_terms(metadata, config, now))


def prune_count(active: int, fraction: float = PRUNE_FRACTION) -> int:
    """Quante memorie escono su `active`: la frazione, arrotondata per eccesso.
    """
    return math.ceil(active * fraction)


def prune_archive(log: List[OperationLogEntry], limit: int, config: MemoryConfig,
                  now: Optional[datetime] = None) -> List[str]:
    """Oltre il limite, toglie le memorie attive con lo score piu' basso.

    Ne toglie prune_count delle attive; con tutti i termini spenti non ce n'e'
    nessuna con uno score, e non esce niente. Restituisce gli id rimossi, e per
    ciascuno aggiunge al log una voce `prune` con lo score. Come
    evict_archived_tombstones non solleva se lo store fallisce, e il log registra
    solo le rimozioni riuscite.

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
        terms = eviction_terms(metadata, config, now)
        score = combine_terms(terms)
        if score is not None:
            scored.append((score, doc_id, content, terms))
    targets = sorted(scored, key=lambda target: target[:2])[:prune_count(len(ids))]
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
