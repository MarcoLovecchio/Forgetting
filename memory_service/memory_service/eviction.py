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

Le memorie con safety vera non escono mai: sono tolte dalle candidate prima di
ogni calcolo, senza uno score. La quota pero' resta calcolata su tutte le attive,
perche' e' quella che serve a rientrare nel limite.

Lo score
--------
Ogni termine e' una funzione a parte, con valori in [0, 1], alto vuol dire da
tenere. eviction_terms calcola i termini accesi, per nome, e combine_terms li
combina. time_decay e frequency hanno il loro switch in MemoryConfig, come
eviction; utility segue consolidation_utility. Un termine nuovo va aggiunto in
eviction_terms e nei pesi, e da li' arriva anche nel log.

    utility_term      il voto da 1 a 5 che il classificatore da' a ogni fatto
                      con consolidation_utility `single`, diviso per 5: da 0.2
                      a 1. Entra solo in quella modalita'.

    time_decay        il tempo da un timestamp, con una delle due curve secondo
                      eviction_time_decay_function; tutte e due valgono 1 al
                      momento del timestamp e occupano lo stesso posto nella
                      media:
                        weibull_decay_term  exp(-(t/scala)^forma), forma 0.5 e
                                            scala 1 ora: 1/e dopo un'ora;
                        power_decay_term    (1 + t/scala)^-d, d 0.5 e scala
                                            1 ora: 0.5 dopo 3 ore, 0.2 dopo
                                            un giorno, con la coda lunga della
                                            power law.
                      Da sole ordinano le memorie allo stesso modo, perche'
                      dipendono dallo stesso timestamp: la scelta conta quando
                      il decadimento e' mediato con altri termini. Il
                      timestamp e' updated_at o retrieved_at, secondo
                      eviction_time_decay_field, e se ne usa sempre uno solo.
                      Quando gli usi si contano, retrieved_at e' l'ultimo uso.

    frequency_term    un conteggio diviso per il piu' alto fra le attive: lineare
                      e relativo all'archivio, come LFU e N_visit di MemoryOS.
                      Il conteggio e' n_used quando gli usi si contano
                      (track_used, con generate_answer), n_retrieve altrimenti:
                      come per il time decay, se ne usa sempre uno solo. Entra
                      con eviction_frequency.

combine_terms segue eviction_combination:

    weighted_sum      0.5 utility + 0.25 time_decay + 0.25 frequency
    product           utility * (0.5 time_decay + 0.5 frequency)

I pesi si riproporzionano sui termini accesi: senza frequency la somma pesata e'
2/3 utility + 1/3 time_decay, il prodotto utility * time_decay. Senza utility le
due formule sono la media dei termini accesi.

Con tutti i termini spenti non esce niente. Le memorie escono tutte insieme: gli
score si calcolano una volta sola, sulle candidate di quel momento.

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
concorrenza con la chiamata successiva sulla stessa collezione. Da li' il router
exceed_archive_limit manda al nodo prune_archive solo se le attive superano il
limite, come gli altri due limiti del ramo. Il ramo retrieve non la esegue: non
produce tombstone e non aggiunge memorie. Nodi e router leggono gli switch dal
runtime context, la MemoryConfig che run_memory_agent passa a invoke.

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
from memory_service.consolidation import RATING_SCALES, OperationLogEntry

EVICTED_STATUSES = ("superseded", "deleted")

PRUNE_FRACTION = 0.10
DECAY_SHAPE = 0.5
DECAY_SCALE_HOURS = 1
POWER_EXPONENT = 0.5
POWER_SCALE_HOURS = 1
WEIGHTED_SUM_WEIGHTS = {"utility": 0.5, "time_decay": 0.25, "frequency": 0.25}
PRODUCT_WEIGHTS = {"time_decay": 0.5, "frequency": 0.5}


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


def hours_since(timestamp: str, now: datetime) -> float:
    """Ore trascorse da timestamp, mai negative.

    Un timestamp nel futuro conta 0: succede davvero, perche' i timestamp sono ora
    locale e a fine ora legale l'orologio torna indietro.
    """
    elapsed = now - datetime.fromisoformat(timestamp)
    return max(0.0, elapsed.total_seconds() / 3600)


def weibull_decay_term(timestamp: str, now: datetime,
                       scale_hours: float = DECAY_SCALE_HOURS,
                       shape: float = DECAY_SHAPE) -> float:
    """Weibull sul tempo trascorso da timestamp: 1 al momento, 1/e a scale_hours."""
    return math.exp(-((hours_since(timestamp, now) / scale_hours) ** shape))


def power_decay_term(timestamp: str, now: datetime,
                     scale_hours: float = POWER_SCALE_HOURS,
                     exponent: float = POWER_EXPONENT) -> float:
    """Power law sul tempo trascorso da timestamp: (1 + t/scale_hours)^-exponent, 1 al momento."""
    return (1 + hours_since(timestamp, now) / scale_hours) ** -exponent


DECAY_FUNCTIONS = {"weibull": weibull_decay_term, "power": power_decay_term}


def frequency_field(config: MemoryConfig) -> str:
    """Il conteggio del termine di frequenza: n_used quando gli usi si contano."""
    return "n_used" if config.counts_used else "n_retrieve"


def frequency_term(count: int, count_max: int) -> float:
    """Il conteggio di una memoria rispetto al piu' alto: count / count_max.

    Se il piu' alto e' 0, nessuna e' mai stata contata: vale 0 per tutte.
    """
    if count_max == 0:
        return 0.0
    return count / count_max


def utility_term(utility: int) -> float:
    """Il voto del classificatore da 0.2 a 1: utility / 5."""
    return utility / RATING_SCALES["utility"]


def eviction_terms(metadata: dict, config: MemoryConfig, now: datetime,
                   frequency_max: Optional[int] = None) -> Dict[str, float]:
    """I termini accesi, per nome: sono anche i sotto-score che il log riporta.

    `frequency_max`, il conteggio piu' alto fra le attive sul campo di
    frequency_field, dipende dalle altre memorie, quindi lo calcola chi le ha
    tutte. Serve solo con eviction_frequency.
    """
    terms = {}
    if config.consolidation_utility == "single":
        terms["utility"] = utility_term(metadata["utility"])
    if config.eviction_time_decay:
        decay = DECAY_FUNCTIONS[config.eviction_time_decay_function]
        terms["time_decay"] = decay(metadata[config.eviction_time_decay_field], now)
    if config.eviction_frequency:
        terms["frequency"] = frequency_term(metadata[frequency_field(config)], frequency_max)
    return terms


def weighted_mean(terms: Dict[str, float], weights: Dict[str, float]) -> float:
    """Media dei termini con i loro pesi, riproporzionati su quelli presenti."""
    total = sum(weights[name] for name in terms)
    return sum(weights[name] * value for name, value in terms.items()) / total


def combine_terms(terms: Dict[str, float], combination: str) -> Optional[float]:
    """Lo score dai termini, in [0, 1]: None se non ce n'e' nessuno.

    Senza utility le due combinazioni sono la stessa media.
    """
    if not terms:
        return None
    if combination == "product" and "utility" in terms:
        rest = {name: value for name, value in terms.items() if name != "utility"}
        return terms["utility"] * (weighted_mean(rest, PRODUCT_WEIGHTS) if rest else 1)
    return weighted_mean(terms, WEIGHTED_SUM_WEIGHTS)


def eviction_score(metadata: dict, config: MemoryConfig, now: datetime) -> Optional[float]:
    """Lo score di una memoria: il piu' basso esce per primo. None se nessun termine e' acceso."""
    return combine_terms(eviction_terms(metadata, config, now), config.eviction_combination)


def prune_count(active: int, fraction: float = PRUNE_FRACTION) -> int:
    """Quante memorie escono su `active`: la frazione, arrotondata per eccesso.
    """
    return math.ceil(active * fraction)


def prune_archive(log: List[OperationLogEntry], config: MemoryConfig,
                  now: Optional[datetime] = None) -> List[str]:
    """Toglie le memorie attive con lo score piu' basso.

    Ne toglie prune_count delle attive, tutte insieme, scegliendole fra quelle
    senza safety: le protette restano fuori prima di ogni calcolo, anche dal
    conteggio piu' alto della frequenza. Con tutti i termini spenti non ce n'e'
    nessuna con uno score, e non esce niente. Restituisce gli id rimossi, e per
    ciascuno aggiunge al log una voce `prune` con lo score e i termini. Come
    evict_archived_tombstones non solleva se lo store fallisce, e il log registra
    solo le rimozioni riuscite.

    Non guarda il limite: se l'archivio lo supera lo decide il grafo, con
    archive_over_limit, prima di arrivare qui. Il filtro `status: active` e' lo
    stesso insieme su cui quella conta le memorie.
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
    candidates = [(doc_id, content, metadata)
                  for doc_id, content, metadata in zip(ids, documents, metadatas)
                  if not metadata.get("safety")]
    frequency_max = (max((metadata[frequency_field(config)] for _, _, metadata in candidates),
                         default=0)
                     if config.eviction_frequency else None)
    now = now or datetime.now()

    scored = []
    for doc_id, content, metadata in candidates:
        terms = eviction_terms(metadata, config, now, frequency_max)
        score = combine_terms(terms, config.eviction_combination)
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
