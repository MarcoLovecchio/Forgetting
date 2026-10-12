"""FAMA: quanto la memoria ricorda, e quanto dimentica davvero.

La metrica viene da Memora (Uddin et al., *From Recall to Forgetting:
Benchmarking Long-Term Memory for Personalized Agents*, ACL 2026,
arXiv:2604.20006). Per ogni domanda di valutazione si scrivono due famiglie di
criteri binari:

    memory_presence     l'informazione valida che la risposta DEVE contenere
    forgetting_absence  l'informazione superata che NON deve contenere

e da quelle si ricava un punteggio unico:

    MPA  = frazione di criteri di presenza soddisfatti
    FAA  = frazione di criteri di assenza soddisfatti
    lam  = N_forget / (N_presence + N_forget)
    FAMA = max(0, MPA - lam * (1 - FAA))

Il peso `lam` fa in modo che una domanda tutta di dimenticanza sia giudicata
tutta sulla dimenticanza: con N_presence = 0 il conto degenera in FAMA = FAA, e
con N_forget = 0 in FAMA = MPA. Il `max` tiene il punteggio in [0, 1].

Perche' serve, qui
------------------
FAMA nasce come metrica **sulla risposta**, e la risposta e' l'ultimo anello di
una catena lunga: consolidation scrive lo store, retrieval decide se interrogare
l'archivio, generate_answer compone. Un numero solo su quella catena non dice
mai di chi e' la colpa - e questo test l'ha gia' visto succedere, con il delete
dell'indirizzo consolidato correttamente e la risposta che diceva lo stesso dove
abitava.

Per questo qui i criteri si applicano **due volte**, agli stessi criteri e con
la stessa formula, ma a due oggetti diversi:

    FAMA-risposta   sul testo composto da generate_answer
    FAMA-memoria    sulle memorie attive, appena consolidato il messaggio che
                    precede la domanda (memory_snapshot_turn)

Il secondo e' quello che parla della consolidation, ed e' deterministico: lo
store e' strutturato, `status` distingue gia' l'attivo dal cancellato, non serve
nessun giudice. Lo scarto fra i due isola il colpevole:

    memoria alta + risposta alta   tutto a posto
    memoria alta + risposta bassa  colpa di generate_answer o del retrieval
    memoria bassa + risposta bassa colpa della consolidation
    memoria bassa + risposta alta  se la memoria ha mancato un fatto, la
                                   risposta veniva dal contesto e non dalla
                                   memoria; se se l'e' tenuto dopo un delete,
                                   e' di nuovo consolidation, con una risposta
                                   fortunata che la prossima volta non lo sara'

Come si giudica un criterio
---------------------------
Memora usa tre giudici LLM a maggioranza. Qui no: i criteri sono scritti su
valori discreti (2200, Torino, golden retriever) e il match testuale basta,
costa zero chiamate e non aggiunge varianza a una misura che serve proprio a
misurare la varianza.

Il caso che il match testuale non prende da solo e' il valore vecchio nominato
come vecchio: *"non bevi piu' caffe'"* e' la risposta giusta alla domanda sul
caffe', ma la parola "caffe'" c'e'. Da qui la distinzione fra i due modi di
scrivere un criterio di dimenticanza:

    forgets()     dopo un delete: il valore non deve comparire, punto.
                  La memoria non c'e' piu', nominarla e' gia' una perdita.
    supersedes()  dopo un update o una contraddizione: il valore puo' comparire
                  solo come superato - nella sua proposizione ("non piu' X",
                  "prima X"), oppure in una frase che lo ritratta altrove o
                  che nomina il valore che l'ha sostituito ("mangi pesce dopo
                  sei anni di vegetarianesimo").

La differenza non e' arbitraria, e' la semantica delle due operazioni: una
cancellazione non deve lasciare traccia, un aggiornamento puo' essere raccontato
come cambiamento.

Due lingue, non una
-------------------
Le due colonne leggono testi in lingue diverse, e non per scelta: la risposta
segue la lingua dell'utente ed e' italiana, le memorie consolidate seguono la
regola del campo `fact` e sono inglesi. Quindi ogni criterio porta le varianti
di entrambe le lingue - "pesce" e "fish", "padre" e "father", "corr*" e "run*" -
e cosi' fanno i marcatori di superamento, che senza l'inglese leggerebbero "The
user stopped drinking coffee" come una fuga invece che come una negazione.

Affiancare invece di duplicare e' obbligatorio: un criterio e' un fatto, non una
stringa. Sdoppiarlo in due criteri raddoppierebbe il denominatore di MPA e
darebbe 50% a una risposta perfetta ma monolingue. Nella forma giusta
l'operazione va nella direzione giusta da entrambi i lati - sulla presenza
allarga, perche' basta una variante qualsiasi; sull'assenza stringe, perche' un
valore da dimenticare non si salva traducendosi.

Dove le due lingue condividono la radice basta una voce: "vegetarian*" prende
"vegetariana" e "vegetarian", "clinic*" prende "clinica" e "clinic", "eliminat*"
prende "eliminato" ed "eliminated".

Resta una distorsione, ed e' bene saperla: il controllo di superamento guarda la
frase del valore, quindi una frase che si contraddice da sola ("bevi caffe' la
mattina, ma hai eliminato il caffe'") passa, e il valore vecchio nominato in una
frase tutta sua, senza marcatori, viene contato come errore.

Fedelta' al riferimento
-----------------------
Il conto e' quello di `fama_score` e di `overall_metrics` in
evals/model_eval/model_based_evaluator.py. Coincidono: la formula per domanda,
lam, il taglio a zero, FAA = 1 quando non ci sono criteri di assenza, il
punteggio complessivo come media per domanda moltiplicata per cento, e MPA e FAA
complessivi come micro-medie sui criteri invece che medie per domanda.

Tre cose non coincidono, e sono tutte dichiarate:

1. **Il giudice.** Memora usa GPT-4.1, Claude Haiku 4.5 e Gemini 2.5 Flash a
   maggioranza su criteri in linguaggio naturale. Qui il match e' testuale su
   varianti scritte a mano. Stessa formula, criteri giudicati diversamente.

2. **MPA su un insieme vuoto.** Il loro codice lo mette a 0, e siccome li' lam
   vale 1 quelle domande valgono 0 comunque vada. Nel loro dataset il ramo non
   gira mai; qui girerebbe nove volte su 56, bloccando il totale sotto gli 84
   punti a prescindere. Vale 1 per vacuita', come dice la prosa del paper.
   Il conto e' in `score`, il perche' pure.

3. **I tipi di task.** Memora divide in Remembering, Reasoning e Recommending e
   riporta un punteggio per ciascuno. Qui quasi tutte le domande sono
   Remembering, e dieci sono compiti di tipo Recommending ("mi consigli
   cosa cucinare per cena?"), giudicati con gli stessi criteri: cosa la
   risposta e la memoria devono contenere. Il numero resta uno solo.

Quindi il numero e' calcolato come nel paper, ma **non e' confrontabile con le
sue tabelle**: dati diversi, una persona invece di dieci, 56 domande
invece di centocinquanta, e un altro giudice. E' un numero per confrontare
questa architettura con se stessa fra una run e l'altra.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Normalizzazione e match
# --------------------------------------------------------------------------- #

# La conversazione e' scritta in ASCII con l'apostrofo ("caffe'", "e'"), il
# modello risponde con gli accenti veri ("caffe'" -> "caffeè"). Senza
# normalizzare, meta' dei criteri non aggancerebbe niente per un motivo
# tipografico.
_COMBINING = "Mn"
_SEPARATORS = ("'", "’", "ʼ", "`", "-", "‐", "–", "—")

# Quanto puo' essere largo il buco di una variante con il '*' in mezzo:
# "non*piu" deve prendere "non fai piu'" senza arrivare alla frase dopo.
_GAP = 24

# Su cosa si spezza una proposizione. La virgola c'e' apposta: i marcatori di
# superamento si cercano nella proposizione del valore, e senza la virgola
# "bevi caffe', non te'" passerebbe per una negazione del caffe'.
_CLAUSE_BREAK = re.compile(r"[.!?;:,\n]")

# Su cosa si spezza una frase: mai oltre un a capo, che nella memoria separa due
# fatti diversi.
_SENTENCE_BREAK = re.compile(r"[.!?\n]")

# Il valore compare, ma negato: la memoria e' stata cancellata o smentita.
# I marcatori portano il proprio '*': stelletta dove serve la famiglia di
# parole ("eliminat*"), parola intera dove il prefisso pescherebbe troppo ("non"
# senza stelletta, altrimenti prende "nonna"; "no" inglese, altrimenti prende
# "not", "now", "north" e mezzo dizionario).
#
# Sono in due lingue perche' le due colonne del resoconto leggono testi in
# lingue diverse: la risposta segue la lingua dell'utente ed e' italiana, le
# memorie consolidate seguono la regola del campo `fact` e sono inglesi. Con i
# soli marcatori italiani, "The user stopped drinking coffee" verrebbe contata
# come una fuga invece che come una negazione, e la colonna della memoria
# crollerebbe per motivi lessicali proprio dove la consolidation funziona.
_NEGATION = (
    # italiano
    "non", "mai", "nessun*", "niente", "nulla", "senza", "smesso", "smettere",
    "eliminat*", "tolt*", "rimoss*", "dimenticat*", "abbandonat*",
    "rinunciat*", "esclus*",
    # inglese ("cancel*" copre anche "cancellato", "eliminat*" anche
    # "eliminated": dove le due lingue condividono la radice basta una voce)
    "no", "not", "never", "none", "nothing", "without", "anymore", "longer",
    "stop*", "quit*", "remov*", "delet*", "cancel*", "forgot*", "forget*",
    "drop*", "abandon*", "exclud*",
    # contrazioni: l'apostrofo diventa spazio in `normalize`, quindi "doesn't"
    # arriva qui come "doesn t" e la parola intera da agganciare e' "doesn"
    "doesn", "don", "didn", "isn", "wasn", "aren",
)

# Il valore compare, ma come valore vecchio: c'e' stato un cambiamento.
# Fuori di proposito i marcatori del presente - "ora", "adesso", "now": stanno
# nella proposizione del valore NUOVO, non di quello vecchio, e ammetterli
# lascerebbe passare "ora vive a Mondello" come se fosse un superamento.
_TRANSITION = (
    # italiano
    "piu", "prima", "era", "erano", "ero", "eri", "invece", "sostituit*",
    "cambiat*", "aggiornat*", "modificat*", "passat*", "precedent*",
    "trasferit*", "vecchi*", "fino", "allora", "inizialmente", "originariamente",
    # inglese
    "previously", "was", "were", "instead", "replac*", "chang*", "updat*",
    "moved", "relocat*", "former", "formerly", "old", "until", "used",
    "initially", "originally",
)

_SUPERSEDED_MARKERS = _NEGATION + _TRANSITION


def normalize(text) -> str:
    """Minuscolo, senza accenti, senza apostrofi e trattini, spazi compattati.

    Serve a far coincidere le grafie che girano in questo test: "caffe'" della
    conversazione, "caffè" della risposta, "Caffe" di una memoria consolidata,
    "part-time" e "part time".

    Gli a capo restano: nella memoria separano un fatto dall'altro, e
    _CLAUSE_BREAK ci spezza sopra. Comprimerli come gli altri spazi fondeva lo
    store in una proposizione sola, e un "no longer" in una memoria qualsiasi
    faceva passare per superato ogni valore dell'archivio.
    """
    lowered = unicodedata.normalize("NFD", str(text or "").lower())
    stripped = "".join(ch for ch in lowered if unicodedata.category(ch) != _COMBINING)
    for separator in _SEPARATORS:
        stripped = stripped.replace(separator, " ")
    lines = (re.sub(r"\s+", " ", line).strip() for line in stripped.split("\n"))
    return "\n".join(line for line in lines if line)


def _pattern(variant: str) -> str:
    """Regex di una variante. Il '*' significa due cose, a seconda di dove sta.

    In fondo e' un prefisso: "corr*" prende correre, corri, corro senza doverli
    elencare. In mezzo e' un buco: "non*piu" prende "non fai piu'" e "non lo
    suono piu'", che sono lo stesso modo di dire con parole in mezzo. Senza il
    '*' la variante e' una parola intera.
    """
    variant = normalize(variant)
    open_end = variant.endswith("*")
    parts = [piece.strip() for piece in variant.split("*") if piece.strip()]
    if not parts:
        return r"(?!x)x"  # una variante fatta di soli '*' non aggancia niente
    body = (r".{0," + str(_GAP) + r"}?").join(re.escape(piece) for piece in parts)
    return r"\b" + body + ("" if open_end else r"\b")


def _occurrences(haystack: str, variants: Sequence[str]) -> List[int]:
    """Le posizioni in cui una qualunque delle varianti compare nel testo."""
    found = []
    for variant in variants:
        found.extend(match.start() for match in re.finditer(_pattern(variant), haystack))
    return sorted(found)


def _clause_around(text: str, position: int) -> str:
    """La proposizione che contiene quella posizione."""
    start = 0
    for match in _CLAUSE_BREAK.finditer(text, 0, position):
        start = match.end()
    end = _CLAUSE_BREAK.search(text, position)
    return text[start:end.start() if end else len(text)]


def _sentence_around(text: str, position: int) -> Tuple[int, int]:
    """Inizio e fine della frase che contiene quella posizione."""
    start = 0
    for match in _SENTENCE_BREAK.finditer(text, 0, position):
        start = match.end()
    end = _SENTENCE_BREAK.search(text, position)
    return start, end.start() if end else len(text)


# Una proposizione che segue il valore e comincia cosi' lo sta correggendo...
_ADVERSATIVE = ("ma", "pero", "tuttavia", "eppure", "but", "however", "though", "yet")

# ...purche' dica che e' finito: una negazione semplice non basta, "bevi caffe',
# ma non alcolici" nega un'altra cosa.
_CESSATION = ("piu", "smesso", "smettere", "eliminat*", "tolt*", "abbandonat*",
              "rinunciat*", "anymore", "longer", "stop*", "quit*", "gave up")


def _has_marker(clause: str, markers: Sequence[str]) -> bool:
    return any(re.search(_pattern(marker), clause) for marker in markers)


def _following_clause(text: str, position: int) -> str:
    """La proposizione dopo quella del valore, se sta nella stessa frase.

    Mai oltre un punto o un a capo: nella memoria ogni riga e' un fatto diverso.
    """
    end = _CLAUSE_BREAK.search(text, position)
    if end is None or end.group() not in ",;:":
        return ""
    after = _CLAUSE_BREAK.search(text, end.end())
    return text[end.end():after.start() if after else len(text)].strip()


def _is_superseded(text: str, position: int) -> bool:
    """Il valore in quella posizione e' nominato come superato o negato?

    Nella sua proposizione, oppure nella successiva se questa lo corregge:
    "bevi solo caffe' la mattina, ma ora non lo prendi piu'" dice che il caffe'
    e' finito, con la negazione dopo la virgola e un pronome al posto del nome.
    """
    if _has_marker(_clause_around(text, position), _SUPERSEDED_MARKERS):
        return True
    following = _following_clause(text, position)
    return (following.split(" ", 1)[0] in _ADVERSATIVE
            and _has_marker(following, _CESSATION))


# --------------------------------------------------------------------------- #
# I criteri
# --------------------------------------------------------------------------- #

PRESENCE = "memory_presence"
FORGETTING = "forgetting_absence"


@dataclass(frozen=True)
class Criterion:
    """Un criterio binario, nella forma dei due `evaluation_type` di Memora."""

    label: str
    variants: Tuple[str, ...]
    kind: str
    superseded_ok: bool = False
    # Solo per recalls_some: i fatti fra cui scegliere, ognuno con le sue
    # varianti, e quanti ne devono comparire.
    groups: Tuple[Tuple[str, ...], ...] = ()
    minimum: int = 1
    # Solo per supersedes, e solo dove il valore nuovo esclude il vecchio (2200 al
    # posto di 2000): chitarra e pianoforte convivevano, non si sostituiscono.
    replaced_by: Tuple[str, ...] = ()

    def satisfied(self, text: str) -> bool:
        """Il criterio e' rispettato da questo testo?"""
        haystack = normalize(text)
        if self.kind == PRESENCE and self.groups:
            found = sum(1 for group in self.groups if _occurrences(haystack, group))
            return found >= self.minimum
        hits = _occurrences(haystack, self.variants)
        if self.kind == PRESENCE:
            return bool(hits)
        if not self.superseded_ok:
            return not hits
        # Una menzione non superata nella sua proposizione passa se la stessa frase
        # ritratta il valore altrove, o nomina il valore che l'ha sostituito.
        retracted = [position for position in hits if _is_superseded(haystack, position)]
        narrated = retracted + _occurrences(haystack, self.replaced_by)
        for position in hits:
            if position in retracted:
                continue
            start, end = _sentence_around(haystack, position)
            if not any(start <= other < end for other in narrated):
                return False
        return True


def recalls(label: str, *variants: str) -> Criterion:
    """Criterio di presenza: questa informazione ci deve essere."""
    return Criterion(label, tuple(variants) or (label,), PRESENCE)


def recalls_some(label: str, minimum: int, *facts) -> Criterion:
    """Criterio di presenza per le domande aperte: almeno `minimum` di questi fatti.

    Ogni fatto e' una variante o una tupla di varianti, e conta una volta sola
    anche se compare in due lingue. generate_answer risponde in una o due frasi:
    a "cosa sai della mia alimentazione?" sceglie lui quali fatti dire, e un
    criterio che pretende quelli precisi misura la scelta, non la memoria.
    """
    groups = tuple((fact,) if isinstance(fact, str) else tuple(fact) for fact in facts)
    variants = tuple(variant for group in groups for variant in group)
    return Criterion(label, variants, PRESENCE, groups=groups, minimum=minimum)


def forgets(label: str, *variants: str) -> Criterion:
    """Criterio di assenza dopo un delete: il valore non deve comparire.

    Nemmeno negato. La memoria e' stata cancellata su richiesta dell'utente:
    rispondere "non abiti piu' a Mondello" rivela esattamente quello che
    doveva sparire.
    """
    return Criterion(label, tuple(variants) or (label,), FORGETTING, superseded_ok=False)


def supersedes(label: str, *variants: str, replaced_by: Sequence[str] = ()) -> Criterion:
    """Criterio di assenza dopo un update o una contraddizione.

    Il valore vecchio puo' comparire, ma solo come vecchio: raccontare il
    cambiamento e' legittimo, riproporre il valore superato come attuale no.
    """
    return Criterion(label, tuple(variants) or (label,), FORGETTING, superseded_ok=True,
                     replaced_by=tuple(replaced_by))


@dataclass(frozen=True)
class EvaluationQuestion:
    """Una domanda della conversazione con i suoi criteri."""

    index: int
    note: str
    criteria: Tuple[Criterion, ...]

    @property
    def presence(self) -> List[Criterion]:
        """I criteri di presenza."""
        return [item for item in self.criteria if item.kind == PRESENCE]

    @property
    def forgetting(self) -> List[Criterion]:
        """I criteri di assenza."""
        return [item for item in self.criteria if item.kind == FORGETTING]


# --------------------------------------------------------------------------- #
# Il punteggio
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Score:
    """L'esito di una domanda: le tre quantita', i conti e cosa ha ceduto."""

    mpa: float
    faa: float
    lam: float
    fama: float
    n_presence: int
    n_forgetting: int
    missed: Tuple[str, ...]
    leaked: Tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Nessun criterio violato."""
        return not self.missed and not self.leaked

    @property
    def presence_satisfied(self) -> int:
        """Quanti criteri di presenza sono stati soddisfatti."""
        return self.n_presence - len(self.missed)

    @property
    def forgetting_satisfied(self) -> int:
        """Quanti criteri di assenza sono stati soddisfatti."""
        return self.n_forgetting - len(self.leaked)


def score(question: EvaluationQuestion, text: str) -> Score:
    """FAMA di una domanda contro un testo: la risposta, o le memorie attive.

    Ricalca `fama_score` di evals/model_eval/model_based_evaluator.py, con una
    deviazione dichiarata sul caso senza criteri di presenza - vedi sotto.
    """
    presence, forgetting = question.presence, question.forgetting
    n_p, n_f = len(presence), len(forgetting)

    missed = tuple(item.label for item in presence if not item.satisfied(text))
    leaked = tuple(item.label for item in forgetting if not item.satisfied(text))

    if not n_p and not n_f:
        # Come il riferimento: una domanda senza criteri non vale niente,
        # invece di valere tutto per vacuita'.
        return Score(0.0, 0.0, 0.0, 0.0, 0, 0, missed, leaked)

    # DEVIAZIONE DICHIARATA. Il codice di Memora mette MPA a 0 quando non ci
    # sono criteri di presenza, e siccome li' lam vale 1, quelle domande
    # valgono 0 comunque vada. Nel loro dataset non succede mai - le domande
    # sono a lista ("quali task restano?") e qualcosa resta sempre - quindi e'
    # un ramo difensivo che non gira. Qui invece succede nove volte su
    # 56: dopo una cancellazione non c'e' nessun valore valido da
    # ricordare, e "non lo so" e' la risposta giusta. Con MPA a 0 quelle nove
    # domande sarebbero incorreggibili e il totale bloccato sotto gli 84 punti,
    # qualunque cosa faccia l'architettura.
    #
    # Qui MPA vale 1 per vacuita', che e' la lettura della prosa del paper:
    # "the fraction of memory presence criteria satisfied" su un insieme vuoto,
    # con lam = 1 che consegna tutto il peso a FAA. FAMA degenera in FAA, ed e'
    # esattamente cio' che quelle domande devono misurare.
    mpa = 1.0 - len(missed) / n_p if n_p else 1.0
    faa = 1.0 - len(leaked) / n_f if n_f else 1.0
    lam = n_f / (n_p + n_f)

    return Score(mpa, faa, lam, max(0.0, mpa - lam * (1.0 - faa)),
                 n_p, n_f, missed, leaked)


def aggregate(scores: Sequence[Score]) -> Dict[str, float]:
    """Il punteggio dell'architettura, non della singola domanda.

    Segue `overall_metrics` del riferimento, e le tre quantita' non si mediano
    tutte allo stesso modo:

        fama      media delle FAMA per domanda, per cento. Ogni domanda pesa
                  uguale, che e' il senso di un punteggio gia' normalizzato.
        mpa, faa  rapporto fra criteri soddisfatti e criteri totali su tutte le
                  domande insieme (micro-media): li' a pesare e' il criterio,
                  non la domanda, cosi' una domanda con sette criteri conta per
                  sette e non per uno.
        accuracy  gli stessi criteri, presenza e assenza in un mucchio solo.
                  E' l'`overall_accuracy` del riferimento: la frazione grezza di
                  criteri passati, senza il peso lam.

    Mediare anche mpa e faa per domanda darebbe numeri diversi e piu' gentili:
    una domanda con un solo criterio sbagliato peserebbe quanto una con sette
    criteri giusti.
    """
    empty = {"questions": 0, "criteria": 0, "fama": 0.0,
             "mpa": None, "faa": None, "accuracy": 0.0}
    if not scores:
        return empty

    presence_total = sum(item.n_presence for item in scores)
    forgetting_total = sum(item.n_forgetting for item in scores)
    presence_ok = sum(item.presence_satisfied for item in scores)
    forgetting_ok = sum(item.forgetting_satisfied for item in scores)
    criteria = presence_total + forgetting_total

    return {
        "questions": len(scores),
        "criteria": criteria,
        "fama": 100.0 * sum(item.fama for item in scores) / len(scores),
        "mpa": 100.0 * presence_ok / presence_total if presence_total else None,
        "faa": 100.0 * forgetting_ok / forgetting_total if forgetting_total else None,
        "accuracy": 100.0 * (presence_ok + forgetting_ok) / criteria if criteria else 0.0,
    }


def verdict(memory: Score, answer: Score, tolerance: float = 0.01) -> str:
    """Di chi e' la colpa, leggendo insieme il punteggio sui due oggetti.

    E' il quadrante della tabella nel docstring del modulo, tradotto in una
    parola sola da mettere in fondo alla riga del resoconto.

    Il quadrante "memoria sbagliata, risposta giusta" pero' non e' uno solo, e
    confonderlo costa una diagnosi: se la memoria ha **mancato** un fatto e la
    risposta l'ha detto lo stesso, la risposta veniva dal contesto della
    conversazione e non dalla memoria; se invece la memoria si e' **tenuta** un
    fatto cancellato e la risposta non l'ha usato, lo store e' sbagliato e la
    risposta e' stata fortunata - e quella e' consolidation, non fortuna da
    festeggiare. La prossima domanda sullo stesso fatto la perde.
    """
    memory_ok = memory.fama >= 1.0 - tolerance
    answer_ok = answer.fama >= 1.0 - tolerance
    if memory_ok and answer_ok:
        return "ok"
    if memory_ok:
        return "risposta"
    if not answer_ok:
        return "consolidation"
    return "consolidation" if memory.leaked else "fuori memoria"


def memory_snapshot_turn(question: int, maximum_historical_messages: int) -> int:
    """Dopo l'insert di quale turno si fotografa la memoria per questa domanda.

    Non al momento della domanda. Il consolidamento e' in ritardo per
    costruzione: summarize_memories_node tiene gli ultimi
    maximum_historical_messages messaggi e consolida quelli prima, e ogni turno
    ne aggiunge due (la domanda e la risposta di generate_answer). Il messaggio
    subito prima della domanda - quasi sempre l'update o il delete che la domanda
    mette alla prova - al momento della domanda e' ancora solo nella
    conversazione, e la memoria lo "manca" per definizione. Misurarla li'
    misurava il ritardo, non la consolidation.

    Con la finestra del test lungo (2 messaggi) e' l'insert del turno stesso
    della domanda, che consolida il messaggio prima e non ancora la domanda.
    """
    return question + max(1, maximum_historical_messages // 2) - 1


# --------------------------------------------------------------------------- #
# Le 56 domande
# --------------------------------------------------------------------------- #

QUESTIONS: Dict[int, EvaluationQuestion] = {

    36: EvaluationQuestion(
        36, "i fatti della presentazione, niente e' ancora superato",
        (
            recalls_some(
                "almeno due fatti sull'alimentazione", 2,
                "vegetarian*", ("arachid*", "peanut*"), "2000", "protein*",
                ("alcol*", "alcohol*"), ("caffe", "coffee"), ("te verde", "green tea"),
                ("parmigian*", "parmesan"), ("liquirizia", "licorice", "liquorice"),
                ("glutine", "gluten", "celiac*"), ("lattosio", "lactose"),
            ),
        ),
    ),

    39: EvaluationQuestion(
        39, "msg 38 ha aggiornato le calorie da 2000 a 2200",
        (
            recalls("2200", "2200"),
            supersedes("2000", "2000", replaced_by=("2200",)),
        ),
    ),

    42: EvaluationQuestion(
        42, "msg 41 ha ripetuto l'asma del msg 9",
        (
            recalls("asma", "asma", "asthma*"),
        ),
    ),

    46: EvaluationQuestion(
        46, "msg 45 ha ripetuto il nome",
        (
            recalls("bianca"),
        ),
    ),

    50: EvaluationQuestion(
        50, "compito: un film, e gli horror del msg 49 le mettono ansia",
        (
            recalls("niente horror", "horror*", "orror*"),
        ),
    ),

    58: EvaluationQuestion(
        58, "msg 57 ha smentito la dieta vegetariana: ora mangia pesce",
        (
            recalls("mangia pesce", "pesce", "pescetarian*", "fish", "pescatarian*"),
            supersedes("vegetarian*",
                       replaced_by=("pesce", "fish", "pescatarian*", "pescetarian*")),
        ),
    ),

    60: EvaluationQuestion(
        60, "compito: una cena, con il pesce del msg 57 e l'allergia alle arachidi",
        (
            recalls("mangia pesce", "pesce", "fish", "pescatarian*", "pescetarian*"),
            recalls("niente arachidi", "arachid*", "peanut*"),
            supersedes("vegetarian*",
                       replaced_by=("pesce", "fish", "pescatarian*", "pescetarian*")),
        ),
    ),

    65: EvaluationQuestion(
        65, "msg 64 ha cancellato l'indirizzo: Palermo centro, poi Mondello",
        (
            forgets("indirizzo cancellato", "mondello", "palermo",
                    "centro", "centre", "center", "downtown"),
        ),
    ),

    67: EvaluationQuestion(
        67, "msg 66 ha portato la levotiroxina a 88 microgrammi",
        (
            recalls("88 microgrammi", "88"),
        ),
    ),

    71: EvaluationQuestion(
        71, "msg 70 ha eliminato il caffe': nessuna bevanda nuova al suo posto",
        (
            supersedes("caffe' della mattina", "caffe", "coffee"),
        ),
    ),

    73: EvaluationQuestion(
        73, "msg 72: allergica anche alla penicillina",
        (
            recalls("penicillina", "penicillin*"),
        ),
    ),

    79: EvaluationQuestion(
        79, "msg 78: ad agosto vorrebbe andare in Grecia, forse a Naxos",
        (
            recalls("Naxos o la Grecia", "naxos", "grecia", "greece"),
        ),
    ),

    81: EvaluationQuestion(
        81, "compito: arrivare a Naxos con la paura di volare del msg 75",
        (
            recalls("niente aereo", "volare", "fly*", "flight*", "aere*", "plane*",
                    "nave", "traghett*", "ferry*", "boat*"),
        ),
    ),

    84: EvaluationQuestion(
        84, "msg 83 ha cancellato l'eta': prima 29, poi 30",
        (
            # Mai il numero da solo: "6:30" diventa "6 30", e la sveglia passava per l'eta'.
            forgets("eta' cancellata", "30 anni", "29 anni", "30 year*", "29 year*",
                    "is 30", "is 29", "turned 30", "turned 29", "aged 30", "age of 30",
                    "trent anni", "trentanni", "ventinove anni",
                    "thirty year*", "twenty nine year*"),
        ),
    ),

    90: EvaluationQuestion(
        90, "msg 89 ha spostato la sorella da Milano a Torino",
        (
            recalls("vive a Torino", "torino", "turin"),
            supersedes("vecchia citta' Milano", "milano", "milan",
                       replaced_by=("torino", "turin")),
        ),
    ),

    92: EvaluationQuestion(
        92, "msg 85 e 91: il corso di fotografia del lunedi' sera",
        (
            recalls("fotografia", "fotograf*", "photograph*"),
        ),
    ),

    101: EvaluationQuestion(
        101, "msg 95: Giulia si sposa il 14 giugno",
        (
            recalls("14 giugno", "14 giugno", "14 june", "june 14*", "14th of june",
                    "14th june"),
        ),
    ),

    105: EvaluationQuestion(
        105, "msg 104 ha cancellato il nome della sorella",
        (
            forgets("nome cancellato", "chiara"),
        ),
    ),

    108: EvaluationQuestion(
        108, "msg 102: volontariato in un canile il martedi' sera",
        (
            recalls("volontariato", "volontari*", "volunteer*", "canile", "shelter*"),
        ),
    ),

    111: EvaluationQuestion(
        111, "msg 110 ha esteso l'allergia all'olio di arachidi, msg 72 ha aggiunto la "
             "penicillina",
        (
            recalls("arachidi", "arachid*", "peanut*"),
            recalls("penicillina", "penicillin*"),
        ),
    ),

    114: EvaluationQuestion(
        114, "msg 99: si e' ricreduta sulla liquirizia, ora le piace",
        (
            recalls("liquirizia", "liquirizia", "licorice", "liquorice"),
            recalls("le piace", "piace", "like*", "enjoy*", "ricredut*"),
        ),
    ),

    118: EvaluationQuestion(
        118, "msg 117 ha smentito la razza: non labrador ma golden retriever",
        (
            recalls("golden retriever", "golden"),
            supersedes("vecchia razza labrador", "labrador", replaced_by=("golden",)),
        ),
    ),

    119: EvaluationQuestion(
        119, "msg 18: Argo non puo' mangiare il pollo",
        (
            recalls("pollo", "pollo", "chicken"),
        ),
    ),

    123: EvaluationQuestion(
        123, "compito: lo zaino per la montagna, con l'asma del msg 9",
        (
            recalls("inalatore", "inalator*", "inhaler*", "asma", "asthma*"),
        ),
    ),

    127: EvaluationQuestion(
        127, "msg 126 ha smentito lo yoga: smesso a settembre",
        (
            recalls("ha smesso", "smesso", "settembre", "non*piu", "interrott*",
                    "abbandonat*", "stop*", "quit*", "september", "no longer",
                    "anymore", "gave up"),
            supersedes("giorni dello yoga", "sabato", "mercoledi",
                       "saturday", "wednesday"),
        ),
    ),

    131: EvaluationQuestion(
        131, "compito: la spesa del giovedi', con il budget del msg 130",
        (
            recalls("budget di 50 euro", "50 eur*"),
        ),
    ),

    133: EvaluationQuestion(
        133, "msg 120: il richiamo del vaccino di Argo il 20 ottobre",
        (
            recalls("20 ottobre", "20 ottobre", "20 october", "october 20*",
                    "20th of october", "20th october"),
        ),
    ),

    137: EvaluationQuestion(
        137, "msg 128: dentista martedi' prossimo alle tre del pomeriggio",
        (
            recalls("martedi'", "martedi", "tuesday"),
            recalls("alle tre", "alle tre", "tre del pomeriggio", "15", "3 pm", "3pm",
                    "three in the afternoon", "3 in the afternoon"),
        ),
    ),

    140: EvaluationQuestion(
        140, "msg 139 ha cancellato il lavoro: clinica, poi part-time, poi in proprio",
        (

            forgets("lavoro cancellato", "fisioterapist*", "physiotherap*",
                    "physical therap*", "clinic*", "in proprio", "part time",
                    "self employed", "own practice", "freelance*"),
        ),
    ),

    148: EvaluationQuestion(
        148, "msg 147 ha chiuso col pianoforte: restano chitarra e vinili jazz",
        (
            recalls("chitarra", "chitarra", "guitar*"),
            recalls("jazz in vinile", "jazz", "vinil*", "vinyl*"),
            supersedes("pianoforte abbandonato", "pianoforte", "piano"),
        ),
    ),

    151: EvaluationQuestion(
        151, "compito: a chi lasciare Argo, dal msg 142",
        (
            recalls("la vicina", "vicin*", "neighbo*"),
        ),
    ),

    154: EvaluationQuestion(
        154, "msg 153 ha portato le calorie a 1800, dopo 2000 e 2200",
        (
            recalls("obiettivo 1800", "1800"),
            supersedes("obiettivo 2200", "2200", replaced_by=("1800",)),
            supersedes("obiettivo 2000", "2000", replaced_by=("1800",)),
        ),
    ),

    157: EvaluationQuestion(
        157, "msg 156 ha portato la levotiroxina a 100 microgrammi, dopo gli 88 del msg 66",
        (
            recalls("levotiroxina", "levotiroxin*", "levothyroxin*", "tiroide",
                    "thyroid*"),
            recalls("100 microgrammi", "100"),
            supersedes("88 microgrammi", "88", replaced_by=("100",)),
        ),
    ),

    169: EvaluationQuestion(
        169, "msg 168 ha cancellato la celiachia: prima la madre, poi il padre",
        (
            forgets("familiare celiaco cancellato", "padre", "madre", "papa",
                    "mamma", "father", "mother", "dad", "mom", "mum", "parent*"),
        ),
    ),

    171: EvaluationQuestion(
        171, "msg 25: la migliore amica e' Giulia",
        (
            recalls("giulia"),
        ),
    ),

    172: EvaluationQuestion(
        172, "msg 53 e 158: lo spagnolo, ora al livello B1",
        (
            recalls("spagnolo", "spagnol*", "spanish"),
        ),
    ),

    177: EvaluationQuestion(
        177, "msg 176 ha ripreso la corsa; il nuoto e' a tre volte, lo yoga e' finito",
        (
            recalls("corsa ripresa", "corr*", "corsa", "run*", "jog*"),
            recalls("nuoto", "nuot*", "swim*"),
            supersedes("yoga interrotto", "yoga"),
        ),
    ),

    179: EvaluationQuestion(
        179, "compito: la settimana sportiva, con il pilates del msg 178 e la corsa ripresa al "
             "msg 176",
        (
            recalls_some("almeno due sport", 2, ("corr*", "corsa", "run*", "jog*"),
                         ("nuot*", "swim*"), "pilates"),
            supersedes("yoga interrotto", "yoga"),
        ),
    ),

    181: EvaluationQuestion(
        181, "msg 180: ai pomodori e al basilico del msg 31 si aggiungono i peperoncini",
        (
            recalls_some("almeno due piante", 2, ("pomodor*", "tomato*"),
                         ("basilic*", "basil"), ("peperoncin*", "chili*", "chilli*")),
        ),
    ),

    183: EvaluationQuestion(
        183, "msg 182: ha smesso di meditare",
        (
            recalls("meditazione", "medit*"),
            recalls("ha smesso", "smesso", "stop*", "no longer", "non*piu", "anymore",
                    "quit*", "gave up"),
        ),
    ),

    188: EvaluationQuestion(
        188, "msg 187 ha cancellato il volontariato al canile",
        (
            forgets("volontariato cancellato", "volontari*", "volunteer*", "canile",
                    "shelter*", "kennel*"),
        ),
    ),

    192: EvaluationQuestion(
        192, "compito: il lago senza patente (msg 6 e 191)",
        (
            recalls("niente patente", "patente", "licen*"),
        ),
    ),

    194: EvaluationQuestion(
        194, "msg 193: ha superato la paura di volare",
        (
            recalls("volare", "volare", "fly*", "flight*", "aere*", "plane*"),
            recalls("superata", "superat*", "overc*", "got over"),
        ),
    ),

    198: EvaluationQuestion(
        198, "msg 197 ha cancellato i familiari, non gli animali",
        (
            recalls("il cane Argo", "argo"),
            recalls("il gatto Milo", "milo"),
        ),
    ),

    200: EvaluationQuestion(
        200, "msg 199: operata agli occhi, niente piu' occhiali",
        (
            recalls("operata agli occhi", "operat*", "surgery", "intervent*", "laser"),
            supersedes("occhiali", "occhiali", "glasses", "eyeglasses"),
        ),
    ),

    201: EvaluationQuestion(
        201, "msg 3: il compleanno e' il 12 marzo",
        (
            recalls("12 marzo", "12 marzo", "12 march", "march 12*", "12th of march",
                    "12th march"),
        ),
    ),

    203: EvaluationQuestion(
        203, "msg 202 ha cancellato la data del compleanno",
        (
            forgets("compleanno cancellato", "12 marzo", "12 march", "march 12*",
                    "marzo", "march"),
        ),
    ),

    206: EvaluationQuestion(
        206, "msg 205: finito il corso di lingua dei segni",
        (
            recalls("lingua dei segni", "lingua dei segni", "segni", "sign language"),
            recalls("finito", "finit*", "conclus*", "finished", "completed",
                    "primo livello", "first level"),
        ),
    ),

    208: EvaluationQuestion(
        208, "msg 51 ha portato le proteine da 60 a 75 grammi",
        (
            recalls("75 grammi", "75"),
            supersedes("60 grammi", "60", replaced_by=("75",)),
        ),
    ),

    210: EvaluationQuestion(
        210, "msg 209 ha confermato i due litri d'acqua del msg 68",
        (
            recalls("due litri", "due litri", "2 litri", "two lit*", "2 lit*", "2l"),
        ),
    ),

    213: EvaluationQuestion(
        213, "msg 212 ha cancellato l'orto sul balcone",
        (
            forgets("orto cancellato", "orto", "pomodor*", "tomato*", "basilic*",
                    "basil", "peperoncin*", "chili*", "chilli*", "vegetable garden",
                    "vegetable patch"),
        ),
    ),

    217: EvaluationQuestion(
        217, "msg 216: niente Naxos, ad agosto il Portogallo",
        (
            recalls("portogallo", "portogallo", "portugal"),
            supersedes("Naxos e la Grecia", "naxos", "grecia", "greece",
                       replaced_by=("portogallo", "portugal")),
        ),
    ),

    219: EvaluationQuestion(
        219, "compito: una cena, con la trattoria preferita del msg 218",
        (
            recalls("la trattoria di Cefalu'", "cefalu", "trattoria"),
        ),
    ),

    230: EvaluationQuestion(
        230, "msg 229: niente sci in Trentino, a Natale resta a casa",
        (
            recalls("resta a casa", "resto a casa", "resti a casa", "restare a casa",
                    "rimani a casa", "stay at home", "stays at home", "staying at home",
                    "stay home", "stays home", "staying home"),
            supersedes("sci in Trentino", "sciare", "sci", "ski", "skiing", "trentino"),
        ),
    ),

    231: EvaluationQuestion(
        231, "msg 161: lo scrittore preferito e' Camilleri",
        (
            recalls("camilleri"),
        ),
    ),

    232: EvaluationQuestion(
        232, "il quadro d'insieme: cosa e' sopravvissuto e cosa doveva sparire",
        (
            recalls_some(
                "almeno due fatti sopravvissuti", 2,
                "bianca", ("arachid*", "peanut*"), "argo", "milo",
                ("pesce", "fish", "pescatarian*", "pescetarian*"), "protein*",
                ("te verde", "green tea"), ("nuot*", "swim*"), ("parmigian*", "parmesan"),
                ("chitarra", "guitar*"), ("jazz", "vinil*", "vinyl*"),
                ("grecia", "greece", "naxos"), ("montagna", "mountain*"),
                ("acqua", "water"), ("lingua dei segni", "sign language"),
                ("bici*", "bike*", "bicycle*"),
            ),
            forgets("nome della sorella", "chiara"),
            forgets("lavoro", "fisioterapist*", "physiotherap*",
                    "physical therap*", "clinic*"),
            forgets("indirizzo", "mondello", "palermo"),
        ),
    ),
}

