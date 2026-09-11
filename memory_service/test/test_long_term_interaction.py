"""Simulazione di una interazione a lungo termine sullo stack reale.

A differenza di test_memory_llm.py, qui **non ci sono turni mirati**: c'e' una
conversazione lunga, realistica e disordinata, iniettata **un messaggio alla
volta**, e ogni messaggio ripercorre la pipeline vera dell'architettura.

Nella pipeline nessuno "decide" fra insert e retrieve: sono due fasi della
stessa interazione, e avvengono sempre entrambe.

    1. intent_recognition.listener_callback  ->  send_get_request()
       un get_memory su OGNI messaggio, per procurarsi il contesto
    2. explainability.query_explanation_callback
       genera la spiegazione usando quel contesto
    3. explainability                        ->  send_update_request(input, spiegazione)
       un update_memory su OGNI spiegazione prodotta

Il test fa le stesse tre cose per ogni messaggio. Il passo 2 in produzione sta
fuori dal memory service (explainability ha un suo LLM): qui e' una chiamata
diretta al modello con lo stesso ruolo, cosi' c'e' una risposta vera da
consolidare e da verificare.

I primi 25 messaggi sono la presentazione: l'utente si racconta e sono tutti
fatti nuovi. Da li' in poi **e' tutto mescolato**, come in una sessione vera: una
ripetizione, poi un raffinamento, poi due chiacchiere, poi una contraddizione,
poi una cancellazione. Nessun blocco per tipo di operazione.

L'unico vincolo rispettato e' la causalita': un fatto viene introdotto prima di
poter essere ripetuto, raffinato, smentito o cancellato. Le catene si sviluppano
a distanza, come succederebbe davvero - per esempio le calorie:

    msg 7    "il mio obiettivo giornaliero e' 2000 calorie"     (new)
    msg 28   "in realta' l'obiettivo ora e' 2200"               (update)
    msg 82   "ho cambiato obiettivo: 1800, voglio dimagrire"    (contradict)
    msg 93   "non memorizzare piu' quante calorie punto"        (delete)

Ogni messaggio porta in commento il caso che dovrebbe provocare. Le chiacchiere
sparse servono a riempire la core memory fino a far scattare lo split verso
l'archivio.

Sparse fra i fatti ci sono anche **domande**. Non ricevono un trattamento
diverso - anche loro passano da retrieve e poi da insert, come tutto il resto -
ma sono i punti in cui la risposta dice qualcosa: sono piazzate subito dopo un
update, una contraddizione o una cancellazione, e sono le uniche che il resoconto
riporta per esteso:

    "in realta' l'obiettivo ora e' 2200"        (update)
    "qual e' il mio obiettivo calorico?"        -> deve dire 2200, non 2000

    "dimentica il mio indirizzo"                (delete)
    "sai dirmi dove abito?"                     -> non deve piu' saperlo

Nel resoconto si vede, per ogni domanda, cosa ha risposto l'assistente e se
l'archivio e' stato consultato per quel giro. La risposta e' quella del servizio,
composta da generate_answer sulla core memory piu' l'eventuale recupero: il test
non ne simula piu' una propria.

La domanda corrente viene passata al servizio (GetMemory.user_input ->
run_memory_agent(query=...)), quindi la ricerca in archivio parte da quello che
l'utente sta chiedendo adesso e non dall'ultimo messaggio gia' in memoria.

Durante l'esecuzione l'output dell'agente viene soppresso: si stampa solo una
riga di avanzamento ogni 10 messaggi. **Alla fine** viene stampato il resoconto:
operation log completo, core memory, archivio, e domande con le relative
risposte.

Le diciassette domande sono anche i punti di valutazione di **FAMA** (Memora,
arXiv:2604.20006): ognuna porta in fama.py dei criteri binari - cosa la risposta
deve contenere, cosa non deve piu' contenere - e il resoconto ne ricava un
punteggio. Lo calcola due volte, con gli stessi criteri: una sulla risposta, che
e' la metrica del paper e misura tutta la catena, e una sulle memorie attive una
volta consolidato il messaggio che precede la domanda, che non passa da nessun
modello e misura la sola consolidation. Lo scarto fra le due dice quale nodo ha ceduto.

Le assert sono strutturali (invarianti che devono valere qualunque cosa decida il
modello), non sul contenuto: con un LLM vero pretendere una classificazione
esatta renderebbe il test inutilizzabile. **Anche FAMA e' misurata e stampata,
mai asserita**: con diciassette domande e temperature 1.0 una soglia
trasformerebbe la varianza del campionamento in un test che fallisce a caso.

Requisiti: LLM_CONFIG ed EMBEDDING_CONFIG in .config, e i due modelli
raggiungibili all'indirizzo indicato da MEMORY_LLM_BASE_URL /
MEMORY_EMBEDDING_BASE_URL nel .env. Se qualcosa non risponde il test si salta
spiegando cosa.

Attenzione: ogni messaggio costa 4-6 chiamate reali al modello (sufficienza,
eventuale retrieval, risposta della retrieve, risposta dell'assistente,
consolidamento, eventuale split). Su 117 messaggi sono diverse centinaia di
chiamate: conviene partire con MEMORY_LONGRUN_MESSAGES basso. Con i modelli
sul cluster il limite non e' la quota ma la latenza, quindi il tempo dipende
dal carico delle GPU su cui girano.

Esecuzione:
    pytest memory_service/test/test_long_term_interaction.py -v -s

Variabili utili:
    MEMORY_LONGRUN_MESSAGES      quanti messaggi iniettare (default: tutti e 117)
    MEMORY_LONGRUN_DELAY         secondi di pausa fra un messaggio e l'altro
                                 (default 0). Sul cluster non ci sono rate limit
                                 da rispettare, ma serve se l'endpoint e'
                                 condiviso con altri carichi
    MEMORY_LONGRUN_CHROMA_PATH   dove tenere l'archivio (default: quello del
                                 servizio, ripulibile con reset_archive.py, in
                                 una collezione sua)
    MEMORY_CORE_MEMORY_LIMIT     limite di caratteri della core memory (1500)
"""

import ast
import contextlib
import dataclasses
import io
import os
import sys
import time

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import AIMessage  # noqa: E402

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import get_active_items  # noqa: E402

import fama  # noqa: E402
from live_model import live_stack_unavailable  # noqa: E402
from snapshot import SEPARATOR, safe_print  # noqa: E402


pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_CONFIG"),
    reason="LLM_CONFIG is required to run the long term interaction test",
)

PROGRESS_EVERY = 10
ID_WIDTH = 8


# --------------------------------------------------------------------------- #
# La conversazione: 117 messaggi dell'utente (100 fatti e 17 domande)
#
# Quali siano le diciassette lo dice fama.QUESTIONS, non la punteggiatura:
# l'ultimo messaggio chiede un riassunto senza punto interrogativo, ed e' una
# domanda come le altre.
# --------------------------------------------------------------------------- #

CONVERSATION = [
    # --- presentazione: l'utente si racconta, sono tutti fatti nuovi ------- #
    "Ciao, mi chiamo Bianca.",
    "Ho 29 anni.",
    "Vivo a Palermo, in centro.",
    "Lavoro come fisioterapista in una clinica privata.",
    "Sono vegetariana da sei anni.",
    "Sono allergica alle arachidi, e' un'allergia seria.",
    "Il mio obiettivo giornaliero e' 2000 calorie.",
    "Punto a 60 grammi di proteine al giorno.",
    "Non bevo alcolici.",
    "Bevo il caffe' solo la mattina.",
    "Nel pomeriggio preferisco il te' verde.",
    "Ho un cane che si chiama Argo.",
    "Argo e' un labrador di quattro anni.",
    "Vado a correre tre volte a settimana.",
    "Corro la mattina presto, prima del lavoro.",
    "Il mio piatto preferito e' la parmigiana di melanzane.",
    "Odio il gusto della liquirizia.",
    "Mia sorella si chiama Chiara e vive a Milano.",
    "Mia madre e' celiaca, in casa cuciniamo senza glutine.",
    "Suono il pianoforte da quando avevo otto anni.",
    "Leggo soprattutto romanzi storici.",
    "Vado a dormire verso le undici di sera.",
    "Mi sveglio alle sei e mezza.",
    "Sono intollerante al lattosio, ma solo leggermente.",
    "Faccio yoga il sabato mattina.",

    "Oggi al lavoro ho avuto una giornata lunga, sette pazienti.",  # chiacchiere
    "Che cosa sai della mia alimentazione?",  # DOMANDA: fatti della presentazione
    "Ti ripeto che sono allergica alle arachidi.",  # redundant: allergia
    "In realta' l'obiettivo ora e' 2200 calorie, ho aumentato.",  # update: calorie
    "Qual e' il mio obiettivo calorico giornaliero?",  # DOMANDA: 2200, non 2000
    "Ho comprato una bicicletta nuova, una gravel grigia.",  # new
    "Come ti dicevo, mi chiamo Bianca.",  # redundant: nome
    "Argo ha compiuto cinque anni la settimana scorsa.",  # update: eta' del cane
    "Non sopporto i film dell'orrore, mi mettono ansia.",  # new
    "Le proteine le ho portate a 75 grammi al giorno.",  # update: proteine
    "Il caffe' lo prendo la mattina, mai dopo.",  # redundant: caffe'
    "Mi sono trasferita: sempre Palermo, ma ora vivo a Mondello.",  # update: casa
    "Sto seguendo un corso serale di lingua dei segni.",  # new
    "Non sono piu' vegetariana: da questo mese mangio pesce.",  # contradict: dieta
    "Che tipo di dieta seguo adesso?",  # DOMANDA: deve riflettere la contraddizione
    "Il mio colore preferito e' il verde bottiglia.",  # new
    "Corro quattro volte a settimana adesso, non piu' tre.",  # update: corsa
    "Mia sorella Chiara e' sempre a Milano.",  # redundant: sorella
    "Dimentica il mio indirizzo, non voglio che resti in memoria.",  # delete: casa
    "Sai dirmi dove abito?",  # DOMANDA: dopo il delete non deve saperlo
    "Bevo almeno due litri d'acqua al giorno.",  # new
    "Ho compiuto 30 anni il mese scorso.",  # update: eta'
    "Il caffe' l'ho eliminato del tutto, mi agitava troppo.",  # contradict: caffe'
    "Cosa bevo di solito la mattina?",  # DOMANDA: non deve piu' dire caffe'
    "Ho una collezione di dischi in vinile, soprattutto jazz.",  # new
    "Confermo: niente alcolici per me.",  # redundant: alcol
    "Suono il pianoforte e da un anno anche la chitarra.",  # update: musica
    "Ad agosto vorrei andare in Grecia, forse a Naxos.",  # new
    "Ho smesso di correre, mi da' fastidio il ginocchio.",  # contradict: corsa
    "Cancella l'informazione sulla mia eta', non memorizzarla.",  # delete: eta'
    "Quanti anni ho?",  # DOMANDA: dopo il delete non deve saperlo
    "Il sabato mattina c'e' yoga, come ogni settimana.",  # redundant: yoga
    "Ora nuoto invece di correre, due volte a settimana.",  # new
    "Chiara si e' trasferita da Milano a Torino.",  # update: sorella
    "Dove vive mia sorella?",  # DOMANDA: deve dire Torino, non Milano
    "In inverno soffro molto il freddo alle mani.",  # new
    "Ho ricominciato a bere vino, un bicchiere a cena.",  # contradict: alcol
    "Faccio la fisioterapista, come ti ho raccontato.",  # redundant: lavoro
    "Mi sono ricreduta sulla liquirizia, adesso mi piace.",  # contradict: liquirizia
    "Il mio numero fortunato e' il sette.",  # new
    "Yoga il sabato e da poco anche il mercoledi' sera.",  # update: yoga
    "Non conservare il nome di mia sorella, eliminalo.",  # delete: sorella
    "Come si chiama mia sorella?",  # DOMANDA: dopo il delete non deve saperlo
    "Lavoro in clinica, ma ora solo part-time il pomeriggio.",  # update: lavoro
    "Ho una cicatrice sul ginocchio destro da bambina.",  # new
    "L'allergia vale anche per l'olio di arachidi, sto attenta.",  # update: allergia
    "Non sono intollerante al lattosio, gli esami erano sbagliati.",  # contradict
    "Mi piace cucinare, ma solo nel weekend quando ho tempo.",  # new
    "Mi sveglio alle sei adesso, mezz'ora prima.",  # update: sveglia
    "Argo non e' un labrador, e' un golden retriever.",  # contradict: razza
    "Di che razza e' il mio cane?",  # DOMANDA: deve dire golden retriever
    "Dimentica cosa ti ho detto sul vino.",  # delete: vino
    "Preferisco la montagna al mare, anche se vivo sulla costa.",  # new
    "Leggo romanzi storici e ultimamente anche saggi.",  # update: letture
    "Non faccio piu' yoga, ho smesso a settembre.",  # contradict: yoga
    "Faccio ancora yoga il sabato?",  # DOMANDA: deve dire di no
    "Ho l'abitudine di fare la spesa il giovedi' pomeriggio.",  # new
    "Il te' verde lo prendo anche a merenda.",  # update: te'
    "Non lavoro piu' in clinica, mi sono messa in proprio.",  # contradict: lavoro
    "Sto pensando a un secondo cane per fare compagnia ad Argo.",  # new
    "Vado a dormire molto piu' tardi ora, verso l'una.",  # contradict: sonno
    "Rimuovi il dato sul lavoro che faccio, e' privato.",  # delete: lavoro
    "Che lavoro faccio?",  # DOMANDA: dopo il delete non deve saperlo
    "Non uso i social network, li ho cancellati l'anno scorso.",  # new
    "La parmigiana la faccio senza formaggio adesso.",  # update: piatto
    "Ho chiuso con il pianoforte, non lo suono piu'.",  # contradict: musica
    "Che musica ascolto o suono?",  # DOMANDA: probabile recupero dall'archivio
    "Ricorda che mangio pesce ma non carne.",  # redundant: dieta
    "Ho cambiato obiettivo: 1800 calorie, voglio dimagrire.",  # contradict: calorie
    "Quante calorie dovrei assumere ogni giorno?",  # DOMANDA: deve dire 1800
    "Elimina quello che sai sui miei orari di sonno.",  # delete: sonno
    "Non e' mia madre a essere celiaca, e' mio padre.",  # contradict: celiachia
    "Il nuoto l'ho portato a tre volte a settimana.",  # update: nuoto
    "Ho adottato un gatto, si chiama Milo.",  # new
    "Non leggo piu' romanzi storici, ora solo gialli.",  # contradict: letture
    "Milo e' un soriano di due anni.",  # new
    "Cancella il riferimento a mio padre e alla celiachia.",  # delete: celiachia
    "C'e' qualcuno di celiaco nella mia famiglia?",  # DOMANDA: dopo il delete
    "Ti ricordo ancora che sono allergica alle arachidi.",  # redundant: allergia
    "In realta' Milo ha tre anni, mi sono sbagliata.",  # update: eta' del gatto
    "Ho ripreso a correre, il ginocchio sta meglio.",  # contradict: corsa
    "Che sport faccio in questo periodo?",  # DOMANDA: corsa ripresa piu' nuoto
    "Non memorizzare piu' quante calorie punto, per ora.",  # delete: calorie
    "Dimentica il mio colore preferito, non serve.",  # delete: colore
    "Corro sempre la mattina presto, come sempre.",  # redundant: orario corsa
    "Dimentica il fastidio al ginocchio di cui ti parlavo.",  # delete: ginocchio
    "Togli dalla memoria tutto quello che riguarda i familiari.",  # delete: famiglia
    "Quali animali ho in casa?",  # DOMANDA: Argo e Milo, probabile recupero
    "Il mio piatto preferito resta la parmigiana.",  # redundant: piatto
    "Non bevo piu' caffe' nemmeno la mattina, confermo.",  # redundant: caffe'
    "Punto sempre a 75 grammi di proteine.",  # redundant: proteine
    "Riassumi tutto quello che sai di me.",  # DOMANDA finale: quadro complessivo
]


# --------------------------------------------------------------------------- #
# Preparazione
# --------------------------------------------------------------------------- #

def _configure_langsmith():
    """Tracing opzionale, come nel resto dell'architettura."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    model_name = ast.literal_eval(os.getenv("LLM_CONFIG"))["memory_agent"]["model_name"]
    os.environ["LANGSMITH_PROJECT"] = f'MEMORY-LONGRUN:{model_name}'
    os.environ["LANGSMITH_TEST_SUITE"] = "Memory Service long term"


def _archived_documents(config):
    """Quanti documenti ci sono gia' nell'archivio, o None se non e' leggibile."""
    try:
        return backends.get_vector_store()._collection.count()
    except Exception:
        return None


def _build_agent():
    """Agente sullo stack vero, sull'archivio configurato.

    Usa lo stesso percorso del resto del servizio (MEMORY_CHROMA_PATH, o
    ./chroma_db) in una collezione sua, cosi' reset_archive.py lo ripulisce
    insieme agli altri. Il rovescio e' che le run si sommano: senza ripulire, la
    successiva parte con le memorie della precedente e le classificazioni
    cambiano senso. Il resoconto iniziale dice quanti documenti ha trovato.
    """
    from memory_service.memory_manager_llm import MemoryAgent

    chroma_path = os.getenv("MEMORY_LONGRUN_CHROMA_PATH")
    config = dataclasses.replace(
        MemoryConfig.from_environment(),
        maximum_historical_messages=2,  # ogni messaggio fa scattare il consolidamento
        # Fissata, non ereditata: senza la risposta del servizio questo test non
        # ha niente da misurare, e un MEMORY_GENERATE_ANSWER=false nell'ambiente
        # lo svuoterebbe in silenzio.
        generate_answer=True,
        core_memory_limit=int(os.getenv("MEMORY_CORE_MEMORY_LIMIT", "200")),
        collection_name=os.getenv("MEMORY_LONGRUN_COLLECTION", "longterm_test_archive"),
        **({"chroma_path": os.path.abspath(chroma_path)} if chroma_path else {}),
    )

    backends.reset()
    backends.configure(config=config)
    MemoryAgent.reset_instance()
    return MemoryAgent(config=config), config


def _is_question(index: int) -> bool:
    """Solo per il resoconto: quali messaggi vale la pena riportare per esteso.

    NON decide il ramo del grafo. Ogni messaggio, domanda o no, passa da retrieve
    e poi da insert, come nella pipeline vera: e' che le risposte alle domande
    sono le uniche interessanti da leggere fra centodiciassette.

    A deciderlo e' la tabella dei criteri, non il punto interrogativo. Prima era
    il contrario, e il messaggio 117 - "Riassumi tutto quello che sai di me." -
    non finiva nel resoconto per motivi di punteggiatura, pur essendo la domanda
    che mette alla prova tutta la memoria insieme.
    """
    return index in fama.QUESTIONS


def questions_in(messages) -> int:
    return sum(1 for index in range(1, len(messages) + 1) if _is_question(index))


def _active_memory_text(state, vector_store) -> str:
    """Le memorie attive in questo istante: core piu' archivio, una per riga.

    E' l'oggetto su cui si misura FAMA-memoria. Solo le attive: quello che e'
    stato cancellato o superato resta in archivio come tombstone, e contarlo
    significherebbe rimproverare alla consolidation di aver conservato la storia
    invece di averla dimenticata.

    Va letto quando il messaggio che precede la domanda e' stato consolidato, non
    al momento della domanda: vedi fama.memory_snapshot_turn.
    """
    lines = [item.content for item in get_active_items(state["core_memory"])]
    try:
        stored = vector_store.get()
    except Exception:
        return "\n".join(lines)

    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    for content, metadata in zip(documents, metadatas):
        if (metadata or {}).get("status") == "active":
            lines.append(str(content))
    return "\n".join(lines)


def _score_memory(entries, state, vector_store):
    """FAMA-memoria per le domande che aspettavano questo stato."""
    memory = _active_memory_text(state, vector_store)
    for entry in entries:
        entry["on_memory"] = fama.score(fama.QUESTIONS[entry["index"]], memory)


def _message_limit():
    raw = os.getenv("MEMORY_LONGRUN_MESSAGES")
    if not raw:
        return len(CONVERSATION)
    try:
        return max(1, min(int(raw), len(CONVERSATION)))
    except ValueError:
        return len(CONVERSATION)


def _delay():
    try:
        return max(0.0, float(os.getenv("MEMORY_LONGRUN_DELAY", "0")))
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- #
# Resoconto finale
# --------------------------------------------------------------------------- #

def _short(item_id):
    return (item_id or "-")[:ID_WIDTH]


def _clock(timestamp):
    return timestamp.strftime("%H:%M:%S") if timestamp else "--:--:--"


def _clock_iso(text):
    """Solo l'orario di un timestamp ISO: nei metadata di Chroma e' una stringa."""
    if not text:
        return "--:--:--"
    return str(text).split("T")[-1][:8]


def _last_answer(state):
    """La risposta composta da generate_answer: e' l'ultimo messaggio del turno.

    Prima qui c'era un doppio di explainability, una chiamata al modello scritta
    dentro il test. Era una finzione due volte: explainability non risponde dalla
    memoria - risponde dai risultati di una query al database, e la memoria non
    la legge affatto - e quel doppio non riceveva l'archivio, quindi il resoconto
    mostrava recuperi che non entravano in nessuna risposta. Adesso la risposta
    e' quella del servizio, e il resoconto misura il servizio.
    """
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return str(message.content).strip()
    return ""


def _print_progress(index, total, agent):
    """Unica cosa stampata durante l'esecuzione, una riga ogni PROGRESS_EVERY."""
    active = len(get_active_items(agent.state["core_memory"]))
    operations = len(agent.state["operation_log"])
    safe_print(f"  [{index:>3}/{total}] core: {active} item | operazioni: {operations}")


def _print_operation_log(log):
    safe_print(f"\n--- OPERATION LOG ({len(log)} operazioni) ---")

    counters = {}
    for entry in log:
        counters[entry.op_type] = counters.get(entry.op_type, 0) + 1
    if counters:
        summary = ", ".join(f"{name} {count}" for name, count in sorted(counters.items()))
        safe_print(f"Riepilogo: {summary}")
    safe_print("")

    if not log:
        safe_print("  (nessuna operazione registrata)")
        return

    for number, entry in enumerate(log, 1):
        safe_print(f"  {number:>3}  {entry.op_type:<10} | item {_short(entry.item_id)}"
                   f" | related {_short(entry.related_item_id)}"
                   f" | {_clock(entry.timestamp)} | {entry.content}")


def _print_core_memory(state):
    active = get_active_items(state["core_memory"])
    limit = state.get("core_memory_limit")
    used = len("\n".join(item.content for item in active))
    safe_print(f"\n--- CORE MEMORY ({len(active)} item attivi, {used}/{limit} caratteri) ---")

    if not active:
        safe_print("  (core memory vuota)")
        return

    for item in active:
        safe_print(f"  - [{_short(item.id)}] {item.content}")
        safe_print(f"      created {_clock(item.created_at)}"
                   f" | updated {_clock(item.updated_at)}")


def _print_archive(vector_store):
    try:
        stored = vector_store.get()
    except Exception as error:
        safe_print(f"\n--- ARCHIVE MEMORY --- (non ispezionabile: {error})")
        return

    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    safe_print(f"\n--- ARCHIVE MEMORY ({len(ids)} documenti) ---")
    if not ids:
        safe_print("  (archivio vuoto)")
        return

    counters = {}
    for doc_id, content, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        status = metadata["status"]
        counters[status] = counters.get(status, 0) + 1
        safe_print(f"  - [{_short(doc_id)}] ({status}) {content}")
        safe_print(f"      created {_clock_iso(metadata.get('created_at'))}"
                   f" | updated {_clock_iso(metadata.get('updated_at'))}")

    summary = ", ".join(f"{name} {count}" for name, count in sorted(counters.items()))
    safe_print(f"\n  Riepilogo status: {summary}")


def _print_questions(answers):
    """Le domande, cosa ha risposto l'assistente e cosa e' arrivato dall'archivio."""
    consulted = sum(1 for answer in answers if answer["retrieved"])
    safe_print(f"\n--- DOMANDE E RISPOSTE ({len(answers)}) ---")
    safe_print(f"Archivio interrogato per {consulted} domande su {len(answers)}\n")

    if not answers:
        safe_print("  (nessuna domanda posta)")
        return

    for answer in answers:
        safe_print(f"  msg {answer['index']:>3}  D: {answer['question']}")
        safe_print(f"           R: {answer['answer']}")
        retrieved = answer["retrieved"]
        if retrieved:
            for line in str(retrieved).splitlines():
                safe_print(f"           archivio: {line}")
        else:
            safe_print("           archivio: (non interrogato, e' bastata la core memory)")

        on_answer, on_memory = answer["on_answer"], answer["on_memory"]
        safe_print(f"           FAMA: risposta {on_answer.fama:.2f}"
                   f" | memoria {on_memory.fama:.2f}"
                   f" | {fama.verdict(on_memory, on_answer)}")
        for name, scored in (("risposta", on_answer), ("memoria", on_memory)):
            if scored.missed:
                safe_print(f"             {name}: non ricorda {list(scored.missed)}")
            if scored.leaked:
                safe_print(f"             {name}: non ha dimenticato {list(scored.leaked)}")
        safe_print("")


def _cell(value, width=7):
    """Una cifra della tabella FAMA, o un trattino se il conto non esiste.

    MPA senza criteri di presenza e FAA senza criteri di assenza sono None, non
    zero: stamparli come 0.0 direbbe "e' andata malissimo" dove invece non c'era
    niente da misurare.
    """
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.1f}"


def _print_fama(answers):
    """FAMA domanda per domanda, sulla risposta e sulle memorie attive.

    Le stesse due colonne di sempre non basterebbero: un numero solo sulla
    risposta somma consolidation, retrieval e generate_answer, e quando cala non
    dice quale dei tre ha ceduto. La colonna "memoria" applica gli stessi criteri
    allo store appena consolidato il messaggio che precede la domanda, dove non
    c'e' nessun modello di mezzo:
    e' li' che si legge se la consolidation ha fatto il suo lavoro.

    Il verdetto in fondo alla riga e' la lettura congiunta delle due:
    "risposta" quando la memoria era giusta e la risposta no, "consolidation"
    quando gia' lo store era sbagliato, "fuori memoria" quando la risposta era
    giusta pur senza avere in memoria di che dirlo - cioe' ha risposto dal
    contesto della conversazione, non dalla memoria.
    """
    safe_print("")
    safe_print("--- FAMA (Memora, arXiv:2604.20006) ---")
    if not answers:
        safe_print("  (nessuna domanda valutata)")
        return

    safe_print("  FAMA = max(0, MPA - lam * (1 - FAA)),  lam = N_assenza / N_criteri")
    safe_print("")
    safe_print(f"  {'':>5} {'':>6} {'':>6} |{'sulla risposta':^21} |"
               f"{'sulle memorie attive':^21} |")
    safe_print(f"  {'msg':>5} {'P/F':>6} {'lam':>6} |{'MPA':>7}{'FAA':>7}{'FAMA':>7} |"
               f"{'MPA':>7}{'FAA':>7}{'FAMA':>7} | verdetto")

    verdicts = {}
    for answer in answers:
        on_answer, on_memory = answer["on_answer"], answer["on_memory"]
        subject = fama.QUESTIONS[answer["index"]]
        name = fama.verdict(on_memory, on_answer)
        verdicts[name] = verdicts.get(name, 0) + 1
        shape = f"{len(subject.presence)}/{len(subject.forgetting)}"
        safe_print(f"  {answer['index']:>5} {shape:>6} {on_answer.lam:>6.2f} |"
                   f"{on_answer.mpa:>7.2f}{on_answer.faa:>7.2f}{on_answer.fama:>7.2f} |"
                   f"{on_memory.mpa:>7.2f}{on_memory.faa:>7.2f}{on_memory.fama:>7.2f} |"
                   f" {name}")

    reply = fama.aggregate([answer["on_answer"] for answer in answers])
    store = fama.aggregate([answer["on_memory"] for answer in answers])
    safe_print(f"  {'-' * 82}")
    safe_print(f"  {'tot':>5} {'':>6} {'':>6} |"
               f"{_cell(reply['mpa'])}{_cell(reply['faa'])}{reply['fama']:>7.1f} |"
               f"{_cell(store['mpa'])}{_cell(store['faa'])}{store['fama']:>7.1f} |"
               f" su {reply['questions']} domande")

    safe_print("")
    safe_print(f"  --- PUNTEGGIO COMPLESSIVO ({reply['questions']} domande, "
               f"{reply['criteria']} criteri) ---")
    safe_print(f"  {'':<38}{'risposta':>12}{'memoria':>12}")
    for label, key in (("FAMA", "fama"),
                       ("MPA  (presenza, micro-media)", "mpa"),
                       ("FAA  (assenza, micro-media)", "faa"),
                       ("Criteri passati in tutto", "accuracy")):
        safe_print(f"  {label:<38}{_cell(reply[key], 12)}{_cell(store[key], 12)}")

    if verdicts:
        summary = ", ".join(f"{name} {count}"
                            for name, count in sorted(verdicts.items(), key=lambda x: -x[1]))
        safe_print(f"\n  Verdetti: {summary}")

    safe_print("")


def _print_node_costs(elapsed):
    """Dove sono finiti i secondi, nodo per nodo.

    Il totale di una run mescola quattro chiamate con bisogni diversi e il carico
    di una GPU condivisa. I token generati non dipendono dal carico, quindi le
    due colonne insieme dicono se un nodo e' costato perche' ha ragionato tanto o
    perche' il cluster era occupato - che e' quello che serve per tarare
    NODE_SAMPLING un nodo alla volta.
    """
    from memory_service.memory_manager_llm import NODE_STATS

    if not NODE_STATS:
        safe_print("")
        safe_print("--- COSTO PER NODO ---")
        safe_print("  (nessuna chiamata registrata)")
        return

    safe_print("")
    safe_print("--- COSTO PER NODO ---")
    safe_print(f"  {'nodo':<18}{'chiam.':>7}{'secondi':>10}{'s/chiam':>9}"
               f"{'tok out':>10}{'tok/chiam':>11}{'quota':>7}")

    rows = sorted(NODE_STATS.items(), key=lambda item: -item[1]["seconds"])
    totals = {"calls": 0, "seconds": 0.0, "output_tokens": 0}
    for name, stats in rows:
        for key in totals:
            totals[key] += stats[key]

    for name, stats in rows:
        calls = stats["calls"] or 1
        share = stats["seconds"] / elapsed * 100 if elapsed else 0
        safe_print(f"  {name:<18}{stats['calls']:>7}{stats['seconds']:>10.1f}"
                   f"{stats['seconds'] / calls:>9.1f}{stats['output_tokens']:>10}"
                   f"{stats['output_tokens'] / calls:>11.0f}{share:>6.0f}%")

    calls = totals["calls"] or 1
    safe_print(f"  {'-' * 60}")
    safe_print(f"  {'totale':<18}{totals['calls']:>7}{totals['seconds']:>10.1f}"
               f"{totals['seconds'] / calls:>9.1f}{totals['output_tokens']:>10}"
               f"{totals['output_tokens'] / calls:>11.0f}"
               f"{totals['seconds'] / elapsed * 100 if elapsed else 0:>6.0f}%")

    if not totals["output_tokens"]:
        safe_print("  (il server non ha restituito il conteggio dei token: "
                   "restano i tempi)")


def _print_final_report(state, vector_store, config, injected, failures, elapsed,
                       answers, inherited=0):
    safe_print("\n" + SEPARATOR)
    safe_print("=== RESOCONTO FINALE - LONG TERM INTERACTION ===")
    safe_print(SEPARATOR)

    safe_print(f"\nMessaggi iniettati: {injected} (falliti: {len(failures)}), "
               f"di cui {len(answers)} domande")
    safe_print(f"Durata: {elapsed:.1f}s")
    safe_print(f"Modello: {config.llm_config.get('model_name')}")
    safe_print(f"Archivio: {config.chroma_path} / {config.collection_name}")
    if inherited:
        safe_print(f"\033[33mATTENZIONE: l'archivio conteneva gia' {inherited} documenti "
                   f"all'avvio. Le classificazioni di questa run sono state fatte anche "
                   f"contro quelli: lancia reset_archive.py prima di rilanciare.\033[0m")

    if failures:
        safe_print(f"\n--- MESSAGGI FALLITI ({len(failures)}) ---")
        for index, message, error in failures[:10]:
            safe_print(f"  {index:>3}  {message[:60]!r}: {error[:120]}")
        if len(failures) > 10:
            safe_print(f"  ... e altri {len(failures) - 10}")

    _print_operation_log(state["operation_log"])
    _print_core_memory(state)
    _print_archive(vector_store)
    _print_questions(answers)
    _print_fama(answers)
    _print_node_costs(elapsed)

    safe_print("\n" + SEPARATOR + "\n")


# --------------------------------------------------------------------------- #
# Invarianti
# --------------------------------------------------------------------------- #

def _assert_state_is_consistent(state):
    core_memory = state["core_memory"]
    assert all(item.status == "active" for item in core_memory), (
        "la core memory deve contenere solo item attivi: superseded e deleted "
        "finiscono in archivio")
    assert len({item.id for item in core_memory}) == len(core_memory), "id duplicati in core"
    for item in core_memory:
        assert item.content.strip(), "un item senza contenuto non ha senso"
        assert item.updated_at >= item.created_at


def _assert_everything_that_left_core_is_in_the_archive(state, vector_store):
    """Nessun item deve sparire: se esce dalla core memory, sta nell'archivio."""
    expected = set()
    for entry in state["operation_log"]:
        if entry.op_type in ("delete", "archive"):
            expected.add(entry.item_id)
        elif entry.op_type in ("update", "contradict") and entry.related_item_id:
            expected.add(entry.related_item_id)

    if not expected:
        return 0

    found = set(vector_store.get(ids=sorted(expected)).get("ids") or [])
    missing = sorted(expected - found)
    assert not missing, (
        f"{len(missing)} item usciti dalla core memory non si trovano in archivio: "
        f"{missing[:5]}")
    return len(expected)


# --------------------------------------------------------------------------- #
# Il test
# --------------------------------------------------------------------------- #

def test_long_term_interaction():
    reason = live_stack_unavailable()
    if reason:
        pytest.skip(reason)

    _configure_langsmith()
    agent, config = _build_agent()
    from memory_service.memory_manager_llm import reset_node_stats

    reset_node_stats()
    inherited = _archived_documents(config) or 0
    vector_store = backends.get_vector_store()

    messages = CONVERSATION[:_message_limit()]
    delay = _delay()

    safe_print(f"\nIniezione di {len(messages)} messaggi uno alla volta, "
               f"di cui {questions_in(messages)} domande "
               f"(finestra storico: {config.maximum_historical_messages}, "
               f"limite core: {config.core_memory_limit} caratteri)")
    safe_print("Ogni messaggio ripercorre la pipeline: get_memory come fa "
               "intent_recognition,\nrisposta come fa explainability, "
               "update_memory con la coppia domanda/risposta.")
    safe_print("L'output dell'agente e' soppresso: si stampa solo l'avanzamento.\n")

    noise = io.StringIO()
    failures = []
    answers = []
    started = time.time()

    snapshot_due = {}

    for index, message in enumerate(messages, 1):
        try:
            with contextlib.redirect_stdout(noise):
                # --- come intent_recognition.listener_callback ---------------
                state = agent.run_memory_agent("retrieve", query=message)
                retrieved = state.get("retrieved_memory", "")

                # --- la risposta la compone il servizio ----------------------
                answer = _last_answer(state)

                # --- come explainability: send_update_request(input, risposta) -
                agent.run_memory_agent("insert")

            if _is_question(index):
                entry = {
                    "index": index,
                    "question": message,
                    "answer": answer,
                    "retrieved": retrieved,
                    "on_answer": fama.score(fama.QUESTIONS[index], answer),
                    "on_memory": None,
                }
                answers.append(entry)
                turn = fama.memory_snapshot_turn(index, config.maximum_historical_messages)
                snapshot_due.setdefault(turn, []).append(entry)
        except Exception as error:
            failures.append((index, message, repr(error)))

        # agent.state e non `state`: l'insert ha sostituito il dizionario.
        if index in snapshot_due:
            _score_memory(snapshot_due.pop(index), agent.state, vector_store)

        if index % PROGRESS_EVERY == 0 or index == len(messages):
            _print_progress(index, len(messages), agent)
        if delay:
            time.sleep(delay)

    elapsed = time.time() - started
    state = agent.state

    # Fotografie che cadevano oltre l'ultimo messaggio iniettato: stato finale.
    for entries in snapshot_due.values():
        _score_memory(entries, state, vector_store)

    _print_final_report(state, vector_store, config, len(messages), failures, elapsed,
                        answers, inherited)

    assert len(failures) < len(messages) / 2, (
        f"{len(failures)} messaggi su {len(messages)} sono falliti: "
        "probabilmente il server non risponde o la configurazione e' sbagliata")
    assert state["operation_log"], "una sessione lunga deve aver prodotto delle operazioni"
    _assert_state_is_consistent(state)
    _assert_everything_that_left_core_is_in_the_archive(state, vector_store)

    expected_questions = questions_in(messages)
    assert len(answers) == expected_questions - len(
        [f for f in failures if _is_question(f[0])]), (
        "ogni domanda andata a buon fine deve comparire nel resoconto")
    unanswered = [answer for answer in answers if not answer["answer"]]
    assert not unanswered, (
        f"{len(unanswered)} domande sono rimaste senza risposta: "
        f"{[answer['question'] for answer in unanswered][:3]}")
