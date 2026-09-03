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
l'archivio e' stato consultato per quel giro.

La domanda corrente viene passata al servizio (GetMemory.user_input ->
run_memory_agent(query=...)), quindi la ricerca in archivio parte da quello che
l'utente sta chiedendo adesso e non dall'ultimo messaggio gia' in memoria.

Durante l'esecuzione l'output dell'agente viene soppresso: si stampa solo una
riga di avanzamento ogni 10 messaggi. **Alla fine** viene stampato il resoconto:
operation log completo, core memory, archivio, e domande con le relative
risposte.

Le assert sono strutturali (invarianti che devono valere qualunque cosa decida il
modello), non sul contenuto: con un LLM vero pretendere una classificazione
esatta renderebbe il test inutilizzabile.

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

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    get_active_items,
    serialize_core_memory_for_prompt,
)

from live_model import live_stack_unavailable  # noqa: E402
from snapshot import SEPARATOR, safe_print  # noqa: E402


# Lo skip a livello di modulo resta economico (guarda solo la configurazione):
# la raggiungibilita' dei modelli costa una chiamata e si verifica dentro il test.
pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_CONFIG"),
    reason="LLM_CONFIG is required to run the long term interaction test",
)

PROGRESS_EVERY = 10
ID_WIDTH = 8


# --------------------------------------------------------------------------- #
# La conversazione: 117 messaggi dell'utente (101 fatti e 16 domande),
# nell'ordine in cui li direbbe
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
    "Corro sempre la mattina presto, prima del lavoro.",
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

    # --- da qui in poi tutto mescolato, come in una sessione vera ---------- #
    # Il commento segna il caso atteso: l'ordine rispetta le dipendenze, un
    # fatto viene sempre introdotto prima di essere ripetuto, raffinato,
    # smentito o cancellato. I messaggi che finiscono con "?" sono domande:
    # non consolidano, interrogano la memoria (vedi _is_question).
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
        core_memory_limit=int(os.getenv("MEMORY_CORE_MEMORY_LIMIT", "200")),
        collection_name=os.getenv("MEMORY_LONGRUN_COLLECTION", "longterm_test_archive"),
        **({"chroma_path": os.path.abspath(chroma_path)} if chroma_path else {}),
    )

    backends.reset()
    backends.configure(config=config)
    MemoryAgent.reset_instance()
    return MemoryAgent(config=config), config


def _is_question(message: str) -> bool:
    """Solo per il resoconto: quali messaggi vale la pena riportare per esteso.

    NON decide il ramo del grafo. Ogni messaggio, domanda o no, passa da retrieve
    e poi da insert, come nella pipeline vera: e' che le risposte alle domande
    sono le uniche interessanti da leggere fra centodiciassette.
    """
    return message.strip().endswith("?")


def questions_in(messages) -> int:
    return sum(1 for message in messages if _is_question(message))


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


def _answer_like_explainability(message, memories):
    """Genera la risposta all'utente, come fa il nodo explainability.

    In produzione questo passo non e' nel memory service: `explainability` ha un
    suo LLM, riceve il contesto che `intent_recognition` ha preso dalla memoria e
    produce la spiegazione che poi torna indietro come `update_memory`. Qui lo
    stesso ruolo e' svolto da una chiamata diretta al modello, per avere una
    risposta vera da consolidare e da verificare.
    """
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sei l'assistente personale dell'utente. Rispondi in italiano,
         in una o due frasi, usando SOLO quello che sai di lui:

         {memories}

         Se l'informazione non c'e', dillo chiaramente invece di inventarla."""),
        ("human", "{input}"),
    ])
    chain = prompt | backends.get_llm()
    response = chain.invoke({"input": message, "memories": "\n".join(memories)})
    return str(response.content).strip()


def _context_from_memory(state):
    """Quello che intent_recognition passa a valle: memory_list + last_messages."""
    memories = serialize_core_memory_for_prompt(state.get("core_memory", []))
    messages = [str(getattr(m, "content", m)) for m in state.get("messages", [])]
    return memories + messages


def _last_answer(state, question):
    """Ultimo messaggio dell'assistente, se la retrieve ne ha prodotto uno."""
    for message in reversed(state.get("messages", [])):
        content = str(getattr(message, "content", "")).strip()
        if content and content != question.strip():
            return content
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
                   f" | updated {_clock(item.updated_at)}"
                   f" | supersedes {_short(item.supersedes)}")


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
        status = metadata.get("status", "N/A")
        counters[status] = counters.get(status, 0) + 1
        safe_print(f"  - [{_short(doc_id)}] ({status}) {content}")
        safe_print(f"      created {_clock_iso(metadata.get('created_at'))}"
                   f" | updated {_clock_iso(metadata.get('updated_at'))}"
                   f" | supersedes {_short(metadata.get('supersedes'))}")

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
        safe_print("")


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
        assert item.supersedes != item.id, "un item non puo' superare se stesso"


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
    # Contato prima di iniettare: dopo non si distinguerebbe piu' quello che
    # ha scritto questa run da quello che ha trovato.
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

    for index, message in enumerate(messages, 1):
        try:
            # Tutto quello che l'agente stampa finisce nel buffer: il resoconto
            # arriva alla fine, non messaggio per messaggio.
            with contextlib.redirect_stdout(noise):
                # --- come intent_recognition.listener_callback ---------------
                # Un get_memory su OGNI messaggio, senza condizioni: e' il modo
                # in cui la pipeline si procura il contesto. Azzerare prima
                # retrieved_memory serve solo al resoconto, per capire se
                # l'archivio e' stato interrogato per questo giro o per uno
                # precedente.
                agent.state["retrieved_memory"] = ""
                state = agent.run_memory_agent("retrieve", query=message)
                retrieved = state.get("retrieved_memory", "")
                context = _context_from_memory(state)

                # --- come explainability: produce la risposta -----------------
                answer = _answer_like_explainability(message, context)

                # --- come explainability: send_update_request(input, risposta) -
                agent.append_message(message, "user")
                agent.append_message(answer, "assistant")
                agent.run_memory_agent("insert")

            if _is_question(message):
                answers.append({
                    "index": index,
                    "question": message,
                    "answer": answer,
                    "retrieved": retrieved,
                })
        except Exception as error:
            failures.append((index, message, repr(error)))

        if index % PROGRESS_EVERY == 0 or index == len(messages):
            _print_progress(index, len(messages), agent)
        if delay:
            time.sleep(delay)

    elapsed = time.time() - started
    state = agent.state

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
        [f for f in failures if _is_question(f[1])]), (
        "ogni domanda andata a buon fine deve comparire nel resoconto")
    unanswered = [answer for answer in answers if not answer["answer"]]
    assert not unanswered, (
        f"{len(unanswered)} domande sono rimaste senza risposta: "
        f"{[answer['question'] for answer in unanswered][:3]}")
