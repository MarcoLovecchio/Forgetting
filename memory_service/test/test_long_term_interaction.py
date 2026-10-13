"""Simulation of a long-term interaction on the real stack.

Unlike test_memory_llm.py, here **there are no targeted turns**: there is a
long, realistic and messy conversation, injected **one message at a time**,
and every message goes through the real pipeline of the architecture.

In the pipeline nobody "decides" between insert and retrieve: they are two
phases of the same interaction, and both always happen.

    1. intent_recognition.listener_callback  ->  send_get_request()
       a get_memory on EVERY message, to obtain the context
    2. explainability.query_explanation_callback
       generates the explanation using that context
    3. explainability                        ->  send_update_request(input, explanation)
       an update_memory on EVERY explanation produced

The test does the same three things for every message. In production step 2
sits outside the memory service (explainability has its own LLM): here the
answer is the one generate_answer composes inside the service, so there is a
real answer to consolidate and to check.

The first 34 messages are the presentation: the user talks about themselves and
they are all new facts. From there on **everything is mixed**, as in a real
session: a repetition, then a refinement, then some chatter, then a
contradiction, then a deletion. No blocks per operation type.

The only constraint respected is causality: a fact is introduced before it can
be repeated, refined, contradicted or deleted. The chains unfold at a distance,
as would really happen - for example the calories:

    msg 11   "il mio obiettivo giornaliero e' 2000 calorie"     (new)
    msg 38   "in realta' l'obiettivo ora e' 2200"               (update)
    msg 153  "ho cambiato obiettivo: 1800, voglio dimagrire"    (update)
    msg 184  "non memorizzare piu' quante calorie punto"        (delete)

Every message carries in a comment the case it should trigger. The scattered
chatter serves to fill the core memory until the split towards the archive
triggers.

Scattered among the facts there are also **questions**. They get no different
treatment - they too go through retrieve and then insert, like everything else -
but they are the points where the answer says something, and they are the only
ones the report shows in full. Many are placed right after an update, a
contradiction or a deletion:

    "in realta' l'obiettivo ora e' 2200"        (update)
    "qual e' il mio obiettivo calorico?"        -> must say 2200, not 2000

    "dimentica il mio indirizzo"                (delete)
    "sai dirmi dove abito?"                     -> must no longer know it

Others recall a fact written much earlier, which may have ended up in the
archive ("come si chiama la mia migliore amica?"), and others still are
assistant tasks that use the facts without naming them: "mi consigli cosa
cucinare per cena?" wants fish and no peanuts, "come mi conviene andare al
lago?" wants the missing driving licence. This way the task decides which
memories are needed, not whoever wrote the question.

The report shows, for each question, what the assistant answered and whether
the archive was consulted on that round. The answer is the service's one,
composed by generate_answer on the core memory plus the possible retrieval: the
test no longer simulates one of its own.

The current question is passed to the service (GetMemory.user_input ->
run_memory_agent(query=...)), so the archive search starts from what the user
is asking right now and not from the last user message already in memory.

During the run the agent's output is suppressed: only one progress line every
10 messages is printed. **At the end** the report is printed: full operation
log, core memory, archive, questions with their answers, the FAMA table, the
confusion matrix and the cost per node.

The 56 questions are also the evaluation points of **FAMA** (Memora,
arXiv:2604.20006): each one carries binary criteria in fama.py - what the
answer must contain, what it must no longer contain - and the report derives a
score from them. It computes it twice, with the same criteria: once on the
answer, which is the paper's metric and measures the whole chain, and once on
the active memories once the message preceding the question is consolidated,
which goes through no model and measures consolidation alone. The gap between
the two says which node gave way.

Every message is also a row of the **confusion matrix** of the operations
(operation_matrix.py): the expected operation, from the comment next to it,
against the one consolidation really made. The run is appended to a file that
operation_matrix.py can sum with the others of the same commit.

The asserts are structural (invariants that must hold whatever the model
decides), not on the content: with a real LLM demanding an exact classification
would make the test unusable. **FAMA too is measured and printed, never
asserted**: with 56 questions and a sampling far from greedy a threshold would
turn the sampling variance into a test that fails at random.

Requirements: LLM_CONFIG and EMBEDDING_CONFIG in .config, and the two models
reachable at the address given by MEMORY_LLM_BASE_URL /
MEMORY_EMBEDDING_BASE_URL in the .env. If something does not answer the test is
skipped explaining what.

Warning: each message costs 3-4 real calls to the model (the retrieval
decision, the answer, consolidation, and the core split when the core memory
overflows). Over 232 messages that is several hundred calls: better to start
with a low MEMORY_LONGRUN_MESSAGES. With the models on the cluster the
limit is not the quota but the latency, so the time depends on the load of the
GPUs they run on.

Run:
    pytest memory_service/test/test_long_term_interaction.py -v -s

Useful variables:
    MEMORY_LONGRUN_MESSAGES      how many messages to inject, out of 232
    MEMORY_LONGRUN_DELAY         seconds of pause between one message and the
                                 next. On the cluster there are no rate limits
                                 to respect, but it is needed if the endpoint is
                                 shared with other loads
    MEMORY_LONGRUN_CHROMA_PATH   another folder for the archive, instead of the
                                 service's one that reset_archive.py cleans
    MEMORY_LONGRUN_COLLECTION    the collection of the archive, kept apart from
                                 the production one
    MEMORY_CORE_MEMORY_LIMIT     character limit of the core memory
    MEMORY_LONGRUN_RESULTS       the file to append the run's operations to, for
                                 operation_matrix.py
    MEMORY_LONGRUN_CALIBRATION   if set, the file to append the trace to for
                                 calibrate_utility.py: the active memories at
                                 every question, with their ratings. The run
                                 goes with eviction off, because a memory
                                 removed before a question could never turn out
                                 served
"""

import ast
import contextlib
import dataclasses
import io
import os
import subprocess
import sys
import time

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import AIMessage

from memory_service import backends
from memory_service.config import NODE_SAMPLING, MemoryConfig
from memory_service.consolidation import ARCHIVE_CANDIDATES_K, get_active_items

import fama
from live_model import live_stack_unavailable
import operation_matrix
from snapshot import SEPARATOR, safe_print


pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_CONFIG"),
    reason="LLM_CONFIG is required to run the long term interaction test",
)

PROGRESS_EVERY = 10
ID_WIDTH = 8

# --------------------------------------------------------------------------- #
# The conversation: 232 user messages (176 facts and 56 questions)
#
# Which ones are the 56 is said by fama.QUESTIONS, not by the punctuation:
# the last message asks for a summary without a question mark, and it is a
# question like the others.
# --------------------------------------------------------------------------- #

CONVERSATION = [
    # --- presentation: the user introduces themselves, all new facts ------- #
    "Ciao, mi chiamo Bianca.",
    "Ho 29 anni.",
    "Il mio compleanno e' il 12 marzo.",
    "Vivo a Palermo, in centro.",
    "Lavoro come fisioterapista in una clinica privata.",
    "Non ho la patente, mi muovo in bici o con i mezzi.",
    "Sono vegetariana da sei anni.",
    "Sono allergica alle arachidi, e' un'allergia seria.",
    "Ho l'asma da quando ero piccola, porto sempre l'inalatore.",
    "Prendo la levotiroxina ogni mattina a digiuno per la tiroide.",
    "Il mio obiettivo giornaliero e' 2000 calorie.",
    "Punto a 60 grammi di proteine al giorno.",
    "Non bevo alcolici.",
    "Bevo il caffe' solo la mattina.",
    "Nel pomeriggio preferisco il te' verde.",
    "Ho un cane che si chiama Argo.",
    "Argo e' un labrador di quattro anni.",
    "Argo non puo' mangiare il pollo, gli fa male.",
    "Vado a correre tre volte a settimana.",
    "Corro la mattina presto, prima del lavoro.",
    "Il mio piatto preferito e' la parmigiana di melanzane.",
    "Odio il gusto della liquirizia.",
    "Mia sorella si chiama Chiara e vive a Milano.",
    "Mia madre e' celiaca, in casa cuciniamo senza glutine.",
    "La mia migliore amica si chiama Giulia, ci conosciamo dal liceo.",
    "Suono il pianoforte da quando avevo otto anni.",
    "Leggo soprattutto romanzi storici.",
    "Porto gli occhiali da vista, sono miope.",
    "Vado a dormire verso le undici di sera.",
    "Mi sveglio alle sei e mezza.",
    "Sul balcone ho un piccolo orto con pomodori e basilico.",
    "Sono intollerante al lattosio, ma solo leggermente.",
    "Faccio yoga il sabato mattina.",
    "La domenica guardo sempre la Formula 1.",

    "Oggi al lavoro ho avuto una giornata lunga, sette pazienti.",  # chatter
    "Che cosa sai della mia alimentazione?",  # QUESTION: facts of the presentation
    "Ti ripeto che sono allergica alle arachidi.",  # redundant: allergy
    "In realta' l'obiettivo ora e' 2200 calorie, ho aumentato.",  # update: calories
    "Qual e' il mio obiettivo calorico giornaliero?",  # QUESTION: 2200, not 2000
    "Buongiorno!",  # chatter
    "Ricordati che ho l'asma, l'inalatore ce l'ho sempre in borsa.",  # redundant: asthma
    "Ho qualche problema respiratorio?",  # QUESTION: the asthma
    "Ho comprato una bicicletta nuova, una gravel grigia.",  # new
    "Oggi ho comprato un ombrello rosso.",  # new
    "Come ti dicevo, mi chiamo Bianca.",  # redundant: name
    "Ti ricordi come mi chiamo?",  # QUESTION: the name
    "Argo ha compiuto cinque anni la settimana scorsa.",  # update: the dog's age
    "Medito dieci minuti appena sveglia.",  # new
    "Non sopporto i film dell'orrore, mi mettono ansia.",  # new
    "Mi consigli un film da vedere stasera?",  # QUESTION (task): no horror
    "Le proteine le ho portate a 75 grammi al giorno.",  # update: proteins
    "Il caffe' lo prendo la mattina, mai dopo.",  # redundant: coffee
    "Sto imparando lo spagnolo con un'app, dieci minuti al giorno.",  # new
    "Mi sono trasferita: sempre Palermo, ma ora vivo a Mondello.",  # update: home
    "Sto seguendo un corso serale di lingua dei segni.",  # new
    "La sera ascolto podcast di storia.",  # new
    "Non sono piu' vegetariana: da questo mese mangio pesce.",  # contradict: diet
    "Che tipo di dieta seguo adesso?",  # QUESTION: must reflect the contradiction
    "Oggi piove da stamattina, che noia.",  # chatter
    "Mi consigli cosa cucinare per cena stasera?",  # QUESTION (task): fish, no peanuts
    "Il mio colore preferito e' il verde bottiglia.",  # new
    "Corro quattro volte a settimana adesso, non piu' tre.",  # update: running
    "Mia sorella Chiara e' sempre a Milano.",  # redundant: sister
    "Dimentica il mio indirizzo.",  # delete: home
    "Sai dirmi dove abito?",  # QUESTION: after the delete it must not know it
    "Il medico mi ha aumentato la levotiroxina a 88 microgrammi.",  # update: levothyroxine
    "Che dose di levotiroxina prendo?",  # QUESTION: 88 micrograms
    "Bevo almeno due litri d'acqua al giorno.",  # new
    "Ho compiuto 30 anni il mese scorso.",  # update: age
    "Il caffe' l'ho eliminato del tutto, mi agitava troppo.",  # contradict: coffee
    "Cosa bevo di solito la mattina?",  # QUESTION: must no longer say coffee
    "Sono allergica anche alla penicillina, l'ho scoperto da poco.",  # new
    "A quali farmaci sono allergica?",  # QUESTION: penicillin
    "Ho una collezione di dischi in vinile, soprattutto jazz.",  # new
    "Ho paura di volare, per viaggiare preferisco treno e nave.",  # new
    "Confermo: niente alcolici per me.",  # redundant: alcohol
    "Suono il pianoforte e da un anno anche la chitarra.",  # update: music
    "Ad agosto vorrei andare in Grecia, forse a Naxos.",  # new
    "Dove vorrei andare in vacanza ad agosto?",  # QUESTION: Naxos
    "Come sai, l'aereo mi terrorizza.",  # redundant: fear of flying
    "Come mi conviene arrivare a Naxos?",  # QUESTION (task): no plane
    "Ho smesso di correre, mi da' fastidio il ginocchio.",  # contradict: running
    "Cancella l'informazione sulla mia eta'",  # delete: age
    "Quanti anni ho?",  # QUESTION: after the delete it must not know it
    "Ho iniziato un corso di fotografia il lunedi' sera.",  # new
    "Stamattina ho perso le chiavi, poi le ho ritrovate in tasca.",  # new
    "Il sabato mattina c'e' yoga, come ogni settimana.",  # redundant: yoga
    "Ora nuoto invece di correre, due volte a settimana.",  # new
    "Chiara si e' trasferita da Milano a Torino.",  # update: sister
    "Dove vive mia sorella?",  # QUESTION: must say Torino, not Milano
    "Ho comprato una reflex usata per il corso di fotografia.",  # new
    "Che corso faccio il lunedi' sera?",  # QUESTION: photography
    "In inverno soffro molto il freddo alle mani.",  # new
    "Ho comprato un tappetino nuovo per la cucina.",  # new
    "Giulia si sposa il 14 giugno e io saro' la sua testimone.",  # new
    "Ho ricominciato a bere vino, un bicchiere a cena.",  # contradict: alcohol
    "Va bene, a dopo.",  # chatter
    "Faccio la fisioterapista, come ti ho raccontato.",  # redundant: work
    "Mi sono ricreduta sulla liquirizia, adesso mi piace.",  # contradict: liquorice
    "Il mio numero fortunato e' il sette.",  # new
    "Quando si sposa la mia amica Giulia?",  # QUESTION: 14 June
    "Il martedi' sera faccio volontariato in un canile.",  # new
    "Yoga il sabato e da poco anche il mercoledi' sera.",  # update: yoga
    "Non conservare il nome di mia sorella, eliminalo.",  # delete: sister
    "Come si chiama mia sorella?",  # QUESTION: after the delete it must not know it
    "Che caldo oggi, non si respira.",  # chatter
    "Lavoro in clinica, ma ora solo part-time il pomeriggio.",  # update: work
    "Cosa faccio il martedi' sera?",  # QUESTION: volunteering
    "Ho una cicatrice sul ginocchio destro da bambina.",  # new
    "L'allergia vale anche per l'olio di arachidi, sto attenta.",  # update: allergy
    "Sono allergica a qualcosa?",  # QUESTION: peanuts and penicillin
    "Non sono intollerante al lattosio, gli esami erano sbagliati.",  # contradict
    "Mi piace cucinare, ma solo nel weekend quando ho tempo.",  # new
    "Mi piace la liquirizia?",  # QUESTION: yes, a change of mind
    "Mi sono tagliata i capelli corti.",  # new
    "Mi sveglio alle sei adesso, mezz'ora prima.",  # update: alarm
    "Argo non e' un labrador, e' un golden retriever.",  # contradict: breed
    "Di che razza e' il mio cane?",  # QUESTION: must say golden retriever
    "Cosa non puo' mangiare Argo?",  # QUESTION: chicken
    "Argo deve fare il richiamo del vaccino il 20 ottobre.",  # new
    "Dimentica cosa ti ho detto sul vino.",  # delete: wine
    "Preferisco la montagna al mare, anche se vivo sulla costa.",  # new
    "Cosa metto nello zaino per un'escursione in montagna?",  # QUESTION (task): the inhaler
    "Leggo romanzi storici e ultimamente anche saggi.",  # update: reading
    "Ah ok, capito.",  # chatter
    "Non faccio piu' yoga, ho smesso a settembre.",  # contradict: yoga
    "Faccio ancora yoga il sabato?",  # QUESTION: must say no
    "Ho il dentista martedi' prossimo alle tre del pomeriggio.",  # new
    "Ho l'abitudine di fare la spesa il giovedi' pomeriggio.",  # new
    "Cerco di non spendere piu' di 50 euro a settimana per la spesa.",  # new
    "Mi aiuti a fare la lista della spesa per giovedi'?",  # QUESTION (task): 50 euro budget
    "Il te' verde lo prendo anche a merenda.",  # update: tea
    "Quando deve fare il vaccino Argo?",  # QUESTION: 20 October
    "Non lavoro piu' in clinica, mi sono messa in proprio.",  # contradict: work
    "Ho comprato una pianta grassa per la scrivania.",  # new
    "Sto pensando a un secondo cane per fare compagnia ad Argo.",  # new
    "Quando ho l'appuntamento dal dentista?",  # QUESTION: Tuesday at three
    "Vado a dormire molto piu' tardi ora, verso l'una.",  # contradict: sleep
    "Rimuovi il dato sul lavoro che faccio.",  # delete: work
    "Che lavoro faccio?",  # QUESTION: after the delete it must not know it
    "Sono andata dal dentista, era solo un controllo e va tutto bene.",  # update: dentist
    "La mia vicina di casa tiene Argo quando sono in viaggio.",  # new
    "Non uso i social network, li ho cancellati l'anno scorso.",  # new
    "Uso un'agenda di carta per gli appuntamenti.",  # new
    "Ho cambiato telefono, ora uso un iPhone.",  # new
    "La parmigiana la faccio senza formaggio adesso.",  # update: dish
    "Ho chiuso con il pianoforte, non lo suono piu'.",  # contradict: music
    "Che musica ascolto o suono?",  # QUESTION: likely retrieval from the archive
    "Mi sono comprata una chitarra elettrica.",  # new
    "Ho prenotato: ad agosto vado a Naxos con Giulia.",  # update: trip
    "A chi posso lasciare Argo quando parto ad agosto?",  # QUESTION (task): the neighbour
    "Ricorda che mangio pesce ma non carne.",  # redundant: diet
    "Ho cambiato obiettivo: 1800 calorie, voglio dimagrire.",  # contradict: calories
    "Quante calorie dovrei assumere ogni giorno?",  # QUESTION: must say 1800
    "Ok, grazie mille.",  # chatter
    "Il medico ha portato la levotiroxina a 100 microgrammi.",  # update: levothyroxine
    "Che farmaco devo ricordarmi di prendere domattina?",  # QUESTION (task): levothyroxine 100
    "Sono arrivata al livello B1 di spagnolo.",  # update: Spanish
    "Elimina quello che sai sui miei orari di sonno.",  # delete: sleep
    "Non e' mia madre a essere celiaca, e' mio padre.",  # contradict: coeliac disease
    "Il mio scrittore preferito e' Andrea Camilleri.",  # new
    "Il nuoto l'ho portato a tre volte a settimana.",  # update: swimming
    "Ho adottato un gatto, si chiama Milo.",  # new
    "Uso solo detersivi ecologici.",  # new
    "Non leggo piu' romanzi storici, ora solo gialli.",  # contradict: reading
    "Milo e' un soriano di due anni.",  # new
    "Grazie, sei gentilissima.",  # chatter
    "Cancella il riferimento a mio padre e alla celiachia.",  # delete: coeliac disease
    "C'e' qualcuno di celiaco nella mia famiglia?",  # QUESTION: after the delete
    "Il matrimonio di Giulia e' stato bellissimo, ho anche pianto.",  # update: wedding
    "Come si chiama la mia migliore amica?",  # QUESTION: Giulia
    "Che lingua sto imparando?",  # QUESTION: Spanish
    "Ti ricordo ancora che sono allergica alle arachidi.",  # redundant: allergy
    "Ti ricordo che sono allergica alla penicillina.",  # redundant: penicillin
    "In realta' Milo ha tre anni, mi sono sbagliata.",  # update: the cat's age
    "Ho ripreso a correre, il ginocchio sta meglio.",  # contradict: running
    "Che sport faccio in questo periodo?",  # QUESTION: running taken up again plus swimming
    "Da questo mese faccio anche pilates il giovedi' sera.",  # new
    "Mi organizzi gli allenamenti della settimana?",  # QUESTION (task): running, swimming, pilates
    "Sul balcone adesso coltivo anche i peperoncini.",  # update: vegetable garden
    "Cosa coltivo sul balcone?",  # QUESTION: tomatoes, basil, chili peppers
    "Ho smesso di meditare, non riesco a essere costante.",  # contradict: meditation
    "Medito ancora la mattina?",  # QUESTION: stopped
    "Non memorizzare piu' quante calorie punto.",  # delete: calories
    "Argo ha fatto il vaccino, tutto a posto.",  # update: vaccine
    "Dimentica il mio colore preferito, non serve.",  # delete: colour
    "Dimentica la storia del volontariato al canile.",  # delete: volunteering
    "Ho impegni fissi il martedi' sera?",  # QUESTION: after the delete it must not know it
    "Corro sempre la mattina presto, come sempre.",  # redundant: running time
    "Ci sentiamo dopo.",  # chatter
    "Te l'ho gia' detto, la patente non ce l'ho.",  # redundant: driving licence
    "Come mi conviene andare al lago domenica?",  # QUESTION (task): no driving licence
    "Ho fatto un corso e ho superato la paura di volare.",  # contradict: fear of flying
    "Ho ancora paura di volare?",  # QUESTION: overcome
    "Dimentica il fastidio al ginocchio di cui ti parlavo.",  # delete: knee
    "Stamattina il bus era in ritardo di venti minuti.",  # new
    "Togli dalla memoria tutto quello che riguarda i familiari.",  # delete: family
    "Quali animali ho in casa?",  # QUESTION: Argo and Milo, likely retrieval
    "Mi sono operata agli occhi, non porto piu' gli occhiali.",  # contradict: glasses
    "Porto gli occhiali?",  # QUESTION: had surgery, no glasses
    "Quando e' il mio compleanno?",  # QUESTION: 12 March
    "Dimentica la data del mio compleanno.",  # delete: birthday
    "Sai quando compio gli anni?",  # QUESTION: after the delete it must not know it
    "La Formula 1 non la guardo piu', mi annoia.",  # contradict: formula 1
    "Ho finito il corso di lingua dei segni, ho il primo livello.",  # update: sign language
    "Sto ancora facendo il corso di lingua dei segni?",  # QUESTION: finished it
    "Il mio piatto preferito resta la parmigiana.",  # redundant: dish
    "Quante proteine devo assumere al giorno?",  # QUESTION: 75 grams
    "Continuo a bere i miei due litri d'acqua al giorno.",  # redundant: water
    "Quanta acqua bevo al giorno?",  # QUESTION: two litres
    "Non bevo piu' caffe' nemmeno la mattina, confermo.",  # redundant: coffee
    "Ho tolto l'orto dal balcone, dimenticalo pure.",  # delete: vegetable garden
    "Ho ancora l'orto sul balcone?",  # QUESTION: after the delete it must not know it
    "Punto sempre a 75 grammi di proteine.",  # redundant: proteins
    "Mi piacciono molto i girasoli.",  # new
    "Naxos salta, ad agosto andro' in Portogallo.",  # contradict: trip
    "Dove vado in vacanza ad agosto?",  # QUESTION: Portugal, not Naxos
    "Il mio ristorante preferito e' una trattoria di pesce a Cefalu'.",  # new
    "Mi consigli dove andare a cena domenica?",  # QUESTION (task): the trattoria
    "A Natale vorrei andare a sciare in Trentino.",  # new
    "Non mangio pesce crudo, il sushi non mi piace.",  # new
    "Dimentica quello che ti ho detto sulla Formula 1.",  # delete: formula 1
    "Non ho mai fumato in vita mia.",  # new
    "Cancella il fatto che ho un iPhone.",  # delete: phone
    "Come ti dicevo, Giulia e' la mia migliore amica.",  # redundant: Giulia
    "La levotiroxina la prendo sempre a digiuno, la mattina.",  # redundant: levothyroxine
    "In casa tengo il riscaldamento basso, mi piace il fresco.",  # new
    "Ieri ho visto un tramonto bellissimo dal lungomare.",  # new
    "Niente sci a Natale, quest'anno resto a casa.",  # contradict: Christmas
    "Cosa faccio a Natale?",  # QUESTION: stays at home
    "Chi e' il mio scrittore preferito?",  # QUESTION: Camilleri
    "Riassumi tutto quello che sai di me.",  # QUESTION (final): overall picture
]


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def _configure_langsmith():
    """Optional tracing, as in the rest of the architecture."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    model_name = ast.literal_eval(os.getenv("LLM_CONFIG"))["memory_agent"]["model_name"]
    os.environ["LANGSMITH_PROJECT"] = f'MEMORY-LONGRUN:{model_name}'
    os.environ["LANGSMITH_TEST_SUITE"] = "Memory Service long term"


def _archived_documents(config):
    """How many documents are already in the archive, or None if it is not readable."""
    try:
        return backends.get_vector_store()._collection.count()
    except Exception:
        return None


def _build_agent():
    """Agent on the real stack, on the configured archive.

    It uses the same path as the rest of the service (MEMORY_CHROMA_PATH) in a
    collection of its own, so reset_archive.py cleans it
    together with the others. The flip side is that the runs add up: without
    cleaning, the next one starts with the memories of the previous one and the
    classifications change meaning. The initial report says how many documents
    it found.
    """
    from memory_service.memory_manager_llm import MemoryAgent

    chroma_path = os.getenv("MEMORY_LONGRUN_CHROMA_PATH")
    config = dataclasses.replace(
        MemoryConfig.from_environment(),
        maximum_historical_messages=2,
        generate_answer=True,
        core_memory_limit=int(os.getenv("MEMORY_CORE_MEMORY_LIMIT", "400")),
        collection_name=os.getenv("MEMORY_LONGRUN_COLLECTION", "longterm_test_archive"),
        **({"chroma_path": os.path.abspath(chroma_path)} if chroma_path else {}),
        **({"eviction": False} if os.getenv("MEMORY_LONGRUN_CALIBRATION") else {}),
    )

    backends.reset()
    backends.configure(config=config)
    MemoryAgent.reset_instance()
    return MemoryAgent(config=config), config


def _is_question(index: int) -> bool:
    """Only for the report: which messages are worth showing in full.

    It does NOT decide the branch of the graph. Every message, question or not,
    goes through retrieve and then insert, as in the real pipeline: it is just
    that the answers to the questions are the only interesting ones to read
    among 232.

    The criteria table decides it, not the question mark. It used to be the
    other way round, and message 232 - "Riassumi tutto quello che sai di me." -
    did not end up in the report for reasons of punctuation, despite being the
    question that tests the whole memory together.
    """
    return index in fama.QUESTIONS


def questions_in(messages) -> int:
    return sum(1 for index in range(1, len(messages) + 1) if _is_question(index))


def _active_memories(state, vector_store):
    """The memories active at this instant, core then archive, with the ratings.

    Every rating not given is None, as in the fields of CoreMemoryItem.
    """
    judged = ("safety", "utility", "need", "gain")
    memories = [{"id": item.id, "content": item.content,
                 **{name: getattr(item, name) for name in judged}}
                for item in get_active_items(state["core_memory"])]
    try:
        stored = vector_store.get()
    except Exception:
        return memories

    for doc_id, content, metadata in zip(stored.get("ids") or [],
                                         stored.get("documents") or [],
                                         stored.get("metadatas") or []):
        metadata = metadata or {}
        if metadata.get("status") == "active":
            memories.append({"id": doc_id, "content": str(content),
                             **{name: metadata.get(name) for name in judged}})
    return memories


def _active_memory_text(memories) -> str:
    """The active memories of _active_memories, one per line.

    It is the object FAMA-memory is measured on. Only the active ones: what was
    deleted or superseded stays in the archive as a tombstone, and counting it
    would mean blaming consolidation for keeping the history instead of
    forgetting it.

    It must be read when the message preceding the question has been
    consolidated, not at the moment of the question: see
    fama.memory_snapshot_turn.
    """
    return "\n".join(memory["content"] for memory in memories)


def _score_memory(entries, state, vector_store):
    """FAMA-memory for the questions that were waiting for this state, and the state itself."""
    memories = _active_memories(state, vector_store)
    memory = _active_memory_text(memories)
    for entry in entries:
        entry["on_memory"] = fama.score(fama.QUESTIONS[entry["index"]], memory)
        entry["snapshot"] = memories


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
# Final report
# --------------------------------------------------------------------------- #

def _short(item_id):
    return (item_id or "-")[:ID_WIDTH]


def _clock(timestamp):
    return timestamp.strftime("%H:%M:%S") if timestamp else "--:--:--"


def _clock_iso(text):
    """Only the time of an ISO timestamp: in Chroma's metadata it is a string."""
    if not text:
        return "--:--:--"
    return str(text).split("T")[-1][:8]


def _judgment(values):
    """safety, utility, need and gain given by the classifier; empty if there are none."""
    given = [f"{name} {values.get(name)}" for name in ("safety", "utility", "need", "gain")
             if values.get(name) is not None]
    return " | " + " ".join(given) if given else ""


def _last_answer(state):
    """The answer composed by generate_answer: it is the last message of the turn.

    Here there used to be a double of explainability, a model call written
    inside the test. It was a fiction twice over: explainability does not answer
    from the memory - it answers from the results of a database query, and does
    not read the memory at all - and that double did not receive the archive, so
    the report showed retrievals that entered no answer. Now the answer is the
    service's one, and the report measures the service.
    """
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return str(message.content).strip()
    return ""


def _print_progress(index, total, agent):
    """The only thing printed during the run, one line every PROGRESS_EVERY."""
    active = len(get_active_items(agent.state["core_memory"]))
    operations = len(agent.state["operation_log"])
    safe_print(f"  [{index:>3}/{total}] core: {active} item | operations: {operations}")


def _print_operation_log(log):
    safe_print(f"\n--- OPERATION LOG ({len(log)} operations) ---")

    counters = {}
    for entry in log:
        counters[entry.op_type] = counters.get(entry.op_type, 0) + 1
    if counters:
        summary = ", ".join(f"{name} {count}" for name, count in sorted(counters.items()))
        safe_print(f"Summary: {summary}")
    safe_print("")

    if not log:
        safe_print("  (no operations recorded)")
        return

    for number, entry in enumerate(log, 1):
        score = ""
        if entry.score is not None:
            terms = ", ".join(f"{name} {value:.3f}"
                              for name, value in (entry.score_terms or {}).items())
            score = f" | score {entry.score:.3f}" + (f" ({terms})" if terms else "")
        safe_print(f"  {number:>3}  {entry.op_type:<10} | item {_short(entry.item_id)}"
                   f" | related {_short(entry.related_item_id)}"
                   f" | {_clock(entry.timestamp)}{score}{_judgment(entry.ratings or {})}"
                   f" | {entry.content}")


def _print_core_memory(state, limit):
    active = get_active_items(state["core_memory"])
    used = len("\n".join(item.content for item in active))
    safe_print(f"\n--- CORE MEMORY ({len(active)} active items, {used}/{limit} characters) ---")

    if not active:
        safe_print("  (core memory empty)")
        return

    for item in active:
        safe_print(f"  - [{_short(item.id)}] {item.content}")
        safe_print(f"      updated {_clock(item.updated_at)}"
                   f" | retrieved {_clock(item.retrieved_at)} ({item.n_retrieve})"
                   f" | used {item.n_used}{_judgment(item.model_dump())}")


def _print_archive(vector_store):
    try:
        stored = vector_store.get()
    except Exception as error:
        safe_print(f"\n--- ARCHIVE MEMORY --- (not inspectable: {error})")
        return

    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    safe_print(f"\n--- ARCHIVE MEMORY ({len(ids)} documents) ---")
    if not ids:
        safe_print("  (archive empty)")
        return

    counters = {}
    for doc_id, content, metadata in zip(ids, documents, metadatas):
        metadata = metadata or {}
        status = metadata["status"]
        counters[status] = counters.get(status, 0) + 1
        safe_print(f"  - [{_short(doc_id)}] ({status}) {content}")
        safe_print(f"      updated {_clock_iso(metadata.get('updated_at'))}"
                   f" | retrieved {_clock_iso(metadata.get('retrieved_at'))}"
                   f" ({metadata.get('n_retrieve')}) | used {metadata.get('n_used')}"
                   f"{_judgment(metadata)}")

    summary = ", ".join(f"{name} {count}" for name, count in sorted(counters.items()))
    safe_print(f"\n  Status summary: {summary}")


def _print_questions(answers):
    """The questions, what the assistant answered and what came from the archive."""
    consulted = sum(1 for answer in answers if answer["retrieved"])
    safe_print(f"\n--- QUESTIONS AND ANSWERS ({len(answers)}) ---")
    safe_print(f"Archive queried for {consulted} questions out of {len(answers)}\n")

    if not answers:
        safe_print("  (no questions asked)")
        return

    for answer in answers:
        safe_print(f"  msg {answer['index']:>3}  Q: {answer['question']}")
        safe_print(f"           A: {answer['answer']}")
        retrieved = answer["retrieved"]
        if retrieved:
            for line in str(retrieved).splitlines():
                safe_print(f"           archive: {line}")
        else:
            safe_print("           archive: (not queried, the core memory was enough)")

        on_answer, on_memory = answer["on_answer"], answer["on_memory"]
        safe_print(f"           FAMA: answer {on_answer.fama:.2f}"
                   f" | memory {on_memory.fama:.2f}"
                   f" | {fama.verdict(on_memory, on_answer)}")
        for name, scored in (("answer", on_answer), ("memory", on_memory)):
            if scored.missed:
                safe_print(f"             {name}: does not remember {list(scored.missed)}")
            if scored.leaked:
                safe_print(f"             {name}: did not forget {list(scored.leaked)}")
        safe_print("")


def _cell(value, width=7):
    """A figure of the FAMA table, or a dash if the count does not exist.

    MPA without presence criteria and FAA without absence criteria are None, not
    zero: printing them as 0.0 would say "it went terribly" where instead there
    was nothing to measure.
    """
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.1f}"


def _print_fama(answers):
    """FAMA question by question, on the answer and on the active memories.

    The usual columns would not be enough: a single number on the answer sums
    consolidation, retrieval and generate_answer, and when it drops it does not
    say which of the three gave way. The "on active memories" column applies the
    same criteria to the store right after consolidating the message that
    precedes the question, where there is no model in between:
    that is where one reads whether consolidation did its job.

    The verdict at the end of the row is the joint reading of the two:
    "answer" when the memory was right and the answer was not,
    "consolidation" when the store was already wrong, "outside memory" when the
    answer was right without having in memory what to say it from - that is,
    it answered from the conversation's context, not from the memory.
    """
    safe_print("")
    safe_print("--- FAMA (Memora, arXiv:2604.20006) ---")
    if not answers:
        safe_print("  (no questions evaluated)")
        return

    safe_print("  FAMA = max(0, MPA - lam * (1 - FAA)),  lam = N_absence / N_criteria")
    safe_print("")
    safe_print(f"  {'':>5} {'':>6} {'':>6} |{'on the answer':^21} |"
               f"{'on active memories':^21} |")
    safe_print(f"  {'msg':>5} {'P/F':>6} {'lam':>6} |{'MPA':>7}{'FAA':>7}{'FAMA':>7} |"
               f"{'MPA':>7}{'FAA':>7}{'FAMA':>7} | verdict")

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
               f" over {reply['questions']} questions")

    safe_print("")
    safe_print(f"  --- OVERALL SCORE ({reply['questions']} questions, "
               f"{reply['criteria']} criteria) ---")
    safe_print(f"  {'':<38}{'answer':>12}{'memory':>12}")
    for label, key in (("FAMA", "fama"),
                       ("MPA  (presence, micro-average)", "mpa"),
                       ("FAA  (absence, micro-average)", "faa"),
                       ("Criteria passed overall", "accuracy")):
        safe_print(f"  {label:<38}{_cell(reply[key], 12)}{_cell(store[key], 12)}")

    if verdicts:
        summary = ", ".join(f"{name} {count}"
                            for name, count in sorted(verdicts.items(), key=lambda x: -x[1]))
        safe_print(f"\n  Verdicts: {summary}")

    safe_print("")


def _print_node_costs(elapsed):
    """Where the seconds went, node by node.

    The total of a run mixes four calls with different needs and the load of a
    shared GPU. The generated tokens do not depend on the load, so the two
    columns together say whether a node cost because it reasoned a lot or
    because the cluster was busy - which is what is needed to tune
    NODE_SAMPLING one node at a time.
    """
    from memory_service.memory_manager_llm import NODE_STATS

    if not NODE_STATS:
        safe_print("")
        safe_print("--- COST PER NODE ---")
        safe_print("  (no calls recorded)")
        return

    safe_print("")
    safe_print("--- COST PER NODE ---")
    safe_print(f"  {'node':<18}{'calls':>7}{'seconds':>10}{'s/call':>9}"
               f"{'tok out':>10}{'tok/call':>11}{'share':>7}")

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
    safe_print(f"  {'total':<18}{totals['calls']:>7}{totals['seconds']:>10.1f}"
               f"{totals['seconds'] / calls:>9.1f}{totals['output_tokens']:>10}"
               f"{totals['output_tokens'] / calls:>11.0f}"
               f"{totals['seconds'] / elapsed * 100 if elapsed else 0:>6.0f}%")

    if not totals["output_tokens"]:
        safe_print("  (the server did not return the token count: "
                   "only the times remain)")


def _print_operation_matrix(records):
    """Expected operation against executed operation, message by message."""
    safe_print("")
    safe_print(f"--- CONFUSION MATRIX OF THE OPERATIONS ({len(records)} messages "
               f"consolidated) ---")
    if not records:
        safe_print("  (no messages consolidated)")
        return
    for line in operation_matrix.format_report(records):
        safe_print(line)

    wrong = [record for record in records if not operation_matrix.is_strict(record)]
    if wrong:
        safe_print("\n  Wrong (expected -> actual):")
    for record in wrong:
        note = " (acceptable)" if operation_matrix.is_lenient(record) else ""
        done = "+".join(record["types"]) or "none"
        text = CONVERSATION[record["index"] - 1][:55]
        safe_print(f"    msg {record['index']:>3}  "
                   f"{operation_matrix.display(record['expected']):<9} -> "
                   f"{done:<16}{note}  {text!r}")


def _git_commit():
    """The commit of the run, with a '+' if the tree has uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PACKAGE_ROOT,
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PACKAGE_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return commit + ("+" if commit and dirty else "")


def _save_run(config, records, elapsed):
    """Append the run to the operations file, to sum it with the others."""
    path = os.getenv("MEMORY_LONGRUN_RESULTS") or operation_matrix.DEFAULT_RESULTS
    run = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": _git_commit(),
        "model": config.llm_config.get("model_name"),
        "window": config.maximum_historical_messages,
        "core_memory_limit": config.core_memory_limit,
        "retrieval_mode": config.retrieval_mode,
        "track_used": config.counts_used,
        "consolidation_utility": config.consolidation_utility,
        "archive_candidates_k": ARCHIVE_CANDIDATES_K,
        "node_sampling": NODE_SAMPLING,
        "seconds": round(elapsed, 1),
        "messages": list(records),
    }
    try:
        operation_matrix.save_run(path, run)
        safe_print(f"Run operations appended to {path} (commit {run['commit'] or '?'})")
    except OSError as error:
        safe_print(f"Run operations not saved in {path}: {error}")


def _save_calibration_trace(config, answers):
    """Append the run's trace, if MEMORY_LONGRUN_CALIBRATION asks for it.

    For each question the memories active at its snapshot: it is all that
    calibrate_utility.py needs to remove one memory at a time.
    """
    path = os.getenv("MEMORY_LONGRUN_CALIBRATION")
    if not path:
        return
    run = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": _git_commit(),
        "model": config.llm_config.get("model_name"),
        "consolidation_utility": config.consolidation_utility,
        "questions": [{"index": entry["index"], "fama": entry["on_memory"].fama,
                       "memories": entry["snapshot"]}
                      for entry in answers if entry.get("snapshot") is not None],
    }
    try:
        operation_matrix.save_run(path, run)
        safe_print(f"Calibration trace appended to {path}")
    except OSError as error:
        safe_print(f"Calibration trace not saved in {path}: {error}")


def _print_final_report(state, vector_store, config, injected, failures, elapsed,
                       answers, inherited=0, operations=()):
    safe_print("\n" + SEPARATOR)
    safe_print("=== FINAL REPORT - LONG TERM INTERACTION ===")
    safe_print(SEPARATOR)

    safe_print(f"\nMessages injected: {injected} (failed: {len(failures)}), "
               f"of which {len(answers)} questions")
    safe_print(f"Duration: {elapsed:.1f}s")
    safe_print(f"Model: {config.llm_config.get('model_name')}")
    safe_print(f"Retrieval: {config.retrieval_mode} | uses counted: {config.counts_used}"
               f" | utility: {config.consolidation_utility}")
    safe_print(f"Archive: {config.chroma_path} / {config.collection_name}")
    if inherited:
        safe_print(f"\033[33mWARNING: the archive already held {inherited} documents "
                   f"at startup. This run's classifications were also made "
                   f"against those: run reset_archive.py before running again.\033[0m")

    if failures:
        safe_print(f"\n--- FAILED MESSAGES ({len(failures)}) ---")
        for index, message, error in failures[:10]:
            safe_print(f"  {index:>3}  {message[:60]!r}: {error[:120]}")
        if len(failures) > 10:
            safe_print(f"  ... and {len(failures) - 10} more")

    _print_operation_log(state["operation_log"])
    _print_core_memory(state, config.core_memory_limit)
    _print_archive(vector_store)
    _print_questions(answers)
    _print_fama(answers)
    _print_operation_matrix(operations)
    _print_node_costs(elapsed)

    safe_print("\n" + SEPARATOR + "\n")


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

def _assert_state_is_consistent(state):
    core_memory = state["core_memory"]
    assert all(item.status == "active" for item in core_memory), (
        "the core memory must contain only active items: superseded and deleted "
        "end up in the archive")
    assert len({item.id for item in core_memory}) == len(core_memory), "duplicate ids in core"
    for item in core_memory:
        assert item.content.strip(), "an item without content makes no sense"


def _assert_everything_that_left_core_is_archived_or_evicted(state, vector_store):
    """No item vanishes without a trace: if it leaves the core memory, it is in
    the archive or has an evict or prune entry in the log. It holds with eviction
    on and off."""
    expected = set()
    evicted = set()
    for entry in state["operation_log"]:
        if entry.op_type in ("delete", "archive"):
            expected.add(entry.item_id)
        elif entry.op_type == "update" and entry.related_item_id:
            expected.add(entry.related_item_id)
        elif entry.op_type in ("evict", "prune"):
            evicted.add(entry.item_id)

    if not expected:
        return 0

    found = set(vector_store.get(ids=sorted(expected)).get("ids") or [])
    missing = sorted(expected - found - evicted)
    assert not missing, (
        f"{len(missing)} items that left the core memory are not in the archive "
        f"and do not appear removed by eviction: {missing[:5]}")
    return len(expected)


# --------------------------------------------------------------------------- #
# The test
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

    safe_print(f"\nInjecting {len(messages)} messages one at a time, "
               f"of which {questions_in(messages)} questions "
               f"(history window: {config.maximum_historical_messages}, "
               f"core limit: {config.core_memory_limit} characters)")
    safe_print("Each message goes through the pipeline: get_memory as "
               "intent_recognition does,\nanswer as explainability does, "
               "update_memory with the question/answer pair.")
    safe_print("The agent's output is suppressed: only the progress is printed.\n")

    noise = io.StringIO()
    failures = []
    answers = []
    operations = []
    started = time.time()

    snapshot_due = {}

    for index, message in enumerate(messages, 1):
        try:
            with contextlib.redirect_stdout(noise):
                # --- like intent_recognition.listener_callback ---------------
                state = agent.run_memory_agent("retrieve", query=message)
                retrieved = state.get("retrieved_memory", "")

                # --- the service composes the answer -------------------------
                answer = _last_answer(state)

                # --- like explainability: send_update_request(input, answer) ---
                agent.run_memory_agent("insert")

            consolidated = operation_matrix.consolidated_message(
                index, config.maximum_historical_messages)
            if consolidated >= 1:
                operations.append(operation_matrix.row(
                    consolidated, [logged.op_type for logged in agent.last_operations()]))

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

        # agent.state and not `state`: the insert replaced the dictionary.
        if index in snapshot_due:
            _score_memory(snapshot_due.pop(index), agent.state, vector_store)

        if index % PROGRESS_EVERY == 0 or index == len(messages):
            _print_progress(index, len(messages), agent)
        if delay:
            time.sleep(delay)

    elapsed = time.time() - started
    state = agent.state

    # Snapshots that fell beyond the last injected message: final state.
    for entries in snapshot_due.values():
        _score_memory(entries, state, vector_store)

    _print_final_report(state, vector_store, config, len(messages), failures, elapsed,
                        answers, inherited, operations)
    _save_run(config, operations, elapsed)
    _save_calibration_trace(config, answers)

    assert len(failures) < len(messages) / 2, (
        f"{len(failures)} messages out of {len(messages)} failed: "
        "probably the server does not answer or the configuration is wrong")
    assert state["operation_log"], "a long session must have produced operations"
    _assert_state_is_consistent(state)
    _assert_everything_that_left_core_is_archived_or_evicted(state, vector_store)

    expected_questions = questions_in(messages)
    assert len(answers) == expected_questions - len(
        [f for f in failures if _is_question(f[0])]), (
        "every successful question must appear in the report")
    unanswered = [answer for answer in answers if not answer["answer"]]
    assert not unanswered, (
        f"{len(unanswered)} questions were left without an answer: "
        f"{[answer['question'] for answer in unanswered][:3]}")
