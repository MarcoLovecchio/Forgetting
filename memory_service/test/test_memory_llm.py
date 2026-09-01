"""Ciclo di vita della memoria sullo stack reale: chat model, embedding, ChromaDB.

Stessa turnistica del test offline (test_consolidation.py), ma qui **nessuno
scripta le risposte**: a classificare i fatti e' il modello vero. Ogni turno
propone all'assistente una conversazione pensata per provocare una specifica
operazione, e stampa lo stato completo della memoria dopo il consolidamento:

    1. new         l'utente si presenta, i fatti sono tutti nuovi
    2. redundant   l'utente ripete un fatto gia' detto
    3. update      l'utente raffina un fatto (2000 -> 2200 calorie)
    4. contradict  l'utente smentisce un fatto (non piu' vegetariana)
    5. delete      l'utente chiede esplicitamente di dimenticare il nome
    6. retrieve    l'utente fa una domanda e la risposta pesca dalla memoria

Le assert sono volutamente larghe: la classificazione la decide il modello, e
pretendere una sequenza esatta di operazioni renderebbe il test instabile. Il
valore vero e' la traccia stampata, da leggere per capire se il modello ha
classificato come ci si aspetta. Le invarianti strutturali (core memory con soli
item attivi, log coerente, nessun crash) sono invece verificate sul serio.

Requisiti: LLM_CONFIG ed EMBEDDING_CONFIG in .config, e i due modelli
raggiungibili. Con Ollama: server attivo e modelli gia' scaricati con
`ollama pull`; l'indirizzo sta in MEMORY_LLM_BASE_URL / MEMORY_EMBEDDING_BASE_URL
nel .env. Se qualcosa non risponde il test si salta spiegando cosa, invece di
fallire: un server spento non e' un difetto del codice.

LANGSMITH_API_KEY resta opzionale e abilita solo il tracing.

Esecuzione:  pytest memory_service/test/test_memory_llm.py -v -s
"""

import ast
import dataclasses
import os
import sys

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import get_active_items  # noqa: E402

from live_model import live_stack_unavailable  # noqa: E402
from snapshot import print_memory_snapshot, print_turn_header  # noqa: E402


# Il test parla con i modelli veri, quindi ha senso solo con l'ambiente
# configurato. Senza LLM_CONFIG il modulo sollevava all'import e faceva saltare
# l'intera sessione di test.
# Lo skip a livello di modulo resta economico (guarda solo la configurazione):
# la raggiungibilita' dei modelli costa una chiamata e si verifica dentro i test.
pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_CONFIG"),
    reason="LLM_CONFIG is required to run the memory agent tests",
)


def _require_live_stack():
    reason = live_stack_unavailable()
    if reason:
        pytest.skip(reason)


def _configure_langsmith():
    # LangSmith tracing is optional, as in the rest of the architecture: it is
    # only enabled when a LANGSMITH_API_KEY is actually provided.
    if not os.getenv("LANGSMITH_API_KEY"):
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    model_name = ast.literal_eval(os.getenv("LLM_CONFIG"))["memory_agent"]["model_name"]
    os.environ["LANGSMITH_PROJECT"] = f'MEMORY:{model_name}'
    os.environ["LANGSMITH_TEST_SUITE"] = "Memory Service"


def _build_agent():
    """Agente sullo stack vero, con una finestra di storico corta.

    maximum_historical_messages=1 fa scattare il consolidamento a ogni turno,
    altrimenti servirebbero decine di messaggi prima di vedere qualcosa.
    """
    from memory_service.memory_manager_llm import MemoryAgent

    config = dataclasses.replace(
        MemoryConfig.from_environment(),
        maximum_historical_messages=1,
        core_memory_limit=int(os.getenv("MEMORY_CORE_MEMORY_LIMIT", "1500")),
    )
    MemoryAgent.reset_instance()
    return MemoryAgent(config=config), config


def _turn(agent, number, operation, description, user_message, assistant_message):
    """Un turno: l'utente parla, l'assistente risponde, la memoria consolida."""
    print_turn_header(number, operation, description)
    agent.append_message(user_message, "user")
    agent.append_message(assistant_message, "assistant")

    state = agent.run_memory_agent("insert")

    print_memory_snapshot(f"turno {number} - {operation}", state, backends.get_vector_store())
    _assert_state_is_consistent(state)
    return state


def _assert_state_is_consistent(state):
    """Invarianti che devono valere qualunque cosa decida il modello."""
    core_memory = state["core_memory"]
    assert all(item.status == "active" for item in core_memory), (
        "core memory deve contenere solo item attivi: superseded e deleted "
        "finiscono in archivio")
    assert len({item.id for item in core_memory}) == len(core_memory), "id duplicati in core"
    for item in core_memory:
        assert item.content.strip(), "un item senza contenuto non ha senso"
        assert item.updated_at >= item.created_at
        assert item.supersedes != item.id, "un item non puo' superare se stesso"


def _contents(state):
    return [item.content for item in get_active_items(state["core_memory"])]


def _op_types(state, since=0):
    return [entry.op_type for entry in state["operation_log"][since:]]


def test_memory_lifecycle_with_a_real_llm():
    _require_live_stack()
    _configure_langsmith()
    agent, config = _build_agent()

    print(f"\nModello: {config.llm_config.get('model_name')} "
          f"({config.llm_config.get('model_provider')})")
    print(f"Archivio: {config.chroma_path} / {config.collection_name}")

    # --- TURNO 1: new ----------------------------------------------------- #
    state = _turn(
        agent, 1, "new", "l'utente si presenta: sono tutti fatti nuovi",
        "Ciao, mi chiamo Bianca, sono vegetariana e il mio obiettivo e' 2000 calorie al giorno.",
        "Piacere Bianca, ho preso nota delle tue preferenze alimentari.",
    )
    assert _contents(state), "dopo la presentazione la core memory non puo' essere vuota"
    assert _op_types(state), "il consolidamento deve lasciare traccia nell'operation log"
    seen = len(state["operation_log"])

    # --- TURNO 2: redundant ----------------------------------------------- #
    state = _turn(
        agent, 2, "redundant", "l'utente ripete un fatto gia' noto",
        "Ti ricordo che mi chiamo Bianca.",
        "Certo Bianca, me lo ricordo.",
    )
    print(f"  -> operazioni di questo turno: {_op_types(state, seen)}")
    seen = len(state["operation_log"])

    # --- TURNO 3: update --------------------------------------------------- #
    state = _turn(
        agent, 3, "update", "l'utente raffina un fatto gia' in memoria",
        "Ho alzato il mio obiettivo giornaliero da 2000 a 2200 calorie.",
        "Aggiorno il tuo obiettivo a 2200 calorie.",
    )
    print(f"  -> operazioni di questo turno: {_op_types(state, seen)}")
    seen = len(state["operation_log"])

    # --- TURNO 4: contradict ----------------------------------------------- #
    state = _turn(
        agent, 4, "contradict", "l'utente smentisce un fatto gia' in memoria",
        "Non sono piu' vegetariana, da questo mese mangio pesce.",
        "Capito, non sei piu' vegetariana.",
    )
    print(f"  -> operazioni di questo turno: {_op_types(state, seen)}")
    seen = len(state["operation_log"])

    # --- TURNO 5: delete ---------------------------------------------------- #
    state = _turn(
        agent, 5, "delete", "l'utente chiede esplicitamente di dimenticare un fatto",
        "Dimentica il mio nome per favore, non voglio che tu lo memorizzi.",
        "Va bene, dimentico il tuo nome.",
    )
    print(f"  -> operazioni di questo turno: {_op_types(state, seen)}")

    # Qualunque cosa il modello abbia deciso turno per turno, alla fine devono
    # essere successe delle cose: la memoria non puo' essere rimasta ferma.
    all_ops = _op_types(state)
    print(f"\n  -> operation log completo: {all_ops}")
    assert len(all_ops) > 1, "cinque turni di conversazione devono muovere la memoria"

    # --- TURNO 6: retrieve --------------------------------------------------- #
    print_turn_header(6, "retrieve", "l'utente fa una domanda: la risposta usa la memoria")
    agent.state["messages"] = []
    agent.append_message("Qual e' il mio obiettivo calorico giornaliero?", "user")
    agent.state["retrieved_memory"] = ""

    state = agent.run_memory_agent("retrieve")
    print_memory_snapshot("turno 6 - retrieve", state, backends.get_vector_store())

    answer = state["messages"][-1].content
    print(f"\n  -> risposta dell'assistente: {answer}")
    assert answer, "la retrieve deve produrre una risposta"
    _assert_state_is_consistent(state)


def test_deleted_memories_are_never_retrieved():
    """Un fatto cancellato non deve piu' tornare fuori dall'archivio."""
    _require_live_stack()
    _configure_langsmith()

    from memory_service.consolidation import (
        apply_memory_operations,
        retrieve_active_archival_memories,
    )

    log = []
    core, _ = apply_memory_operations(
        [{"fact": "Il colore preferito dell'utente e' il magenta fluorescente",
          "operation": "new"}], [], log)
    fact = core[0]

    core, _ = apply_memory_operations(
        [{"fact": "dimentica il colore preferito", "operation": "delete",
          "target_item_id": fact.id}], core, log)

    print(f"\nItem cancellato: [{fact.id}] {fact.content}")
    print(f"Operation log: {[entry.op_type for entry in log]}")

    assert core == [], "l'item cancellato esce dalla core memory"

    retrieved = retrieve_active_archival_memories("colore preferito magenta", k=10)
    print(f"Recupero dall'archivio: {retrieved}")
    assert "magenta fluorescente" not in retrieved, "il tombstone non deve tornare"
