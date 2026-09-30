"""Offline tests for the memory agent.

They run the real LangGraph graph with scripted backends, so they need neither
API keys nor network access, ChromaDB, rclpy or any other package of the
architecture. The turn by turn lifecycle of the memory lives in
test_consolidation.py; this file keeps the unit level checks.

    python memory_service/run_tests.py
    pytest memory_service
"""

import contextlib
import dataclasses
import io
import os
import sys
import unittest
from datetime import datetime, timedelta

# Allow running this file directly, or through a runner that does not pick up
# the conftest.py of the package.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from memory_service import backends  # noqa: E402
from memory_service.config import NODE_SAMPLING, MemoryConfig  # noqa: E402

from memory_service.consolidation import (  # noqa: E402
    CoreMemoryItem,
    archive_items,
)
from memory_service.memory_manager_llm import (  # noqa: E402
    NODE_STATS,
    MemoryAgent,
    _timed,
    query_and_history,
    reset_node_stats,
    retrieve_memory,
    split_by_speaker,
)

from fakes import FakeVectorStore, ScriptedChatModel  # noqa: E402


# generate_answer e' acceso qui perche' molte classi esercitano il ramo
# retrieve fino alla risposta.
TEST_CONFIG = MemoryConfig(
    node_name="memory_agent",
    generate_answer=True,
    track_used=False,
    retrieval_mode="decide",
    eviction=False,
    maximum_historical_messages=5,
    core_memory_limit=150,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)


def contents(state):
    """Text of the core memories, the part assertions care about."""
    return [item.content for item in state["core_memory"]]


class MemoryServiceTestCase(unittest.TestCase):
    """Base case wiring the scripted backends and isolating the singleton."""

    tool_responses = {}
    default_content = "fake answer"

    def setUp(self):
        self.llm = ScriptedChatModel(
            tool_responses=self.tool_responses,
            default_content=self.default_content,
        )
        self.vector_store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=self.llm, vector_store=self.vector_store, config=TEST_CONFIG)
        MemoryAgent.reset_instance()
        self.agent = MemoryAgent(config=TEST_CONFIG)

    def tearDown(self):
        MemoryAgent.reset_instance()
        backends.reset()

    def archive(self, contents, status="active"):
        """Memorie in archivio con i metadata completi, come le scrive archive_items."""
        archive_items([CoreMemoryItem(id=item_id, content=content, status=status)
                       for item_id, content in contents.items()])

    def conversation(self, turns):
        """Build an alternating human/AI conversation of the given length."""
        messages = []
        for index in range(turns):
            if index % 2 == 0:
                messages.append(HumanMessage(content=f"human message {index}"))
            else:
                messages.append(AIMessage(content=f"ai message {index}"))
        return messages


class InsertInteractionTest(MemoryServiceTestCase):
    tool_responses = {
        "InsertCoreMemories": {
            "memories": [
                {"fact": "Bianca is vegetarian", "operation": "new"},
                {"fact": "Bianca is allergic to peanuts", "operation": "new"},
            ],
        },
    }

    def test_short_history_is_left_untouched(self):
        self.agent.state["messages"] = self.conversation(4)

        state = self.agent.run_memory_agent("insert")

        self.assertEqual(state["core_memory"], [])
        self.assertEqual(len(state["messages"]), 4)
        self.assertEqual(self.llm.invocations, [], "no LLM call is needed below the history limit")

    def test_long_history_is_summarized_into_core_memory(self):
        self.agent.state["messages"] = self.conversation(9)

        state = self.agent.run_memory_agent("insert")

        self.assertEqual(
            contents(state),
            ["Bianca is vegetarian", "Bianca is allergic to peanuts"],
        )
        self.assertEqual(len(state["messages"]), 5, "only the last N messages are kept")
        self.assertIn("InsertCoreMemories", self.llm.bound_tool_names())

    def test_tool_calls_are_cleared_after_the_run(self):
        self.agent.state["messages"] = self.conversation(9)

        state = self.agent.run_memory_agent("insert")

        self.assertEqual(state["tool_calls"], [])

    def test_no_messages_returns_the_current_state(self):
        state = self.agent.run_memory_agent("insert")

        self.assertEqual(state["messages"], [])
        self.assertEqual(self.llm.invocations, [])


class SplitLimitTest(MemoryServiceTestCase):
    """Lo split parte solo oltre il limite della core, e dietro non c'e' un taglio forzato."""

    def test_within_the_limit_the_split_is_not_attempted(self):
        self.agent.state["core_memory"] = [CoreMemoryItem(content="x" * 100)]
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({"InsertCoreMemories": {"memories": []}})

        self.agent.run_memory_agent("insert")

        self.assertNotIn("SplitCoreAndArchivalMemory", self.llm.bound_tool_names())

    def test_ignoring_the_limit_is_reported_but_not_forced(self):
        # Il modello decide di non archiviare nulla: la scelta viene rispettata,
        # ma non deve passare in silenzio.
        item = CoreMemoryItem(content="x" * 200)
        self.agent.state["core_memory"] = [item]
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({
            "InsertCoreMemories": {"memories": []},
            "SplitCoreAndArchivalMemory": {"decisions": []},
        })

        with contextlib.redirect_stdout(io.StringIO()) as captured:
            state = self.agent.run_memory_agent("insert")

        self.assertEqual([i.id for i in state["core_memory"]], [item.id],
                         "la decisione del modello viene rispettata, non corretta")
        self.assertIn("still over the limit", captured.getvalue())


class RetrieveInteractionTest(MemoryServiceTestCase):
    tool_responses = {"NoSearchNeeded": {"reason": "Bianca is vegetarian"}}
    default_content = "Yes, you are vegetarian."

    def test_deciding_not_to_search_leaves_the_archive_alone(self):
        self.agent.state["core_memory"] = [CoreMemoryItem(content="Bianca is vegetarian")]
        self.agent.state["messages"] = [HumanMessage(content="What are my dietary preferences?")]

        state = self.agent.run_memory_agent("retrieve")

        self.assertEqual(state["messages"][-1].content, "Yes, you are vegetarian.")
        self.assertEqual(self.vector_store.searches, [], "the archive must not be queried")

    def test_retrieve_result_is_cached_until_the_next_insert(self):
        self.agent.state["messages"] = [HumanMessage(content="What are my dietary preferences?")]

        self.agent.run_memory_agent("retrieve")
        calls_after_first = len(self.llm.invocations)
        self.agent.run_memory_agent("retrieve")

        self.assertEqual(
            len(self.llm.invocations), calls_after_first, "second retrieve is served from cache")

        self.agent.run_memory_agent("insert")
        self.agent.run_memory_agent("retrieve")

        self.assertGreater(
            len(self.llm.invocations), calls_after_first,
            "dopo un insert la memoria puo' essere cambiata: la risposta in "
            "cache non vale piu'")


class RetrieveWithArchiveTest(MemoryServiceTestCase):
    tool_responses = {
        # k deliberately arrives as a string, as the LLMs often do
        "retrieve_memory": {"query": "afternoon drink", "k": "2"},
    }
    default_content = "You like black tea in the afternoon."

    def test_a_decision_to_search_reaches_the_archive(self):
        self.archive({"memory_a": "User likes black tea in the afternoon",
                      "memory_b": "User prefers coffee in the morning"})
        self.agent.state["messages"] = [HumanMessage(content="What can I drink in the afternoon?")]

        state = self.agent.run_memory_agent("retrieve")

        self.assertEqual(len(self.vector_store.searches), 1)
        # The string "2" reached the store as an int: the fake refuses anything
        # else. The filter travels with it, so no widening is needed.
        self.assertEqual(self.vector_store.searches[0]["k"], 2)
        self.assertEqual(self.vector_store.searches[0]["filter"], {"status": "active"})
        self.assertIn("black tea", state["retrieved_memory"])
        self.assertEqual(state["messages"][-1].content, "You like black tea in the afternoon.")


class InsertDoesNotCountRetrievalsTest(MemoryServiceTestCase):
    """Il consolidamento cerca in archivio i candidati, ma non e' un recupero."""

    tool_responses = {"InsertCoreMemories": {"memories": []}}

    def test_an_insert_leaves_n_retrieve_alone(self):
        self.archive({"memory_a": "User likes black tea in the afternoon"})
        self.agent.state["messages"] = self.conversation(8)

        self.agent.run_memory_agent("insert")

        self.assertTrue(self.vector_store.searches, "il consolidamento ha cercato i candidati")
        self.assertEqual(self.vector_store.metadatas["memory_a"]["n_retrieve"], 0)


class AnswerGenerationSwitchTest(MemoryServiceTestCase):
    """Il ramo retrieve puo' non comporre la risposta.

    Quella risposta non e' restituita da nessun campo del servizio: finisce solo
    in last_messages, dove il chiamante la vede come un turno in piu' della
    conversazione - uno che l'utente non ha mai letto. Spegnerla toglie una
    chiamata per retrieve e lascia una coppia (utente, assistente) per scambio
    invece di tre messaggi.
    """

    tool_responses = {"NoSearchNeeded": {"reason": "basta la core memory"}}
    default_content = "una risposta qualsiasi"

    def run_retrieve(self, generate_answer):
        config = dataclasses.replace(TEST_CONFIG, generate_answer=generate_answer)
        backends.configure(config=config)
        MemoryAgent.reset_instance()
        agent = MemoryAgent(config=config)
        agent.state["messages"] = [HumanMessage(content="cosa mi piace bere?")]

        before = len(self.llm.invocations)
        state = agent.run_memory_agent("retrieve")
        return state, len(self.llm.invocations) - before

    def test_switched_on_the_answer_is_appended(self):
        state, calls = self.run_retrieve(generate_answer=True)

        self.assertEqual(len(state["messages"]), 2, "domanda piu' risposta")
        self.assertEqual(calls, 2, "la decisione piu' la generazione")

    def test_a_question_passed_in_is_appended_before_its_answer(self):
        # La domanda arrivata come current_query - cioe' dal campo user_input
        # della GetMemory - non e' ancora in messages. Appendere la sola
        # risposta lascerebbe la coppia invertita, e il consolidamento
        # leggerebbe una risposta senza la domanda a cui risponde.
        config = dataclasses.replace(TEST_CONFIG, generate_answer=True)
        backends.configure(config=config)
        MemoryAgent.reset_instance()
        agent = MemoryAgent(config=config)

        state = agent.run_memory_agent("retrieve", query="cosa mi piace bere?")

        self.assertEqual(len(state["messages"]), 2, "un turno, non uno solo")
        self.assertIsInstance(state["messages"][0], HumanMessage)
        self.assertEqual(state["messages"][0].content, "cosa mi piace bere?")
        self.assertIsInstance(state["messages"][1], AIMessage)

    def test_switched_off_it_costs_one_call_less_and_appends_nothing(self):
        state, calls = self.run_retrieve(generate_answer=False)

        self.assertEqual(len(state["messages"]), 1, "resta solo la domanda dell'utente")
        self.assertEqual(calls, 1, "solo la decisione")


class StaleRetrievalTest(MemoryServiceTestCase):
    """Il recupero di un giro non deve sopravvivere al giro dopo.

    tool_node conserva retrieved_memory quando la ricerca non avviene, e niente
    lo azzerava fra una get_memory e l'altra. Finche' generate_answer era spento
    non lo leggeva nessuno e il difetto era invisibile; adesso il giro che decide
    di non cercare risponderebbe con le memorie tirate su per la domanda
    precedente, e le pubblicherebbe in retrieved_memories come proprie.
    """

    tool_responses = {"retrieve_memory": {"query": "te", "k": 2}}

    def test_a_turn_that_does_not_search_starts_from_nothing(self):
        self.archive({"memory_a": "All'utente piace il te nero"})

        first = self.agent.run_memory_agent("retrieve", query="cosa bevo?")
        self.assertIn("te nero", first["retrieved_memory"])

        self.llm.script({"NoSearchNeeded": {"reason": "basta la core memory"}})
        second = self.agent.run_memory_agent("retrieve", query="come mi chiamo?")

        self.assertEqual(
            second["retrieved_memory"], "",
            "la risposta di questo giro non deve vedere il recupero del giro prima")


class UsedMemoriesTest(MemoryServiceTestCase):
    """track_used: la risposta dice quali memorie ha usato, e ciascuna conta un uso."""

    ANSWER = "Il pomeriggio bevi te nero."

    def setUp(self):
        super().setUp()
        self.long_ago = datetime.now() - timedelta(days=3)
        self.archive_at(self.long_ago, {"memory_a": "User likes black tea in the afternoon",
                                        "memory_b": "User drinks green tea in the afternoon"})

    def archive_at(self, updated_at, contents):
        archive_items([CoreMemoryItem(id=item_id, content=content, updated_at=updated_at)
                       for item_id, content in contents.items()])

    def run_retrieve(self, used_ids, track_used=True, generate_answer=True):
        config = dataclasses.replace(TEST_CONFIG, track_used=track_used,
                                     generate_answer=generate_answer)
        backends.configure(config=config)
        MemoryAgent.reset_instance()
        agent = MemoryAgent(config=config)
        agent.state["core_memory"] = [
            CoreMemoryItem(id=item_id, content=content, updated_at=self.long_ago)
            for item_id, content in (("core_name", "User is called Bianca"),
                                     ("core_diet", "User is vegetarian"))]
        self.llm.script({"retrieve_memory": {"query": "tea afternoon", "k": 2},
                         "AnswerWithUsedMemories": {"answer": self.ANSWER,
                                                    "used_memory_ids": used_ids}})
        return agent.run_memory_agent("retrieve", query="Cosa bevo il pomeriggio?")

    def core(self, state, item_id):
        return next(item for item in state["core_memory"] if item.id == item_id)

    def archived(self, item_id):
        metadata = self.vector_store.metadatas[item_id]
        return metadata["n_retrieve"], metadata["n_used"], metadata["retrieved_at"]

    def test_the_user_gets_the_answer_alone(self):
        state = self.run_retrieve(["memory_a"])

        self.assertIsInstance(state["messages"][-1], AIMessage)
        self.assertEqual(state["messages"][-1].content, self.ANSWER)

    def test_a_used_core_memory_counts_one_use(self):
        state = self.run_retrieve(["core_name"])

        used, unused = self.core(state, "core_name"), self.core(state, "core_diet")
        self.assertEqual(used.n_used, 1)
        self.assertGreater(used.retrieved_at, self.long_ago)
        self.assertEqual((unused.n_used, unused.retrieved_at), (0, self.long_ago))

    def test_only_the_used_archived_memory_is_stamped_both_count_the_retrieval(self):
        self.run_retrieve(["memory_a"])

        n_retrieve, n_used, retrieved_at = self.archived("memory_a")
        self.assertEqual((n_retrieve, n_used), (1, 1))
        self.assertGreater(retrieved_at, self.long_ago.isoformat())
        self.assertEqual(self.archived("memory_b"), (1, 0, self.long_ago.isoformat()),
                         "recuperata e non usata: conta il recupero, retrieved_at resta")

    def test_an_id_repeated_counts_once_and_one_never_shown_changes_nothing(self):
        self.archive_at(self.long_ago, {"memory_far": "User runs in the morning"})

        state = self.run_retrieve(["core_name", "core_name", "invented", "memory_far"])

        self.assertEqual(self.core(state, "core_name").n_used, 1)
        self.assertEqual(self.archived("memory_far")[1], 1,
                         "l'id esiste in archivio: il filtro e' solo l'esistenza")

    def test_switched_off_the_retrieval_stamps_and_no_use_is_counted(self):
        state = self.run_retrieve(["memory_a", "core_name"], track_used=False)

        answer_call = self.llm.invocations[-1]
        self.assertEqual(answer_call["tools"], [])
        for item_id in ("memory_a", "memory_b"):
            n_retrieve, n_used, retrieved_at = self.archived(item_id)
            self.assertEqual((n_retrieve, n_used), (1, 0))
            self.assertGreater(retrieved_at, self.long_ago.isoformat())
        self.assertEqual(self.core(state, "core_name").n_used, 0)

    def test_without_the_answer_the_switch_does_nothing(self):
        self.run_retrieve(["memory_a"], generate_answer=False)

        self.assertNotIn("AnswerWithUsedMemories", self.llm.bound_tool_names())
        n_retrieve, n_used, retrieved_at = self.archived("memory_b")
        self.assertEqual((n_retrieve, n_used), (1, 0))
        self.assertGreater(retrieved_at, self.long_ago.isoformat(),
                           "retrieved_at torna ad aggiornarsi al recupero")


class CurrentQueryTest(MemoryServiceTestCase):
    """Il recupero deve partire dalla domanda corrente, non dall'ultimo messaggio.

    Senza la query, l'unico appiglio era state["messages"][-1] - che al momento
    della get_memory e' la risposta del giro precedente, non quello che l'utente
    sta chiedendo adesso.
    """

    tool_responses = {
        "retrieve_memory": {"query": "irrilevante", "k": 3},
    }
    default_content = "risposta"

    def test_a_question_is_answered_even_with_an_empty_conversation(self):
        # L'archivio puo' contenere roba di sessioni precedenti: una domanda a
        # freddo deve comunque poterlo interrogare.
        self.archive({"memory_a": "L'utente e' allergico alle arachidi"})

        state = self.agent.run_memory_agent("retrieve", query="a cosa sono allergico?")

        self.assertTrue(self.llm.invocations, "il grafo deve girare, non uscire subito")
        self.assertIn("arachidi", state["retrieved_memory"])

    def test_a_retrieve_with_no_question_spends_nothing(self):
        # Solo messaggi dell'assistente: non c'e' niente a cui rispondere, e il
        # ramo girerebbe su un input vuoto cercando in archivio con niente.
        self.agent.state["messages"] = [AIMessage(content="ciao")]

        self.agent.run_memory_agent("retrieve")

        self.assertEqual(self.llm.invocations, [])


def _tool_name(tool):
    """Nome di uno strumento, sia esso un modello pydantic o un @tool."""
    return getattr(tool, "__name__", None) or getattr(tool, "name", str(tool))


class QueryFallbackTest(unittest.TestCase):
    """Senza user_input la domanda e' l'ultimo messaggio DELL'UTENTE.

    intent_recognition chiama get_memory senza passare user_input, e in quel
    momento l'ultimo messaggio in memoria e' la spiegazione che explainability ha
    appeso al giro prima. Ripiegando sull'ultimo messaggio e basta si cercherebbe
    in archivio con le parole dell'assistente e gli si risponderebbe come a una
    domanda: la stessa retroazione tolta dal lato del consolidamento, rimasta in
    piedi da questo.
    """

    def test_the_query_from_the_caller_wins(self):
        state = {"current_query": "quante calorie?",
                 "messages": [HumanMessage(content="e il cane?")]}

        query, history = query_and_history(state)

        self.assertEqual(query, "quante calorie?")
        self.assertEqual(len(history), 1,
                         "con la domanda dal chiamante, i messaggi sono tutti storico")

    def test_the_last_assistant_line_is_not_taken_as_a_question(self):
        state = {"current_query": "",
                 "messages": [HumanMessage(content="cosa bevo la mattina?"),
                              AIMessage(content="bevi caffe")]}

        query, history = query_and_history(state)

        self.assertEqual(query, "cosa bevo la mattina?")
        self.assertEqual(history, [],
                         "lo storico e' cio' che precede la domanda, e la "
                         "risposta a quella domanda non la precede")

    def test_without_any_user_message_there_is_no_question(self):
        state = {"current_query": "", "messages": [AIMessage(content="ciao")]}

        self.assertEqual(query_and_history(state), ("", []))


class SpeakerSeparationTest(MemoryServiceTestCase):
    """I fatti vengono dall'utente, l'assistente e' contesto in un campo suo.

    Senza la separazione l'assistente rientra: le sue risposte ripetono quello
    che gia' sa - cioe' la memoria - e il consolidamento le riestrae come fatti.
    """

    tool_responses = {"InsertCoreMemories": {"memories": []}}

    ONLY_THE_ASSISTANT_SAYS_THIS = "PAROLA-DETTA-SOLO-DALL-ASSISTENTE"

    def consolidate(self):
        # Otto messaggi con una finestra di cinque: ne escono tre, cioe' almeno
        # un turno di ciascuna voce.
        self.agent.state["messages"] = [
            HumanMessage(content="mi chiamo Bianca"),
            AIMessage(content=f"piacere, {self.ONLY_THE_ASSISTANT_SAYS_THIS}"),
            HumanMessage(content="e vivo a Palermo"),
            AIMessage(content="bella citta'"),
            HumanMessage(content="a domani"),
            AIMessage(content="a domani"),
            HumanMessage(content="ciao"),
            AIMessage(content="ciao"),
        ]
        self.agent.run_memory_agent("insert")

    def test_the_two_voices_end_up_in_different_places(self):
        said_by_user, said_by_assistant = split_by_speaker([
            HumanMessage(content="mi chiamo Bianca"),
            AIMessage(content=f"piacere, {self.ONLY_THE_ASSISTANT_SAYS_THIS}"),
        ])

        self.assertIn("mi chiamo Bianca", said_by_user)
        self.assertNotIn(self.ONLY_THE_ASSISTANT_SAYS_THIS, said_by_user,
                         "la risposta dell'assistente non deve stare fra le fonti dei fatti")
        self.assertIn(self.ONLY_THE_ASSISTANT_SAYS_THIS, said_by_assistant)
        self.assertNotIn("mi chiamo Bianca", said_by_assistant)

    def test_the_archive_is_searched_with_the_words_of_the_user(self):
        # Cercare con la risposta dell'assistente riporterebbe a galla proprio
        # le memorie che stava citando, e le renderebbe candidate per un
        # redundant: il giro si chiuderebbe dal lato della ricerca.
        self.consolidate()
        query = self.vector_store.searches[-1]["query"]

        self.assertIn("mi chiamo Bianca", query)
        self.assertNotIn(self.ONLY_THE_ASSISTANT_SAYS_THIS, query)


class NodeSamplingWiringTest(MemoryServiceTestCase):
    """Ogni nodo deve chiedere il proprio nome, e quel nome deve esistere.

    Il doppio dei test viene restituito per qualunque nome, quindi un refuso -
    get_llm("retrival") - non farebbe fallire niente: quel nodo erediterebbe in
    silenzio le impostazioni generali, e la manopola che credi di girare in
    NODE_SAMPLING non sarebbe collegata a nulla.
    """

    tool_responses = {
        "retrieve_memory": {"query": "q", "k": 2},
        "InsertCoreMemories": {"memories": []},
        "SplitCoreAndArchivalMemory": {"decisions": []},
    }

    def setUp(self):
        super().setUp()
        self.nodes = []
        original = backends.get_llm

        def spy(node=""):
            self.nodes.append(node)
            return original(node)

        backends.get_llm = spy
        self.addCleanup(setattr, backends, "get_llm", original)

    def test_every_node_asks_for_a_name_that_exists(self):
        # Una core memory oltre il limite serve a far scattare anche lo split,
        # che e' il quarto e il piu' facile da dimenticare.
        self.agent.state["core_memory"] = [CoreMemoryItem(content="x" * 200)]
        self.agent.run_memory_agent("retrieve", query="cosa bevo?")

        self.agent.state["messages"] = self.conversation(8)
        self.agent.run_memory_agent("insert")

        self.assertEqual(
            set(self.nodes),
            set(NODE_SAMPLING),
            "i nomi usati dai nodi e le chiavi di NODE_SAMPLING devono coincidere")


class NodeStatsTest(MemoryServiceTestCase):
    """Il costo di una run va attribuito al nodo che l'ha speso.

    Il totale mescola quattro chiamate con bisogni diversi e il carico di una GPU
    condivisa. Senza la ripartizione, girare una manopola di NODE_SAMPLING
    produce un numero solo e nessun modo di sapere quale nodo l'ha mosso - e i
    token, che dal carico non dipendono, separano "ha ragionato di meno" da "il
    cluster era piu' libero".
    """

    tool_responses = {
        "retrieve_memory": {"query": "q", "k": 2},
        "InsertCoreMemories": {"memories": []},
    }

    def setUp(self):
        super().setUp()
        reset_node_stats()
        self.addCleanup(reset_node_stats)

    def test_each_node_counts_its_own_calls(self):
        self.agent.run_memory_agent("retrieve", query="cosa bevo?")

        self.assertEqual(NODE_STATS["retrieval"]["calls"], 1)
        self.assertEqual(NODE_STATS["generate_answer"]["calls"], 1)
        self.assertNotIn("consolidation", NODE_STATS,
                         "il ramo insert non ha girato, non deve comparire")

    def test_the_time_is_recorded(self):
        self.agent.run_memory_agent("retrieve", query="cosa bevo?")

        self.assertGreater(NODE_STATS["retrieval"]["seconds"], 0)

    def test_the_tokens_come_from_the_usage_metadata(self):
        class Response:
            usage_metadata = {"input_tokens": 120, "output_tokens": 45}

        _timed("finto", Response)

        self.assertEqual(NODE_STATS["finto"]["input_tokens"], 120)
        self.assertEqual(NODE_STATS["finto"]["output_tokens"], 45)

    def test_a_response_without_usage_metadata_counts_zero(self):
        # I doppi non lo popolano, e nemmeno ogni provider lo restituisce:
        # l'assenza deve valere zero token, non fermare la run a meta'.
        self.agent.run_memory_agent("retrieve", query="cosa bevo?")

        self.assertEqual(NODE_STATS["retrieval"]["output_tokens"], 0)


class ToolChoiceSpy:
    """Registra come vengono legati gli strumenti, senza cambiare il risultato.

    Il doppio e' un modello pydantic e rifiuta le assegnazioni sull'istanza, per
    cui la sostituzione avviene sulla classe e viene annullata a fine test.
    """

    def __init__(self, test_case, model):
        self.calls = calls = {}
        model_class = type(model)
        original = model_class.bind_tools

        def spy(model_self, tools, **kwargs):
            for tool in tools:
                calls[_tool_name(tool)] = kwargs.get("tool_choice")
            return original(model_self, tools, **kwargs)

        model_class.bind_tools = spy
        test_case.addCleanup(setattr, model_class, "bind_tools", original)


class ToolChoiceTest(MemoryServiceTestCase):
    """Le tool call sempre attese vengono forzate, non sperate.

    Con una temperatura deliberatamente diversa da zero il modello ogni tanto
    risponde in prosa invece di emettere la chiamata. Dove una tool call e'
    l'unico esito sensato quello e' perdita silenziosa di dati: il nodo non
    salva niente e i messaggi vengono comunque tagliati.
    """

    tool_responses = {
        "InsertCoreMemories": {
            "memories": [{"fact": "L'utente si chiama Bianca", "operation": "new"}]},
    }

    def test_the_consolidation_requires_its_tool_call(self):
        spy = ToolChoiceSpy(self, self.llm)

        # Oltre la finestra, altrimenti il consolidamento non scatta affatto.
        self.agent.state["messages"] = self.conversation(8)
        self.agent.run_memory_agent("insert")

        self.assertEqual(spy.calls.get("InsertCoreMemories"), "required")


class RetrieveToolChoiceTest(MemoryServiceTestCase):
    """Sul ramo di recupero la chiamata e' obbligatoria, la scelta resta libera.

    Prima questo nodo era l'unico non forzato, perche' non cercare doveva restare
    possibile. Ma si esprimeva non emettendo la tool call, e un modello che
    sbaglia a emetterla lascia esattamente lo stesso stato: la decisione e il
    guasto erano indistinguibili. Con due strumenti la stessa liberta' passa da
    NoSearchNeeded, che si vede.
    """

    tool_responses = {"retrieve_memory": {"query": "te nero", "k": 2}}
    default_content = "Bevi te nero il pomeriggio."

    def test_both_ways_out_are_bound_and_one_of_them_is_mandatory(self):
        self.archive({"memory_a": "All'utente piace il te nero"})
        self.agent.state["messages"] = [HumanMessage(content="cosa bevo il pomeriggio?")]

        spy = ToolChoiceSpy(self, self.llm)
        self.agent.run_memory_agent("retrieve")

        self.assertEqual(spy.calls.get("retrieve_memory"), "required")
        self.assertEqual(spy.calls.get("NoSearchNeeded"), "required",
                         "non cercare deve essere una risposta, non un silenzio")


class RetrievalModeTest(MemoryServiceTestCase):
    """Con always_llm_query la ricerca in archivio avviene sempre: il modello scrive solo la query."""

    tool_responses = {"retrieve_memory": {"query": "black tea", "k": 2}}
    default_content = "Bevi te nero il pomeriggio."

    def run_retrieve(self, mode):
        config = dataclasses.replace(TEST_CONFIG, retrieval_mode=mode)
        backends.configure(config=config)
        MemoryAgent.reset_instance()
        agent = MemoryAgent(config=config)
        self.archive({"memory_a": "User likes black tea in the afternoon"})
        return agent.run_memory_agent("retrieve", query="cosa bevo il pomeriggio?")

    def test_with_the_model_query_searching_is_the_only_way_out(self):
        spy = ToolChoiceSpy(self, self.llm)

        state = self.run_retrieve("always_llm_query")

        self.assertEqual(spy.calls, {"retrieve_memory": "required"}, "NoSearchNeeded non c'e'")
        self.assertEqual(self.vector_store.searches[0]["query"], "black tea",
                         "la query la scrive il modello")
        self.assertEqual(self.vector_store.searches[0]["k"], 2)
        self.assertIn("black tea", state["retrieved_memory"])


class AppendMessageTest(MemoryServiceTestCase):
    def test_messages_are_typed_after_their_sender(self):
        self.agent.append_message("hello", "user")
        self.agent.append_message("hi there", "assistant")

        self.assertIsInstance(self.agent.state["messages"][0], HumanMessage)
        self.assertIsInstance(self.agent.state["messages"][1], AIMessage)


class RetrieveMemoryToolTest(MemoryServiceTestCase):

    def test_no_results(self):
        self.assertEqual(
            retrieve_memory.invoke({"query": "anything"}), "No relevant active memories found.")


class LastOperationsTest(MemoryServiceTestCase):
    """What the ROS response publishes: the operations of THIS call only."""

    def test_a_later_run_does_not_report_the_previous_operations(self):
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({"InsertCoreMemories": {"memories": [
            {"fact": "Bianca is vegetarian", "operation": "new"}]}})
        self.agent.run_memory_agent("insert")

        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({"InsertCoreMemories": {"memories": [
            {"fact": "Bianca lives in Palermo", "operation": "new"}]}})
        state = self.agent.run_memory_agent("insert")

        operations = self.agent.last_operations()
        self.assertEqual([entry.content for entry in operations], ["Bianca lives in Palermo"])
        self.assertEqual(len(state["operation_log"]), 2, "il log completo li tiene entrambi")

    def test_a_run_that_changes_nothing_reports_no_operation(self):
        self.agent.state["messages"] = self.conversation(2)  # sotto il limite

        self.agent.run_memory_agent("insert")

        self.assertEqual(self.agent.last_operations(), [])

    def test_no_operations_before_the_first_run(self):
        self.assertEqual(self.agent.last_operations(), [])


if __name__ == "__main__":
    unittest.main()
