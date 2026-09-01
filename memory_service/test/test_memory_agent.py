"""Offline tests for the memory agent.

They run the real LangGraph graph with scripted backends, so they need neither
API keys nor network access, ChromaDB, rclpy or any other package of the
architecture. The turn by turn lifecycle of the memory lives in
test_consolidation.py; this file keeps the unit level checks.

    python memory_service/run_tests.py
    pytest memory_service
"""

import contextlib
import io
import os
import sys
import unittest

# Allow running this file directly, or through a runner that does not pick up
# the conftest.py of the package.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import CoreMemoryItem  # noqa: E402
from memory_service.memory_manager_llm import (  # noqa: E402
    MemoryAgent,
    messages_to_str,
    retrieve_memory,
)

from fakes import FakeVectorStore, ScriptedChatModel  # noqa: E402


TEST_CONFIG = MemoryConfig(
    node_name="memory_agent",
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

    def conversation(self, turns):
        """Build an alternating human/AI conversation of the given length."""
        messages = []
        for index in range(turns):
            if index % 2 == 0:
                messages.append(HumanMessage(content=f"human message {index}"))
            else:
                messages.append(AIMessage(content=f"ai message {index}"))
        return messages


class ImportIsolationTest(unittest.TestCase):
    """The package must not drag in the rest of the architecture."""

    def test_agent_module_has_no_architecture_dependencies(self):
        import memory_service.memory_manager_llm as module

        with open(module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        # I provider stanno solo in backends.py: il modulo dell'agente non deve
        # sapere se dietro c'e' Ollama, Mistral o altro.
        forbidden_imports = (
            "shared_utils", "db_adapters", "rclpy", "chromadb",
            "langchain_mistralai", "langchain_ollama")
        for forbidden in forbidden_imports:
            self.assertNotIn(
                forbidden, source, f"{forbidden} must not be imported by the agent module")

    def test_consolidation_module_has_no_architecture_dependencies(self):
        import memory_service.consolidation as module

        with open(module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for forbidden in ("shared_utils", "db_adapters", "rclpy", "chromadb"):
            self.assertNotIn(
                forbidden, source, f"{forbidden} must not be imported by consolidation")

    def test_importing_does_not_build_backends(self):
        backends.reset()
        self.assertIsNone(backends._llm)
        self.assertIsNone(backends._vector_store)


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

    def test_every_extracted_fact_is_logged(self):
        self.agent.state["messages"] = self.conversation(9)

        state = self.agent.run_memory_agent("insert")

        self.assertEqual([entry.op_type for entry in state["operation_log"]], ["create", "create"])

    def test_tool_calls_are_cleared_after_the_run(self):
        self.agent.state["messages"] = self.conversation(9)

        state = self.agent.run_memory_agent("insert")

        self.assertEqual(state["tool_calls"], [])

    def test_no_messages_returns_the_current_state(self):
        state = self.agent.run_memory_agent("insert")

        self.assertEqual(state["messages"], [])
        self.assertEqual(self.llm.invocations, [])


class CoreMemoryOverflowTest(MemoryServiceTestCase):
    """Regression test for the summarize/execute cycle that never terminated."""

    def test_core_memory_over_the_limit_is_split_once_and_terminates(self):
        long_item = CoreMemoryItem(content="x" * 300)
        self.agent.state["core_memory"] = [long_item]
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({
            "InsertCoreMemories": {"memories": []},
            "SplitCoreAndArchivalMemory": {
                "decisions": [{"item_id": long_item.id, "destination": "archive"}]},
        })

        state = self.agent.run_memory_agent("insert")

        self.assertEqual(state["core_memory"], [], "the item moved to the archive")
        self.assertEqual(self.vector_store.documents[long_item.id], long_item.content)
        self.assertEqual(
            self.llm.bound_tool_names().count("SplitCoreAndArchivalMemory"),
            1,
            "the split must happen once, not loop until the recursion limit",
        )

    def test_archived_item_keeps_the_core_memory_item_fields(self):
        item = CoreMemoryItem(content="y" * 300, supersedes="an-older-item")
        self.agent.state["core_memory"] = [item]
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({
            "InsertCoreMemories": {"memories": []},
            "SplitCoreAndArchivalMemory": {
                "decisions": [{"item_id": item.id, "destination": "archive"}]},
        })

        self.agent.run_memory_agent("insert")

        metadata = self.vector_store.metadatas[item.id]
        self.assertEqual(metadata["status"], "active")
        self.assertEqual(metadata["supersedes"], "an-older-item")
        self.assertEqual(metadata["created_at"], item.created_at.isoformat())
        self.assertEqual(metadata["updated_at"], item.updated_at.isoformat())


class SplitPromptTest(MemoryServiceTestCase):
    """Il limite e' una richiesta al modello: il prompt deve dargli i numeri.

    Non c'e' nessuna eviction deterministica dietro, quindi se il prompt non
    mette il modello in condizione di contare, il limite non viene rispettato.
    """

    def _run_split(self, items, limit=150):
        self.agent.state["core_memory"] = list(items)
        self.agent.state["core_memory_limit"] = limit
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({
            "InsertCoreMemories": {"memories": []},
            "SplitCoreAndArchivalMemory": {"decisions": []},
        })
        self.agent.run_memory_agent("insert")
        return "\n".join(
            invocation["prompt"] for invocation in self.llm.invocations
            if "SplitCoreAndArchivalMemory" in invocation["tools"])

    def test_the_prompt_carries_the_numbers(self):
        prompt = self._run_split([CoreMemoryItem(content="x" * 200)], limit=150)

        self.assertIn("200", prompt, "la lunghezza attuale deve essere nel prompt")
        self.assertIn("150", prompt, "il limite deve essere nel prompt")
        self.assertIn("50", prompt, "quanto liberare deve essere nel prompt")

    def test_every_memory_carries_its_own_length(self):
        prompt = self._run_split(
            [CoreMemoryItem(content="a" * 90), CoreMemoryItem(content="b" * 80)], limit=100)

        self.assertIn("(90 characters)", prompt)
        self.assertIn("(80 characters)", prompt)

    def test_the_prompt_states_the_limit_is_hard(self):
        prompt = self._run_split([CoreMemoryItem(content="x" * 200)], limit=150)

        self.assertIn("HARD constraint", prompt)
        self.assertIn("Archiving is not deleting", prompt)

    def test_ignoring_the_limit_is_reported_but_not_forced(self):
        # Il modello decide di non archiviare nulla: la scelta viene rispettata,
        # ma non deve passare in silenzio.
        item = CoreMemoryItem(content="x" * 200)
        self.agent.state["core_memory"] = [item]
        self.agent.state["core_memory_limit"] = 150
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


class MalformedToolCallTest(MemoryServiceTestCase):
    """The graph must survive an LLM that omits arguments."""

    tool_responses = {
        "InsertCoreMemories": {},
        "SplitCoreAndArchivalMemory": {},
    }

    def test_missing_arguments_do_not_crash_the_graph(self):
        kept = CoreMemoryItem(content="kept fact")
        self.agent.state["core_memory"] = [kept]
        self.agent.state["messages"] = self.conversation(9)

        state = self.agent.run_memory_agent("insert")

        self.assertEqual(contents(state), ["kept fact"])
        self.assertEqual(self.vector_store.documents, {})

    def test_malformed_operations_are_skipped(self):
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({"InsertCoreMemories": {"memories": [
            "not a dict",
            {"fact": "", "operation": "new"},
            {"fact": "a fact", "operation": "nonsense"},
            {"fact": "a good fact", "operation": "new"},
        ]}})

        state = self.agent.run_memory_agent("insert")

        self.assertEqual(contents(state), ["a good fact"], "only the valid one survives")


class RetrieveInteractionTest(MemoryServiceTestCase):
    tool_responses = {"InformationSufficiency": {"is_sufficient": True}}
    default_content = "Yes, you are vegetarian."

    def test_sufficient_information_answers_without_retrieval(self):
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


class RetrieveWithArchiveTest(MemoryServiceTestCase):
    tool_responses = {
        "InformationSufficiency": {"is_sufficient": False},
        # k deliberately arrives as a string, as the LLMs often do
        "retrieve_memory": {"query": "afternoon drink", "k": "2"},
    }
    default_content = "You like black tea in the afternoon."

    def test_insufficient_information_goes_through_the_archive(self):
        self.vector_store.add_texts(
            texts=["User likes black tea in the afternoon", "User prefers coffee in the morning"],
            ids=["memory_a", "memory_b"],
            metadatas=[{"status": "active"}, {"status": "active"}],
        )
        self.agent.state["messages"] = [HumanMessage(content="What can I drink in the afternoon?")]

        state = self.agent.run_memory_agent("retrieve")

        self.assertEqual(len(self.vector_store.searches), 1)
        self.assertEqual(self.vector_store.searches[0]["k"], 2, "k must be coerced to int")
        self.assertIn("black tea", state["retrieved_memory"])
        self.assertEqual(state["messages"][-1].content, "You like black tea in the afternoon.")


class CurrentQueryTest(MemoryServiceTestCase):
    """Il recupero deve partire dalla domanda corrente, non dall'ultimo messaggio.

    Senza la query, l'unico appiglio era state["messages"][-1] - che al momento
    della get_memory e' la risposta del giro precedente, non quello che l'utente
    sta chiedendo adesso.
    """

    tool_responses = {
        "InformationSufficiency": {"is_sufficient": False},
        "retrieve_memory": {"query": "irrilevante", "k": 3},
    }
    default_content = "risposta"

    def test_the_passed_query_is_what_reaches_the_prompts(self):
        self.agent.state["messages"] = [
            HumanMessage(content="di cosa parlavamo prima?"),
            AIMessage(content="parlavamo del tuo cane"),
        ]

        self.agent.run_memory_agent("retrieve", query="quante calorie devo assumere?")

        prompts = "\n".join(invocation["prompt"] for invocation in self.llm.invocations)
        self.assertIn("quante calorie devo assumere?", prompts)

    def test_without_a_query_it_falls_back_to_the_last_message(self):
        self.agent.state["messages"] = [HumanMessage(content="e il mio cane?")]

        self.agent.run_memory_agent("retrieve")

        prompts = "\n".join(invocation["prompt"] for invocation in self.llm.invocations)
        self.assertIn("e il mio cane?", prompts)

    def test_a_new_question_is_not_served_from_the_cache(self):
        self.agent.state["messages"] = [HumanMessage(content="un messaggio")]

        self.agent.run_memory_agent("retrieve", query="prima domanda?")
        after_first = len(self.llm.invocations)
        self.agent.run_memory_agent("retrieve", query="seconda domanda?")

        self.assertGreater(len(self.llm.invocations), after_first,
                           "una domanda diversa deve far rigirare il grafo")
        prompts = "\n".join(invocation["prompt"] for invocation in self.llm.invocations)
        self.assertIn("seconda domanda?", prompts)

    def test_the_same_question_is_still_cached(self):
        self.agent.state["messages"] = [HumanMessage(content="un messaggio")]

        self.agent.run_memory_agent("retrieve", query="stessa domanda?")
        after_first = len(self.llm.invocations)
        self.agent.run_memory_agent("retrieve", query="stessa domanda?")

        self.assertEqual(len(self.llm.invocations), after_first)

    def test_a_question_is_answered_even_with_an_empty_conversation(self):
        # L'archivio puo' contenere roba di sessioni precedenti: una domanda a
        # freddo deve comunque poterlo interrogare.
        self.vector_store.add_texts(
            texts=["L'utente e' allergico alle arachidi"], ids=["memory_a"],
            metadatas=[{"status": "active"}])

        state = self.agent.run_memory_agent("retrieve", query="a cosa sono allergico?")

        self.assertTrue(self.llm.invocations, "il grafo deve girare, non uscire subito")
        self.assertIn("arachidi", state["retrieved_memory"])

    def test_an_insert_clears_the_query(self):
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({"InsertCoreMemories": {"memories": []}})

        self.agent.run_memory_agent("retrieve", query="una domanda?")
        state = self.agent.run_memory_agent("insert")

        self.assertEqual(state["current_query"], "",
                         "la domanda vale per il retrieve che l'ha ricevuta")


class AppendMessageTest(MemoryServiceTestCase):
    def test_messages_are_typed_after_their_sender(self):
        self.agent.append_message("hello", "user")
        self.agent.append_message("hi there", "assistant")

        self.assertIsInstance(self.agent.state["messages"][0], HumanMessage)
        self.assertIsInstance(self.agent.state["messages"][1], AIMessage)

    def test_unknown_sender_is_rejected(self):
        with self.assertRaises(ValueError):
            self.agent.append_message("hello", "robot")

    def test_singleton_is_shared_but_resettable(self):
        self.assertIs(MemoryAgent(), self.agent)
        MemoryAgent.reset_instance()
        self.assertIsNot(MemoryAgent(config=TEST_CONFIG), self.agent)


class RetrieveMemoryToolTest(MemoryServiceTestCase):
    def test_string_k_is_accepted(self):
        self.vector_store.add_texts(
            texts=["a fact about tea"], ids=["memory_a"], metadatas=[{"status": "active"}])

        result = retrieve_memory.invoke({"query": "tea", "k": "1"})

        self.assertIn("a fact about tea", result)

    def test_tombstones_are_never_returned(self):
        self.vector_store.add_texts(
            texts=["a deleted fact", "a superseded fact", "a live fact"],
            ids=["memory_a", "memory_b", "memory_c"],
            metadatas=[{"status": "deleted"}, {"status": "superseded"}, {"status": "active"}],
        )

        result = retrieve_memory.invoke({"query": "fact", "k": 10})

        self.assertIn("a live fact", result)
        self.assertNotIn("a deleted fact", result)
        self.assertNotIn("a superseded fact", result)

    def test_entries_without_a_status_are_treated_as_active(self):
        # Whatever was written before consolidation existed must stay readable.
        self.vector_store.add_texts(texts=["a legacy fact"], ids=["memory_legacy"])

        self.assertIn("a legacy fact", retrieve_memory.invoke({"query": "legacy"}))

    def test_no_results(self):
        self.assertEqual(
            retrieve_memory.invoke({"query": "anything"}), "No relevant active memories found.")


class LastOperationsTest(MemoryServiceTestCase):
    """What the ROS response publishes: the operations of THIS call only."""

    def test_operations_of_the_last_run_are_reported(self):
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({"InsertCoreMemories": {"memories": [
            {"fact": "Bianca is vegetarian", "operation": "new"},
            {"fact": "Bianca is allergic to peanuts", "operation": "new"},
        ]}})

        self.agent.run_memory_agent("insert")

        self.assertEqual([entry.op_type for entry in self.agent.last_operations()],
                         ["create", "create"])

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


class MessagesToStrTest(unittest.TestCase):
    def test_messages_are_prefixed_by_their_role(self):
        rendered = messages_to_str([HumanMessage(content="hi"), AIMessage(content="hello")])

        self.assertEqual(rendered, "Human: hi\nAI: hello")

    def test_plain_values_are_stringified(self):
        self.assertEqual(messages_to_str(["a", 1]), "a\n1")


if __name__ == "__main__":
    unittest.main()
