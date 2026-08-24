"""Offline tests for the memory agent.

They run the real LangGraph graph with scripted backends, so they need neither
API keys nor network access, ChromaDB, rclpy or any other package of the
architecture. Run them with:

    python -m unittest discover -s memory_service/test
    pytest memory_service
"""

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
from memory_service.memory_manager_llm import (  # noqa: E402
    MemoryAgent,
    add_memory,
    insert_archival_memories,
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
        forbidden_imports = (
            "shared_utils", "db_adapters", "rclpy", "chromadb", "langchain_mistralai")
        for forbidden in forbidden_imports:
            self.assertNotIn(
                forbidden, source, f"{forbidden} must not be imported by the agent module")

    def test_importing_does_not_build_backends(self):
        backends.reset()
        self.assertIsNone(backends._llm)
        self.assertIsNone(backends._vector_store)


class InsertInteractionTest(MemoryServiceTestCase):
    tool_responses = {
        "InsertCoreMemories": {
            "memories": ["Bianca is vegetarian", "Bianca is allergic to peanuts"],
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
            state["core_memory"],
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


class CoreMemoryOverflowTest(MemoryServiceTestCase):
    """Regression test for the summarize/execute cycle that never terminated."""

    tool_responses = {
        "InsertCoreMemories": {"memories": ["x" * 300]},
        "SplitCoreAndArchivalMemory": {
            "core_memories": ["Bianca is vegetarian"],
            "archival_memories": ["Bianca visited Palermo in 2019"],
        },
    }

    def test_core_memory_over_the_limit_is_split_once_and_terminates(self):
        self.agent.state["messages"] = self.conversation(9)

        state = self.agent.run_memory_agent("insert")

        self.assertEqual(state["core_memory"], ["Bianca is vegetarian"])
        self.assertEqual(
            list(self.vector_store.documents.values()), ["Bianca visited Palermo in 2019"])
        self.assertEqual(
            self.llm.bound_tool_names().count("SplitCoreAndArchivalMemory"),
            1,
            "the split must happen once, not loop until the recursion limit",
        )


class MalformedToolCallTest(MemoryServiceTestCase):
    """The graph must survive an LLM that omits arguments."""

    tool_responses = {
        "InsertCoreMemories": {},
        "SplitCoreAndArchivalMemory": {},
    }

    def test_missing_arguments_do_not_crash_the_graph(self):
        self.agent.state["core_memory"] = ["kept fact"]
        self.agent.state["messages"] = self.conversation(9)

        state = self.agent.run_memory_agent("insert")

        self.assertEqual(state["core_memory"], ["kept fact"])
        self.assertEqual(self.vector_store.documents, {})


class RetrieveInteractionTest(MemoryServiceTestCase):
    tool_responses = {"InformationSufficiency": {"is_sufficient": True}}
    default_content = "Yes, you are vegetarian."

    def test_sufficient_information_answers_without_retrieval(self):
        self.agent.state["core_memory"] = ["Bianca is vegetarian"]
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
        )
        self.agent.state["messages"] = [HumanMessage(content="What can I drink in the afternoon?")]

        state = self.agent.run_memory_agent("retrieve")

        self.assertEqual(len(self.vector_store.searches), 1)
        self.assertEqual(self.vector_store.searches[0]["k"], 2, "k must be coerced to int")
        self.assertIn("black tea", state["retrieved_memory"])
        self.assertEqual(state["messages"][-1].content, "You like black tea in the afternoon.")


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


class ToolsTest(MemoryServiceTestCase):
    def test_archival_ids_are_unique(self):
        insert_archival_memories.invoke({"memories": ["fact one", "fact two"]})
        insert_archival_memories.invoke({"memories": ["fact one", "fact two"]})

        self.assertEqual(
            len(self.vector_store.documents), 4, "ids must not collide and overwrite")

    def test_empty_archival_insert_is_a_no_op(self):
        result = insert_archival_memories.invoke({"memories": []})

        self.assertEqual(self.vector_store.documents, {})
        self.assertIn("No memories", result)

    def test_blank_archival_memories_are_dropped(self):
        insert_archival_memories.invoke({"memories": ["   ", "real fact"]})

        self.assertEqual(list(self.vector_store.documents.values()), ["real fact"])

    def test_add_memory_does_not_overwrite_previous_entries(self):
        add_memory.invoke({"memory_content": "same content"})
        add_memory.invoke({"memory_content": "same content"})

        self.assertEqual(len(self.vector_store.documents), 2)

    def test_retrieve_memory_accepts_a_string_k(self):
        self.vector_store.add_texts(texts=["a fact about tea"], ids=["memory_a"])

        result = retrieve_memory.invoke({"query": "tea", "k": "1"})

        self.assertIn("a fact about tea", result)

    def test_retrieve_memory_without_results(self):
        self.assertEqual(
            retrieve_memory.invoke({"query": "anything"}), "No relevant memories found.")


class MessagesToStrTest(unittest.TestCase):
    def test_messages_are_prefixed_by_their_role(self):
        rendered = messages_to_str([HumanMessage(content="hi"), AIMessage(content="hello")])

        self.assertEqual(rendered, "Human: hi\nAI: hello")

    def test_plain_values_are_stringified(self):
        self.assertEqual(messages_to_str(["a", 1]), "a\n1")


if __name__ == "__main__":
    unittest.main()
