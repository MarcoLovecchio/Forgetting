"""Tests for the ROS node.

They need a working ROS 2 environment (rclpy and the generated
memory_service_interfaces): the whole module is skipped otherwise, so the rest
of the suite still runs on a bare Python installation.

The node is built with a stub agent, so no LLM is contacted.
"""

import os
import sys
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import json

from memory_service.consolidation import (
    CoreMemoryItem,
    OperationLogEntry,
)

try:
    import rclpy

    from memory_service_interfaces.srv import GetMemory, UpdateMemory

    from memory_service.memory_server import MemoryServer, to_string_list

    ROS_AVAILABLE = True
except ImportError as error:  # pragma: no cover - depends on the environment
    ROS_AVAILABLE = False
    ROS_IMPORT_ERROR = error


class StubMessage:
    def __init__(self, content):
        self.content = content


class StubAgent:
    """Agent double recording what the callbacks asked for."""

    def __init__(self, state=None, raises=None, operations=None):
        self.state = state if state is not None else {"core_memory": [], "messages": []}
        self.raises = raises
        self.operations = operations or []
        self.appended = []
        self.runs = []
        self.queries = []

    def last_operations(self):
        return self.operations

    def append_message(self, message, sender):
        if self.raises:
            raise self.raises
        self.appended.append((message, sender))

    def run_memory_agent(self, interaction_mode="retrieve", query=None):
        if self.raises:
            raise self.raises
        self.runs.append(interaction_mode)
        self.queries.append(query)
        return self.state


@unittest.skipUnless(ROS_AVAILABLE, "rclpy and memory_service_interfaces are required")
class MemoryServerTestCase(unittest.TestCase):
    """Base case bringing up rclpy once for the whole class."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def build_server(self, agent):
        server = MemoryServer(agent=agent)
        self.addCleanup(server.destroy_node)
        return server


class UpdateMemoryCallbackTest(MemoryServerTestCase):
    def test_interaction_is_appended_and_core_memory_returned(self):
        agent = StubAgent(
            {"core_memory": [CoreMemoryItem(content="Bianca is vegetarian")], "messages": []})
        server = self.build_server(agent)

        request = UpdateMemory.Request()
        request.user_input = "hello"
        request.explanation = "hi there"
        response = server.update_memory_callback(request, UpdateMemory.Response())

        self.assertEqual(agent.appended, [("hello", "user"), ("hi there", "assistant")])
        self.assertEqual(agent.runs, ["insert"])
        self.assertEqual(list(response.memory_list), ["Bianca is vegetarian"])
        self.assertEqual(list(response.memory_ids), [agent.state["core_memory"][0].id])

    def test_operations_of_this_call_are_published_as_json(self):
        item = CoreMemoryItem(content="Bianca is vegetarian")
        older = CoreMemoryItem(content="Bianca lives in Palermo")
        agent = StubAgent(
            {"core_memory": [item], "messages": []},
            operations=[
                OperationLogEntry(op_type="update", item_id=item.id,
                                  related_item_id=older.id, content=item.content),
            ],
        )
        server = self.build_server(agent)

        request = UpdateMemory.Request()
        request.user_input = "a"
        request.explanation = "b"
        response = server.update_memory_callback(request, UpdateMemory.Response())

        self.assertEqual(len(response.operation_log), 1)
        operation = json.loads(response.operation_log[0])
        self.assertEqual(operation["op_type"], "update")
        self.assertEqual(operation["item_id"], item.id)
        self.assertEqual(operation["related_item_id"], older.id)
        self.assertIn("timestamp", operation)

    def test_no_operation_means_an_empty_log(self):
        server = self.build_server(StubAgent({"core_memory": [], "messages": []}))

        request = UpdateMemory.Request()
        request.user_input = "a"
        request.explanation = "b"
        response = server.update_memory_callback(request, UpdateMemory.Response())

        self.assertEqual(list(response.operation_log), [])

    def test_agent_failure_returns_an_empty_response(self):
        server = self.build_server(StubAgent(raises=RuntimeError("boom")))

        request = UpdateMemory.Request()
        request.user_input = "a"
        request.explanation = "b"
        response = server.update_memory_callback(request, UpdateMemory.Response())

        self.assertEqual(list(response.memory_list), [])
        self.assertEqual(list(response.memory_ids), [])
        self.assertEqual(list(response.operation_log), [])


class GetMemoryCallbackTest(MemoryServerTestCase):
    def test_core_memory_and_messages_are_returned(self):
        agent = StubAgent(
            {
                "core_memory": [CoreMemoryItem(content="Bianca is vegetarian")],
                "messages": [StubMessage("hello"), StubMessage("hi there")],
            }
        )
        server = self.build_server(agent)

        response = server.get_memory_callback(GetMemory.Request(), GetMemory.Response())

        self.assertEqual(agent.runs, ["retrieve"])
        self.assertEqual(list(response.memory_list), ["Bianca is vegetarian"])
        self.assertEqual(list(response.memory_ids), [agent.state["core_memory"][0].id])
        self.assertEqual(list(response.last_messages), ["hello", "hi there"])
        self.assertEqual(list(response.operation_log), [], "una retrieve non consolida")

    def test_the_user_input_reaches_the_agent_as_the_query(self):
        # Senza questo, il recupero cercherebbe in archivio partendo dall'ultimo
        # messaggio gia' in memoria invece che dalla domanda appena arrivata.
        agent = StubAgent({"core_memory": [], "messages": []})
        server = self.build_server(agent)

        request = GetMemory.Request()
        request.user_input = "a cosa sono allergico?"
        server.get_memory_callback(request, GetMemory.Response())

        self.assertEqual(agent.queries, ["a cosa sono allergico?"])

    def test_an_empty_user_input_is_still_accepted(self):
        agent = StubAgent({"core_memory": [], "messages": []})
        server = self.build_server(agent)

        server.get_memory_callback(GetMemory.Request(), GetMemory.Response())

        self.assertEqual(agent.queries, [""])

    def test_non_string_message_content_is_coerced(self):
        # Assigning a non-string to a ROS string[] field would raise: this is
        # what the coercion protects from.
        agent = StubAgent({"core_memory": [], "messages": [StubMessage([{"type": "text"}])]})
        server = self.build_server(agent)

        response = server.get_memory_callback(GetMemory.Request(), GetMemory.Response())

        self.assertEqual(list(response.last_messages), ["[{'type': 'text'}]"])

    def test_agent_failure_clears_both_fields(self):
        server = self.build_server(StubAgent(raises=RuntimeError("boom")))

        response = server.get_memory_callback(GetMemory.Request(), GetMemory.Response())

        self.assertEqual(list(response.memory_list), [])
        self.assertEqual(list(response.memory_ids), [])
        self.assertEqual(list(response.last_messages), [])
        self.assertEqual(list(response.operation_log), [])


@unittest.skipUnless(ROS_AVAILABLE, "rclpy and memory_service_interfaces are required")
class RetrievedMemoriesFieldTest(MemoryServerTestCase):
    """Quello che l'archivio ha restituito deve uscire dal servizio.

    Senza questo campo il recupero avveniva, costava una ricerca, e restava
    nello stato: al chiamante arrivava solo se qualcuno lo impacchettava dentro
    una risposta generata.
    """

    def respond_to_get(self, retrieved):
        agent = StubAgent(
            {"core_memory": [], "messages": [], "retrieved_memory": retrieved})
        server = self.build_server(agent)
        return server.get_memory_callback(GetMemory.Request(), GetMemory.Response())

    def test_what_the_archive_returned_is_published(self):
        response = self.respond_to_get(
            "ID: abc, Content: all'utente piace il te nero\n"
            "ID: def, Content: l'utente beve caffe' la mattina")

        self.assertEqual(list(response.retrieved_memories), [
            "ID: abc, Content: all'utente piace il te nero",
            "ID: def, Content: l'utente beve caffe' la mattina"])


@unittest.skipUnless(ROS_AVAILABLE, "rclpy and memory_service_interfaces are required")
class ToStringListTest(unittest.TestCase):
    def test_an_empty_list_stays_empty(self):
        self.assertEqual(to_string_list([]), [])

    def test_strings_are_left_untouched(self):
        self.assertEqual(to_string_list(["a", "b"]), ["a", "b"])

    def test_content_blocks_are_stringified(self):
        self.assertEqual(to_string_list([[{"type": "text"}]]), ["[{'type': 'text'}]"])


if __name__ == "__main__":
    unittest.main()
