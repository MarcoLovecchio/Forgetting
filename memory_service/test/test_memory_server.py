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

from memory_service.consolidation import CoreMemoryItem

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

    def __init__(self, state=None, raises=None):
        self.state = state if state is not None else {"core_memory": [], "messages": []}
        self.raises = raises
        self.appended = []
        self.runs = []

    def append_message(self, message, sender):
        if self.raises:
            raise self.raises
        self.appended.append((message, sender))

    def run_memory_agent(self, interaction_mode="retrieve"):
        if self.raises:
            raise self.raises
        self.runs.append(interaction_mode)
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

    def test_only_the_active_memories_are_published(self):
        # Superseded and deleted items are history: the service must not expose them.
        server = self.build_server(StubAgent({"core_memory": [
            CoreMemoryItem(content="active fact"),
            CoreMemoryItem(content="old fact", status="superseded"),
            CoreMemoryItem(content="forgotten fact", status="deleted"),
        ], "messages": []}))

        request = UpdateMemory.Request()
        request.user_input = "a"
        request.explanation = "b"
        response = server.update_memory_callback(request, UpdateMemory.Response())

        self.assertEqual(list(response.memory_list), ["active fact"])

    def test_missing_core_memory_key_is_tolerated(self):
        server = self.build_server(StubAgent({}))

        request = UpdateMemory.Request()
        request.user_input = "a"
        request.explanation = "b"
        response = server.update_memory_callback(request, UpdateMemory.Response())

        self.assertEqual(list(response.memory_list), [])

    def test_agent_failure_returns_an_empty_response(self):
        server = self.build_server(StubAgent(raises=RuntimeError("boom")))

        request = UpdateMemory.Request()
        request.user_input = "a"
        request.explanation = "b"
        response = server.update_memory_callback(request, UpdateMemory.Response())

        self.assertEqual(list(response.memory_list), [])


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
        self.assertEqual(list(response.last_messages), ["hello", "hi there"])

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
        self.assertEqual(list(response.last_messages), [])


@unittest.skipUnless(ROS_AVAILABLE, "rclpy and memory_service_interfaces are required")
class ToStringListTest(unittest.TestCase):
    def test_none_and_empty_become_an_empty_list(self):
        self.assertEqual(to_string_list(None), [])
        self.assertEqual(to_string_list([]), [])

    def test_strings_are_left_untouched(self):
        self.assertEqual(to_string_list(["a", "b"]), ["a", "b"])

    def test_non_strings_are_stringified(self):
        self.assertEqual(to_string_list([1, ["block"], None]), ["1", "['block']", "None"])


if __name__ == "__main__":
    unittest.main()
