#!/usr/bin/env python3
"""Run the test suite of the memory service.

No API keys and no network: the chat model and the archival store are replaced
by test doubles, and the ROS node is driven with a stub agent. Usage:

    python memory_service/run_tests.py [-v]

The node tests need rclpy and the generated interfaces, so they are skipped
outside a ROS 2 environment. Everything that talks to a real model is left out
on purpose - the model gate (test_tool_calling_gate.py), the integration test
(test_memory_llm.py) and the long run (test_long_term_interaction.py) - together
with the ament linters: those belong to the colcon workflow and need a reachable
model.
"""

import os
import sys
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.path.join(PACKAGE_ROOT, "test")

for path in (PACKAGE_ROOT, TEST_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

TEST_MODULES = (
    "test_memory_agent",
    "test_consolidation",
    "test_config",
    "test_memory_server",  # skipped when rclpy is not available
)


def build_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module_name in TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(module_name))
    return suite


if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(build_suite())
    sys.exit(0 if result.wasSuccessful() else 1)
