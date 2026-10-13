"""Deletion on the real stack: chat model, embedding, ChromaDB.

The offline tests run on a double of the archive. This one checks the real
Chroma: a deleted fact must never come out of the archive again, which only
holds if the `status` filter works inside the real store.

Requirements: LLM_CONFIG and EMBEDDING_CONFIG in .config, and the two models
reachable at the address given by MEMORY_LLM_BASE_URL /
MEMORY_EMBEDDING_BASE_URL in the .env. If something does not answer the test is
skipped explaining what, instead of failing: an unreachable endpoint is not a
defect of the code.

LANGSMITH_API_KEY stays optional and only enables tracing.

Run:  pytest memory_service/test/test_memory_llm.py -v -s
"""

import ast
import os
import sys

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from live_model import live_stack_unavailable  # noqa: E402


# The test talks to the real models, so it makes sense only with the environment
# configured. Without LLM_CONFIG the module raised at import and blew up the
# whole test session.
# The module-level skip stays cheap (it only looks at the configuration): the
# reachability of the models costs a call and is checked inside the tests.
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


def test_deleted_memories_are_never_retrieved():
    """A deleted fact must never come out of the archive again."""
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

    print(f"\nDeleted item: [{fact.id}] {fact.content}")
    print(f"Operation log: {[entry.op_type for entry in log]}")

    assert core == [], "the deleted item leaves the core memory"

    retrieved = retrieve_active_archival_memories("colore preferito magenta", k=10)
    print(f"Retrieval from the archive: {retrieved}")
    assert "magenta fluorescente" not in retrieved, "the tombstone must not come back"
