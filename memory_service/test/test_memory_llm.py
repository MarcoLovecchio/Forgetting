import ast
import os

import pytest
from langchain_core.messages import HumanMessage, AIMessage


# This test drives the real LLM/LangSmith stack, so it is only meaningful when
# the environment is configured. Without these variables the module used to raise
# at import time and break the whole test session.
pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_CONFIG") or not os.getenv("LANGSMITH_API_KEY"),
    reason="LLM_CONFIG and LANGSMITH_API_KEY are required to run the memory agent tests",
)


def _configure_langsmith():
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    model_name = ast.literal_eval(os.getenv("LLM_CONFIG"))["memory_agent"]["model_name"]
    os.environ["LANGSMITH_PROJECT"] = f'MEMORY:{model_name}'
    os.environ["LANGSMITH_TEST_SUITE"] = "Memory Service"


def test_memory_agent_insert_and_retrieve():
    _configure_langsmith()

    from memory_service.memory_manager_llm import MemoryAgent

    memory_agent = MemoryAgent()

    memory_agent.state["messages"] = [
        HumanMessage(content="Hi, can you add me to your database?"),
        AIMessage(content="Sure, I will need your name and dietary preferences."),
        HumanMessage(content="My name is Bianca and I am a vegetarian."),
        AIMessage(content="Got it, Bianca. How many calories, protein, carbs, and fat do you want per day?"),
        HumanMessage(content="I want 2000 calories, 50g protein, 250g carbs, and 70g fat."),
        AIMessage(content="Thanks, Bianca. Do you have any allergies?"),
        HumanMessage(content="Yes, I am allergic to peanuts."),
        AIMessage(content="Thanks for letting me know. I have added you to the database."),
        HumanMessage(content="Can you tell me my dietary preferences?"),
    ]

    print(memory_agent.state)

    new_state = memory_agent.run_memory_agent("insert")
    print("\n\n")
    print(new_state)
    assert "core_memory" in new_state

    state2 = memory_agent.run_memory_agent("retrieve")
    print("\n\n")
    print(state2)
    assert state2["messages"]

    memory_agent.state["messages"].append(HumanMessage(content="What can i drink in the afternoon?"))
    print(memory_agent.state)

    state3 = memory_agent.run_memory_agent("retrieve")
    print("\n\n")
    print(state3)
    assert state3["messages"]
