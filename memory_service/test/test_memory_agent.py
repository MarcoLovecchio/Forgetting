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
    split_by_speaker,
)

from fakes import FakeVectorStore, ScriptedChatModel  # noqa: E402


# generate_answer is on here because many classes exercise the retrieve branch
# up to the answer.
TEST_CONFIG = MemoryConfig(
    node_name="memory_agent",
    generate_answer=True,
    track_used=False,
    retrieval_mode="decide",
    eviction=False,
    consolidation_utility="none",
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
        """Archived memories with complete metadata, as archive_items writes them."""
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


class SplitLimitTest(MemoryServiceTestCase):
    """The split starts only over the core limit, and there is no forced cut behind it."""

    def test_within_the_limit_the_split_is_not_attempted(self):
        self.agent.state["core_memory"] = [CoreMemoryItem(content="x" * 100)]
        self.agent.state["messages"] = self.conversation(9)
        self.llm.script({"InsertCoreMemories": {"memories": []}})

        self.agent.run_memory_agent("insert")

        self.assertNotIn("SplitCoreAndArchivalMemory", self.llm.bound_tool_names())

    def test_ignoring_the_limit_is_reported_but_not_forced(self):
        # The model decides to archive nothing: the choice is respected,
        # but it must not go unnoticed.
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
                         "the model's decision is respected, not corrected")
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
            "after an insert the memory may have changed: the cached answer "
            "no longer holds")


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
    """Consolidation searches the archive for candidates, but it is not a retrieval."""

    tool_responses = {"InsertCoreMemories": {"memories": []}}

    def test_an_insert_leaves_n_retrieve_alone(self):
        self.archive({"memory_a": "User likes black tea in the afternoon"})
        self.agent.state["messages"] = self.conversation(8)

        self.agent.run_memory_agent("insert")

        self.assertTrue(self.vector_store.searches, "consolidation searched for the candidates")
        self.assertEqual(self.vector_store.metadatas["memory_a"]["n_retrieve"], 0)


class AnswerGenerationSwitchTest(MemoryServiceTestCase):
    """The retrieve branch may not compose the answer.

    That answer is returned by no field of the service: it ends up only in
    last_messages, where the caller sees it as one more turn of the
    conversation - one the user never read. Turning it off removes one call
    per retrieve and leaves one (user, assistant) pair per exchange instead of
    three messages.
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

        self.assertEqual(len(state["messages"]), 2, "question plus answer")
        self.assertEqual(calls, 2, "the decision plus the generation")

    def test_a_question_passed_in_is_appended_before_its_answer(self):
        # The question that arrived as current_query - that is, from the
        # user_input field of GetMemory - is not in messages yet. Appending only
        # the answer would leave the pair inverted, and consolidation would read
        # an answer without the question it answers.
        config = dataclasses.replace(TEST_CONFIG, generate_answer=True)
        backends.configure(config=config)
        MemoryAgent.reset_instance()
        agent = MemoryAgent(config=config)

        state = agent.run_memory_agent("retrieve", query="cosa mi piace bere?")

        self.assertEqual(len(state["messages"]), 2, "a turn, not a single one")
        self.assertIsInstance(state["messages"][0], HumanMessage)
        self.assertEqual(state["messages"][0].content, "cosa mi piace bere?")
        self.assertIsInstance(state["messages"][1], AIMessage)

    def test_switched_off_it_costs_one_call_less_and_appends_nothing(self):
        state, calls = self.run_retrieve(generate_answer=False)

        self.assertEqual(len(state["messages"]), 1, "only the user's question stays")
        self.assertEqual(calls, 1, "only the decision")


class StaleRetrievalTest(MemoryServiceTestCase):
    """A round's retrieval must not survive the next round.

    tool_node keeps retrieved_memory when the search does not happen, and
    nothing reset it between one get_memory and the next. As long as
    generate_answer was off nobody read it and the defect was invisible; now the
    round that decides not to search would answer with the memories pulled up
    for the previous question, and would publish them in retrieved_memories as
    its own.
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
            "this round's answer must not see the previous round's retrieval")


class UsedMemoriesTest(MemoryServiceTestCase):
    """track_used: the answer says which memories it used, and each one counts a use."""

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
                         "retrieved and not used: the retrieval counts, retrieved_at stays")

    def test_an_id_repeated_counts_once_and_one_never_shown_changes_nothing(self):
        self.archive_at(self.long_ago, {"memory_far": "User runs in the morning"})

        state = self.run_retrieve(["core_name", "core_name", "invented", "memory_far"])

        self.assertEqual(self.core(state, "core_name").n_used, 1)
        self.assertEqual(self.archived("memory_far")[1], 1,
                         "the id exists in the archive: the filter is only existence")

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
                           "retrieved_at goes back to updating on retrieval")


class CurrentQueryTest(MemoryServiceTestCase):
    """Retrieval must start from the current question, not from the last message.

    Without the query, the only foothold was state["messages"][-1] - which at
    the time of get_memory is the answer of the previous round, not what the
    user is asking right now.
    """

    tool_responses = {
        "retrieve_memory": {"query": "irrilevante", "k": 3},
    }
    default_content = "risposta"

    def test_a_question_is_answered_even_with_an_empty_conversation(self):
        # The archive can hold stuff from previous sessions: a cold question
        # must still be able to query it.
        self.archive({"memory_a": "L'utente e' allergico alle arachidi"})

        state = self.agent.run_memory_agent("retrieve", query="a cosa sono allergico?")

        self.assertTrue(self.llm.invocations, "the graph must run, not exit right away")
        self.assertIn("arachidi", state["retrieved_memory"])

    def test_a_retrieve_with_no_question_spends_nothing(self):
        # Only assistant messages: there is nothing to answer, and the branch
        # would run on an empty input searching the archive with nothing.
        self.agent.state["messages"] = [AIMessage(content="ciao")]

        self.agent.run_memory_agent("retrieve")

        self.assertEqual(self.llm.invocations, [])


def _tool_name(tool):
    """Name of a tool, be it a pydantic model or a @tool."""
    return getattr(tool, "__name__", None) or getattr(tool, "name", str(tool))


class QueryFallbackTest(unittest.TestCase):
    """Without user_input the question is the last message OF THE USER.

    Some callers - explainability, inner_speech, intent_recognition at startup -
    call get_memory without passing user_input, and the last message in memory
    can be the explanation that explainability appended on the previous round.
    Falling back on just the last message, the
    archive would be searched with the assistant's words and they would be
    answered as a question: the same feedback loop removed from the
    consolidation side, still standing on this one.
    """

    def test_the_query_from_the_caller_wins(self):
        state = {"current_query": "quante calorie?",
                 "messages": [HumanMessage(content="e il cane?")]}

        query, history = query_and_history(state)

        self.assertEqual(query, "quante calorie?")
        self.assertEqual(len(history), 1,
                         "with the question from the caller, the messages are all history")

    def test_the_last_assistant_line_is_not_taken_as_a_question(self):
        state = {"current_query": "",
                 "messages": [HumanMessage(content="cosa bevo la mattina?"),
                              AIMessage(content="bevi caffe")]}

        query, history = query_and_history(state)

        self.assertEqual(query, "cosa bevo la mattina?")
        self.assertEqual(history, [],
                         "the history is what precedes the question, and the "
                         "answer to that question does not precede it")

    def test_without_any_user_message_there_is_no_question(self):
        state = {"current_query": "", "messages": [AIMessage(content="ciao")]}

        self.assertEqual(query_and_history(state), ("", []))


class SpeakerSeparationTest(MemoryServiceTestCase):
    """The facts come from the user, the assistant is context in a field of its own.

    Without the separation the assistant comes back in: its answers repeat what
    it already knows - that is, the memory - and consolidation extracts them
    again as facts.
    """

    tool_responses = {"InsertCoreMemories": {"memories": []}}

    ONLY_THE_ASSISTANT_SAYS_THIS = "PAROLA-DETTA-SOLO-DALL-ASSISTENTE"

    def consolidate(self):
        # Eight messages with a window of five: three leave, that is at least
        # one turn of each entry.
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
                         "the assistant's answer must not be among the sources of the facts")
        self.assertIn(self.ONLY_THE_ASSISTANT_SAYS_THIS, said_by_assistant)
        self.assertNotIn("mi chiamo Bianca", said_by_assistant)

    def test_the_archive_is_searched_with_the_words_of_the_user(self):
        # Searching with the assistant's answer would bring up exactly the
        # memories it was quoting, and would make them candidates for a
        # redundant: the loop would close on the search side.
        self.consolidate()
        query = self.vector_store.searches[-1]["query"]

        self.assertIn("mi chiamo Bianca", query)
        self.assertNotIn(self.ONLY_THE_ASSISTANT_SAYS_THIS, query)


class NodeSamplingWiringTest(MemoryServiceTestCase):
    """Each node must ask for its own name, and that name must exist.

    The test double is returned for any name, so a typo -
    get_llm("retrival") - would make nothing fail: that node would silently
    inherit the general settings, and the knob you think you are turning in
    NODE_SAMPLING would be connected to nothing.
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
        # A core memory over the limit also triggers the split,
        # which is the fourth and the easiest to forget.
        self.agent.state["core_memory"] = [CoreMemoryItem(content="x" * 200)]
        self.agent.run_memory_agent("retrieve", query="cosa bevo?")

        self.agent.state["messages"] = self.conversation(8)
        self.agent.run_memory_agent("insert")

        self.assertEqual(
            set(self.nodes),
            set(NODE_SAMPLING),
            "the names used by the nodes and the keys of NODE_SAMPLING must match")


class NodeStatsTest(MemoryServiceTestCase):
    """The cost of a run must be attributed to the node that spent it.

    The total mixes four calls with different needs and the load of a shared
    GPU. Without the breakdown, turning a NODE_SAMPLING knob produces a single
    number and no way to know which node moved it - and the tokens, which do
    not depend on the load, separate "it reasoned less" from "the cluster was
    freer".
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
                         "the insert branch did not run, it must not appear")

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
        # The doubles do not fill it, and not every provider returns it either:
        # its absence must count as zero tokens, not stop the run halfway.
        self.agent.run_memory_agent("retrieve", query="cosa bevo?")

        self.assertEqual(NODE_STATS["retrieval"]["output_tokens"], 0)


class ToolChoiceSpy:
    """Record how the tools are bound, without changing the result.

    The double is a pydantic model and refuses assignments on the instance, so
    the replacement happens on the class and is undone at the end of the test.
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
    """The tool calls that are always expected are forced, not hoped for.

    With a temperature deliberately different from zero the model now and then
    answers in prose instead of emitting the call. Where a tool call is the only
    sensible outcome that is silent data loss: the node saves nothing and the
    messages are cut anyway.
    """

    tool_responses = {
        "InsertCoreMemories": {
            "memories": [{"fact": "L'utente si chiama Bianca", "operation": "new"}]},
    }

    def test_the_consolidation_requires_its_tool_call(self):
        spy = ToolChoiceSpy(self, self.llm)

        # Beyond the window, otherwise consolidation does not trigger at all.
        self.agent.state["messages"] = self.conversation(8)
        self.agent.run_memory_agent("insert")

        self.assertEqual(spy.calls.get("InsertCoreMemories"), "required")


class RetrieveToolChoiceTest(MemoryServiceTestCase):
    """On the retrieval branch the call is mandatory, the choice stays free.

    Before, this node was the only one not forced, because not searching had to
    stay possible. But it was expressed by not emitting the tool call, and a
    model that fails to emit it leaves exactly the same state: the decision and
    the failure were indistinguishable. With two tools the same freedom goes
    through NoSearchNeeded, which is visible.
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
                         "not searching must be an answer, not a silence")


class RetrievalModeTest(MemoryServiceTestCase):
    """With always_llm_query the archive search always happens: the model writes only the query."""

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

        self.assertEqual(spy.calls, {"retrieve_memory": "required"}, "NoSearchNeeded is not there")
        self.assertEqual(self.vector_store.searches[0]["query"], "black tea",
                         "the model writes the query")
        self.assertEqual(self.vector_store.searches[0]["k"], 2)
        self.assertIn("black tea", state["retrieved_memory"])


class AppendMessageTest(MemoryServiceTestCase):
    def test_messages_are_typed_after_their_sender(self):
        self.agent.append_message("hello", "user")
        self.agent.append_message("hi there", "assistant")

        self.assertIsInstance(self.agent.state["messages"][0], HumanMessage)
        self.assertIsInstance(self.agent.state["messages"][1], AIMessage)


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
        self.assertEqual(len(state["operation_log"]), 2, "the full log keeps both")

    def test_a_run_that_changes_nothing_reports_no_operation(self):
        self.agent.state["messages"] = self.conversation(2)  # under the limit

        self.agent.run_memory_agent("insert")

        self.assertEqual(self.agent.last_operations(), [])


if __name__ == "__main__":
    unittest.main()
