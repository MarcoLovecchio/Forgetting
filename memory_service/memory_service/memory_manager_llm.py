import time

from pydantic import BaseModel, Field
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated, Dict, Any, List, Literal, Optional, TypedDict

from memory_service import backends
from memory_service.config import MemoryConfig
from memory_service.consolidation import (
    CONSOLIDATION_TOOLS,
    CoreMemoryItem,
    OperationLogEntry,
    SEARCH_STATS,
    apply_memory_operations,
    build_candidate_memories,
    core_memory_length,
    get_active_items,
    memory_line,
    record_uses,
    retrieve_active_archival_memories,
    serialize_core_memory_for_prompt,
    serialize_core_memory_with_dates,
    split_core_by_recency,
)
from memory_service.eviction import archive_over_limit, evict_archived_tombstones, prune_archive

class AgentState(TypedDict):
    messages: list
    tool_calls: list[Dict[str, Any]]
    core_memory: list[CoreMemoryItem]
    operation_log: list[OperationLogEntry]
    retrieved_memory: str
    current_query: str
    current_interaction: Literal["insert", "retrieve"]


def get_llm(node: str = ""):
    """Chat model used by the nodes of the graph, one set of settings per node.

    The name picks an entry of NODE_SAMPLING: see config.py to turn reasoning on
    or off, or to change the sampling, of a single node.
    """
    return backends.get_llm(node)


def get_vector_store():
    """Vector store holding the archival memories."""
    return backends.get_vector_store()

def _describe_tool_response(response) -> str:
    """One-line summary of an LLM tool-call response, for logging.

    The full AIMessage repr (id, additional_kwargs, response_metadata, ...) is
    mostly noise: what matters for tracing is which tool was picked and with
    which arguments.
    """
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return f"no tool call, content={response.content!r}"
    return "; ".join(f"{call['name']}(args={call['args']})" for call in tool_calls)

def messages_to_str(messages) -> str:
    """Convert a list of messages to a single string for prompt input."""
    if isinstance(messages, list):
        return "\n".join(messages_to_str(msg) for msg in messages)
    if isinstance(messages, HumanMessage):
        return f"Human: {messages.content}"
    return f"AI: {messages.content}"


REQUIRED = "required"

NODE_STATS: Dict[str, Dict[str, float]] = {}


def reset_node_stats() -> None:
    """Reset the counters, the archive searches too: what is measured is a run."""
    NODE_STATS.clear()
    SEARCH_STATS.update(calls=0, seconds=0.0)


def _timed(node: str, call):
    """Run the model call, recording duration and tokens."""
    started = time.perf_counter()
    response = call()
    elapsed = time.perf_counter() - started

    stats = NODE_STATS.setdefault(
        node, {"calls": 0, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0})
    stats["calls"] += 1
    stats["seconds"] += elapsed
    usage = getattr(response, "usage_metadata", None) or {}
    stats["input_tokens"] += int(usage.get("input_tokens") or 0)
    stats["output_tokens"] += int(usage.get("output_tokens") or 0)
    return response

NO_SEARCH_NEEDED = "No archive search: the facts at hand were enough."


def split_by_speaker(messages):
    """The two entries kept apart: the facts come from the user, the rest is context."""
    said_by_user = [m for m in messages if isinstance(m, HumanMessage)]
    said_by_assistant = [m for m in messages if not isinstance(m, HumanMessage)]
    return messages_to_str(said_by_user), messages_to_str(said_by_assistant)


class NoSearchNeeded(BaseModel):
    """Call this INSTEAD of retrieve_memory when the facts already at hand answer
    the query on their own, and searching the archive would add nothing.

    Deciding not to search is a real decision and it is recorded as one. It is
    not the same as not knowing: if the listed facts do not contain what the
    query asks for, the archive is where to look, so call retrieve_memory."""

    reason: str = Field(description="Which of the facts at hand already answers, in one line.")


def query_and_history(state: AgentState):
    """Question to answer and the history that precedes it.

    The retrieval branch needs to know what the user is asking. If the caller
    passed it (GetMemory.user_input -> current_query) that is used, and all the
    messages in memory are history.

    Otherwise it falls back on the last message **of the user**, not just on the
    last message. The difference matters: some callers - explainability,
    inner_speech, intent_recognition at startup - call get_memory without
    user_input, and the last message in memory can be the explanation that
    explainability appended on the previous turn. Falling back on that, the
    archive would be searched with the assistant's words and they would be
    answered as if they were a question - the same feedback loop we removed from
    the consolidation side, still standing on this one.
    """
    query = str(state.get("current_query") or "").strip()
    messages = state.get("messages") or []
    if query:
        return query, messages

    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return str(messages[index].content), messages[:index]
    return "", []


def interaction_type_node(state: AgentState) -> str:
    return state["current_interaction"]

def empty_node(state: AgentState) -> AgentState:
    return state

# k when nobody chooses it: retrieval_mode "always" without retrieve_k.
RETRIEVE_K = 20


# Tool to retrieve memories from the archive
@tool
def retrieve_memory(k: int = RETRIEVE_K,
                    query: Annotated[str, InjectedToolArg] = "",
                    stamp_retrieved_at: Annotated[bool, InjectedToolArg] = True,
                    dated: Annotated[bool, InjectedToolArg] = False) -> str:
    """Search the archive for what the user asked. Returns up to k memories.
    Think about the best value of k based on the complexity of the question."""
    results = retrieve_active_archival_memories(query, k, stamp_retrieved_at, dated)
    print(results)
    return results

# retrieval_mode "decide": the model chooses between searching and NoSearchNeeded.
DECIDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You decide whether the assistant already has what it needs, and when
         it does not, you search its archival memory.

         Facts kept at hand: {core_memory}

         Your previous interactions:

         {previous_messages}

         Those facts are only the memories kept always at hand. Everything older or less
         frequently used lives in an ARCHIVE that is not listed above and that
         retrieve_memory searches: calling it is what opens the archive, so calling it is
         not a failure. Replying "I don't know" is not an answer, it is a lookup that was
         not made. Call NoSearchNeeded only when the facts above already contain what the
         query asks for."""),
    ("human", """{input}

Do the facts kept at hand already contain what this asks for? If they do, call
NoSearchNeeded. If they do not, or you are not sure, call retrieve_memory: the answer may
be in the archive, and not looking is how it gets missed."""),
])


# Define the retrieval node
def retrieval_node(state: AgentState, runtime: Runtime[MemoryConfig]):
    print("\tRetrieval node activated")
    if runtime.context.retrieval_mode == "always":
        # Nothing for the model to decide or write: the search is the user's words.
        search = AIMessage(content="", tool_calls=[
            {"name": "retrieve_memory", "args": {"k": RETRIEVE_K}, "id": "always"}])
        return {"tool_calls": state["tool_calls"] + [search]}

    user_query, history = query_and_history(state)
    chain = DECIDE_PROMPT | get_llm("retrieval").bind_tools(
        [retrieve_memory, NoSearchNeeded], tool_choice=REQUIRED)
    response = _timed("retrieval", lambda: chain.invoke(
        {"input": user_query,
         "core_memory": serialize_core_memory_for_prompt(state["core_memory"]),
         "previous_messages": messages_to_str(history)}))

    return {"tool_calls": state["tool_calls"] + [response]}

# Define tool execution node
def tool_node(state: AgentState, runtime: Runtime[MemoryConfig]):
    print("\tTool node activated")
    messages = state["tool_calls"]
    last_message = messages[-1]

    tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "retrieve_memory":
            # A query the model still writes is dropped: the user's words find more.
            tool_args = {"query": query_and_history(state)[0],
                         "k": tool_args.get("k") or RETRIEVE_K}
            if runtime.context.retrieve_k is not None:
                tool_args["k"] = runtime.context.retrieve_k
            print(f"Invoking retrieve_memory with args: {tool_args}")
            # When uses are counted, generate_answer updates retrieved_at.
            result = retrieve_memory.invoke(
                {**tool_args, "stamp_retrieved_at": not runtime.context.counts_used,
                 "dated": runtime.context.answer_with_dates})
        elif tool_name == "NoSearchNeeded":
            print(f"\tNo archive search: {tool_args.get('reason', '')}")
            result = NO_SEARCH_NEEDED
        elif tool_name == "InsertCoreMemories":
            print(f"Invoking InsertCoreMemories with args: {tool_args}")
            state["core_memory"], result = apply_memory_operations(
                tool_args.get("memories", []), state["core_memory"], state["operation_log"])
        else:
            result = "Unknown tool"

        results.append(result)

    results = [str(result) for result in results]

    # Update memory context with retrieval results if applicable
    if "retrieve_memory" in [tc["name"] for tc in tool_calls]:
        retrieved_memory = "\n".join(results)
    else:
        retrieved_memory = state.get("retrieved_memory", "")

    return {"tool_calls": messages + [AIMessage(content="\n".join(results))],
            "retrieved_memory": retrieved_memory, "core_memory": state["core_memory"],
            "operation_log": state["operation_log"]}

# answer_with_dates: what the date in front of every fact and memory means.
DATES_NOTE = """

         Every fact and memory starts with the day it was last written: use it when the
         question asks when something happened, and to tell which of two facts that
         disagree is the newer one. The last messages are newer than any of those days."""

# track_used off: the answer alone.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the user's personal assistant. You answer questions about the user only
         from what you are given here, and nothing else.

         You are given three sources, from the oldest to the newest:
         1. Facts kept at hand - stored some time ago.
         2. Memories recalled from the archive for this question - stored some time ago. They
            are listed as ID and Content: use only the content, never mention the id.
         3. What the user said in the last messages - the NEWEST information you have. Memory
            is updated with a delay, so these messages are not in the facts or the memories yet.{dates}

         When the sources disagree about the same fact, the newest wins:
         - if the user recently gave a new value, answer with the new value only;
         - if the user recently told you something no fact mentions, you know it: answer with it;
         - if the user has just retracted a piece of previous information, answer with what they retracted;
         - if the user recently asked you to forget something, you no longer know it: say that
           you do not keep that information anymore, even if a fact or a memory still contains it.

         A recent message changes only the fact it talks about: everything else you know on the
         same topic is still true. When the question asks for several things answer with every 
         item that is still true, from all the sources.
         Say directly what is true now: never state an outdated value and then correct it,
         and never repeat a forgotten fact.

         Answer in one or two sentences, in the language the user wrote in. If none of the
         sources contains the answer, say so plainly instead of inventing it.

         Facts kept at hand: {core_memory}

         Memories recalled from the archive for this question: {retrieved_memory}

         Last messages (Human is the user, AI is you): {messages}"""),
    ("human", """{input}

Before answering, check the conversation above against the facts or among the recalled memories.
If the user has just given a newer value, use the newer one.
If the user has just asked you to forget something, it is gone: do not use it,
even if it is still listed among the facts or among the recalled memories.""")
])

# track_used on: the same answer, then the ids of the memories it is based on.
ANSWER_WITH_USED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the user's personal assistant. You answer questions about the user only
         from what you are given here, and nothing else.

         You are given three sources, from the oldest to the newest:
         1. Facts kept at hand - stored some time ago.
         2. Memories recalled from the archive for this question - stored some time ago.
         3. What the user said in the last messages - the NEWEST information you have. Memory
            is updated with a delay, so these messages are not in the facts or the memories yet.
         Facts and memories are listed as ID and Content: use only the content in the answer,
         never mention an id there.{dates}

         When the sources disagree about the same fact, the newest wins:
         - if the user recently gave a new value, answer with the new value only;
         - if the user recently told you something no fact mentions, you know it: answer with it;
         - if the user has just retracted a piece of previous information, answer with what they retracted;
         - if the user recently asked you to forget something, you no longer know it: say that
           you do not keep that information anymore, even if a fact or a memory still contains it.

         A recent message changes only the fact it talks about: everything else you know on the
         same topic is still true. When the question asks for several things answer with every 
         item that is still true, from all the sources.
         Say directly what is true now: never state an outdated value and then correct it,
         and never repeat a forgotten fact.

         Answer in one or two sentences, in the language the user wrote in. If none of the
         sources contains the answer, say so plainly instead of inventing it.

         Then list the ID of every fact and memory the answer is based on, copied exactly.
         Leave out the ones you read but did not use, and the ones a recent message replaced
         or asked to forget. If the answer comes only from the last messages, or from no
         source at all, the list is empty.

         Facts kept at hand: {core_memory}

         Memories recalled from the archive for this question: {retrieved_memory}

         Last messages (Human is the user, AI is you): {messages}"""),
    ("human", """{input}

Before answering, check the conversation above against the facts or among the recalled memories.
If the user has just given a newer value, use the newer one.
If the user has just asked you to forget something, it is gone: do not use it,
even if it is still listed among the facts or among the recalled memories.
Then list the ids of the facts and memories your answer used.""")
])


class AnswerWithUsedMemories(BaseModel):
    """The answer for the user, then the facts and memories it is based on."""

    answer: str = Field(description="The answer for the user, with no id in it.")
    used_memory_ids: List[str] = Field(description=(
        "The ID of every fact and memory the answer is based on, copied exactly. Empty "
        "when the answer uses none of them."))


def generate_answer(state: AgentState, runtime: Runtime[MemoryConfig]):
    if not runtime.context.generate_answer:
        print("\tAnswer generation disabled, skipping")
        return {}

    print("\tAnswer agent node activated")
    user_query, history = query_and_history(state)
    dated = runtime.context.answer_with_dates
    inputs = {"input": user_query,
              "retrieved_memory": state.get("retrieved_memory", ""),
              "messages": messages_to_str(history),
              "dates": DATES_NOTE if dated else ""}

    if runtime.context.track_used:
        inputs["core_memory"] = "\n".join(
            memory_line(item.content, item.id, item.updated_at if dated else None)
            for item in get_active_items(state["core_memory"])) or "(none)"
        chain = ANSWER_WITH_USED_PROMPT | get_llm("generate_answer").bind_tools(
            [AnswerWithUsedMemories], tool_choice=REQUIRED)
        args = _timed("generate_answer", lambda: chain.invoke(inputs)).tool_calls[0]["args"]
        print(f"\tUsed memories: {args['used_memory_ids']}")
        record_uses(args["used_memory_ids"], state["core_memory"])
        response = AIMessage(content=args["answer"])
    else:
        serialize = (serialize_core_memory_with_dates if dated
                     else serialize_core_memory_for_prompt)
        inputs["core_memory"] = serialize(state["core_memory"])
        chain = ANSWER_PROMPT | get_llm("generate_answer")
        response = _timed("generate_answer", lambda: chain.invoke(inputs))

    asked_out_of_band = bool(str(state.get("current_query") or "").strip())
    turn = [HumanMessage(content=user_query)] if asked_out_of_band else []

    return {"messages": state["messages"] + turn + [response]}

def exceed_memory_limit(state: AgentState, runtime: Runtime[MemoryConfig]) -> bool:
    print("\tInsert memories interaction selected")

    if len(state['messages']) > runtime.context.maximum_historical_messages:
        return True
    return False

def exceed_core_memory_limit(state: AgentState, runtime: Runtime[MemoryConfig]) -> bool:
    print("\tChecking core memory limit")

    if core_memory_length(state["core_memory"]) > runtime.context.core_memory_limit:
        return True
    return False

# The classification: the same in every consolidation_utility mode.
CONSOLIDATION_SYSTEM = """You are a tool that extracts facts about the user and classifies each one
        against the memories the assistant already has, given to you as id: content pairs
        coming from both core and archival memory.
        For each fact you extract, decide which operation applies; the operations are
        defined with the tool you must call.

        Work in this order.

        1. What kind of message is it?
           - It contains no information about the user at all - only a question, a greeting
             or small talk: return an empty list.
             A greeting or a question that also tells something about the user ("Hi, my name
             is Bianca") is information: go on with step 2.
           - It asks to forget, delete or stop storing something: find the known memory it
             refers to - by name or by category - and emit a delete on it, with that memory's text as the fact.
             If no known memory matches, return an empty list. Never store the request itself.
           - It tells something about the user: go on with step 2.

        2. For each piece of information, look among the known memories for one about the
           same thing, and ask: does what the user said make any part of it false - it
           changed, ended, or was wrong?
           - Yes: update, written as the complete memory with only the false part replaced.
           - No, and its text already says everything the user said: redundant.
           - No, and the user said something it does not say: new, even on the same topic.
             Telling more about something is a new memory; it never rewrites an old one.
           - No memory is about it: new.
           A change told as an event still makes the old memory false: "I moved to Rome"
           updates the memory with the city, "I sold my car" the one saying the user owns
           it. Two facts that can change independently are two memories.

        Facts come only from what the USER said: the assistant's replies are context, and a
        fact that appears only there gets no operation - not even a redundant.
        A known memory the user did not refer to gets no operation.
        Facts that are always true together belong in one memory.
        Two operations never share the same fact text."""

CONSOLIDATION_HUMAN = """For context only, what the assistant replied - do NOT extract facts from here:

{assistant_messages}

Known memories (id: content):
{core_memory}

What the user said - this is where the facts come from:

{user_messages}

First decide what kind of message this is, then store every piece of information about the user it contains."""

# After the classification, only with consolidation_utility: safety first, then the ratings.
SAFETY_STEP = """

        Then judge every new and every update - never a redundant or a delete - on the fact
        as you wrote it: for an update, the complete new memory.

        3. Safety: would forgetting this fact put the user at risk? Set safety to true only
           when, without it, the assistant could harm the user or put them in danger: an
           allergy, a medical condition or treatment, a physical limitation, something the
           user must avoid for their health or safety. Everything else is false, however
           important it is to the user. When safety is true, stop here: the memory is
           protected whatever its usefulness, so give it no rating."""

UTILITY_STEP = """

        4. Utility: rate utility, how valuable keeping this fact will be in future
           conversations, on the scale the tool describes. The rating is about the fact, not
           about this message: a fact is not more useful because the user repeated it or just
           said it, and not less useful because it came up in small talk. An update is worth
           what the whole new memory is worth: that it changed adds nothing."""

NEED_GAIN_STEP = """

        4. Need and gain: rate need and gain separately, on the scales the tool describes:
           need is how likely the fact will come up again, gain is what it holds that an
           answer could not guess. Rate one without looking at the other. The ratings are
           about the fact, not about this message: a fact is not more useful because the
           user repeated it or just said it, and not less useful because it came up in
           small talk. An update is worth what the whole new memory is worth: that it changed
           adds nothing."""

CONSOLIDATION_PROMPTS = {
    mode: ChatPromptTemplate.from_messages([
        ("system", CONSOLIDATION_SYSTEM + steps),
        ("human", CONSOLIDATION_HUMAN + request)])
    for mode, steps, request in (
        ("none", "", ""),
        ("single", SAFETY_STEP + UTILITY_STEP,
         " Then flag every new and update for safety and, unless safety is true, rate its"
         " utility."),
        ("need_gain", SAFETY_STEP + NEED_GAIN_STEP,
         " Then flag every new and update for safety and, unless safety is true, rate its"
         " need and its gain."),
    )
}


def summarize_memories_node(state: AgentState, runtime: Runtime[MemoryConfig]):

    keep = max(1, runtime.context.maximum_historical_messages)
    exceeding_messages = state['messages'][:-keep]
    print(f"\tSummarizing {len(exceeding_messages)} exceeding messages")
    user_messages, assistant_messages = split_by_speaker(exceeding_messages)

    mode = runtime.context.consolidation_utility
    summarizer_llm = get_llm("consolidation").bind_tools(
        [CONSOLIDATION_TOOLS[mode]], tool_choice=REQUIRED)
    chain = CONSOLIDATION_PROMPTS[mode] | summarizer_llm
    core_memories = build_candidate_memories(
        state["core_memory"], user_messages,
        count=runtime.context.count_candidate_search,
        stamp=not runtime.context.counts_used)
    known_memories = "\n".join(f"{item_id}: {content}" for item_id, content in core_memories.items())
    response = _timed("consolidation", lambda: chain.invoke({
        "user_messages": user_messages,
        "assistant_messages": assistant_messages or "(none)",
        "core_memory": known_memories or "(none)",
    }))
    print(f"\tSummarization result: {_describe_tool_response(response)}")
    return {"tool_calls": state["tool_calls"] + [response], "messages": state["messages"][-keep:]}  # Keep only the last N messages

def split_core_node(state: AgentState, runtime: Runtime[MemoryConfig]):
    """Over its character limit, the core gives its least recent memories to the archive."""
    limit = runtime.context.core_memory_limit
    print(f"\tCore memory over its limit ({core_memory_length(state['core_memory'])}/{limit} "
          f"characters): the least recent memories move to the archive")
    state["core_memory"], result = split_core_by_recency(
        state["core_memory"], limit, state["operation_log"])
    print(f"\t{result}")
    return {"core_memory": state["core_memory"], "operation_log": state["operation_log"]}


def eviction_node(state: AgentState, runtime: Runtime[MemoryConfig]):
    """End of the insert branch: the tombstones go."""
    if not runtime.context.eviction:
        return {}
    log = list(state["operation_log"])
    evict_archived_tombstones(log)
    return {"operation_log": log}


def exceed_archive_limit(state: AgentState, runtime: Runtime[MemoryConfig]) -> bool:
    if not runtime.context.eviction:
        return False
    return archive_over_limit(runtime.context.archive_memory_limit) > 0


def prune_node(state: AgentState, runtime: Runtime[MemoryConfig]):
    log = list(state["operation_log"])
    prune_archive(log, runtime.context)
    return {"operation_log": log}


# Build the graph
graph = StateGraph(AgentState, context_schema=MemoryConfig)

graph.add_node("retrieve", retrieval_node)
graph.add_node("execute_tool", tool_node)
graph.add_node("generate_answer", generate_answer)

graph.add_edge("retrieve", "execute_tool")
graph.add_edge("execute_tool", "generate_answer")
graph.add_edge("generate_answer", END)

# Insertion nodes
graph.add_node("insert_memories", empty_node)
graph.add_node("summarize_memories", summarize_memories_node)
graph.add_node("split_core", split_core_node)
graph.add_node("execute_insertion_tool", tool_node)
graph.add_node("evict_archive", eviction_node)
graph.add_node("prune_archive", prune_node)
graph.add_edge("summarize_memories",  "execute_insertion_tool")
graph.add_edge("split_core", "evict_archive")
graph.add_edge("prune_archive", END)

graph.add_conditional_edges(START, interaction_type_node, {'insert': "insert_memories", 'retrieve': "retrieve"})

graph.add_conditional_edges('insert_memories', exceed_memory_limit, {True: "summarize_memories", False: "evict_archive"})
graph.add_conditional_edges('execute_insertion_tool', exceed_core_memory_limit, {True: "split_core", False: "evict_archive"})
graph.add_conditional_edges('evict_archive', exceed_archive_limit, {True: "prune_archive", False: END})

# Compile the graph
memory_agent = graph.compile()

class MemoryAgent():
    _instance = None  # class-level reference to hold the singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # if no instance exists, create one
            cls._instance = super(MemoryAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[MemoryConfig] = None):
        if not hasattr(self, "state"):  # prevents re-initialization
            self.config = config or backends.get_config()
            self.state = {
                "core_memory": [],
                "operation_log": [],
                "messages": [],
                "retrieved_memory": "",
                "current_query": "",
                "tool_calls": []
            }
            self.up_to_date = False
            self._log_offset = 0

    @classmethod
    def reset_instance(cls):
        """Drop the singleton, so that a fresh agent can be built.

        The singleton is convenient for the ROS node (a single memory shared by
        every service call) but it leaks state across test cases.
        """
        cls._instance = None

    def last_operations(self):
        """Consolidation operations produced by the most recent run.

        The full log lives in state["operation_log"] and grows for the whole life
        of the node; what a caller usually wants is what changed just now.
        """
        return self.state["operation_log"][self._log_offset:]

    # Function to run the agent
    def run_memory_agent(self, interaction_mode="retrieve", query=None):
        """Run the graph.

        `query` is the user message the retrieve branch has to answer. Without
        it, retrieval falls back on the last user message in memory - the
        question of the previous turn, not what the user is asking right now
        (see query_and_history).
        """
        # Anything logged from here on belongs to this run.
        self._log_offset = len(self.state["operation_log"])
        query = str(query or "").strip()

        if interaction_mode == "retrieve":
            if self.up_to_date and query == self.state.get("current_query", ""):
                return self.state
            self.up_to_date = True
            self.state["retrieved_memory"] = ""

        if interaction_mode == "insert":
            self.up_to_date = False

        self.state["current_interaction"] = interaction_mode
        self.state["current_query"] = query if interaction_mode == "retrieve" else ""

        if interaction_mode == "retrieve" and not query_and_history(self.state)[0]:
            print("No question to answer.")
            return self.state

        if not self.state["messages"] and not self.state["current_query"]:
            print("No messages to process.")
            return self.state

        print(f"\t[{interaction_mode}] running with {len(self.state['messages'])} messages, "
              f"{len(get_active_items(self.state['core_memory']))} active core memories")
        self.state = memory_agent.invoke(self.state, context=self.config)
        self.state["tool_calls"] = []
        return self.state

    def append_message(self, message: str, sender: Literal["user", "assistant"]):
        if sender == "user":
            self.state["messages"].append(
                HumanMessage(content=message, additional_kwargs={}, response_metadata={})
            )
        else:
            self.state["messages"].append(
                AIMessage(content=message, additional_kwargs={}, response_metadata={})
            )

# Fill archive with fake memories
# fake_memories = [
#     "User prefers coffee over tea in the morning.",
#     "User likes black tea in the afternoon.",
#     "User's favourite dish is risotto with mushrooms.",
#     "User feels unwell when eating goat cheese.",
#     "User's preferred snack is dark chocolate.",
# ]

# insert_archival_memories.invoke({"memories": fake_memories})
