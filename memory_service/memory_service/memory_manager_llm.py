from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any, Literal, Optional, TypedDict

from memory_service import backends
from memory_service.config import MemoryConfig
from memory_service.consolidation import (
    CoreMemoryItem,
    InsertCoreMemories,
    OperationLogEntry,
    SplitCoreAndArchivalMemory,
    apply_memory_operations,
    apply_split_decisions,
    build_candidate_memories,
    core_memory_length,
    get_active_items,
    retrieve_active_archival_memories,
    serialize_core_memory_for_prompt,
)

# Define the agent state.
# NOTE: TypedDict does not support default values, so every key has to be provided
# by the caller (see MemoryAgent.__init__ for the initial values).
class AgentState(TypedDict):
    messages: list
    tool_calls: list[Dict[str, Any]]
    core_memory: list[CoreMemoryItem]  # active items only, see memory_service.consolidation
    operation_log: list[OperationLogEntry]
    retrieved_memory: str
    current_query: str  # domanda a cui il ramo retrieve deve rispondere
    current_interaction: Literal["insert", "retrieve"]
    maximum_historical_messages: int  # Limit the number of historical messages to keep
    core_memory_limit: int  # Character limit for core memory


# The chat model and the archival vector store are built on first use, see
# memory_service.backends. Tests (and any other application) can replace them
# with backends.configure(llm=..., vector_store=...).
def get_llm():
    """Chat model used by every node of the graph."""
    return backends.get_llm()


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
    elif isinstance(messages, HumanMessage):
        return f"Human: {messages.content}"
    elif isinstance(messages, AIMessage):
        return f"AI: {messages.content}"
    else:
        return str(messages)


# Asking for the tool is not the same as getting it: with a temperature that
# is deliberately not zero, the model occasionally answers in prose instead of
# emitting the call. Where a tool call is ALWAYS the expected outcome, that is
# silent data loss - the node stores nothing and the messages are trimmed all
# the same - so the call is required rather than hoped for. An empty result is
# still expressible: the model can return no memories, no decisions.
REQUIRED = "required"


# Tool to check if information is sufficient to answer the query
class InformationSufficiency(BaseModel):
    """This tool checks if the information provided is sufficient to answer the user's query.
    Answer True if sufficient, False otherwise.
    Estimate sufficiency ONLY based on the provided context."""
    is_sufficient: bool


def query_and_history(state: AgentState):
    """Domanda a cui rispondere e storico che la precede.

    Il ramo di recupero ha bisogno di sapere che cosa sta chiedendo l'utente.
    Se il chiamante l'ha passata (GetMemory.user_input -> current_query) si usa
    quella, e tutti i messaggi in memoria sono storico. Altrimenti si ripiega
    sull'ultimo messaggio noto, che e' quello che succedeva sempre prima che il
    servizio potesse ricevere la domanda.
    """
    query = str(state.get("current_query") or "").strip()
    messages = state.get("messages") or []
    if query:
        return query, messages
    if not messages:
        return "", []
    return messages[-1].content, messages[:-1]


def interaction_type_node(state: AgentState) -> str:
    return state["current_interaction"]

def empty_node(state: AgentState) -> AgentState:
    return state

def check_information_sufficiency(state: AgentState) -> bool:
    print("\tChecking information sufficiency")

    user_query, history = query_and_history(state)
    if not user_query:
        return False

    core_memory = serialize_core_memory_for_prompt(state["core_memory"])
    previous_messages = messages_to_str(history)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a tool that checks if the information provided is sufficient to answer the user's query."),
        ("human", """User query: {user_query}
Known facts: {core_memory}
Your previous interactions:

{previous_messages}

Is the information sufficient to answer the query?""")
    ])

    router_llm = get_llm().bind_tools([InformationSufficiency], tool_choice=REQUIRED)
    chain = prompt | router_llm
    response = chain.invoke({"user_query": user_query, "core_memory": core_memory, "previous_messages": previous_messages})
    # extract and coerce
    try:
        is_sufficient = response.tool_calls[0]['args']["is_sufficient"]
        if isinstance(is_sufficient, str):
            is_sufficient = is_sufficient.strip().lower() in ["true", "1", "yes"]
    except Exception:
        is_sufficient = False
    print(f"\tInformation sufficiency: {is_sufficient}")

    return bool(is_sufficient)


# Tool to retrieve memories from the archive
@tool
def retrieve_memory(query: str, k: int = 3) -> str:
    """Retrieve relevant memories from the archival vector store based on a query.
    Returns up to k relevant memories.
    Think about the best value of k based on the complexity of the query."""
    # Superseded and deleted memories are tombstones: they must not come back.
    results = retrieve_active_archival_memories(query, k)
    print(results)
    return results

# Define the retrieval node
def retrieval_node(state: AgentState):
    print("\tRetrieval node activated")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an agent who retrieves memories from the archive to enhance currently available context.
         Current memory context: {core_memory}"""),
        ("human", "{input}")
    ])

    # Bind tools to LLM
    # Non forzato, a differenza degli altri: qui decidere di NON recuperare e'
    # una risposta valida, ed e' il comportamento dell'architettura originale.
    llm_with_tools = get_llm().bind_tools([retrieve_memory])

    chain = prompt | llm_with_tools
    user_query, _ = query_and_history(state)
    response = chain.invoke({"input": user_query,
                             "core_memory": serialize_core_memory_for_prompt(state["core_memory"])})

    return {"tool_calls": state["tool_calls"] + [response]}

# Define tool execution node
def tool_node(state: AgentState):
    print("\tTool node activated")
    messages = state["tool_calls"]
    last_message = messages[-1]

    tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "retrieve_memory":
            print(f"Invoking retrieve_memory with args: {tool_args}")
            result = retrieve_memory.invoke(tool_args)
        elif tool_name == "InsertCoreMemories":
            print(f"Invoking InsertCoreMemories with args: {tool_args}")
            # Each extracted fact carries its own classification: consolidation
            # decides whether it becomes a new item, reinforces, supersedes or
            # deletes an existing one, in core memory or in the archive.
            state["core_memory"], result = apply_memory_operations(
                tool_args.get("memories", []), state["core_memory"], state["operation_log"])
        elif tool_name == "SplitCoreAndArchivalMemory":
            print(f"Invoking SplitCoreAndArchivalMemory with args: {tool_args}")
            state["core_memory"], result = apply_split_decisions(
                tool_args.get("decisions", []), state["core_memory"], state["operation_log"])
            # Il limite resta una richiesta al modello, non un vincolo imposto:
            # se non lo rispetta non si interviene, ma non deve passare in
            # silenzio.
            remaining = core_memory_length(state["core_memory"])
            if remaining > state["core_memory_limit"]:
                print(f"\tWARNING: core memory still over the limit after the split "
                      f"({remaining}/{state['core_memory_limit']} characters)")
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

def generate_answer(state: AgentState):
    print("\tAnswer agent node activated")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful agent that replies to user queries.
         You answer only based on the following information:
         Facts about the user: {core_memory}
         Retrieved memories: {retrieved_memory}
         Your previous interactions: {messages}
         Select the most relevant information to answer the user's query as best as you can.
         If you don't know the answer, simply say you don't know.
         Do not make up an answer."""),
        ("human", "{input}")
    ])

    chain = prompt | get_llm()

    user_query, history = query_and_history(state)
    response = chain.invoke({"input": user_query,
                             "core_memory": serialize_core_memory_for_prompt(state["core_memory"]),
                             "retrieved_memory": state.get("retrieved_memory", ""),
                             "messages": messages_to_str(history)})

    return {"messages": state["messages"] + [response]}

def exceed_memory_limit(state: AgentState) -> bool:
    print("\tInsert memories interaction selected")

    if len(state['messages']) > state['maximum_historical_messages']:
        return True
    return False

def exceed_core_memory_limit(state: AgentState) -> bool:
    print("\tChecking core memory limit")

    if core_memory_length(state["core_memory"]) > state["core_memory_limit"]:
        return True
    return False

def summarize_memories_node(state: AgentState):

    keep = max(1, state['maximum_historical_messages'])
    exceeding_messages = state['messages'][:-keep]
    print(f"\tSummarizing {len(exceeding_messages)} exceeding messages")
    exceeding_messages = messages_to_str(exceeding_messages)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a tool that extracts new facts from the messages and classifies
        each one against the memories the assistant already has, given to you as id: content
        pairs coming from both core and archival memory.
        For each fact decide whether it is new, redundant, an update, a contradict or - only
        when the user explicitly asked to forget something - a delete with respect to an
        existing memory, and reference that memory's id whenever the fact is not new.
        Only include facts that are relevant and likely to be referenced in future interactions."""),
        ("human", """The messages exceeding the limit are:

        {exceeding_messages}

Extract any new facts about the user from these messages and classify each one.
Known memories (id: content) are: {core_memory}.
Focus on preferences, opinions, or personal facts mentioned by the user.""")
    ])
    summarizer_llm = get_llm().bind_tools([InsertCoreMemories], tool_choice=REQUIRED)
    chain = prompt | summarizer_llm
    # The classifier can point at core memories and at related archival ones.
    core_memories = build_candidate_memories(state["core_memory"], exceeding_messages)
    response = chain.invoke({"exceeding_messages": exceeding_messages, "core_memory": core_memories})
    print(f"\tSummarization result: {_describe_tool_response(response)}")
    return {"tool_calls": state["tool_calls"] + [response], "messages": state["messages"][-keep:]}  # Keep only the last N messages

def summarize_core_memories_node(state: AgentState):

    active_items = get_active_items(state["core_memory"])
    # La lunghezza di ogni memoria e' nel prompt apposta: senza, al modello si
    # chiede di rispettare un limite in caratteri senza dargli modo di contarli.
    core_memory_display = "\n".join(
        f"{item.id}: {item.content} ({len(item.content)} characters)" for item in active_items)
    used = core_memory_length(state["core_memory"])
    limit = state["core_memory_limit"]
    to_free = max(0, used - limit)
    print(f"\tSummarizing {len(active_items)} core memories for the core/archival split "
          f"({used}/{limit} characters, {to_free} to free)")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a tool that decides what stays in core memory and what moves to
        archival memory. The memories are given to you as `id: content (N characters)` and you
        must refer to them by id.

        The core memory limit is a HARD constraint: after your decision, the memories you keep
        in core MUST total {core_memory_limit} characters or less. It is not a target to
        approach, it is a ceiling that must always hold. Keeping everything in core is never a
        valid answer when the limit is exceeded.

        Archiving is not deleting: an archived memory keeps its content, stays searchable and
        comes back whenever it is relevant again. It only gives up its seat in the always-on
        context. When you hesitate about a memory, archive it: archiving something still useful
        costs far less than breaking the limit.

        Keep in core the facts that are most important and most likely to be needed in the next
        few interactions - identity, constraints, active preferences. Move everything else.

        Before answering, add up the lengths of the memories you decided to keep and verify the
        total is within the limit."""),
        ("human", """Current core memories:

{core_memory}

Total: {core_memory_length} characters. Limit: {core_memory_limit} characters.
You must move to the archive at least {to_free} characters worth of memories.
Return one decision for every memory listed above.""")])

    summarizer_llm = get_llm().bind_tools(
        [SplitCoreAndArchivalMemory], tool_choice=REQUIRED)
    chain = prompt | summarizer_llm
    response = chain.invoke({"core_memory": core_memory_display,
                             "core_memory_length": used,
                             "core_memory_limit": limit,
                             "to_free": to_free})
    print(f"\tCore memory summarization result: {_describe_tool_response(response)}")
    return {"tool_calls": state["tool_calls"] + [response]}


# Build the graph
graph = StateGraph(AgentState)


graph.add_node("retrieve", retrieval_node)
graph.add_node("execute_tool", tool_node)
graph.add_node("generate_answer", generate_answer)
graph.add_node("router", empty_node)

graph.add_edge("retrieve", "execute_tool")
graph.add_edge("execute_tool", "generate_answer")
graph.add_edge("generate_answer", END)

# Insertion nodes
graph.add_node("insert_memories", empty_node)
graph.add_node("summarize_memories", summarize_memories_node)
graph.add_node("summarize_core_memories", summarize_core_memories_node)
graph.add_node("execute_insertion_tool", tool_node)
# Dedicated node for the core/archival split
graph.add_node("execute_core_split_tool", tool_node)
graph.add_edge("summarize_memories",  "execute_insertion_tool")
graph.add_edge("summarize_core_memories",  "execute_core_split_tool")
graph.add_edge("execute_core_split_tool", END)

graph.add_conditional_edges(START, interaction_type_node, {'insert': "insert_memories", 'retrieve': "router"})
graph.add_conditional_edges('router', check_information_sufficiency, {True: "generate_answer", False: "retrieve"})

graph.add_conditional_edges('insert_memories', exceed_memory_limit, {True: "summarize_memories", False: END})
graph.add_conditional_edges('execute_insertion_tool', exceed_core_memory_limit, {True: "summarize_core_memories", False: END})

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
                "maximum_historical_messages": self.config.maximum_historical_messages,
                "core_memory_limit": self.config.core_memory_limit,
                "retrieved_memory": "",
                "current_query": "",
                "tool_calls": []
            }
            self.up_to_date = False
            # Where the operation log stood before the last run, see last_operations()
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
        """Esegue il grafo.

        `query` e' il messaggio dell'utente a cui il ramo retrieve deve
        rispondere. Senza, il recupero ripiega sull'ultimo messaggio in memoria -
        che e' la risposta del giro precedente, non quello che l'utente sta
        chiedendo adesso.
        """
        # Anything logged from here on belongs to this run.
        self._log_offset = len(self.state["operation_log"])
        query = str(query or "").strip()

        if interaction_mode == "retrieve":
            # La cache vale solo finche' la domanda e' la stessa: due domande
            # diverse di fila devono produrre due risposte diverse.
            if self.up_to_date and query == self.state.get("current_query", ""):
                return self.state
            self.up_to_date = True

        if interaction_mode == "insert":
            self.up_to_date = False

        self.state["current_interaction"] = interaction_mode
        self.state["current_query"] = query if interaction_mode == "retrieve" else ""

        if not self.state["messages"] and not self.state["current_query"]:
            print("No messages to process.")
            return self.state

        print(f"\t[{interaction_mode}] running with {len(self.state['messages'])} messages, "
              f"{len(get_active_items(self.state['core_memory']))} active core memories")
        self.state = memory_agent.invoke(self.state)
        self.state["tool_calls"] = []

        return self.state

    def append_message(self, message: str, sender: Literal["user", "assistant"]):
        if sender == "user":
            self.state["messages"].append(
                HumanMessage(content=message, additional_kwargs={}, response_metadata={})
            )
        elif sender == "assistant":
            self.state["messages"].append(
                AIMessage(content=message, additional_kwargs={}, response_metadata={})
            )
        else:
            raise ValueError("Sender must be 'user' or 'assistant'")

# Fill archive with fake memories
# fake_memories = [
#     "User prefers coffee over tea in the morning.",
#     "User likes black tea in the afternoon.",
#     "User's favourite dish is risotto with mushrooms.",
#     "User feels unwell when eating goat cheese.",
#     "User's preferred snack is dark chocolate.",
# ]

# insert_archival_memories.invoke({"memories": fake_memories})
