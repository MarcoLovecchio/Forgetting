"""Test doubles that let the memory agent run completely offline.

They replace the two external backends of the service (the chat model and the
archival vector store) so the graph can be exercised without API keys, network
access, ChromaDB or any other package of the architecture.
"""

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


def tool_name_of(tool: Any) -> str:
    """Name a bound tool is exposed with: the one the real model sees.

    Not the class name: a model may carry another title in its schema, as the
    consolidation tools do to share the name InsertCoreMemories.
    """
    return convert_to_openai_tool(tool)["function"]["name"]


class ScriptedChatModel(BaseChatModel):
    """Chat model that answers according to the tools it is bound to.

    ``tool_responses`` maps a tool name to the arguments the model should
    "decide" to call it with. When none of the bound tools is scripted, the
    model replies with ``default_content``. Every invocation is recorded in
    ``invocations``, with the names of the tools it was bound to and the tools
    themselves.
    """

    tool_responses: Dict[str, Dict[str, Any]] = {}
    default_content: str = "fake answer"
    invocations: List[Dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        bound = list(kwargs.get("tools") or [])
        bound_tools = [tool_name_of(tool) for tool in bound]
        self.invocations.append({"tools": bound_tools, "bound": bound})

        message = AIMessage(content=self.default_content)
        for name in bound_tools:
            if name in self.tool_responses:
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": self.tool_responses[name],
                            "id": f"call_{len(self.invocations)}",
                            "type": "tool_call",
                        }
                    ],
                )
                break

        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> Any:
        return self.bind(tools=list(tools), **kwargs)

    # -- helpers used by the tests --------------------------------------------
    def script(
        self,
        tool_responses: Optional[Dict[str, Dict[str, Any]]] = None,
        default_content: Optional[str] = None,
    ) -> None:
        """Decide what the model will "want" to do from the next call on.

        Turn based tests call this before every turn, so each turn exercises a
        different classification.
        """
        self.tool_responses = dict(tool_responses or {})
        if default_content is not None:
            self.default_content = default_content

    def reset(self) -> None:
        self.invocations.clear()

    def bound_tool_names(self) -> List[str]:
        return [name for invocation in self.invocations for name in invocation["tools"]]


class FakeCollection:
    """The piece of the chromadb API the service uses from under langchain_chroma.

    It is needed to update the metadata without recomputing the embedding:
    langchain exposes no way to do it, so the code goes down one level and the
    double has to go down with it.
    """

    def __init__(self, store):
        self.store = store
        self.updates = []

    def count(self):
        return len(self.store.documents)

    def update(self, ids, metadatas=None, **kwargs):
        # Only the metadata: document and vector stay where they are. A double that
        # also rewrote the text would hide the reason why this call does not go
        # through add_texts.
        for doc_id, metadata in zip(list(ids), list(metadatas or [])):
            self.updates.append(doc_id)
            if doc_id not in self.store.documents:
                continue  # chroma logs and ignores, it does not raise
            self.store.metadatas[doc_id] = dict(metadata)


class FakeVectorStore:
    """In-memory stand-in for the Chroma archival store.

    Mirrors the slice of the langchain Chroma API the service uses: ``add_texts``
    (which upserts by id), ``get``, ``delete`` and ``similarity_search``. Metadata
    is stored and returned, because consolidation reads the item status from there.
    """

    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        self.searches: List[Dict[str, Any]] = []
        # Every add_texts is a recomputed embedding: counting them checks that a
        # status change does not pay for one.
        self.writes: List[List[str]] = []
        # Every delete received, with its ids: eviction must never make an empty
        # one, and this is the way to see it.
        self.deletes: List[List[str]] = []
        self._collection = FakeCollection(self)

    def add_texts(self, texts, ids=None, metadatas=None, **kwargs):
        texts = list(texts)
        if ids is None:
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(texts))]
        ids = list(ids)
        metadatas = list(metadatas) if metadatas is not None else [{}] * len(texts)
        if not (len(texts) == len(ids) == len(metadatas)):
            raise ValueError("texts, ids and metadatas must have the same length")
        self.writes.append(list(ids))
        for doc_id, text, metadata in zip(ids, texts, metadatas):
            self.documents[doc_id] = text
            self.metadatas[doc_id] = dict(metadata or {})
        return ids

    def get(self, ids=None, where=None, **kwargs):
        """Same contract as Chroma.get: no ids means the whole collection.

        `where` only with equality, which is what the service uses. An operator
        ("$in", "$ne") raises instead of being ignored: ignoring it would mean
        answering with the whole collection, and the double would stop resembling
        the store exactly at the point being tested.
        """
        if ids is None:
            selected = list(self.documents)
        else:
            selected = [doc_id for doc_id in ids if doc_id in self.documents]
        for key, value in (where or {}).items():
            if key.startswith("$") or isinstance(value, dict):
                raise NotImplementedError(f"FakeVectorStore.get: unsupported filter {where!r}")
            selected = [doc_id for doc_id in selected
                        if self.metadatas.get(doc_id, {}).get(key) == value]
        return {
            "ids": selected,
            "documents": [self.documents[doc_id] for doc_id in selected],
            "metadatas": [self.metadatas.get(doc_id, {}) for doc_id in selected],
        }

    def delete(self, ids=None, **kwargs):
        """Like Chroma.delete, by id.

        Without ids it raises: it is the case eviction defends itself from, and a
        double that accepted it silently could never show it. But the call is
        recorded **before** raising: eviction absorbs the store's exceptions, and
        a double that raised without leaving a trace would make the empty delete
        disappear exactly from the list the tests look at.
        """
        ids = list(ids or [])
        self.deletes.append(ids)
        if not ids:
            raise ValueError("FakeVectorStore.delete called without ids")
        for doc_id in ids:
            self.documents.pop(doc_id, None)
            self.metadatas.pop(doc_id, None)

    def similarity_search(self, query, k=3, filter=None, **kwargs):
        if not isinstance(k, int):
            raise TypeError(f"k must be an int, got {type(k).__name__}")
        self.searches.append({"query": query, "k": k, "filter": filter})
        words = {word.lower() for word in str(query).split()}

        def score(text):
            return len(words & {word.lower() for word in text.split()})

        def matches(doc_id):
            metadata = self.metadatas.get(doc_id, {})
            return all(metadata.get(key) == value for key, value in (filter or {}).items())

        ranked = sorted(self.documents.items(), key=lambda item: score(item[1]), reverse=True)
        # The filter BEFORE the cut: that is the whole point of asking the store
        # for it instead of applying it afterwards. A double that cut first would
        # let through the bug this filter exists to avoid.
        ranked = [(doc_id, text) for doc_id, text in ranked if matches(doc_id)]
        return [
            Document(id=doc_id, page_content=text, metadata=self.metadatas.get(doc_id, {}))
            for doc_id, text in ranked[:k]
        ]

    # -- helpers used by the tests --------------------------------------------
    def status_of(self, doc_id: str) -> Optional[str]:
        """Status an archived item was stored with, None if it is not there.

        A document that exists always has a status: archive_metadata writes it.
        If it were missing, this line raises - better than returning None and
        having an assert compare against a value that is not there.
        """
        if doc_id not in self.documents:
            return None
        return self.metadatas[doc_id]["status"]
