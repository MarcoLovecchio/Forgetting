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


def tool_name_of(tool: Any) -> str:
    """Name a bound tool is exposed with, for both @tool functions and models."""
    for attribute in ("name", "__name__"):
        value = getattr(tool, attribute, None)
        if isinstance(value, str):
            return value
    if isinstance(tool, dict):
        return tool.get("name", "")
    return type(tool).__name__


class ScriptedChatModel(BaseChatModel):
    """Chat model that answers according to the tools it is bound to.

    ``tool_responses`` maps a tool name to the arguments the model should
    "decide" to call it with. When none of the bound tools is scripted, the
    model replies with ``default_content``. Every invocation is recorded in
    ``invocations`` so tests can assert on what the graph actually asked.
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
        bound_tools = [tool_name_of(tool) for tool in (kwargs.get("tools") or [])]
        self.invocations.append(
            {
                "tools": bound_tools,
                "prompt": "\n".join(str(message.content) for message in messages),
            }
        )

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
    def reset(self) -> None:
        self.invocations.clear()

    def bound_tool_names(self) -> List[str]:
        return [name for invocation in self.invocations for name in invocation["tools"]]


class FakeVectorStore:
    """In-memory stand-in for the Chroma archival store."""

    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        self.searches: List[Dict[str, Any]] = []

    def add_texts(self, texts, ids=None, metadatas=None, **kwargs):
        texts = list(texts)
        if ids is None:
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(texts))]
        ids = list(ids)
        metadatas = list(metadatas) if metadatas is not None else [{}] * len(texts)
        if not (len(texts) == len(ids) == len(metadatas)):
            raise ValueError("texts, ids and metadatas must have the same length")
        for doc_id, text, metadata in zip(ids, texts, metadatas):
            self.documents[doc_id] = text
            self.metadatas[doc_id] = metadata
        return ids

    def similarity_search(self, query, k=3, **kwargs):
        if not isinstance(k, int):
            raise TypeError(f"k must be an int, got {type(k).__name__}")
        self.searches.append({"query": query, "k": k})
        words = {word.lower() for word in str(query).split()}

        def score(text):
            return len(words & {word.lower() for word in text.split()})

        ranked = sorted(self.documents.items(), key=lambda item: score(item[1]), reverse=True)
        return [Document(id=doc_id, page_content=text) for doc_id, text in ranked[:k]]
