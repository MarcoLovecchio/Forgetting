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
    """Il pezzo di API chromadb che il servizio usa da sotto langchain_chroma.

    Serve per aggiornare i metadata senza ricalcolare l'embedding: langchain non
    espone un modo per farlo, quindi il codice scende di un livello e il doppio
    deve scendere con lui.
    """

    def __init__(self, store):
        self.store = store
        self.updates = []

    def count(self):
        return len(self.store.documents)

    def update(self, ids, metadatas=None, **kwargs):
        # Solo i metadata: documento e vettore restano dove sono. Un doppio che
        # riscrivesse anche il testo nasconderebbe il motivo per cui questa
        # chiamata non passa da add_texts.
        for doc_id, metadata in zip(list(ids), list(metadatas or [])):
            self.updates.append(doc_id)
            if doc_id not in self.store.documents:
                continue  # chroma logga e ignora, non solleva
            self.store.metadatas[doc_id] = dict(metadata)


class FakeVectorStore:
    """In-memory stand-in for the Chroma archival store.

    Mirrors the slice of the langchain Chroma API the service uses: ``add_texts``
    (which upserts by id), ``get`` and ``similarity_search``. Metadata is stored
    and returned, because consolidation reads the item status from there.
    """

    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        self.searches: List[Dict[str, Any]] = []
        # Ogni add_texts e' un embedding ricalcolato: contarle serve a verificare
        # che un cambio di status non ne paghi uno.
        self.writes: List[List[str]] = []
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

    def get(self, ids=None, **kwargs):
        """Same contract as Chroma.get: no ids means the whole collection."""
        if ids is None:
            selected = list(self.documents)
        else:
            selected = [doc_id for doc_id in ids if doc_id in self.documents]
        return {
            "ids": selected,
            "documents": [self.documents[doc_id] for doc_id in selected],
            "metadatas": [self.metadatas.get(doc_id, {}) for doc_id in selected],
        }

    def similarity_search_with_score(self, query, k=3, filter=None, **kwargs):
        """Come similarity_search, ma con la distanza accanto a ogni documento.

        Il punteggio finto e' derivato dal punteggio di sovrapposizione: piu'
        parole in comune, distanza minore. Non ha nessun significato assoluto,
        serve solo perche' i test possano verificare che il numero viaggi.
        """
        docs = self.similarity_search(query, k=k, filter=filter, **kwargs)
        words = {word.lower() for word in str(query).split()}

        def distance(text):
            shared = len(words & {word.lower() for word in text.split()})
            return round(1.0 / (1 + shared), 3)

        return [(doc, distance(doc.page_content)) for doc in docs]

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
        # Il filtro PRIMA del taglio: e' tutto il punto di chiederlo allo store
        # invece di applicarlo dopo. Un doppio che tagliasse prima farebbe
        # passare il bug che questo filtro esiste per evitare.
        ranked = [(doc_id, text) for doc_id, text in ranked if matches(doc_id)]
        return [
            Document(id=doc_id, page_content=text, metadata=self.metadatas.get(doc_id, {}))
            for doc_id, text in ranked[:k]
        ]

    # -- helpers used by the tests --------------------------------------------
    def status_of(self, doc_id: str) -> Optional[str]:
        """Status an archived item was stored with, None if it is not there.

        Un documento che esiste ha sempre uno status: lo scrive archive_metadata.
        Se mancasse, questa riga solleva - meglio che restituire None e far
        confrontare un'assert con un valore che non c'e'.
        """
        if doc_id not in self.documents:
            return None
        return self.metadatas[doc_id]["status"]
