"""I modelli veri sono utilizzabili adesso?

I test che parlano con lo stack reale devono **saltare** quando il modello non
c'e', non fallire: un server spento non e' un difetto del codice. Con i provider
hosted bastava guardare se la chiave era impostata; con un modello servito in
locale la chiave non esiste e l'unico modo per saperlo e' provare.

Il risultato viene memorizzato: i test lo chiedono piu' volte e non ha senso
pagare una chiamata ogni volta.
"""

import os
from typing import Optional

from memory_service import backends
from memory_service.config import load_environment

# I file .env/.config vengono caricati appena questo modulo viene importato, e
# l'import sta in cima ai test che ne hanno bisogno. Senza, os.getenv("LLM_CONFIG")
# sarebbe None al momento in cui pytest valuta lo skipif: i test sullo stack reale
# si salterebbero **sempre**, anche con .config a posto, dicendo che manca una
# configurazione che invece c'e'.
load_environment()

_chat_reason: Optional[str] = None
_chat_checked = False
_embeddings_reason: Optional[str] = None
_embeddings_checked = False


def _describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def chat_model_unavailable() -> Optional[str]:
    """Perche' il chat model non e' utilizzabile, o None se lo e'."""
    global _chat_reason, _chat_checked
    if _chat_checked:
        return _chat_reason

    _chat_checked = True
    if not os.getenv("LLM_CONFIG"):
        _chat_reason = "LLM_CONFIG non e' impostata"
        return _chat_reason

    try:
        backends.get_llm().invoke("Rispondi con la parola: pronto.")
        _chat_reason = None
    except Exception as error:
        _chat_reason = f"chat model non raggiungibile ({_describe(error)})"
    return _chat_reason


def embeddings_unavailable() -> Optional[str]:
    """Perche' il modello di embedding non e' utilizzabile, o None se lo e'.

    Serve a tutto quello che tocca l'archivio: consolidare un delete, archiviare
    per lo split, cercare fra le memorie archiviate.
    """
    global _embeddings_reason, _embeddings_checked
    if _embeddings_checked:
        return _embeddings_reason

    _embeddings_checked = True
    try:
        # Direttamente il modello, senza passare dal vector store: verificare un
        # embedding non deve richiedere anche chromadb, ne' dipendere da come
        # Chroma espone la sua funzione di embedding.
        backends._build_embeddings(backends.get_config()).embed_query("prova")
        _embeddings_reason = None
    except Exception as error:
        _embeddings_reason = f"embedding non raggiungibile ({_describe(error)})"
    return _embeddings_reason


def live_stack_unavailable() -> Optional[str]:
    """Perche' non si puo' girare sullo stack completo, o None se si puo'."""
    return chat_model_unavailable() or embeddings_unavailable()


def reset() -> None:
    """Dimentica il risultato memorizzato (utile fra configurazioni diverse)."""
    global _chat_reason, _chat_checked, _embeddings_reason, _embeddings_checked
    _chat_reason = _embeddings_reason = None
    _chat_checked = _embeddings_checked = False
