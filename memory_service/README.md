# memory_service

Servizio di memoria (core + archival) dell'assistente. Il pacchetto gira in ROS 2
ma è **indipendente dagli altri pacchetti dell'architettura**: non importa
`shared_utils` né `db_adapters`, e non contatta LLM o ChromaDB finché non serve
davvero. Questo permette di testarlo seriamente, senza chiavi API e senza rete.

## Struttura

| Modulo | Ruolo | Dipendenze esterne |
| --- | --- | --- |
| `config.py` | configurazione (limiti, path Chroma, `LLM_CONFIG`) | solo `python-dotenv` |
| `backends.py` | costruzione **pigra** di chat model e vector store, sostituibili | importa Chroma/Mistral/langchain solo se usati |
| `consolidation.py` | modello dati della memoria e operazioni di consolidamento | `pydantic`, `langchain-core` |
| `memory_manager_llm.py` | grafo LangGraph e `MemoryAgent` | `langgraph`, `langchain-core` |
| `memory_server.py` | nodo ROS con i due servizi | `rclpy`, `memory_service_interfaces` |
| `memory_client.py` | client ROS di esempio | `rclpy`, `memory_service_interfaces` |

## Test

```bash
python memory_service/run_tests.py -v
```

50 test, tutti sul **grafo LangGraph reale** con backend simulati
(`test/fakes.py`): nessuna chiamata di rete, nessuna API key, nessun ChromaDB.

`test/test_consolidation.py` e' lo scenario a turni: ogni turno esercita una
classificazione diversa (`new`, `redundant`, `update`, `contradict`, `delete`,
archiviazione, recupero) e stampa lo stato completo della memoria - core,
messaggi, archivio con i metadata, recupero, operation log. `test_memory_agent.py`
tiene invece i controlli unitari (limiti, tool call malformate, cache della
retrieve, tombstone mai recuperati) e `test_config.py` la configurazione.

I test del nodo (`test/test_memory_server.py`) costruiscono `MemoryServer` con un
agente stub e invocano le callback con request/response vere: richiedono rclpy e
le interfacce generate, quindi vanno eseguiti dentro il workspace ROS (fuori si
saltano da soli).

Con pytest installato l'intero pacchetto è raccoglibile (`conftest.py` sistema il
`sys.path`):

```bash
pytest memory_service
colcon test --packages-select memory_service
```

`test/test_memory_llm.py` è invece il test di **integrazione** con lo stack
reale (Groq + Mistral + ChromaDB): stessa turnistica, ma la classificazione la
decide il modello vero, quindi le assert sono larghe (invarianti strutturali) e
il valore sta nella traccia stampata. Viene saltato automaticamente se
`LLM_CONFIG` non è impostata. Con `GROQ_API_KEY` e `MISTRAL_API_KEY` in `.env`
gira per intero; `LANGSMITH_API_KEY` resta opzionale, come nel resto
dell'architettura — se presente abilita solo il tracing.

```bash
pytest memory_service/test/test_memory_llm.py -v -s
```

## Backend sostituibili

Chat model e vector store vengono creati alla prima richiesta e possono essere
iniettati dall'esterno:

```python
from memory_service import backends
from memory_service.memory_manager_llm import MemoryAgent

backends.configure(llm=my_chat_model, vector_store=my_store)
agent = MemoryAgent()
```

Lo stesso meccanismo è usato dai test e permette di provare un provider diverso
senza toccare il codice del grafo. Anche il nodo accetta un agente esplicito
(`MemoryServer(agent=...)`), altrimenti usa il singleton `MemoryAgent`.

## Configurazione

Tutti i valori hanno un default e possono essere sovrascritti da variabili
d'ambiente (lette da `.env` / `.config`):

| Variabile | Default | Significato |
| --- | --- | --- |
| `LLM_CONFIG` | — | dizionario dei modelli; viene usata la voce `memory_agent` |
| `GROQ_API_KEY` | — | chiave passata al provider (nome configurabile con `MEMORY_API_KEY_ENV`) |
| `MEMORY_LLM_NODE` | `memory_agent` | voce di `LLM_CONFIG` da usare |
| `MEMORY_MAX_HISTORICAL_MESSAGES` | `5` | messaggi mantenuti prima del riassunto |
| `MEMORY_CORE_MEMORY_LIMIT` | `150` | caratteri massimi della core memory |
| `MEMORY_CHROMA_PATH` | `./chroma_db` | cartella dell'archivio (risolta in path assoluto) |
| `MEMORY_COLLECTION_NAME` | `memory_archive` | collezione ChromaDB |
| `MEMORY_ENV_FILE` | — | percorso esplicito del file di ambiente |

`.env` e `.config` vengono cercati risalendo le directory a partire dalla
working directory e dalla posizione del pacchetto. Se il nodo viene lanciato da
un install space che non contiene quei file, conviene indicarli esplicitamente
con `MEMORY_ENV_FILE`.

Attenzione a `MEMORY_CHROMA_PATH`: il default resta relativo alla working
directory (come prima), quindi lanciare il nodo da cartelle diverse significa
usare archivi diversi. Impostare un percorso assoluto per avere una memoria
stabile.

## Esecuzione

```bash
ros2 run memory_service memory_server
ros2 run memory_service memory_client
```
