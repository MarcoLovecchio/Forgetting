# memory_service

Servizio di memoria (core + archival) dell'assistente. Il pacchetto gira in ROS 2
ma è **indipendente dagli altri pacchetti dell'architettura**: non importa
`shared_utils` né `db_adapters`, e non contatta LLM o ChromaDB finché non serve
davvero. Questo permette di testarlo seriamente, senza chiavi API e senza rete.

## Struttura

| Modulo | Ruolo | Dipendenze esterne |
| --- | --- | --- |
| `config.py` | configurazione (limiti, path Chroma, `LLM_CONFIG`) | solo `python-dotenv` |
| `backends.py` | costruzione **pigra** di chat model e vector store, sostituibili | importa il provider (Ollama, Mistral, OpenAI) e Chroma solo se usati |
| `consolidation.py` | modello dati della memoria e operazioni di consolidamento | `pydantic`, `langchain-core` |
| `memory_manager_llm.py` | grafo LangGraph e `MemoryAgent` | `langgraph`, `langchain-core` |
| `memory_server.py` | nodo ROS con i due servizi | `rclpy`, `memory_service_interfaces` |
| `memory_client.py` | client ROS di esempio | `rclpy`, `memory_service_interfaces` |

## Interfaccia ROS

I due servizi (`memory_service_interfaces`) espongono la memoria consolidata:

**`GetMemory`** — request: `user_input`, il messaggio a cui la memoria deve
rispondere. Passarlo e' quello che permette al ramo di recupero di cercare in
archivio in base a cosa l'utente sta chiedendo **adesso**; lasciandolo vuoto il
servizio ripiega sull'ultimo messaggio gia' in memoria, che e' la risposta del
giro precedente.

| campo (response) | contenuto |
| --- | --- |
| `memory_list` | contenuto delle core memory **attive** (superseded e deleted restano fuori) |
| `memory_ids` | id di ciascuna, stesso ordine di `memory_list`: le due liste si zippano |
| `last_messages` | messaggi ancora nella finestra di conversazione |
| `operation_log` | operazioni di consolidamento di **questa** chiamata, un JSON per riga |

**`UpdateMemory`** — request `user_input`, `queries`, `results`, `explanation`; la
response ha `memory_list`, `memory_ids` e `operation_log` con la stessa semantica.

Ogni entry di `operation_log` è il JSON di un `OperationLogEntry`:

```json
{"op_type": "contradict", "item_id": "c8f90d30-...", "related_item_id": "7ca3ad4f-...",
 "content": "L'utente mangia pesce", "timestamp": "2026-08-24T18:23:39.059368"}
```

`op_type` è uno fra `create`, `redundant`, `update`, `contradict`, `delete`, `archive`.

Il log riporta **solo le operazioni della chiamata corrente**, non tutto lo
storico: `state["operation_log"]` cresce per tutta la vita del nodo e `get_memory`
viene invocato di frequente, quindi spedire l'intera storia a ogni chiamata
costerebbe sempre di più. Lo storico completo resta disponibile in-process via
`MemoryAgent.state["operation_log"]`.

`memory_list` e `last_messages` mantengono nome e semantica di prima, quindi i
pacchetti che li leggono continuano a funzionare: serve solo un `colcon build`
per rigenerare le interfacce.

Per sfruttare `user_input` serve una riga in `intent_recognition.listener_callback`:

```python
self.get_response = self.memory_client.send_get_request(msg.data.strip())
```

Senza, tutto continua a funzionare come prima (il campo resta vuoto e vale il
fallback), ma il recupero dall'archivio non parte dalla domanda corrente.

## Test

```bash
python memory_service/run_tests.py -v
```

90 test, tutti sul **grafo LangGraph reale** con backend simulati
(`test/fakes.py`): nessuna chiamata di rete, nessun modello, nessun ChromaDB.

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

`test/test_tool_calling_gate.py` è il **gate da superare prima di tutto il
resto**: verifica che il modello configurato produca tool call valide per gli
schemi veri (`InsertCoreMemories`, `SplitCoreAndArchivalMemory`,
`InformationSufficiency`) e che ricopi gli id esattamente. Sono due fallimenti
diversi e li misura separatamente, perché portano a rimedi diversi: la struttura
sbagliata si cura con lo structured output vincolato, gli id sbagliati con
l'aliasing dei prompt. Ogni controllo è ripetuto più volte, perché con un LLM una
singola risposta giusta non dice niente.

```bash
pytest memory_service/test/test_tool_calling_gate.py -v -s

# lettura più affidabile prima di decidere
MEMORY_GATE_ATTEMPTS=10 pytest memory_service/test/test_tool_calling_gate.py -v -s
```

`test/test_memory_llm.py` è invece il test di **integrazione** con lo stack
reale (chat model, embedding, ChromaDB): stessa turnistica, ma la
classificazione la decide il modello vero, quindi le assert sono larghe
(invarianti strutturali) e il valore sta nella traccia stampata. Viene saltato se
`LLM_CONFIG` non è impostata, e anche se i modelli non rispondono — con un
messaggio che dice quale dei due. `LANGSMITH_API_KEY` resta opzionale, come nel
resto dell'architettura: se presente abilita solo il tracing.

```bash
pytest memory_service/test/test_memory_llm.py -v -s
```

`test/test_long_term_interaction.py` e' invece la simulazione di una **sessione
lunga**: 117 messaggi dell'utente iniettati uno alla volta — 101 fatti e 16
domande — mescolati come capiterebbero davvero. I primi 25 sono la presentazione;
da li' in poi fatti nuovi, ripetizioni, raffinamenti, contraddizioni, richieste
di cancellazione e chiacchiere si alternano senza blocchi, rispettando solo la
causalita' (un fatto viene introdotto prima di essere modificato).

I messaggi che finiscono con `?` sono **domande**: non consolidano, fanno girare
il ramo `retrieve` e verificano cosa l'assistente sa in quel momento. Sono
piazzate subito dopo un update, una contraddizione o una cancellazione — "sai
dirmi dove abito?" arriva dopo "dimentica il mio indirizzo" — e quelle su fatti
finiti in archivio costringono la retrieve a interrogarlo davvero.

Durante l'esecuzione stampa solo una riga di avanzamento ogni 10 messaggi; alla
fine stampa il resoconto: operation log completo, core memory, archivio, e ogni
domanda con la risposta ricevuta e cosa e' arrivato dall'archivio. Le assert sono
strutturali (ogni item uscito dalla core memory deve trovarsi in archivio, core
memory con soli item attivi, id unici, nessuna domanda senza risposta).

Sono parecchie chiamate reali al modello — le domande ne fanno due o tre a testa
— quindi diversi minuti. L'archivio finisce in una cartella temporanea nuova,
cosi' il `chroma_db` di produzione non viene toccato.

```bash
pytest memory_service/test/test_long_term_interaction.py -v -s

# versione ridotta, per una prova veloce
MEMORY_LONGRUN_MESSAGES=20 pytest memory_service/test/test_long_term_interaction.py -v -s

# con una pausa fra i messaggi, se il server Ollama e' condiviso con altri
MEMORY_LONGRUN_DELAY=1 pytest memory_service/test/test_long_term_interaction.py -v -s
```

## Vincoli delle versioni (Ollama)

Verificato sulla documentazione e sui sorgenti di `langchain-ollama`:

- `num_ctx` e `base_url` sono parametri **top-level** di `ChatOllama`, e `model` /
  `base_url` lo sono di `OllamaEmbeddings`: si passano direttamente al
  costruttore, non dentro un dizionario di opzioni. È come li passa `backends.py`.
- Il pin a **0.3.3** non è arbitrario: è l'ultima versione compatibile con il
  `langchain-core==0.3.62` di questo progetto (richiede `>=0.3.60`). La 0.3.10
  vuole `langchain-core>=0.3.76`, le 1.x vogliono `>=1.2.21`.
- **Il thinking di Qwen non è un problema per il tool calling.** Le tool call
  arrivano da `response["message"]["tool_calls"]`, un campo strutturato separato
  dal testo: i blocchi `<think>` finiscono in `content` e non corrompono
  l'estrazione. Quello che costano è contesto e latenza, e il fatto che il testo
  del ragionamento entra nelle risposte all'utente — quindi anche in memoria,
  perché la risposta viene consolidata.
- Se `content` sporco dà fastidio, la leva in 0.3.3 è **`extract_reasoning=True`**,
  che sposta il ragionamento in `additional_kwargs["reasoning_content"]`. Non
  `reasoning=False`: oltre a non esistere nella 0.3.3, risulta segnalato come non
  funzionante con Qwen anche nelle versioni che ce l'hanno.

## Il limite della core memory

Il limite in caratteri e' **chiesto al modello, non imposto dal codice**: non c'e'
nessuna eviction deterministica dietro. Quando la core memory sfora, il nodo
`summarize_core_memories` passa allo split tutto quello che serve per rispettarlo:

- ogni memoria con la propria lunghezza (`id: content (N characters)`)
- il totale attuale, il limite, e **quanti caratteri vanno liberati**
- l'indicazione esplicita che il vincolo e' hard e che tenere tutto in core non e'
  una risposta valida
- il fatto che archiviare non e' cancellare: l'archivio resta consultabile, quindi
  nel dubbio conviene archiviare

Se il modello decide comunque di non archiviare abbastanza, **la sua decisione
viene rispettata** e la core memory resta sopra il limite. Non passa pero' in
silenzio: compare un warning nei log.

```
WARNING: core memory still over the limit after the split (500/100 characters)
```

## I candidati dell'archivio

Quando un fatto viene classificato, il modello lo confronta con le memorie core
attive **piu' le prime K dell'archivio** (`ARCHIVE_CANDIDATES_K = 5`), unite in un
solo elenco `id: contenuto` da `build_candidate_memories`. Da entrambe le fonti
passano **solo le attive**: superseded e deleted restano fuori.

Il vettoriale pero' ordina per somiglianza e delle lapidi non sa niente, quindi
puo' benissimo metterle nei primi K posti. Filtrare dopo una ricerca stretta
restituirebbe meno di K candidati — e il divario **peggiora con l'uso**, perche'
i tombstone non vengono mai rimossi: proprio su un archivio molto usato il
classificatore si ritroverebbe senza bersagli, senza che niente lo segnali.

Per questo `search_archive` cerca **piu' largo di quanto serve**
(`ARCHIVE_OVERFETCH = 3`), filtra, e solo allora tronca a K. Chi chiama non se ne
accorge: chiede K e riceve K, finche' K memorie attive esistono. Se non esistono
riceve quelle che ci sono, in ordine di somiglianza — meno candidati e' un prompt
piu' povero, non un errore.

Una cosa che resta com'e': la query usata per cercare nell'archivio e' **il blocco
di messaggi in consolidamento**, non il singolo fatto estratto. I K candidati sono
quindi scelti una volta per batch. E' il motivo per cui un `update` verso una
memoria archiviata puo' non trovare il suo bersaglio quando nel batch si parla
anche d'altro.

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
| `EMBEDDING_CONFIG` | — | dizionario dei modelli di embedding; viene usata la voce `memory_agent` |
| `MEMORY_LLM_BASE_URL` | — | endpoint del chat model servito in locale (Ollama) |
| `MEMORY_EMBEDDING_BASE_URL` | — | endpoint del modello di embedding |
| `MEMORY_NUM_CTX` | `8192` | context window chiesto a Ollama, che di default userebbe 2048 |
| `GROQ_API_KEY` | — | chiave del provider, se ne serve una (nome configurabile con `MEMORY_API_KEY_ENV`). **Non** viene inviata quando il provider è `ollama`: resta impostata per gli altri nodi dell'architettura, che parlano ancora con Groq |
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
