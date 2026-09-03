"""Il modello configurato regge lo structured output? (fase A2 della migrazione)

Questo e' il gate da superare prima di spendere ore nel test lungo. Tutta la
consolidation passa da `bind_tools()`: se il modello non produce tool call
valide, o non ricopia gli id esattamente, il resto non puo' funzionare - e
fallisce in modi silenziosi, non con un errore.

Quello che verifica, in ordine di difficolta' crescente per il modello:

    1. risponde                      una invocazione banale, senza tool
    2. InformationSufficiency        un solo campo booleano
    3. InsertCoreMemories            lista di oggetti annidati, enum a 5 valori
    4. trascrizione degli id         copiare un id dal prompt alla tool call
    5. SplitCoreAndArchivalMemory    una decisione per ogni memoria data
    6. embedding                     il secondo modello risponde e con che
                                     dimensionalita'

I punti 3 e 4 sono quelli che decidono la migrazione. Un modello piccolo puo'
benissimo produrre JSON valido e sbagliare comunque gli id: sono due fallimenti
diversi e vanno misurati separatamente, perche' portano a rimedi diversi (il
primo allo structured output vincolato, il secondo all'aliasing dei prompt).

Ogni controllo viene ripetuto piu' volte: con un LLM una singola risposta giusta
non dice niente. Quello che conta e' il tasso.

Requisiti: LLM_CONFIG ed EMBEDDING_CONFIG configurate, e il modello
raggiungibile all'indirizzo del cluster. Se non lo e', l'intero file si salta
spiegando perche'.

Esecuzione:
    pytest memory_service/test/test_tool_calling_gate.py -v -s

Variabili utili:
    MEMORY_GATE_ATTEMPTS    quante volte ripetere ogni controllo (default 3)
    MEMORY_GATE_THRESHOLD   tasso minimo di successo per passare (default 0.8)
"""

import os
import sys
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from memory_service import backends  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    InsertCoreMemories,
    SplitCoreAndArchivalMemory,
    new_item_id,
    normalize_operation,
)
from memory_service.memory_manager_llm import InformationSufficiency  # noqa: E402

from snapshot import SEPARATOR, safe_print  # noqa: E402


ATTEMPTS = int(os.getenv("MEMORY_GATE_ATTEMPTS", "3"))
THRESHOLD = float(os.getenv("MEMORY_GATE_THRESHOLD", "0.8"))

# Memorie finte con id veri: servono a vedere se il modello li ricopia.
KNOWN_MEMORIES = {new_item_id(): content for content in (
    "L'utente si chiama Bianca",
    "L'utente e' vegetariana",
    "L'obiettivo giornaliero e' 2000 calorie",
)}

# Raccolti da tutti i controlli e stampati alla fine.
RESULTS = []
# Osservazioni che non sono ne' promozioni ne' bocciature.
NOTES = []


def _memories_for_prompt():
    return "\n".join(f"{item_id}: {content}" for item_id, content in KNOWN_MEMORIES.items())


def _tool_call(response, expected_name):
    """La tool call attesa, o None se il modello non l'ha prodotta."""
    for call in getattr(response, "tool_calls", None) or []:
        if call.get("name") == expected_name:
            return call
    return None


class ModelGateTestCase(unittest.TestCase):
    """Base: costruisce il modello una volta, e salta tutto se non risponde."""

    @classmethod
    def setUpClass(cls):
        # Nessun reset: in una run normale non c'e' niente da resettare, e senza
        # reset si puo' iniettare un modello finto per verificare che il gate
        # discrimini davvero fra una risposta buona e una sbagliata.
        try:
            cls.llm = backends.get_llm()
            cls.llm.invoke("Rispondi con la parola: pronto.")
        except Exception as error:
            raise unittest.SkipTest(
                f"modello non raggiungibile o non configurato: {type(error).__name__}: {error}")

    @classmethod
    def tearDownClass(cls):
        backends.reset()

    def measure(self, label, attempt):
        """Ripete `attempt` e restituisce (successi, dettagli dei fallimenti)."""
        successes, failures = 0, []
        for index in range(ATTEMPTS):
            try:
                problem = attempt()
            except Exception as error:
                problem = f"{type(error).__name__}: {error}"
            if problem is None:
                successes += 1
            else:
                failures.append(f"tentativo {index + 1}: {problem}")

        rate = successes / ATTEMPTS
        RESULTS.append((label, successes, ATTEMPTS, failures))
        safe_print(f"\n  {label}: {successes}/{ATTEMPTS} ({rate:.0%})")
        for failure in failures:
            safe_print(f"    - {failure}")
        return rate, failures

    def assertRate(self, label, attempt):
        rate, failures = self.measure(label, attempt)
        self.assertGreaterEqual(
            rate, THRESHOLD,
            f"{label}: {rate:.0%} sotto la soglia del {THRESHOLD:.0%}. "
            f"Fallimenti: {failures[:2]}")


class ToolCallingGateTest(ModelGateTestCase):

    def test_1_the_model_answers(self):
        def attempt():
            response = self.llm.invoke("Rispondi solo con la parola: pronto.")
            if not str(response.content).strip():
                return "risposta vuota"
            return None

        self.assertRate("risponde", attempt)

    def test_1b_reasoning_is_reported_not_judged(self):
        """Il modello ragiona? E' un'informazione, non un difetto.

        Le tool call arrivano da un campo strutturato separato dal testo, quindi
        il ragionamento non corrompe l'estrazione. Quello che costa e' contesto e
        latenza, piu' il fatto che se finisce in `content` entra nelle risposte
        all'utente - e da li' in memoria, perche' la risposta viene consolidata.

        Le due righe stampate qui dicono cose diverse e vanno lette insieme:

            configurato   cosa abbiamo chiesto noi, cioe' la voce
                          enable_thinking di LLM_CONFIG nel .config
            in content    cosa si vede davvero nel testo della risposta

        Se il server gira con --reasoning-parser il ragionamento c'e' ma finisce
        in un campo suo: "configurato: acceso" con "in content: no" non e' una
        contraddizione, e' la situazione migliore. "spento" con "in content: si"
        invece vuol dire che l'opzione non ha avuto effetto, ed e' un problema.

        Serve anche a rendere leggibile un confronto A/B: due run del gate con
        impostazioni diverse altrimenti si distinguono solo a memoria.
        """
        setting = backends.get_config().llm_config.get("enable_thinking")
        configured = {True: "acceso", False: "spento",
                      None: "default del server"}[setting]

        response = self.llm.invoke("Rispondi solo con la parola: pronto.")
        content = str(response.content)
        thinking = "<think>" in content

        safe_print("")
        safe_print(f"  thinking configurato: {configured}")
        safe_print(f"  ragionamento visibile in content: {'si' if thinking else 'no'}")
        if thinking:
            safe_print("    entrera' nelle risposte all'utente e da li' in memoria:")
            safe_print("    valuta --reasoning-parser sul server, o spegnilo del tutto")
        if setting is False and thinking:
            safe_print("    ATTENZIONE: chiesto spento, ma il ragionamento si vede lo stesso")
        # Deliberatamente fuori da RESULTS: finirebbe nel verdetto finale come
        # "sotto soglia" e farebbe sembrare inadatto un modello che va benissimo.
        NOTES.append(f"thinking configurato: {configured}, visibile in content: "
                     f"{'si' if thinking else 'no'}")

    def test_2_information_sufficiency(self):
        """Un solo campo booleano: se fallisce qui, non c'e' structured output."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Sei uno strumento che valuta se le informazioni bastano."),
            ("human", "Domanda: come mi chiamo?\nFatti noti: l'utente si chiama Bianca.\n"
                      "Le informazioni sono sufficienti?"),
        ])
        chain = prompt | self.llm.bind_tools(
            [InformationSufficiency], tool_choice="required")

        def attempt():
            call = _tool_call(chain.invoke({}), "InformationSufficiency")
            if call is None:
                return "nessuna tool call"
            if "is_sufficient" not in call["args"]:
                return f"campo mancante: {call['args']}"
            return None

        self.assertRate("InformationSufficiency", attempt)

    def test_3_insert_core_memories_shape(self):
        """Lista di oggetti annidati con enum: la struttura piu' difficile."""
        chain = self._consolidation_chain()

        def attempt():
            call = _tool_call(
                chain.invoke({"memories": _memories_for_prompt()}),
                "InsertCoreMemories")
            if call is None:
                return "nessuna tool call"
            memories = call["args"].get("memories")
            if not isinstance(memories, list) or not memories:
                return f"memories non e' una lista piena: {call['args']}"
            for operation in memories:
                if not isinstance(operation, dict):
                    return f"operazione non e' un oggetto: {operation!r}"
                if not str(operation.get("fact", "")).strip():
                    return f"operazione senza fact: {operation!r}"
                if normalize_operation(operation.get("operation")) is None:
                    return f"operazione sconosciuta: {operation.get('operation')!r}"
            return None

        self.assertRate("InsertCoreMemories (struttura)", attempt)

    def test_4_ids_are_transcribed_exactly(self):
        """Il fallimento piu' insidioso: JSON valido ma id inventati.

        Un id che non risolve fa degradare l'operazione a `new`, quindi la
        memoria si duplica invece di aggiornarsi, senza nessun errore.
        """
        chain = self._consolidation_chain()

        def attempt():
            call = _tool_call(
                chain.invoke({"memories": _memories_for_prompt()}),
                "InsertCoreMemories")
            if call is None:
                return "nessuna tool call"

            targeted = [operation for operation in call["args"].get("memories", [])
                        if isinstance(operation, dict)
                        and normalize_operation(operation.get("operation")) != "new"]
            if not targeted:
                return "nessuna operazione con target: il modello ha visto tutto come nuovo"

            for operation in targeted:
                target = operation.get("target_item_id")
                if not target:
                    return f"target mancante su '{operation.get('operation')}'"
                if target not in KNOWN_MEMORIES:
                    return f"id inesistente: {target!r}"
            return None

        self.assertRate("trascrizione degli id", attempt)

    def test_5_split_decisions(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Decidi cosa resta in core memory e cosa passa all'archivio. "
                       "Il limite di 60 caratteri e' vincolante."),
            ("human", "Memorie correnti:\n\n{memories}\n\n"
                      "Totale: 95 caratteri, limite: 60. Restituisci una decisione "
                      "per ogni memoria, riferendoti al suo id."),
        ])
        chain = prompt | self.llm.bind_tools(
            [SplitCoreAndArchivalMemory], tool_choice="required")

        def attempt():
            call = _tool_call(chain.invoke({"memories": _memories_for_prompt()}),
                              "SplitCoreAndArchivalMemory")
            if call is None:
                return "nessuna tool call"
            decisions = call["args"].get("decisions")
            if not isinstance(decisions, list) or not decisions:
                return f"decisions non e' una lista piena: {call['args']}"
            for decision in decisions:
                if not isinstance(decision, dict):
                    return f"decisione non e' un oggetto: {decision!r}"
                if decision.get("item_id") not in KNOWN_MEMORIES:
                    return f"id inesistente: {decision.get('item_id')!r}"
                if decision.get("destination") not in ("core", "archive"):
                    return f"destinazione sconosciuta: {decision.get('destination')!r}"
            return None

        self.assertRate("SplitCoreAndArchivalMemory", attempt)

    def _consolidation_chain(self):
        """Prompt della stessa forma di summarize_memories_node."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Estrai i fatti dai messaggi e classifica ognuno rispetto alle "
                       "memorie che ti vengono date come coppie id: contenuto. Se un "
                       "fatto raffina o smentisce una memoria esistente, indica il suo "
                       "id in target_item_id."),
            ("human", "Messaggi:\n\n"
                      "Human: In realta' ora punto a 2200 calorie, ho aumentato.\n\n"
                      "Estrai i fatti e classifica ognuno.\n"
                      "Memorie note (id: contenuto):\n{memories}"),
        ])
        return prompt | self.llm.bind_tools(
            [InsertCoreMemories], tool_choice="required")


class EmbeddingGateTest(unittest.TestCase):
    """Il secondo modello: risponde, e con che dimensionalita'."""

    def test_6_embeddings_answer(self):
        try:
            # Direttamente il modello: il gate verifica l'embedding, non Chroma.
            embeddings = backends._build_embeddings(backends.get_config())
            vector = embeddings.embed_query("L'utente si chiama Bianca")
        except Exception as error:
            raise unittest.SkipTest(
                f"embedding non raggiungibile o non configurato: "
                f"{type(error).__name__}: {error}")

        safe_print(f"\n  embedding: {len(vector)} dimensioni")
        self.assertGreater(len(vector), 0, "il vettore non puo' essere vuoto")
        self.assertTrue(all(isinstance(value, float) for value in vector[:10]),
                        "il vettore deve contenere numeri")


def tearDownModule():
    """Il verdetto, in fondo: e' quello che serve per decidere se proseguire."""
    if not RESULTS:
        return

    safe_print("\n" + SEPARATOR)
    safe_print("=== GATE: STRUCTURED OUTPUT ===")
    safe_print(SEPARATOR + "\n")

    for label, successes, attempts, failures in RESULTS:
        rate = successes / attempts
        verdict = "ok" if rate >= THRESHOLD else "SOTTO SOGLIA"
        safe_print(f"  {label:<34} {successes}/{attempts}  {rate:>4.0%}  {verdict}")

    worst = min(successes / attempts for _, successes, attempts, _ in RESULTS)
    safe_print("")
    for note in NOTES:
        safe_print(f"  nota: {note}")
    if NOTES:
        safe_print("")
    if worst >= THRESHOLD:
        safe_print("  Il modello regge lo structured output: si puo' proseguire.")
    else:
        safe_print("  Structured output non affidabile. Prima di andare avanti:")
        safe_print("   - se fallisce la struttura, valuta with_structured_output o il")
        safe_print("     decoding vincolato (format/guided_json/GBNF)")
        safe_print("   - se falliscono solo gli id, valuta l'aliasing dei prompt")
        safe_print("     ([1], [2], [3] rimappati lato codice)")
    safe_print("\n" + SEPARATOR + "\n")


if __name__ == "__main__":
    unittest.main()
