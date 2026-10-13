"""Does the configured model handle structured output? (phase A2 of the migration)

This is the gate to pass before spending hours on the long test. All of
consolidation goes through `bind_tools()`: if the model does not produce valid
tool calls, or does not copy the ids exactly, the rest cannot work - and it
fails in silent ways, not with an error.

What it checks, in order of increasing difficulty for the model:

    1. answers                       a trivial invocation, without tools
    2. choice between two tools      retrieve_memory or NoSearchNeeded
    3. InsertCoreMemories            list of nested objects, 5-value enum
    4. id transcription              copying an id from the prompt to the tool call
    5. SplitCoreAndArchivalMemory    one decision for each given memory
    6. embedding                     the second model answers, and with what
                                     dimensionality

Points 3 and 4 are the ones that decide the migration. A small model can very
well produce valid JSON and still get the ids wrong: they are two different
failures and must be measured separately, because they lead to different
remedies (the first to constrained structured output, the second to prompt
aliasing).

Every check is repeated several times: with an LLM a single right answer says
nothing. What counts is the rate.

Requirements: LLM_CONFIG and EMBEDDING_CONFIG configured, and the model
reachable at the cluster's address. If it is not, the whole file is skipped
explaining why.

Run:
    pytest memory_service/test/test_tool_calling_gate.py -v -s

Useful variables:
    MEMORY_GATE_ATTEMPTS    how many times to repeat each check
    MEMORY_GATE_THRESHOLD   minimum success rate to pass
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
from memory_service.memory_manager_llm import (  # noqa: E402
    NoSearchNeeded,
    retrieve_memory,
)

from snapshot import SEPARATOR, safe_print  # noqa: E402


ATTEMPTS = int(os.getenv("MEMORY_GATE_ATTEMPTS", "3"))
THRESHOLD = float(os.getenv("MEMORY_GATE_THRESHOLD", "0.8"))

# Fake memories with real ids: they show whether the model copies them.
KNOWN_MEMORIES = {new_item_id(): content for content in (
    "L'utente si chiama Bianca",
    "L'utente e' vegetariana",
    "L'obiettivo giornaliero e' 2000 calorie",
)}

# Collected from all the checks and printed at the end.
RESULTS = []
# Observations that are neither passes nor failures.
NOTES = []


def _memories_for_prompt():
    return "\n".join(f"{item_id}: {content}" for item_id, content in KNOWN_MEMORIES.items())


def _tool_call(response, expected_name):
    """The expected tool call, or None if the model did not produce it."""
    for call in getattr(response, "tool_calls", None) or []:
        if call.get("name") == expected_name:
            return call
    return None


class ModelGateTestCase(unittest.TestCase):
    """Base: build the model once, and skip everything if it does not answer."""

    @classmethod
    def setUpClass(cls):
        # No reset: in a normal run there is nothing to reset, and without a
        # reset a fake model can be injected to check that the gate really
        # discriminates between a good answer and a wrong one.
        try:
            cls.llm = backends.get_llm()
            cls.llm.invoke("Rispondi con la parola: pronto.")
        except Exception as error:
            raise unittest.SkipTest(
                f"model unreachable or not configured: {type(error).__name__}: {error}")

    @classmethod
    def tearDownClass(cls):
        backends.reset()

    def measure(self, label, attempt):
        """Repeat `attempt` and return (successes, details of the failures)."""
        successes, failures = 0, []
        for index in range(ATTEMPTS):
            try:
                problem = attempt()
            except Exception as error:
                problem = f"{type(error).__name__}: {error}"
            if problem is None:
                successes += 1
            else:
                failures.append(f"attempt {index + 1}: {problem}")

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
            f"{label}: {rate:.0%} below the {THRESHOLD:.0%} threshold. "
            f"Failures: {failures[:2]}")


class ToolCallingGateTest(ModelGateTestCase):

    def test_1_the_model_answers(self):
        def attempt():
            response = self.llm.invoke("Rispondi solo con la parola: pronto.")
            if not str(response.content).strip():
                return "empty answer"
            return None

        self.assertRate("answers", attempt)

    def test_1b_reasoning_is_reported_not_judged(self):
        """Does the model reason? It is a piece of information, not a defect.

        The tool calls arrive from a structured field separate from the text, so
        the reasoning does not corrupt the extraction. What it costs is context
        and latency, plus the fact that if it ends up in `content` it enters the
        answers to the user - and from there the memory, because the answer is
        consolidated.

        The two lines printed here say different things and must be read
        together:

            configured    what we asked for, that is the enable_thinking
                          entry of LLM_CONFIG in the .config
            in content    what is really seen in the text of the answer

        If the server runs with --reasoning-parser the reasoning is there but
        ends up in a field of its own: "configured: on" with "in content: no"
        is not a contradiction, it is the best situation. "off" with
        "in content: yes" instead means the option had no effect, and it is a
        problem.

        It also makes an A/B comparison readable: two gate runs with different
        settings could otherwise be told apart only from memory.
        """
        setting = backends.get_config().llm_config.get("enable_thinking")
        configured = {True: "on", False: "off",
                      None: "server default"}[setting]

        response = self.llm.invoke("Rispondi solo con la parola: pronto.")
        content = str(response.content)
        thinking = "<think>" in content

        safe_print("")
        safe_print(f"  thinking configured: {configured}")
        safe_print(f"  reasoning visible in content: {'yes' if thinking else 'no'}")
        if thinking:
            safe_print("    it will enter the answers to the user and from there the memory:")
            safe_print("    consider --reasoning-parser on the server, or turn it off entirely")
        if setting is False and thinking:
            safe_print("    WARNING: asked off, but the reasoning shows anyway")
        # Deliberately outside RESULTS: it would end up in the final verdict as
        # "BELOW THRESHOLD" and would make a model that works fine look unsuitable.
        NOTES.append(f"thinking configured: {configured}, visible in content: "
                     f"{'yes' if thinking else 'no'}")

    def _choice_chain(self):
        """The retrieval node: two tools bound together, one mandatory."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Decidi se i fatti a portata bastano. Se bastano chiama "
                       "NoSearchNeeded, altrimenti cerca in archivio con "
                       "retrieve_memory. Fatti a portata: {core_memory}"),
            ("human", "{query}"),
        ])
        return prompt | self.llm.bind_tools(
            [retrieve_memory, NoSearchNeeded], tool_choice="required")

    def _chose(self, variables, expected):
        """Did it call the right tool?

        It tells not choosing from choosing badly: the first is broken structured
        output, the second is a model that does not see the difference between
        the two tools. They are two different failures and are cured differently.
        """
        chain = self._choice_chain()

        def attempt():
            response = chain.invoke(variables)
            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                return "no tool call"
            if calls[0].get("name") != expected:
                return f"chose {calls[0].get('name')} instead of {expected}"
            return None

        return attempt

    def test_2a_it_can_decide_not_to_search(self):
        """The answer is already among the known facts: searching would be wasted."""
        self.assertRate("choice: do not search", self._chose(
            {"core_memory": "l'utente si chiama Bianca", "query": "come mi chiamo?"},
            "NoSearchNeeded"))

    def test_2b_it_can_decide_to_search(self):
        """The fact is not there. It is the direction the old gate got wrong:
        it answered "sufficient" and the archive was never opened."""
        self.assertRate("choice: search", self._chose(
            {"core_memory": "l'utente si chiama Bianca", "query": "dove abito?"},
            "retrieve_memory"))

    def test_3_insert_core_memories_shape(self):
        """List of nested objects with an enum: the hardest structure."""
        chain = self._consolidation_chain()

        def attempt():
            call = _tool_call(
                chain.invoke({"memories": _memories_for_prompt()}),
                "InsertCoreMemories")
            if call is None:
                return "no tool call"
            memories = call["args"].get("memories")
            if not isinstance(memories, list) or not memories:
                return f"memories is not a non-empty list: {call['args']}"
            for operation in memories:
                if not isinstance(operation, dict):
                    return f"operation is not an object: {operation!r}"
                if not str(operation.get("fact", "")).strip():
                    return f"operation without fact: {operation!r}"
                if normalize_operation(operation.get("operation")) is None:
                    return f"unknown operation: {operation.get('operation')!r}"
            return None

        self.assertRate("InsertCoreMemories (structure)", attempt)

    def test_4_ids_are_transcribed_exactly(self):
        """The most insidious failure: valid JSON but invented ids.

        An id that does not resolve degrades the operation to `new`, so the
        memory is duplicated instead of updated, without any error.
        """
        chain = self._consolidation_chain()

        def attempt():
            call = _tool_call(
                chain.invoke({"memories": _memories_for_prompt()}),
                "InsertCoreMemories")
            if call is None:
                return "no tool call"

            targeted = [operation for operation in call["args"].get("memories", [])
                        if isinstance(operation, dict)
                        and normalize_operation(operation.get("operation")) != "new"]
            if not targeted:
                return "no operation with a target: the model saw everything as new"

            for operation in targeted:
                target = operation.get("target_item_id")
                if not target:
                    return f"missing target on '{operation.get('operation')}'"
                if target not in KNOWN_MEMORIES:
                    return f"nonexistent id: {target!r}"
            return None

        self.assertRate("id transcription", attempt)

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
                return "no tool call"
            decisions = call["args"].get("decisions")
            if not isinstance(decisions, list) or not decisions:
                return f"decisions is not a non-empty list: {call['args']}"
            for decision in decisions:
                if not isinstance(decision, dict):
                    return f"decision is not an object: {decision!r}"
                if decision.get("item_id") not in KNOWN_MEMORIES:
                    return f"nonexistent id: {decision.get('item_id')!r}"
                if decision.get("destination") not in ("core", "archive"):
                    return f"unknown destination: {decision.get('destination')!r}"
            return None

        self.assertRate("SplitCoreAndArchivalMemory", attempt)

    def _consolidation_chain(self):
        """Prompt of the same shape as summarize_memories_node."""
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
    """The second model: it answers, and with what dimensionality."""

    def test_6_embeddings_answer(self):
        try:
            # The model directly: the gate checks the embedding, not Chroma.
            embeddings = backends._build_embeddings(backends.get_config())
            vector = embeddings.embed_query("L'utente si chiama Bianca")
        except Exception as error:
            raise unittest.SkipTest(
                f"embedding unreachable or not configured: "
                f"{type(error).__name__}: {error}")

        safe_print(f"\n  embedding: {len(vector)} dimensions")
        self.assertGreater(len(vector), 0, "the vector cannot be empty")
        self.assertTrue(all(isinstance(value, float) for value in vector[:10]),
                        "the vector must contain numbers")


def tearDownModule():
    """The verdict, at the end: it is what is needed to decide whether to go on."""
    if not RESULTS:
        return

    safe_print("\n" + SEPARATOR)
    safe_print("=== GATE: STRUCTURED OUTPUT ===")
    safe_print(SEPARATOR + "\n")

    for label, successes, attempts, failures in RESULTS:
        rate = successes / attempts
        verdict = "ok" if rate >= THRESHOLD else "BELOW THRESHOLD"
        safe_print(f"  {label:<34} {successes}/{attempts}  {rate:>4.0%}  {verdict}")

    worst = min(successes / attempts for _, successes, attempts, _ in RESULTS)
    safe_print("")
    for note in NOTES:
        safe_print(f"  note: {note}")
    if NOTES:
        safe_print("")
    if worst >= THRESHOLD:
        safe_print("  The model handles structured output: you can go on.")
    else:
        safe_print("  Structured output not reliable. Before going on:")
        safe_print("   - if the structure fails, consider with_structured_output or")
        safe_print("     constrained decoding (format/guided_json/GBNF)")
        safe_print("   - if only the ids fail, consider prompt aliasing")
        safe_print("     ([1], [2], [3] remapped on the code side)")
    safe_print("\n" + SEPARATOR + "\n")


if __name__ == "__main__":
    unittest.main()
