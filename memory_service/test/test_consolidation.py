"""Memory lifecycle, one turn per operation, without API keys.

Each turn the user says something, the classifier (scripted here) returns a
different classification, and after each turn the full state of the memory is
printed: core, message window, vector archive with the metadata, last retrieval
and operation log.

The turns are in sequence and each one starts from the state the previous one
left:

    1. new         three new facts enter the core memory
    2. redundant   the same fact repeated reinforces the existing item
    3. update      a fact is refined: the old one becomes superseded
    4. update      a fact is contradicted: it arrives as contradict, it is an update
    5. delete      the user asks to forget: the item becomes deleted
    6. archive     core memory over the limit: an item moves to the archive
    7. retrieve    the answer draws from the archive, the tombstones stay out

Run: python memory_service/run_tests.py -v
"""

import dataclasses
import json
import os
import sys
import time
import unittest
from datetime import datetime, timedelta

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    NO_ARCHIVAL_RESULTS,
    CoreMemoryItem,
    archive_items,
    get_active_items,
    record_uses,
    reinforce_archived_item,
    retrieve_active_archival_memories,
    serialize_operation_log_for_response,
    serialize_retrieved_for_response,
    supersede_archived_item,
    supersede_item,
)
from memory_service.memory_manager_llm import MemoryAgent  # noqa: E402

from fakes import FakeVectorStore, ScriptedChatModel  # noqa: E402
from snapshot import print_memory_snapshot, print_turn_header  # noqa: E402


# A single history message: so every turn triggers consolidation.
# The character limit is high, turn 6 lowers it on purpose for the split.
LIFECYCLE_CONFIG = MemoryConfig(
    node_name="memory_agent",
    generate_answer=True,  # turn 7 looks at the answer of the retrieve branch
    maximum_historical_messages=1,
    core_memory_limit=2000,
    eviction=False,  # these tests look at the tombstones in the archive
    retrieval_mode="decide",
    consolidation_utility="none",
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)


def find_item(items, needle):
    """Active item whose content contains `needle`."""
    for item in items:
        if needle.lower() in item.content.lower():
            return item
    raise AssertionError(f"no item contains {needle!r}: {[i.content for i in items]}")


class ConsolidationLifecycleTest(unittest.TestCase):
    """A single scenario: the turns follow one another and build the state."""

    def setUp(self):
        self.llm = ScriptedChatModel(tool_responses={}, default_content="ok")
        self.store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=self.llm, vector_store=self.store, config=LIFECYCLE_CONFIG)
        MemoryAgent.reset_instance()
        self.agent = MemoryAgent(config=LIFECYCLE_CONFIG)

    def tearDown(self):
        MemoryAgent.reset_instance()
        backends.reset()

    # -- a turn ------------------------------------------------------------ #

    def consolidate(self, turn, operation, description, user_message, memories,
                    assistant_message="Va bene."):
        """The user says something, the classifier answers `memories`."""
        print_turn_header(turn, operation, description)
        self.llm.script({"InsertCoreMemories": {"memories": memories}})
        self.agent.append_message(user_message, "user")
        self.agent.append_message(assistant_message, "assistant")

        state = self.agent.run_memory_agent("insert")

        print_memory_snapshot(f"turn {turn} - {operation}", state, self.store,
                              self.agent.config.core_memory_limit)
        return state

    def active(self, state):
        return get_active_items(state["core_memory"])

    # -- the scenario ------------------------------------------------------- #

    def test_memory_lifecycle_turn_by_turn(self):
        # --- TURN 1: new -------------------------------------------------- #
        state = self.consolidate(
            1, "new", "three never-seen facts enter the core memory",
            "Mi chiamo Bianca, sono vegetariana e punto a 2000 calorie al giorno.",
            [
                {"fact": "L'utente si chiama Bianca", "operation": "new"},
                {"fact": "L'utente e' vegetariana", "operation": "new"},
                {"fact": "L'obiettivo giornaliero e' 2000 calorie", "operation": "new"},
            ],
        )

        items = self.active(state)
        self.assertEqual(len(items), 3, "the three facts become three distinct items")
        self.assertEqual([entry.op_type for entry in state["operation_log"]], ["create"] * 3)
        self.assertEqual(self.store.documents, {}, "nothing ends up in the archive")

        name_item = find_item(items, "Bianca")
        veg_item = find_item(items, "vegetariana")
        goal_item = find_item(items, "2000 calorie")

        # --- TURN 2: redundant -------------------------------------------- #
        updated_at_before = name_item.updated_at
        time.sleep(0.02)  # updated_at must turn out more recent

        state = self.consolidate(
            2, "redundant", "the user repeats a known fact: it is reinforced, not duplicated",
            "Ti ricordo che mi chiamo Bianca.",
            [{"fact": "L'utente si chiama Bianca", "operation": "redundant",
              "target_item_id": name_item.id}],
        )

        items = self.active(state)
        self.assertEqual(len(items), 3, "no item added")
        reinforced = find_item(items, "Bianca")
        self.assertEqual(reinforced.id, name_item.id, "it is the same item as before")
        self.assertGreater(reinforced.updated_at, updated_at_before, "updated_at refreshed")
        self.assertEqual(state["operation_log"][-1].op_type, "redundant")
        self.assertEqual(self.store.documents, {}, "a reinforcement does not touch the archive")

        # --- TURN 3: update ----------------------------------------------- #
        state = self.consolidate(
            3, "update", "a fact is refined: the old one becomes superseded",
            "Ho alzato l'obiettivo a 2200 calorie al giorno.",
            [{"fact": "L'obiettivo giornaliero e' 2200 calorie", "operation": "update",
              "target_item_id": goal_item.id}],
        )

        items = self.active(state)
        self.assertEqual(len(items), 3, "the old one leaves, the new one enters")
        find_item(items, "2200 calorie")
        self.assertNotIn(goal_item.id, [item.id for item in items], "the old one leaves core")
        self.assertEqual(self.store.status_of(goal_item.id), "superseded")
        self.assertEqual(self.store.documents[goal_item.id], goal_item.content,
                         "the original content stays in the archive")
        last = state["operation_log"][-1]
        self.assertEqual((last.op_type, last.related_item_id), ("update", goal_item.id))

        # --- TURN 4: contradict ------------------------------------------- #
        state = self.consolidate(
            4, "contradict", "a fact is contradicted: the old label reads as update",
            "In realta' non sono piu' vegetariana, ora mangio pesce.",
            [{"fact": "L'utente mangia pesce, non e' piu' vegetariana",
              "operation": "contradict", "target_item_id": veg_item.id}],
        )

        items = self.active(state)
        self.assertEqual(len(items), 3)
        find_item(items, "pesce")
        self.assertEqual(self.store.status_of(veg_item.id), "superseded")
        last = state["operation_log"][-1]
        self.assertEqual((last.op_type, last.related_item_id), ("update", veg_item.id))

        # --- TURN 5: delete ----------------------------------------------- #
        state = self.consolidate(
            5, "delete", "the user explicitly asks to forget a fact",
            "Dimentica il mio nome, non voglio che lo memorizzi.",
            [{"fact": "L'utente si chiama Bianca", "operation": "delete",
              "target_item_id": name_item.id}],
        )

        items = self.active(state)
        self.assertEqual(len(items), 2, "the deleted item leaves core memory")
        self.assertNotIn(name_item.id, [item.id for item in items])
        self.assertEqual(self.store.status_of(name_item.id), "deleted",
                         "it stays in the archive as a tombstone")
        self.assertEqual(state["operation_log"][-1].op_type, "delete")

        # --- TURN 6: archive (split core -> archive) ----------------------- #
        survivor = find_item(items, "pesce")
        moved = find_item(items, "2200 calorie")

        print_turn_header(6, "archive", "core memory over the limit: an item moves to the archive")
        # force the split on the next round
        self.agent.config = dataclasses.replace(self.agent.config, core_memory_limit=40)
        self.llm.script({
            "InsertCoreMemories": {"memories": []},
            "SplitCoreAndArchivalMemory": {
                "decisions": [
                    {"item_id": moved.id, "destination": "archive"},
                    {"item_id": survivor.id, "destination": "core"},
                ]
            },
        })
        self.agent.append_message("Parliamo d'altro.", "user")
        self.agent.append_message("Certo.", "assistant")

        state = self.agent.run_memory_agent("insert")
        print_memory_snapshot("turn 6 - archive", state, self.store,
                              self.agent.config.core_memory_limit)

        items = self.active(state)
        self.assertEqual([item.id for item in items], [survivor.id], "a single item stays in core")
        self.assertEqual(self.store.documents[moved.id], moved.content)
        self.assertEqual(self.store.status_of(moved.id), "active",
                         "archived but still valid, it is not a tombstone")
        self.assertEqual(state["operation_log"][-1].op_type, "archive")

        # --- TURN 7: retrieve ---------------------------------------------- #
        print_turn_header(
            7, "retrieve", "the answer draws from the archive, the tombstones stay out")
        self.llm.script(
            {
                "retrieve_memory": {"query": "calorie obiettivo", "k": 10},
            },
            default_content="Il tuo obiettivo e' 2200 calorie al giorno.",
        )
        self.agent.state["messages"] = [
            HumanMessage(content="Qual e' il mio obiettivo calorico?")]
        self.agent.state["retrieved_memory"] = ""

        state = self.agent.run_memory_agent("retrieve")
        print_memory_snapshot("turn 7 - retrieve", state, self.store,
                              self.agent.config.core_memory_limit)

        retrieved = state["retrieved_memory"]
        self.assertIn("2200 calorie", retrieved, "the active archived item is retrieved")
        self.assertNotIn(name_item.content, retrieved, "the deleted fact does not come back")
        self.assertNotIn(veg_item.content, retrieved, "the contradicted fact does not come back")
        self.assertEqual(state["messages"][-1].content,
                         "Il tuo obiettivo e' 2200 calorie al giorno.")


class TargetResolutionTest(unittest.TestCase):
    """What happens when the classifier does not give a valid target_item_id.

    These are the cases seen most often in practice: the model forgets the id,
    or makes it up.
    """

    def setUp(self):
        self.llm = ScriptedChatModel(tool_responses={}, default_content="ok")
        self.store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=self.llm, vector_store=self.store, config=LIFECYCLE_CONFIG)
        MemoryAgent.reset_instance()
        self.agent = MemoryAgent(config=LIFECYCLE_CONFIG)

    def tearDown(self):
        MemoryAgent.reset_instance()
        backends.reset()

    def consolidate(self, memories):
        self.llm.script({"InsertCoreMemories": {"memories": memories}})
        self.agent.append_message("un messaggio qualsiasi", "user")
        self.agent.append_message("ok", "assistant")
        return self.agent.run_memory_agent("insert")

    def test_redundant_without_id_is_matched_by_content(self):
        state = self.consolidate([{"fact": "L'utente si chiama Bianca", "operation": "new"}])
        existing = state["core_memory"][0]

        state = self.consolidate(
            [{"fact": "L'utente si chiama Bianca", "operation": "redundant"}])

        self.assertEqual([item.id for item in state["core_memory"]], [existing.id],
                         "recognized as the same fact, no duplicate")
        self.assertEqual(state["operation_log"][-1].op_type, "redundant")

    def test_update_on_an_unknown_id_is_kept_as_a_new_fact(self):
        state = self.consolidate(
            [{"fact": "L'utente vive a Palermo", "operation": "update",
              "target_item_id": "id-inventato-dal-modello"}])

        self.assertEqual([item.content for item in state["core_memory"]],
                         ["L'utente vive a Palermo"])
        self.assertEqual(state["operation_log"][-1].op_type, "create",
                         "better to save it as new than to lose the information")

    def test_delete_without_a_target_is_ignored(self):
        state = self.consolidate(
            [{"fact": "qualcosa che non e' in memoria", "operation": "delete"}])

        self.assertEqual(state["core_memory"], [], "nothing is deleted at random")
        self.assertEqual(state["operation_log"], [], "and no operation is recorded")


class MetadataRewriteTest(unittest.TestCase):
    """Changing the status of an archived item must not recompute the vector.

    Rewriting the document with add_texts would recompute the embedding: a
    network call to the model, on the cluster, to change a scalar field. And it
    happens at every redundant, update or delete that hits an item already in
    the archive.
    """

    def setUp(self):
        self.store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=ScriptedChatModel(tool_responses={}),
                           vector_store=self.store, config=LIFECYCLE_CONFIG)
        self.item = CoreMemoryItem(content="L'utente e' vegetariana")
        archive_items([self.item])
        self.writes_after_archiving = len(self.store.writes)

    def tearDown(self):
        backends.reset()

    def test_a_status_change_does_not_rewrite_the_document(self):
        supersede_archived_item(self.item.id, "L'utente mangia pesce", [])

        self.assertEqual(len(self.store.writes), self.writes_after_archiving,
                         "no extra write: no recomputed embedding")
        self.assertIn(self.item.id, self.store._collection.updates,
                      "the change goes through updating only the metadata")


class RetrievedSerializationTest(unittest.TestCase):
    """Retrieval leaves the service as a list, not as a sentence.

    retrieve_active_archival_memories returns text because it ends up straight
    in a prompt: "No relevant active memories found." is a sentence, not an
    empty result, and a caller that received it would treat it as a result.
    """

    def test_the_lines_become_entries(self):
        retrieved = "ID: abc, Content: uno\nID: def, Content: due"

        self.assertEqual(
            serialize_retrieved_for_response(retrieved),
            ["ID: abc, Content: uno", "ID: def, Content: due"])

    def test_the_nothing_found_sentence_becomes_an_empty_list(self):
        self.assertEqual(serialize_retrieved_for_response(NO_ARCHIVAL_RESULTS), [])


class StuckMetadataStore(FakeVectorStore):
    """The store that searches, but cannot rewrite the metadata."""

    def __init__(self):
        super().__init__()

        def refuse(*args, **kwargs):
            raise RuntimeError("metadata stuck")

        self._collection.update = refuse


class RetrievalCountTest(unittest.TestCase):
    """n_retrieve, n_used and retrieved_at: they move only with a retrieval or a use."""

    def setUp(self):
        self.store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=ScriptedChatModel(tool_responses={}),
                           vector_store=self.store, config=LIFECYCLE_CONFIG)

    def tearDown(self):
        backends.reset()

    def archive(self, *contents):
        # Born yesterday: a retrieval now must turn out more recent even with a
        # low-resolution clock.
        yesterday = datetime.now() - timedelta(days=1)
        items = [CoreMemoryItem(content=content, updated_at=yesterday) for content in contents]
        archive_items(items)
        return [item.id for item in items]

    def counters(self, item_id):
        metadata = self.store.metadatas[item_id]
        return metadata.get("n_retrieve"), metadata.get("retrieved_at")

    def test_a_new_item_starts_never_retrieved_nor_used(self):
        # A distant date: with "now" the comparison would pass even by chance.
        created = datetime(2026, 1, 1, 9, 30)
        item = CoreMemoryItem(content="ha un gatto", updated_at=created)

        self.assertEqual((item.n_retrieve, item.n_used), (0, 0))
        self.assertEqual(item.retrieved_at, created)

    def test_the_counters_travel_to_the_archive(self):
        item = CoreMemoryItem(content="ha un gatto", n_retrieve=2, n_used=5)
        archive_items([item])

        self.assertEqual(self.counters(item.id), (2, item.retrieved_at.isoformat()))
        self.assertEqual(self.store.metadatas[item.id]["n_used"], 5)

    def test_every_returned_memory_counts_once_per_retrieval(self):
        cat, dog, far = self.archive("il gatto Milo", "il gatto nero", "corre la mattina")
        created = self.store.metadatas[cat]["retrieved_at"]

        retrieve_active_archival_memories("gatto", k=2)
        retrieve_active_archival_memories("gatto", k=2)

        self.assertEqual(self.store.metadatas[cat]["n_retrieve"], 2)
        self.assertEqual(self.store.metadatas[dog]["n_retrieve"], 2)
        self.assertGreater(self.store.metadatas[cat]["retrieved_at"], created)
        self.assertEqual(self.counters(far), (0, self.store.metadatas[far]["updated_at"]),
                         "not returned, it does not count")

    def test_a_retrieval_is_not_a_write(self):
        # updated_at is the time of the writes, and the vector is not recomputed.
        cat, = self.archive("il gatto Milo")
        updated_at = self.store.metadatas[cat]["updated_at"]
        writes = len(self.store.writes)

        retrieve_active_archival_memories("gatto", k=1)

        self.assertEqual(self.store.metadatas[cat]["updated_at"], updated_at)
        self.assertEqual(len(self.store.writes), writes)

    def test_a_reinforcement_leaves_the_counters_alone(self):
        cat, = self.archive("il gatto Milo")
        retrieve_active_archival_memories("gatto", k=1)
        before = self.counters(cat)

        reinforce_archived_item(cat, [])

        self.assertEqual(self.counters(cat), before)

    def test_an_update_of_a_core_memory_inherits_the_counters(self):
        old = CoreMemoryItem(content="obiettivo 2000 calorie", n_retrieve=3, n_used=1)

        new = supersede_item(old, "obiettivo 2200 calorie", [])

        self.assertEqual((new.n_retrieve, new.n_used), (3, 1))
        self.assertEqual(new.retrieved_at, new.updated_at)

    def test_an_update_of_an_archived_memory_inherits_the_counters(self):
        cat, = self.archive("il gatto Milo ha due anni")
        retrieve_active_archival_memories("gatto", k=1)
        retrieve_active_archival_memories("gatto", k=1)
        record_uses([cat], [])

        new = supersede_archived_item(cat, "il gatto Milo ha tre anni", [])

        self.assertEqual((new.n_retrieve, new.n_used), (2, 1))
        self.assertEqual(new.retrieved_at, new.updated_at)

    def test_a_failed_count_does_not_cost_the_answer(self):
        self.store = StuckMetadataStore()
        backends.configure(vector_store=self.store)
        self.archive("il gatto Milo")

        self.assertIn("il gatto Milo", retrieve_active_archival_memories("gatto", k=1))


class JudgmentTest(unittest.TestCase):
    """safety, utility, need and gain: given on new and update, kept on redundant."""

    def setUp(self):
        self.llm = ScriptedChatModel(tool_responses={}, default_content="ok")
        self.store = FakeVectorStore()
        backends.reset()
        backends.configure(llm=self.llm, vector_store=self.store, config=LIFECYCLE_CONFIG)
        self.use("single")

    def tearDown(self):
        MemoryAgent.reset_instance()
        backends.reset()

    def use(self, mode):
        MemoryAgent.reset_instance()
        self.agent = MemoryAgent(config=dataclasses.replace(
            LIFECYCLE_CONFIG, consolidation_utility=mode))

    def consolidate(self, memories):
        self.llm.script({"InsertCoreMemories": {"memories": memories}})
        self.agent.append_message("un messaggio qualsiasi", "user")
        self.agent.append_message("ok", "assistant")
        return self.agent.run_memory_agent("insert")

    @staticmethod
    def judgment(item):
        return item.safety, item.utility, item.need, item.gain

    def judged_fields(self):
        """The consolidation tool's name and the fields it asks for beyond the classification."""
        tool, = self.llm.invocations[-1]["bound"]
        function = convert_to_openai_tool(tool)["function"]
        fields = set(function["parameters"]["properties"]["memories"]["items"]["properties"])
        return function["name"], fields - {"fact", "operation", "target_item_id"}

    def test_each_mode_binds_its_schema_under_the_same_name(self):
        for mode, judged in (("none", set()), ("single", {"safety", "utility"}),
                             ("need_gain", {"safety", "need", "gain"})):
            with self.subTest(mode=mode):
                self.use(mode)
                self.consolidate([])

                self.assertEqual(self.judged_fields(), ("InsertCoreMemories", judged))

    def test_a_safety_fact_drops_its_ratings(self):
        # It is protected anyway: a rating given all the same must not stay.
        state = self.consolidate([{"fact": "e' allergica alle arachidi", "operation": "new",
                                   "safety": True, "utility": 5, "need": 3, "gain": 3}])

        self.assertEqual(self.judgment(state["core_memory"][0]), (True, None, None, None))

    def test_a_rating_out_of_its_scale_counts_as_not_given(self):
        state = self.consolidate([
            {"fact": "ha un cane", "operation": "new", "safety": "false", "need": "3",
             "gain": 2.0, "utility": True},
            {"fact": "corre la mattina", "operation": "new", "safety": "forse", "utility": 0,
             "need": 4, "gain": 2.5},
        ])

        self.assertEqual([self.judgment(item) for item in state["core_memory"]],
                         [(False, None, 3, 2), (None, None, None, None)])

    def test_an_update_of_an_archived_memory_is_judged_again(self):
        archived = CoreMemoryItem(content="il gatto Milo ha due anni", safety=False, utility=2)
        archive_items([archived])

        state = self.consolidate([{"fact": "il gatto Milo ha tre anni", "operation": "update",
                                   "target_item_id": archived.id, "safety": False,
                                   "utility": 4}])

        self.assertEqual(self.judgment(state["core_memory"][0]), (False, 4, None, None))
        self.assertEqual(state["operation_log"][-1].ratings, {"utility": 4})

    def test_a_redundant_keeps_the_ratings_it_had(self):
        item = self.consolidate([{"fact": "ha un cane", "operation": "new",
                                  "safety": False, "utility": 2}])["core_memory"][0]
        archived = CoreMemoryItem(content="corre la mattina", safety=False, utility=2)
        archive_items([archived])

        state = self.consolidate([
            {"fact": "ha un cane", "operation": "redundant", "target_item_id": item.id,
             "safety": True, "utility": 5},
            {"fact": "corre la mattina", "operation": "redundant",
             "target_item_id": archived.id, "safety": True, "utility": 5},
        ])

        self.assertEqual(self.judgment(state["core_memory"][0]), (False, 2, None, None))
        metadata = self.store.metadatas[archived.id]
        self.assertEqual((metadata["safety"], metadata["utility"]), (False, 2))

    def test_the_log_shows_the_judgment_of_new_and_update_with_safety_only_when_true(self):
        state = self.consolidate([
            {"fact": "ha un cane", "operation": "new", "safety": False, "utility": 4},
            {"fact": "e' allergica alle arachidi", "operation": "new", "safety": True},
        ])
        dog = state["core_memory"][0]
        state = self.consolidate([
            {"fact": "ha un cane di nome Argo", "operation": "update",
             "target_item_id": dog.id, "safety": False, "utility": 5},
            {"fact": "e' allergica alle arachidi", "operation": "redundant",
             "target_item_id": state["core_memory"][1].id},
        ])

        self.assertEqual([(entry.op_type, entry.ratings) for entry in state["operation_log"]],
                         [("create", {"utility": 4}), ("create", {"safety": True}),
                          ("update", {"utility": 5}), ("redundant", None)])
        published = json.loads(serialize_operation_log_for_response(state["operation_log"])[2])
        self.assertEqual(published["ratings"], {"utility": 5})

    def test_with_need_and_gain_the_log_shows_both(self):
        self.use("need_gain")

        state = self.consolidate([{"fact": "ha un cane", "operation": "new", "safety": False,
                                   "need": 2, "gain": 3}])

        self.assertEqual(state["operation_log"][0].ratings, {"need": 2, "gain": 3})

    def test_without_a_judgment_the_log_shows_none(self):
        self.use("none")

        state = self.consolidate([{"fact": "ha un cane", "operation": "new"}])

        self.assertIsNone(state["operation_log"][0].ratings)

    def test_the_ratings_travel_to_the_archive_and_the_missing_ones_stay_out(self):
        # Chroma does not accept None in the metadata.
        item = CoreMemoryItem(content="ha un cane", safety=False, need=3, gain=1)
        archive_items([item])

        metadata = self.store.metadatas[item.id]
        self.assertEqual((metadata["safety"], metadata["need"], metadata["gain"]), (False, 3, 1))
        self.assertNotIn("utility", metadata)


if __name__ == "__main__":
    unittest.main()
