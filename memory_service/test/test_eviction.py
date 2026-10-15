"""Eviction: superseded and deleted memories really leave the archive, and
over the limit the active ones with the lowest score leave.

Offline, on the same Chroma double as the other tests. The tombstones are
produced by the real consolidation functions - supersede_item, delete_item,
supersede_archived_item, delete_archived_item - and not written by hand: if one
day the way they are marked changes, these tests notice it.

Run: python memory_service/run_tests.py -v
"""

import dataclasses
import json
import os
import sys
import unittest
from unittest import mock
from datetime import datetime, timedelta

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from memory_service import backends, eviction  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    CoreMemoryItem,
    archive_items,
    delete_archived_item,
    delete_item,
    supersede_archived_item,
    supersede_item,
)
from memory_service.eviction import (  # noqa: E402
    archive_over_limit,
    combine_terms,
    evict_archived_tombstones,
    eviction_terms,
    frequency_term,
    need_gain_term,
    power_decay_term,
    prune_archive,
    prune_count,
    utility_term,
    weibull_decay_term,
)
from memory_service.memory_manager_llm import MemoryAgent  # noqa: E402

from fakes import FakeVectorStore, ScriptedChatModel  # noqa: E402


EVICTION_CONFIG = MemoryConfig(
    node_name="memory_agent",
    generate_answer=False,
    maximum_historical_messages=1,
    core_memory_limit=2000,
    eviction_time_decay_field="updated_at",
    eviction_time_decay_function="weibull",
    eviction_combination="weighted_sum",
    consolidation_utility="none",
    track_used=False,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)

NOW = datetime(2026, 9, 17, 12, 0)


class EvictionTestCase(unittest.TestCase):
    """Fake archive, empty log, and a shortcut to archive."""

    store_class = FakeVectorStore

    def setUp(self):
        self.store = self.store_class()
        backends.reset()
        backends.configure(vector_store=self.store, config=EVICTION_CONFIG)
        self.log = []

    def tearDown(self):
        backends.reset()

    def archive(self, content, status="active", days_old=None, now=NOW, retrieved_days_old=None,
                **judgment):
        """An item in the archive, as the core split leaves it."""
        item = CoreMemoryItem(content=content, status=status, **judgment)
        if days_old is not None:
            item.updated_at = now - timedelta(days=days_old)
        if retrieved_days_old is not None:
            item.retrieved_at = now - timedelta(days=retrieved_days_old)
        archive_items([item])
        return item.id


class WhatLeavesTest(EvictionTestCase):
    """What leaves the archive and what stays."""

    def test_superseded_and_deleted_leave_the_archive(self):
        old = CoreMemoryItem(content="obiettivo 2000 calorie")
        supersede_item(old, "obiettivo 2200 calorie", self.log)
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        kept = self.archive("ha un cane di nome Argo")

        evicted = evict_archived_tombstones(self.log)

        self.assertCountEqual(evicted, [old.id, gone.id])
        self.assertIsNone(self.store.status_of(old.id))
        self.assertIsNone(self.store.status_of(gone.id))
        self.assertEqual(self.store.status_of(kept), "active",
                         "archived by the core split but still valid: it stays")

    def test_items_retired_while_already_archived_leave_too(self):
        """The other road to becoming a tombstone: changing status in the archive."""
        novels = self.archive("legge romanzi storici")
        piano = self.archive("suona il pianoforte")
        supersede_archived_item(novels, "legge solo gialli", self.log)
        delete_archived_item(piano, self.log)

        self.assertCountEqual(evict_archived_tombstones(self.log), [novels, piano])
        self.assertEqual(self.store.documents, {})


class WhatTheLogSaysTest(EvictionTestCase):
    """One evict entry per document."""

    def test_each_eviction_is_logged_with_the_memory_text(self):
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        before = len(self.log)

        evict_archived_tombstones(self.log)

        added = self.log[before:]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].op_type, "evict")
        self.assertEqual(added[0].item_id, gone.id)
        self.assertEqual(added[0].content, "vive a Mondello")
        self.assertIsNone(added[0].related_item_id)


class NothingToDoTest(EvictionTestCase):
    """Without tombstones nothing happens, and above all no empty delete."""

    def test_an_archive_without_tombstones_is_left_as_it_is(self):
        kept = self.archive("ha un cane di nome Argo")

        self.assertEqual(evict_archived_tombstones(self.log), [])
        self.assertEqual(self.store.status_of(kept), "active")
        self.assertEqual(self.store.deletes, [],
                         "a delete without ids, depending on the version, is everything")
        self.assertEqual(self.log, [])


class UnreadableStore(FakeVectorStore):
    """The store that does not answer the tombstone search."""

    def get(self, ids=None, where=None, **kwargs):
        if where is not None:
            raise RuntimeError("chroma does not answer")
        return super().get(ids=ids, **kwargs)


class StuckStore(FakeVectorStore):
    """The store that finds the tombstones but cannot delete them."""

    def delete(self, ids=None, **kwargs):
        raise RuntimeError("collection stuck")


class WhenTheStoreFailsTest(EvictionTestCase):
    """It never raises, and the log does not tell of deletions that never happened.

    It runs at the end of the update_memory callback: an exception would end up
    in its except and would empty the response of a successful consolidation.
    """

    def use(self, store_class):
        self.store = store_class()
        backends.configure(vector_store=self.store)

    def test_a_store_that_cannot_be_searched_removes_nothing(self):
        self.use(UnreadableStore)
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        before = len(self.log)

        self.assertEqual(evict_archived_tombstones(self.log), [])
        self.assertIn(gone.id, self.store.documents)
        self.assertEqual(len(self.log), before)

    def test_a_failed_delete_logs_nothing(self):
        self.use(StuckStore)
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        before = len(self.log)

        self.assertEqual(evict_archived_tombstones(self.log), [])
        self.assertIn(gone.id, self.store.documents)
        self.assertEqual(len(self.log), before,
                         "the log records only what really happened")

    def test_an_unreadable_store_is_never_over_the_limit(self):
        # Without a count nothing must be removed.
        self.use(UnreadableStore)
        self.archive("ha un cane di nome Argo")

        self.assertEqual(archive_over_limit(0), 0)

    def test_a_failed_prune_logs_nothing(self):
        self.use(StuckStore)
        old = self.archive("ha un cane di nome Argo", days_old=30)

        self.assertEqual(prune_archive(self.log, EVICTION_CONFIG, now=NOW), [])
        self.assertIn(old, self.store.documents)
        self.assertEqual(self.log, [])


class ArchiveLimitTest(EvictionTestCase):
    """The limit counts the active memories and says by how much it is exceeded."""

    def test_within_the_limit_there_is_no_excess(self):
        for fact in ("ha un cane di nome Argo", "ha un gatto di nome Milo"):
            self.archive(fact)

        self.assertEqual(archive_over_limit(2), 0)

    def test_the_excess_is_how_many_active_memories_are_over(self):
        for fact in ("ha un cane", "ha un gatto", "corre la mattina", "nuota"):
            self.archive(fact)

        self.assertEqual(archive_over_limit(1), 3)

    def test_tombstones_do_not_count(self):
        # They leave anyway by status: counting them would trigger the limit for
        # memories that no longer count.
        self.archive("ha un cane di nome Argo")
        delete_item(CoreMemoryItem(content="vive a Mondello"), self.log)
        supersede_item(CoreMemoryItem(content="obiettivo 2000 calorie"),
                       "obiettivo 2200 calorie", self.log)

        self.assertEqual(archive_over_limit(1), 0)


class TimeDecayTermTest(unittest.TestCase):
    """The two decay curves on the time since a timestamp: Weibull and power law."""

    def worth(self, decay, **age):
        return decay((NOW - timedelta(**age)).isoformat(), NOW)

    def test_a_timestamp_in_the_future_is_worth_one(self):
        # A shifted clock must not give more than 1.
        for decay in (weibull_decay_term, power_decay_term):
            with self.subTest(decay=decay.__name__):
                self.assertEqual(self.worth(decay, hours=-5), 1.0)


class TimeDecayFieldTest(unittest.TestCase):
    """Which timestamp the time decay reads."""

    METADATA = {"updated_at": (NOW - timedelta(days=3)).isoformat(),
                "retrieved_at": (NOW - timedelta(days=20)).isoformat(), "n_retrieve": 0}

    def test_the_timestamps_share_one_slot(self):
        # Never a mean between timestamps: the term is one, the selector says which.
        for field in ("updated_at", "retrieved_at"):
            with self.subTest(field=field):
                config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay_field=field)
                self.assertEqual(eviction_terms(self.METADATA, config, NOW, 0)["time_decay"],
                                 weibull_decay_term(self.METADATA[field], NOW))


class FrequencyTermTest(EvictionTestCase):
    """The frequency term: a count relative to the highest, on n_retrieve or n_used."""

    USES_COUNTED = dataclasses.replace(EVICTION_CONFIG, track_used=True, generate_answer=True)
    METADATA = {"updated_at": (NOW - timedelta(days=3)).isoformat(),
                "n_retrieve": 3, "n_used": 1}

    def archive_counts(self, content, n_retrieve, n_used):
        # The same updated_at for all: only the frequency tells them apart.
        item = CoreMemoryItem(content=content, n_retrieve=n_retrieve, n_used=n_used,
                              updated_at=NOW)
        archive_items([item])
        return item.id

    def test_the_count_over_the_highest(self):
        for count, expected in ((6, 1.0), (3, 0.5), (1, 1 / 6), (0, 0.0)):
            with self.subTest(count=count):
                self.assertAlmostEqual(frequency_term(count, 6), expected)

    def test_an_archive_never_counted_is_worth_zero_everywhere(self):
        self.assertEqual(frequency_term(0, 0), 0.0)

    def test_the_two_counts_share_one_slot(self):
        decay = weibull_decay_term(self.METADATA["updated_at"], NOW)
        for field, config in (("n_retrieve", EVICTION_CONFIG), ("n_used", self.USES_COUNTED)):
            with self.subTest(field=field):
                self.assertEqual(eviction_terms(self.METADATA, config, NOW, frequency_max=6),
                                 {"time_decay": decay, "frequency": self.METADATA[field] / 6})

    def test_in_the_prune_the_value_is_relative_to_the_highest(self):
        ids = {n: self.archive_counts(f"fatto {n}", n_retrieve=n, n_used=0) for n in (4, 2, 3)}

        removed = prune_archive(self.log, EVICTION_CONFIG, now=NOW)

        self.assertEqual(removed, [ids[2]])
        self.assertEqual(self.log[0].score_terms["frequency"], 0.5, "2 out of a maximum of 4")

    def test_the_highest_is_taken_on_the_chosen_count(self):
        # All retrieved 9 times: the maximum of retrievals has nothing to do with uses.
        ids = {n: self.archive_counts(f"fatto {n}", n_retrieve=9, n_used=n) for n in (4, 2, 3)}

        removed = prune_archive(self.log, self.USES_COUNTED, now=NOW)

        self.assertEqual(removed, [ids[2]])
        self.assertEqual(self.log[0].score_terms["frequency"], 0.5, "2 uses, the maximum is 4")

    def retrieved_and_used(self):
        """Retrieved often but never used, used often but never retrieved, and one in between."""
        return (self.archive_counts("ha un cane", n_retrieve=4, n_used=0),
                self.archive_counts("ha un gatto", n_retrieve=0, n_used=4),
                self.archive_counts("corre la mattina", n_retrieve=2, n_used=2))

    def test_without_an_answer_the_uses_are_not_counted_and_n_retrieve_stays(self):
        _, never_retrieved, _ = self.retrieved_and_used()
        config = dataclasses.replace(self.USES_COUNTED, generate_answer=False)

        self.assertEqual(prune_archive(self.log, config, now=NOW), [never_retrieved])


class UtilityTermTest(unittest.TestCase):
    """The classifier's 1-5 rating in the score, and how it combines with the other terms."""

    METADATA = {"updated_at": (NOW - timedelta(days=3)).isoformat(), "n_retrieve": 0,
                "utility": 4, "need": 1, "gain": 3}
    TERMS = {"utility": 0.8, "time_decay": 0.4, "frequency": 0.2}

    def test_the_vote_picks_its_entry(self):
        with mock.patch.object(eviction, "UTILITY_TABLE", (0.1, 0.3, 0.5, 0.7, 0.9)):
            self.assertEqual([utility_term(vote) for vote in range(1, 6)],
                             [0.1, 0.3, 0.5, 0.7, 0.9])

    def test_each_mode_brings_its_own_and_none_brings_nothing(self):
        for mode, expected in (("none", None), ("single", utility_term(4)),
                               ("need_gain", need_gain_term(1, 3))):
            with self.subTest(mode=mode):
                config = dataclasses.replace(EVICTION_CONFIG, consolidation_utility=mode)
                self.assertEqual(eviction_terms(self.METADATA, config, NOW, 0).get("utility"),
                                 expected)

    def test_a_memory_without_a_vote_gets_the_middle_one(self):
        # The model can omit the rating, or give it off the scale: the prune must not stop.
        single = dataclasses.replace(EVICTION_CONFIG, consolidation_utility="single")
        need_gain = dataclasses.replace(EVICTION_CONFIG, consolidation_utility="need_gain")
        unrated = {"updated_at": NOW.isoformat(), "n_retrieve": 0}
        asymmetric = ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9))

        with mock.patch.object(eviction, "UTILITY_TABLE", (0.1, 0.3, 0.5, 0.7, 0.9)), \
                mock.patch.object(eviction, "NEED_GAIN_TABLE", asymmetric):
            self.assertEqual(eviction_terms(unrated, single, NOW, 0)["utility"], 0.5)
            self.assertEqual(eviction_terms(unrated, need_gain, NOW, 0)["utility"], 0.5)
            self.assertEqual(eviction_terms({**unrated, "need": 3}, need_gain, NOW, 0)["utility"],
                             0.8)

    def test_need_picks_the_row_and_gain_the_column(self):
        asymmetric = ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9))

        with mock.patch.object(eviction, "NEED_GAIN_TABLE", asymmetric):
            self.assertEqual(need_gain_term(1, 3), 0.3)
            self.assertEqual(need_gain_term(3, 1), 0.7)

    def test_the_tables_are_valid_utilities(self):
        # Calibrated tables too: in (0, 1], never going down as the level goes up.
        table = eviction.NEED_GAIN_TABLE
        self.assertEqual(len(eviction.UTILITY_TABLE), 5)
        self.assertEqual(len(table), 3)
        for line in (eviction.UTILITY_TABLE, *table, *zip(*table)):
            with self.subTest(line=line):
                self.assertTrue(all(0 < value <= 1 for value in line))
                self.assertEqual(list(line), sorted(line))
        self.assertTrue(all(len(row) == 3 for row in table))

    def test_the_weighted_sum_is_half_utility_and_a_quarter_each(self):
        self.assertAlmostEqual(combine_terms(self.TERMS, "weighted_sum"),
                               0.5 * 0.8 + 0.25 * 0.4 + 0.25 * 0.2)

    def test_the_product_is_utility_times_the_mean_of_the_others(self):
        self.assertAlmostEqual(combine_terms(self.TERMS, "product"), 0.8 * (0.4 + 0.2) / 2)

    def test_without_utility_the_two_combinations_are_the_same_mean(self):
        terms = {"time_decay": 0.4, "frequency": 0.2}

        for combination in ("weighted_sum", "product"):
            with self.subTest(combination=combination):
                self.assertAlmostEqual(combine_terms(terms, combination), 0.3)


class PruneCountTest(unittest.TestCase):
    """How many leave: 10% of the active ones, rounded up."""

    def test_ten_percent_rounded_up(self):
        for active, expected in ((1, 1), (5, 1), (50, 5), (51, 6)):
            with self.subTest(active=active):
                self.assertEqual(prune_count(active), expected)


class PruneArchiveTest(EvictionTestCase):
    """The active ones with the lowest score leave, 10%, and only those."""

    def prune(self, config=EVICTION_CONFIG):
        return prune_archive(self.log, config, now=NOW)

    def test_the_oldest_leave_first(self):
        ids = {days: self.archive(f"fatto {days}", days_old=days) for days in range(11)}

        removed = self.prune()

        self.assertCountEqual(removed, [ids[10], ids[9]], "10% of 11, rounded up, is 2")
        self.assertEqual(len(self.store.documents), 9)

    def test_the_sub_scores_survive_the_trip_to_the_ros_response(self):
        from memory_service.consolidation import serialize_operation_log_for_response

        self.archive("ha un cane di nome Argo", days_old=30)
        self.archive("ha un gatto di nome Milo", days_old=0)

        self.prune()

        published = json.loads(serialize_operation_log_for_response(self.log)[0])
        self.assertEqual(published["score_terms"], self.log[0].score_terms)

    def reinforced_and_retrieved(self):
        """Reinforced yesterday but never retrieved since, idle 10 days but retrieved yesterday."""
        return (self.archive("ha un cane di nome Argo", days_old=1, retrieved_days_old=30),
                self.archive("ha un gatto di nome Milo", days_old=10, retrieved_days_old=1))

    def test_on_updated_at_the_longest_untouched_leaves(self):
        _, untouched = self.reinforced_and_retrieved()

        self.assertEqual(self.prune(), [untouched])

    def test_on_retrieved_at_the_longest_unretrieved_leaves(self):
        unretrieved, _ = self.reinforced_and_retrieved()
        config = dataclasses.replace(EVICTION_CONFIG, eviction_time_decay_field="retrieved_at")

        self.assertEqual(self.prune(config), [unretrieved])


class SafetyTest(EvictionTestCase):
    """Memories with safety true are not candidates: no score, never out."""

    def test_the_oldest_stays_when_protected_and_still_counts_in_the_quota(self):
        ids = {days: self.archive(f"fatto {days}", days_old=days, safety=days == 10)
               for days in range(11)}

        removed = prune_archive(self.log, EVICTION_CONFIG, now=NOW)

        self.assertCountEqual(removed, [ids[9], ids[8]], "10% of 11 active, protected included")
        self.assertNotIn(ids[10], [entry.item_id for entry in self.log])

    def test_a_protected_memory_is_out_of_the_frequency_max(self):
        for content, n_retrieve, safety in (("e' allergica alle arachidi", 8, True),
                                            ("ha un cane", 2, False), ("ha un gatto", 4, False)):
            archive_items([CoreMemoryItem(content=content, n_retrieve=n_retrieve, safety=safety,
                                          updated_at=NOW)])

        prune_archive(self.log, EVICTION_CONFIG, now=NOW)

        self.assertEqual(self.log[0].content, "ha un cane")
        self.assertEqual(self.log[0].score_terms["frequency"], 0.5, "2 out of a maximum of 4")


class WiredIntoTheAgentTest(EvictionTestCase):
    """Tombstones and prune in the insert branch, switch on; the prune only over the limit."""

    def setUp(self):
        super().setUp()
        self.llm = ScriptedChatModel(tool_responses={}, default_content="ok")
        backends.configure(llm=self.llm)
        self.use_agent(eviction=True)

    def tearDown(self):
        MemoryAgent.reset_instance()
        super().tearDown()

    def use_agent(self, **overrides):
        config = dataclasses.replace(EVICTION_CONFIG, **overrides)
        backends.configure(config=config)
        MemoryAgent.reset_instance()
        self.agent = MemoryAgent(config=config)

    def insert(self, memories):
        self.llm.script({"InsertCoreMemories": {"memories": memories}})
        self.agent.append_message("un messaggio qualsiasi", "user")
        self.agent.append_message("ok", "assistant")
        return self.agent.run_memory_agent("insert")

    def remember_then_forget(self):
        state = self.insert([{"fact": "vive a Mondello", "operation": "new"}])
        item = state["core_memory"][0]
        self.insert([{"fact": "vive a Mondello", "operation": "delete",
                      "target_item_id": item.id}])
        return item

    def test_the_tombstone_of_this_insert_leaves_in_the_same_insert(self):
        item = self.remember_then_forget()

        self.assertIsNone(self.store.status_of(item.id))
        operations = self.agent.last_operations()
        self.assertEqual([(entry.op_type, entry.item_id) for entry in operations],
                         [("delete", item.id), ("evict", item.id)],
                         "the evict entry comes out in the same response as update_memory")

    def test_every_exit_of_the_insert_branch_goes_through_the_eviction(self):
        paths = {
            "window not exceeded": {"maximum_historical_messages": 10},
            "consolidation without split": {},
            "consolidation with split": {"core_memory_limit": 5},
        }
        for path, overrides in paths.items():
            with self.subTest(path=path):
                self.use_agent(eviction=True, **overrides)
                gone = CoreMemoryItem(content="vive a Mondello")
                delete_item(gone, self.log)

                self.insert([{"fact": "ha un cane", "operation": "new"}])

                self.assertIsNone(self.store.status_of(gone.id))

    def test_with_the_switch_off_nothing_leaves(self):
        self.use_agent(eviction=False, archive_memory_limit=0)
        old = self.archive("ha un cane di nome Argo", days_old=30, now=datetime.now())

        item = self.remember_then_forget()

        self.assertEqual(self.store.status_of(item.id), "deleted")
        self.assertEqual(self.store.status_of(old), "active", "over the limit, but switched off")
        self.assertEqual([entry.op_type for entry in self.agent.state["operation_log"]],
                         ["create", "delete"])

    def test_a_retrieve_removes_nothing(self):
        self.use_agent(eviction=True, archive_memory_limit=0)
        gone = CoreMemoryItem(content="vive a Mondello")
        delete_item(gone, self.log)
        old = self.archive("ha un cane di nome Argo", days_old=30, now=datetime.now())

        self.agent.run_memory_agent("retrieve", query="Dove abito?")

        self.assertEqual(self.store.status_of(gone.id), "deleted")
        self.assertEqual(self.store.status_of(old), "active")

    def test_within_the_limit_nothing_is_pruned(self):
        self.use_agent(eviction=True, archive_memory_limit=2)
        kept = [self.archive("ha un cane di nome Argo", days_old=30, now=datetime.now()),
                self.archive("ha un gatto di nome Milo", days_old=0, now=datetime.now())]

        self.insert([])

        self.assertEqual([self.store.status_of(doc_id) for doc_id in kept], ["active", "active"])
        self.assertEqual(self.store.deletes, [])
        self.assertEqual(self.agent.last_operations(), [])

    def test_over_the_limit_the_oldest_memory_is_pruned_in_the_same_insert(self):
        self.use_agent(eviction=True, archive_memory_limit=1)
        old = self.archive("ha un cane di nome Argo", days_old=30, now=datetime.now())
        recent = self.archive("ha un gatto di nome Milo", days_old=0, now=datetime.now())

        self.insert([])

        self.assertIsNone(self.store.status_of(old))
        self.assertEqual(self.store.status_of(recent), "active")
        operations = self.agent.last_operations()
        self.assertEqual([(entry.op_type, entry.item_id) for entry in operations],
                         [("prune", old)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
