"""The eviction replay, on small invented traces, without a model.

Run: python memory_service/run_tests.py -v
"""

import contextlib
import dataclasses
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PACKAGE_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import eviction_replay as replay_tool  # noqa: E402
from eviction_replay import Insert, KeyRule, Question, Replay, ScoreRule  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402


CONFIG = MemoryConfig(
    node_name="memory_agent",
    core_memory_limit=0,
    archive_memory_limit=100000,
    consolidation_utility="need_gain",
    eviction_combination="weighted_sum",
    eviction_time_decay_field="updated_at",
    eviction_time_decay_function="power",
    eviction_frequency_function="log",
    prune_fraction=0.10,
    track_used=False,
    generate_answer=False,
    count_candidate_search=False,
    chroma_path="/tmp/not-used",
    collection_name="test_archive",
    llm_config={"model_name": "fake", "model_provider": "fake", "temperature": 0.0},
)
START = datetime(2030, 1, 1, 9, 0)


def at(day):
    return START + timedelta(days=day)


def operation(kind, item_id, day, content=None, related=None, ratings=None):
    return {"op_type": kind, "item_id": item_id, "related_item_id": related,
            "content": content or f"memoria {item_id}", "ratings": ratings,
            "timestamp": at(day).isoformat()}


def create(item_id, day, **ratings):
    return operation("create", item_id, day, ratings=ratings or None)


def asked(key, day, session=0):
    return Question(session, key, at(day), {"question_type": "Basic Fact Recall"})


def by_content(key, memories):
    return [item_id for item_id, _ in sorted(memories, key=lambda memory: memory[1])]


def fifo():
    return KeyRule("fifo", lambda replay, item, now: replay.created[item.id], 0.10)


def oracle():
    return KeyRule("oracle", replay_tool.oracle_key, 0.10)


def replayed(events, limit=None, rule=None, needs=None, config=CONFIG):
    return Replay(config, limit, rule, by_content, needs).run(events)


class TraceEventsTest(unittest.TestCase):
    """Every insert of a session, the flush included, then its questions."""

    USER = {"sessions": [{
        "dialogue": [{"timestamp": "Jan 01, 2030, 09:00:00"}, {"timestamp": "x"},
                     {"timestamp": "Jan 01, 2030, 09:05:00"}, {"timestamp": "x"}],
        "end_time": "Jan 01, 2030, 10:00:00"}]}

    def session(self, operations):
        return {"sessions": [{"operations": operations, "questions": [{"asked": "dove?"}]}]}

    def test_the_operations_go_to_their_insert(self):
        operations = [dict(create("a", 0), exchange=1, insert=1),
                      dict(create("b", 0), exchange=1, insert=2)]

        events = replay_tool.trace_events(self.session(operations), self.USER)

        inserts = [event for event in events if isinstance(event, Insert)]
        self.assertEqual([len(event.operations) for event in inserts], [0, 1, 1])
        self.assertEqual(inserts[-1].time, datetime(2030, 1, 1, 10, 0), "the flush, at the end")
        self.assertIsInstance(events[-1], Question)
        self.assertEqual(events[-1].key, "0:0")

    def test_without_the_insert_field_the_flush_joins_the_last_exchange(self):
        operations = [dict(create("a", 0), exchange=1), dict(create("b", 0), exchange=1)]

        events = replay_tool.trace_events(self.session(operations), self.USER)

        self.assertEqual([len(event.operations) for event in events if isinstance(event, Insert)],
                         [0, 2, 0])


class AgainstTheTraceTest(unittest.TestCase):
    """With no limit the replay is the trace: its split archives what the trace archived."""

    TEXTS = {"a": "x" * 20, "b": "y" * 20, "c": "z" * 20}

    def check(self, archived_at):
        config = dataclasses.replace(CONFIG, core_memory_limit=45)
        trace = {"sessions": [{"operations": [
            dict(operation("archive", "a", 2), exchange=archived_at, insert=archived_at)]}]}
        events = [Insert(0, at(day), [operation("create", item_id, day, self.TEXTS[item_id])], day)
                  for day, item_id in enumerate("abc")]
        replay = replayed(events, config=config)
        return replay, replay_tool.trace_checks(replay, trace)["split agreement"]

    def test_the_least_recent_leaves_the_core_as_in_the_trace(self):
        replay, agreement = self.check(archived_at=2)

        self.assertEqual(replay.split_archived, [(0, 2, "a")])
        self.assertEqual(agreement, 1.0)

    def test_the_same_memory_archived_at_another_insert_does_not_agree(self):
        self.assertEqual(self.check(archived_at=1)[1], 0.0)


class OverTheLimitTest(unittest.TestCase):
    """Core limit 0, so every memory reaches the archive at the end of its insert."""

    def events(self, *later):
        return [Insert(0, at(0), [create("a", 0)]), Insert(0, at(1), [create("b", 1)]),
                *later, asked("0:0", 5)]

    def test_the_rule_removes_and_the_question_misses_what_it_needs(self):
        replay = replayed(self.events(), limit=1, rule=fifo(), needs={"0:0": ["a"]})

        self.assertNotIn("a", replay.items)
        self.assertEqual(replay.asked[0].needed, [(0, False, False)])
        self.assertEqual(replay.counts["prunes needed later"], 1)

    def test_the_oracle_keeps_what_a_question_will_need(self):
        replay = replayed(self.events(), limit=1, rule=oracle(), needs={"0:0": ["a"]})

        self.assertEqual(sorted(replay.items), ["a"])
        self.assertEqual(replay.asked[0].needed, [(0, True, True)])

    def test_a_safety_memory_never_leaves(self):
        events = [Insert(0, at(0), [create("a", 0, safety=True)]),
                  Insert(0, at(1), [create("b", 1)])]

        replay = replayed(events, limit=1, rule=fifo())

        self.assertEqual(sorted(replay.items), ["a"])

    def test_the_score_rule_is_the_services_choice(self):
        rule = ScoreRule(CONFIG, consolidation_utility="none")

        replay = replayed(self.events(), limit=1, rule=rule)

        self.assertEqual(sorted(replay.items), ["b"], "the oldest decays the most")
        self.assertEqual(rule.name, "score none 0.10")


class DivergenceTest(unittest.TestCase):
    """An operation on a memory the rule removed: counted, and handled as the system would."""

    def removed_a(self, *later):
        return [Insert(0, at(0), [create("a", 0)]), Insert(0, at(1), [create("b", 1)]), *later]

    def test_an_update_keeps_its_new_memory_with_no_counts(self):
        update = operation("update", "a2", 2, related="a")

        replay = replayed(self.removed_a(Insert(0, at(2), [update])), limit=1, rule=fifo())

        self.assertEqual(replay.counts["updates on removed"], 1)
        self.assertIn("a2", replay.items)
        self.assertEqual(replay.items["a2"].n_retrieve, 0)

    def test_a_redundant_stores_its_fact_again(self):
        redundant = operation("redundant", "a", 3, content="memoria a, ripetuta")

        replay = replayed(self.removed_a(Insert(0, at(3), [redundant]), asked("0:0", 4)),
                          limit=1, rule=fifo(), needs={"0:0": ["a"]})

        self.assertEqual(replay.counts["redundants on removed"], 1)
        self.assertEqual(replay.asked[0].needed[0][1], True, "the fact is back")

    def test_a_delete_finds_nothing(self):
        delete = operation("delete", "a", 2)

        replay = replayed(self.removed_a(Insert(0, at(2), [delete])), limit=1, rule=fifo())

        self.assertEqual(replay.counts["deletes on removed"], 1)
        self.assertEqual(replay.counts["prunes"], 1)


class TwinUpdateTest(unittest.TestCase):

    def test_a_second_update_with_the_same_text_merges_into_the_first(self):
        events = [Insert(0, at(0), [create("b", 0)]), asked("0:0", 1),
                  Insert(0, at(2), [create("a", 2)]),
                  Insert(0, at(3), [operation("update", "c", 3, related="a"),
                                    operation("update", "c", 3, related="b")])]

        replay = replayed(events)

        self.assertEqual(sorted(replay.items), ["c"])
        self.assertEqual(replay.items["c"].n_retrieve, 1, "b's retrieval, not a's zero")
        self.assertEqual(replay.items["c"].n_search, 1, "b's search, not a's zero")


class SearchCountTest(unittest.TestCase):

    def test_a_question_is_a_chance_for_every_archived_memory_and_an_update_keeps_it(self):
        config = dataclasses.replace(CONFIG, core_memory_limit=15)
        events = [Insert(0, at(0), [operation("create", "a", 0, "x" * 10)]),
                  Insert(0, at(1), [operation("create", "b", 1, "y" * 10)]),
                  asked("0:0", 2), asked("0:1", 3),
                  Insert(0, at(4), [operation("update", "a2", 4, "w" * 10, related="a")])]

        replay = replayed(events, config=config)

        self.assertEqual(replay.items["b"].n_search, 0, "in the core, nothing searches it")
        self.assertEqual(replay.items["a2"].n_search, 2)


class IndicatorsTest(unittest.TestCase):

    def test_tally_sums_by_type_and_age(self):
        events = [Insert(0, at(0), [create("a", 0)]), asked("1:0", 9, session=1)]

        counts = replay_tool.tally(replayed(events, needs={"1:0": ["a"]}))

        self.assertEqual((counts["needed"], counts["kept"], counts["found"]), (1, 1, 1))
        self.assertEqual(counts["type Basic Fact Recall kept"], 1)
        self.assertEqual(counts["age 1 needed"], 1)
        self.assertEqual(counts["sessions"], 2)

    def test_auc(self):
        self.assertEqual(replay_tool.auc([(1, False), (2, True)]), 1.0)
        self.assertEqual(replay_tool.auc([(1, False), (1, True)]), 0.5)
        self.assertIsNone(replay_tool.auc([(1, True), (2, True)]))

    def test_only_a_later_session_makes_a_memory_needed(self):
        trace = {"sessions": [
            {"operations": [create("a", 0, need=3, gain=3), create("b", 0, need=2, gain=2),
                            create("s", 0, safety=True)]},
            {"operations": []}]}

        rated = replay_tool.rated_memories(trace, {"0:0": ["a", "b"], "1:0": ["b"]}, "need_gain")

        self.assertEqual([(needed, level) for _, needed, level in rated],
                         [(False, "high"), (True, "2/2")])

    def test_a_merged_twin_counts_once_with_its_last_ratings(self):
        trace = {"sessions": [{"operations": [
            create("c", 0, need=2, gain=2),
            operation("update", "c", 0, related="b", ratings={"need": 3, "gain": 3})]}]}

        rated = replay_tool.rated_memories(trace, {}, "need_gain")

        self.assertEqual([level for _, _, level in rated], ["high"])

    def test_in_single_the_memories_are_rated_by_utility(self):
        trace = {"sessions": [{"operations": [create("a", 0, utility=5),
                                              create("b", 0, utility=3),
                                              create("c", 0, need=3, gain=3)]}]}

        rated = replay_tool.rated_memories(trace, {}, "single")

        self.assertEqual(rated, [({"utility": 5}, False, "high"), ({"utility": 3}, False, "3")])


class RulesTest(unittest.TestCase):
    """The rules the replay tries on a trace."""

    TABLE = ((0.1, 0.1, 0.9), (0.2, 0.2, 0.95), (0.3, 0.3, 1.0))

    def test_every_rule_keeps_the_traces_frequency_function(self):
        rules = replay_tool.score_rules(CONFIG, self.TABLE)

        self.assertEqual({rule.config.eviction_frequency_function for rule in rules}, {"log"})

    def test_none_has_one_rule_per_fraction(self):
        names = [rule.name for rule in replay_tool.score_rules(CONFIG)
                 if rule.config.consolidation_utility == "none"]

        self.assertEqual(names, ["score none 0.05", "score none 0.10", "score none 0.20"])

    def test_single_has_no_expected_gain_beside_its_weighted_sum(self):
        config = dataclasses.replace(CONFIG, consolidation_utility="single")
        rules = [rule for rule in replay_tool.score_rules(config, (0.2, 0.4, 0.6, 0.8, 1.0))
                 if rule.config.consolidation_utility == "single"]

        self.assertEqual({rule.config.eviction_combination for rule in rules},
                         {"weighted_sum", "product"})
        self.assertEqual(len(rules), 12)

    def test_a_single_trace_with_expected_gain_still_has_its_own_rule(self):
        config = dataclasses.replace(CONFIG, consolidation_utility="single",
                                     eviction_combination="expected_gain")

        names = [rule.name for rule in replay_tool.score_rules(config)]

        self.assertIn(ScoreRule(config).name, names)

    def test_without_a_table_there_are_no_calibrated_rules(self):
        names = [rule.name for rule in replay_tool.score_rules(CONFIG)]

        self.assertFalse([name for name in names if "calibrated" in name])

    def test_the_calibrated_rules_are_the_combinations_that_read_the_table(self):
        calibrated = [rule for rule in replay_tool.score_rules(CONFIG, self.TABLE)
                      if rule.table is not None]

        self.assertEqual({rule.config.eviction_combination for rule in calibrated},
                         {"weighted_sum", "product"})
        self.assertEqual({rule.config.consolidation_utility for rule in calibrated},
                         {"need_gain"})
        self.assertIn("score need_gain calibrated weighted_sum 0.10",
                      [rule.name for rule in calibrated])

    def test_a_calibrated_rule_prunes_by_its_table(self):
        events = [Insert(0, at(0), [create("a", 0, need=1, gain=3),
                                    create("b", 0, need=3, gain=1)])]
        config = dataclasses.replace(CONFIG, prune_fraction=0.5)

        provisional = replayed(events, limit=1, rule=ScoreRule(config), config=config)
        calibrated = replayed(events, limit=1, rule=ScoreRule(config, self.TABLE),
                              config=config)

        self.assertEqual(sorted(provisional.items), ["b"], "a tie: the lower id leaves")
        self.assertEqual(sorted(calibrated.items), ["a"])

    def test_utility_alone_reads_the_traces_mode(self):
        config = dataclasses.replace(CONFIG, consolidation_utility="single")
        events = [Insert(0, at(0), [create("a", 0, utility=5), create("b", 0, utility=1)])]
        rule = next(rule for rule in replay_tool.reference_rules(0.5, "single")
                    if rule.name == "utility")

        replay = replayed(events, limit=1, rule=rule, config=config)

        self.assertEqual(sorted(replay.items), ["a"])

    def test_the_calibrated_utility_alone_needs_a_table(self):
        names = [rule.name for rule in replay_tool.reference_rules(0.10, "need_gain")]
        with_table = [rule.name for rule in
                      replay_tool.reference_rules(0.10, "need_gain", self.TABLE)]

        self.assertNotIn("utility calibrated", names)
        self.assertIn("utility calibrated", with_table)

    def test_a_table_in_the_same_order_adds_no_utility_alone(self):
        increasing = [rule.name for rule in replay_tool.reference_rules(
            0.10, "single", (0.0001, 0.15, 0.33, 0.59, 1.0))]
        tied = [rule.name for rule in replay_tool.reference_rules(
            0.10, "single", (0.1, 0.1, 0.33, 0.59, 1.0))]

        self.assertNotIn("utility calibrated", increasing)
        self.assertIn("utility calibrated", tied)

    def test_without_ratings_of_the_traces_model_there_is_no_table(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "ratings.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"uuid": "u0", "session": 0, "mode": "need_gain",
                                         "model": "qwen3.5-9b", "commit": "abc",
                                         "ratings": {}}) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(replay_tool.calibration_table(folder, [path], CONFIG))


class JudgeCandidatesTest(unittest.TestCase):
    """What the judge is shown: the core, the closest to the question, the closest to the
    evidence, once each, with their place in the question's ranking."""

    def test_the_evidence_adds_what_the_question_ranks_lower(self):
        config = dataclasses.replace(CONFIG, core_memory_limit=9)
        replay = replayed([Insert(0, at(day), [create(item_id, day)], day)
                           for day, item_id in enumerate("abcd")], config=config)

        shown = replay_tool.judge_candidates(replay, ["c", "b", "a"], [["a", "c"]], shown=1)

        self.assertEqual([(item_id, place) for item_id, _, place in shown],
                         [("d", 0), ("c", 1), ("a", 3)])

    def test_the_places_go_to_their_bucket(self):
        self.assertEqual([replay_tool.rank_bucket(place) for place in (0, 20, 35, 41)],
                         ["core", "1-20", "31-40", ">40"])

    def test_the_evidence_is_searched_as_the_service_writes(self):
        trace = {"sessions": [{"questions": [
            {"evidence": [{"memory_content": "Rossi Mario likes dogs"},
                          {"memory_content": "Mario has a cat"}]},
            {"evidence": "[{'memory_content': 'Rossi Mario likes dogs'}]"}]}]}

        self.assertEqual(replay_tool.evidence_texts(trace, "Rossi Mario"),
                         ["the user likes dogs", "the user has a cat"])


class JudgeReplyTest(unittest.TestCase):

    CANDIDATES = [("id-1", "uno"), ("id-2", "due")]

    def test_the_numbers_become_ids_and_the_rest_is_ignored(self):
        reply = {"needed": [2, "1", 7, "x", 2]}

        self.assertEqual(replay_tool.needed_ids(reply, self.CANDIDATES), ["id-2", "id-1"])

    def test_a_reply_without_a_list_is_an_error(self):
        with self.assertRaises(ValueError):
            replay_tool.needed_ids({"needed": "2"}, self.CANDIDATES)


if __name__ == "__main__":
    unittest.main()
