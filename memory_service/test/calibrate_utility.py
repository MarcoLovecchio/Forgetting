"""Calibration of UTILITY_TABLE and NEED_GAIN_TABLE on the calibration users of HaluMem.

The two tables, in eviction.py, translate the classifier's ratings into the utility
term of the score. Until they are calibrated they are rating / 5 and (need / 3) *
(gain / 3): products of ordinal codes, with the ties that the literature on risk
matrices objects to. This script derives them from what the memory points of the
first CALIBRATION_USERS users of HaluMem-Medium did, as calibration_ratings.py had the
model rate them. halumem_adapter.py never replays those users, so no user is both
calibrated on and evaluated.

The oracle: the evidence of the questions of later sessions
-----------------------------------------------------------
Every HaluMem question lists its evidence, the memory points an answer needs. A
(point, question) pair is an occasion to serve: a point written in a session before
the question's. The point served if it is among the evidence. A question of the same
session is left out: it asks about what was just said, which recency keeps anyway,
while eviction decides what is kept for long. An update does not end the point it
updates: many HaluMem updates add to the old point or only rephrase it, and the
questions keep citing it. Points with safety true stay out, as they do from eviction,
and so do those without the rating to calibrate; the interference points are never
rated, since a memory should never hold them.

From the pairs to the tables
----------------------------
A point either serves a question or it does not, so what it is worth when it serves
is the same at every level: both ratings enter as how often the point served. For
each level of a rating, the fraction of its pairs that served, pulled toward the
fraction of all the levels together with the weight of PRIOR_WEIGHT pairs, so a level
with little data invents nothing, and then made non-decreasing with the isotonic
regression weighted on the pairs.

    UTILITY_TABLE      the fraction of each utility level
    NEED_GAIN_TABLE    fraction(need i) * fraction(gain j) / fraction(all): Mattar &
                       Daw's product, with need and gain as two independent signs of
                       serving, so each cell predicts the fraction of its own pairs

Both divided by the highest cell, so they lie in (0, 1] like the provisional ones and
can be pasted into eviction.py as they are printed. Next to the product there is the
free table: for each cell its own fraction, pulled toward the product's cell with the
weight of PRIOR_WEIGHT pairs and made non-decreasing along rows and columns.

Every table is then tried on each user left out, estimated on the others: the AUC is
how often a pair that served scores higher than one that did not, ties counting a
half, which is what eviction needs, since it removes the lowest scores. need alone
and gain alone are tried too, to see what each rating adds, and NEED_GAIN_TABLE as
eviction.py holds it, which learns nothing from the pairs: the provisional table beside
the calibrated one. In single there is no such line: the provisional UTILITY_TABLE ranks
the levels as the calibrated one, so its AUC is the same.

    python memory_service/test/calibrate_utility.py --halumem PATH [--ratings FILE ...]

uses, of each file, the ratings of the last saved commit and model; with --model, of the
last commit saved with that model. Several files are several series of the same points:
their pairs go together, so each level's fraction averages the judgments of the series.
"""

import argparse
import json
import os
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import halumem_adapter as adapter
from memory_service import eviction

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RATINGS = os.path.join(HERE, "calibration_work", "halumem_ratings.jsonl")

LEVELS = {"utility": 5, "need": 3, "gain": 3}
MODES = {"single": ("utility",), "need_gain": ("need", "gain")}
TABLE_NAMES = {"single": "UTILITY_TABLE", "need_gain": "NEED_GAIN_TABLE"}
PRIOR_WEIGHT = 2.0

Point = Tuple[int, int]


@dataclass(frozen=True)
class Pair:
    """A point written before a question's session, and whether the question used it."""

    user: int
    ratings: Dict[str, Any]
    served: bool


@dataclass(frozen=True)
class Levels:
    """The fraction served at each level of a rating, already monotone, and its pairs."""

    pairs: Tuple[int, ...]
    served: Tuple[int, ...]
    rate: Tuple[float, ...]


Scorer = Callable[[Pair], Optional[float]]
Model = Callable[[Sequence[Pair]], Scorer]


def load_ratings(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The sessions rated with the commit and the model of the last one saved."""
    if not records:
        return []
    stamp = (records[-1].get("commit"), records[-1].get("model"))
    return [record for record in records if (record.get("commit"), record.get("model")) == stamp]


def of_model(records: Sequence[Dict[str, Any]], model: Optional[str]) -> List[Dict[str, Any]]:
    """The sessions of the last commit saved with `model`; without it, as latest."""
    return latest([record for record in records
                   if model is None or record.get("model") == model])


def series(records: Sequence[Dict[str, Any]], mode: str,
           model: Optional[str] = None) -> List[Dict[str, Any]]:
    """The sessions of `mode` rated with the last commit saved for it, of `model` if given."""
    return latest([record for record in records if record.get("mode") == mode
                   and (model is None or record.get("model") == model)])


def rated_points(records: Sequence[Dict[str, Any]],
                 mode: str) -> Dict[str, Dict[Point, Optional[Dict[str, Any]]]]:
    """uuid -> (session, point) -> the judgment, None if never given, for `mode`."""
    rated: Dict[str, Dict[Point, Optional[Dict[str, Any]]]] = {}
    for record in records:
        if record.get("mode") == mode:
            points = rated.setdefault(record["uuid"], {})
            for point, judgment in record["ratings"].items():
                points[(record["session"], int(point))] = judgment
    return rated


def pairs_of(user: Dict[str, Any], index: int,
             rated: Dict[Point, Optional[Dict[str, Any]]]) -> List[Pair]:
    """The pairs of a user: each rated point with each question of a later session."""
    where: Dict[str, List[Point]] = {}
    for session_index, session in enumerate(user["sessions"]):
        for point_index, point in enumerate(session["memory_points"]):
            if point.get("memory_source") != "interference":
                where.setdefault(point["memory_content"].strip(), []).append(
                    (session_index, point_index))
    usable = {point: judgment for point, judgment in rated.items()
              if judgment is not None and not judgment.get("safety")}

    pairs = []
    for session_index, session in enumerate(user["sessions"]):
        for question in session.get("questions", []):
            cited = {point for evidence in question["evidence"]
                     for point in where.get(evidence["memory_content"].strip(), [])}
            pairs += [Pair(index, judgment, point in cited)
                      for point, judgment in usable.items() if point[0] < session_index]
    return pairs


def isotonic(values: Sequence[float], weights: Sequence[float]) -> List[float]:
    """The closest non-decreasing sequence, in weighted least squares (PAV)."""
    blocks = []
    for value, weight in zip(values, weights):
        blocks.append((value, weight, 1))
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right, left = blocks.pop(), blocks.pop()
            total = left[1] + right[1]
            blocks.append(((left[0] * left[1] + right[0] * right[1]) / total, total,
                           left[2] + right[2]))
    return [mean for mean, _, count in blocks for _ in range(count)]


def shrunk(total: float, count: int, prior: float) -> float:
    """A mean pulled toward `prior` with the weight of PRIOR_WEIGHT observations."""
    return (total + PRIOR_WEIGHT * prior) / (count + PRIOR_WEIGHT)


def by_level(pairs: Sequence[Pair], rating: str) -> Levels:
    """The fraction served at each level of `rating`, on the pairs that have it."""
    top = LEVELS[rating]
    counts, served = [0] * top, [0] * top
    for pair in pairs:
        level = pair.ratings.get(rating)
        if level is not None and 1 <= level <= top:
            counts[level - 1] += 1
            served[level - 1] += pair.served
    if not sum(served):
        raise ValueError(f"no served pair with rating {rating}: more users are needed")
    prior = sum(served) / sum(counts)
    rate = [shrunk(hit, count, prior) for hit, count in zip(served, counts)]
    return Levels(tuple(counts), tuple(served),
                  tuple(isotonic(rate, [count + PRIOR_WEIGHT for count in counts])))


def normalized(table):
    """The table divided by its highest cell, in (0, 1]."""
    if isinstance(table[0], tuple):
        top = max(max(row) for row in table)
        return tuple(tuple(value / top for value in row) for row in table)
    top = max(table)
    return tuple(value / top for value in table)


def utility_table(pairs: Sequence[Pair]) -> Tuple[float, ...]:
    """UTILITY_TABLE: the fraction served per utility level, normalized."""
    return normalized(by_level(pairs, "utility").rate)


def product_table(pairs: Sequence[Pair]) -> Tuple[Tuple[float, ...], ...]:
    """fraction(need i) * fraction(gain j) / fraction(all), the fraction each cell would
    have if need and gain were independent signs of serving; not normalized."""
    need, gain = by_level(pairs, "need").rate, by_level(pairs, "gain").rate
    rated = [pair.served for pair in pairs if _cell(pair) is not None]
    overall = sum(rated) / len(rated)
    return tuple(tuple(n * g / overall for g in gain) for n in need)


def _cell(pair: Pair) -> Optional[Tuple[int, int]]:
    need, gain = pair.ratings.get("need"), pair.ratings.get("gain")
    return None if need is None or gain is None else (need - 1, gain - 1)


def monotone_table(table, weights, rounds: int = 1000):
    """Non-decreasing rows and columns, alternating the isotonic regression on the two."""
    table = [list(row) for row in table]
    for _ in range(rounds):
        before = [row[:] for row in table]
        table = [isotonic(row, weight) for row, weight in zip(table, weights)]
        columns = [isotonic(column, weight)
                   for column, weight in zip(zip(*table), zip(*weights))]
        table = [list(row) for row in zip(*columns)]
        if all(abs(a - b) < 1e-12 for row, old in zip(table, before)
               for a, b in zip(row, old)):
            break
    # The projections stop some 1e-13 away from the order: the cumulative maximum,
    # first on the rows and then on the columns, makes it exact.
    table = [list(accumulate(row, max)) for row in table]
    columns = [list(accumulate(column, max)) for column in zip(*table)]
    return tuple(tuple(row) for row in zip(*columns))


def free_table(pairs: Sequence[Pair]) -> Tuple[Tuple[float, ...], ...]:
    """The fraction of each cell, pulled toward the product and made monotone."""
    prior = product_table(pairs)
    served = [[0] * 3 for _ in range(3)]
    counts = [[0] * 3 for _ in range(3)]
    for pair in pairs:
        cell = _cell(pair)
        if cell is not None:
            served[cell[0]][cell[1]] += pair.served
            counts[cell[0]][cell[1]] += 1
    means = [[shrunk(served[i][j], counts[i][j], prior[i][j]) for j in range(3)]
             for i in range(3)]
    weights = [[counts[i][j] + PRIOR_WEIGHT for j in range(3)] for i in range(3)]
    return monotone_table(means, weights)


def level_model(rating: str) -> Model:
    """A rating alone: the score of a pair is the fraction served at its level."""
    def estimate(train: Sequence[Pair]) -> Scorer:
        rate = by_level(train, rating).rate
        return lambda pair: (None if pair.ratings.get(rating) is None
                             else rate[pair.ratings[rating] - 1])
    return estimate


def table_model(table: Callable[[Sequence[Pair]], Tuple[Tuple[float, ...], ...]]) -> Model:
    """A need x gain table: the score of a pair is its cell."""
    def estimate(train: Sequence[Pair]) -> Scorer:
        values = table(train)

        def score(pair: Pair) -> Optional[float]:
            cell = _cell(pair)
            return None if cell is None else values[cell[0]][cell[1]]
        return score
    return estimate


def eviction_table(train: Sequence[Pair]) -> Tuple[Tuple[float, ...], ...]:
    """NEED_GAIN_TABLE as eviction.py holds it: it learns nothing from the pairs."""
    return eviction.NEED_GAIN_TABLE


# In single the table of eviction.py ranks the levels as the calibrated one: the same AUC.
MODELS: Dict[str, Tuple[Tuple[str, Model], ...]] = {
    "single": (("utility", level_model("utility")),),
    "need_gain": (("need", level_model("need")), ("gain", level_model("gain")),
                  ("provisional", table_model(eviction_table)),
                  ("product", table_model(product_table)), ("free", table_model(free_table))),
}


def auc(scored: Sequence[Tuple[float, bool]]) -> Optional[float]:
    """How often a served pair scores above one that did not serve, ties counting a half."""
    positive = [score for score, served in scored if served]
    negative = sorted(score for score, served in scored if not served)
    if not positive or not negative:
        return None
    wins = 0.0
    for score in positive:
        below, up_to = bisect_left(negative, score), bisect_right(negative, score)
        wins += below + 0.5 * (up_to - below)
    return wins / (len(positive) * len(negative))


def held_out_auc(pairs: Sequence[Pair],
                 model: Model) -> Tuple[Optional[float], Dict[int, Optional[float]]]:
    """The AUC on each user left out, scored by the table of the others, and on all of them.

    Without a second user nothing can be predicted, and the AUC is None.
    """
    pooled: List[Tuple[float, bool]] = []
    by_user: Dict[int, Optional[float]] = {}
    for user in sorted({pair.user for pair in pairs}):
        try:
            score = model([pair for pair in pairs if pair.user != user])
        except ValueError:
            continue
        scored = [(value, pair.served) for pair in pairs if pair.user == user
                  for value in (score(pair),) if value is not None]
        by_user[user] = auc(scored)
        pooled += scored
    return auc(pooled), by_user


def literal(name: str, table) -> List[str]:
    """The table as it is written in eviction.py."""
    if not isinstance(table[0], tuple):
        return [f"{name} = ({', '.join(f'{value:.4f}' for value in table)})"]
    lines = [f"{name} = (", "    # gain 1  gain 2  gain 3"]
    for level, row in enumerate(table, 1):
        lines.append(f"    ({', '.join(f'{value:.4f}' for value in row)}),  # need {level}")
    return lines + [")"]


def level_lines(levels: Levels, rating: str) -> List[str]:
    """Pairs, served, their fraction and the estimate per level."""
    lines = [f"  {rating:<8}{'pairs':>9}{'served':>8}{'fraction':>10}{'estimate':>10}"]
    for level in range(LEVELS[rating]):
        count, hit = levels.pairs[level], levels.served[level]
        fraction = f"{100 * hit / count:.3f}%" if count else "-"
        lines.append(f"  {level + 1:<8}{count:>9}{hit:>8}{fraction:>10}"
                     f"{100 * levels.rate[level]:>9.3f}%")
    return lines


def judged_line(rated: Dict[str, Dict[Point, Optional[Dict[str, Any]]]], mode: str) -> str:
    """How many points were judged, about safety, without the mode's ratings, skipped."""
    judgments = [judgment for points in rated.values() for judgment in points.values()]
    skipped = sum(1 for judgment in judgments if judgment is None)
    safety = sum(1 for judgment in judgments if judgment and judgment.get("safety"))
    unrated = sum(1 for judgment in judgments if judgment and not judgment.get("safety")
                  and any(judgment.get(name) is None for name in MODES[mode]))
    return (f"  {len(judgments)} points: {safety} with safety true, {unrated} without the "
            f"ratings, {skipped} never judged")


def auc_lines(pairs: Sequence[Pair], mode: str) -> List[str]:
    results = [(name, *held_out_auc(pairs, model)) for name, model in MODELS[mode]]
    if all(pooled is None for _, pooled, _ in results):
        return ["  With a single user there is no user to leave out."]
    lines = ["  AUC on the user left out (0.5 is chance):"]
    for name, pooled, by_user in results:
        users = "  ".join("-" if value is None else f"{value:.3f}"
                          for _, value in sorted(by_user.items()))
        shown = "-" if pooled is None else f"{pooled:.3f}"
        lines.append(f"    {name:<11} {shown}   by user: {users}")
    return lines


def usable_pairs(pairs: Sequence[Pair], mode: str) -> List[Pair]:
    return [pair for pair in pairs if all(pair.ratings.get(name) is not None
                                          for name in MODES[mode])]


def mode_table(mode: str, usable: Sequence[Pair]):
    """UTILITY_TABLE or NEED_GAIN_TABLE, as eviction.py takes it."""
    return utility_table(usable) if mode == "single" else normalized(product_table(usable))


def pooled_pairs(users: Sequence[Dict[str, Any]], series_list: Sequence[Sequence[Dict[str, Any]]],
                 mode: str) -> List[Pair]:
    """The pairs of every series together: a point judged in three series counts three times,
    once with each judgment, so the fraction of a level averages the series."""
    pairs = []
    for records in series_list:
        rated = rated_points(records, mode)
        pairs += [pair for index, user in enumerate(users)
                  for pair in pairs_of(user, index, rated.get(user["uuid"], {}))]
    return pairs


def calibrated_table(root: str, series_list: Sequence[Sequence[Dict[str, Any]]], mode: str):
    """The table of `mode` from one series or more of the calibration users, as main prints it.

    It raises ValueError when no pair served at some rating.
    """
    users = adapter.calibration_users(root)
    return mode_table(mode, usable_pairs(pooled_pairs(users, series_list, mode), mode))


def report(mode: str, pairs: Sequence[Pair], users: int) -> List[str]:
    """The report of a consolidation_utility mode, with its table."""
    names = MODES[mode]
    usable = usable_pairs(pairs, mode)
    lines = [f"  {users} users, {len(usable)} pairs, {sum(pair.served for pair in usable)} served"]
    try:
        for name in names:
            lines += level_lines(by_level(usable, name), name)
        lines += [""] + literal(TABLE_NAMES[mode], mode_table(mode, usable))
        if mode == "need_gain":
            lines += ["", "  Free table, for comparison:"]
            lines += ["  " + line for line in literal("free", normalized(free_table(usable)))]
    except ValueError as error:
        return lines + [f"  No table: {error}"]
    return lines + [""] + auc_lines(usable, mode)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Calibrate the tables on the ratings of the last saved commit and model and print them."""
    parser = argparse.ArgumentParser(description="The utility tables from HaluMem's "
                                                 "calibration users.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--ratings", nargs="+", default=[DEFAULT_RATINGS],
                        help="the files calibration_ratings.py writes, one series each: "
                             "their pairs go together")
    parser.add_argument("--model", help="the ratings of this model, instead of the last saved")
    arguments = parser.parse_args(argv)

    loaded = [(path, load_ratings(path)) for path in arguments.ratings]
    model = arguments.model or next((records[-1].get("model")
                                     for _, records in reversed(loaded) if records), None)
    series_list = []
    for path, records in loaded:
        same = of_model(records, model)
        if not same:
            print(f"No ratings saved in {path}" + (f" with model {model}" if model else ""))
            continue
        print(f"{path}: {len(same)} sessions rated with commit "
              f"{same[-1].get('commit') or '(unknown)'} and model "
              f"{same[-1].get('model') or '(unknown)'}, out of {len(records)} saved")
        series_list.append(same)
    if not series_list:
        return 1

    users = adapter.calibration_users(os.path.abspath(os.path.expanduser(arguments.halumem)))
    for mode in MODES:
        rated_series = [(same, rated) for same in series_list
                        for rated in (rated_points(same, mode),) if rated]
        if not rated_series:
            print(f"\n{mode}: no ratings")
            continue
        together = f" {len(rated_series)} series, their pairs together" if len(
            rated_series) > 1 else ""
        print(f"\n{mode}:{together}")
        for _, rated in rated_series:
            print(judged_line(rated, mode))
        pairs = pooled_pairs(users, [same for same, _ in rated_series], mode)
        present = {uuid for _, rated in rated_series for uuid in rated}
        for line in report(mode, pairs, sum(1 for user in users if user["uuid"] in present)):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
