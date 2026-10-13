"""Calibration of UTILITY_TABLE and NEED_GAIN_TABLE from the long test's traces.

The two tables, in eviction.py, translate the classifier's ratings into the
utility term of the score. Until they are calibrated they are rating / 5 and
(need / 3) * (gain / 3): products of ordinal codes, with the ties that the
literature on risk matrices objects to. This script derives them from what the
memories really did.

The oracle: FAMA-memory without the memory
------------------------------------------
With MEMORY_LONGRUN_CALIBRATION the long test saves, for every FAMA question, the
memories active at its snapshot, with the ratings. For every (memory, question)
pair the script removes the memory and recomputes FAMA-memory:

    value = FAMA with the memory - FAMA without

A memory served the question if the value is positive, and the value is its
gain. It is Mattar & Daw's need x gain measured after the fact: how likely it is
to be needed, times how much it is worth when it is. No model call: FAMA-memory
is deterministic.

Every pair is an occasion to serve: a memory active at five questions counts
five times, one written after the last question does not count. Memories with
safety true stay out, as they do from eviction, and so do those without the
rating to calibrate.

A negative value means the memory does harm: usually a value that consolidation
should have superseded or deleted. It is a classification error, not a measure
of utility: it counts as not served, and the report says how many there are.

From the pairs to the tables
----------------------------
For each level of a rating:

    need   the fraction of served pairs
    gain   the mean of the values of the served pairs

both pulled toward the value of all the levels together with the weight of
PRIOR_WEIGHT pairs, so a level with little data invents nothing, and then made
non-decreasing with the isotonic regression weighted on the pairs.

    UTILITY_TABLE      need(u) * gain(u) for the five utility levels
    NEED_GAIN_TABLE    need(i) * gain(j): Mattar & Daw's product, with need
                       estimated on the need levels and gain on the gain ones

Both divided by the highest cell, so they lie in (0, 1] like the provisional
ones and can be pasted into eviction.py as they are printed.

Next to the product there is the free table: for each cell the mean of the
values, with 0 for the pairs not served, pulled toward the product's cell with
the weight of PRIOR_WEIGHT pairs and made non-decreasing along rows and columns.
With at least two runs the report compares the two on each run left out: if the
free one predicts better, the multiplicative hypothesis does not hold on our
data.

Limits
------
- The FAMA questions are 56, among recalls, tasks, updates and deletes: many
  pairs, few served. More runs are needed, and the report says how many pairs
  each level rests on.
- Removing a memory measures a marginal value: if two memories say the same
  thing, neither of them turns out served.
- The calibration holds for the long test's conversation and its questions.

    python memory_service/test/calibrate_utility.py [trace.jsonl]

uses the runs of the last saved commit, like operation_matrix.py.
"""

import os
import sys
from dataclasses import dataclass
from itertools import accumulate
from typing import Callable, List, Optional, Sequence, Tuple

import fama
from operation_matrix import load_runs

DEFAULT_TRACES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "longrun_calibration.jsonl")

LEVELS = {"utility": 5, "need": 3, "gain": 3}
PRIOR_WEIGHT = 2.0


@dataclass(frozen=True)
class Pair:
    """A memory active at a question's snapshot, and how much it served it."""

    run: int
    question: int
    memory: dict
    value: float

    @property
    def served(self) -> bool:
        """Removing it lowers FAMA-memory."""
        return self.value > 0

    @property
    def realized(self) -> float:
        """The value as the calibration counts it: 0 if it did not serve."""
        return max(0.0, self.value)


@dataclass(frozen=True)
class Levels:
    """need and gain of each level of a rating, already monotone, and the pairs behind them."""

    pairs: Tuple[int, ...]
    served: Tuple[int, ...]
    need: Tuple[float, ...]
    gain: Tuple[float, ...]


def memory_text(memories: Sequence[dict]) -> str:
    """The memories as FAMA-memory reads them in the long test: one per line."""
    return "\n".join(memory["content"] for memory in memories)


def pairs_of(run: dict, run_index: int) -> List[Pair]:
    """The (memory, question) pairs of a run, removing one memory at a time."""
    pairs = []
    for snapshot in run["questions"]:
        question = fama.QUESTIONS[snapshot["index"]]
        memories = snapshot["memories"]
        full = fama.score(question, memory_text(memories)).fama
        for position, memory in enumerate(memories):
            if memory.get("safety"):
                continue
            rest = memories[:position] + memories[position + 1:]
            value = full - fama.score(question, memory_text(rest)).fama
            pairs.append(Pair(run_index, snapshot["index"], memory, value))
    return pairs


def changed_snapshots(run: dict) -> int:
    """The snapshots whose FAMA does not match: the criteria changed after the run."""
    return sum(1 for snapshot in run["questions"]
               if abs(fama.score(fama.QUESTIONS[snapshot["index"]],
                                 memory_text(snapshot["memories"])).fama
                      - snapshot["fama"]) > 1e-9)


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
    """need and gain per level of `rating`, on the pairs that have that rating."""
    rated = [pair for pair in pairs if pair.memory.get(rating) is not None]
    gains = [pair.value for pair in rated if pair.served]
    if not gains:
        raise ValueError(f"no served pair with rating {rating}: more runs are needed")
    need_prior, gain_prior = len(gains) / len(rated), sum(gains) / len(gains)

    counts, served, need, gain = [], [], [], []
    for level in range(1, LEVELS[rating] + 1):
        here = [pair for pair in rated if pair.memory[rating] == level]
        hits = [pair.value for pair in here if pair.served]
        counts.append(len(here))
        served.append(len(hits))
        need.append(shrunk(len(hits), len(here), need_prior))
        gain.append(shrunk(sum(hits), len(hits), gain_prior))
    return Levels(tuple(counts), tuple(served),
                  tuple(isotonic(need, [count + PRIOR_WEIGHT for count in counts])),
                  tuple(isotonic(gain, [hit + PRIOR_WEIGHT for hit in served])))


def normalized(table):
    """The table divided by its highest cell, in (0, 1]."""
    if isinstance(table[0], tuple):
        top = max(max(row) for row in table)
        return tuple(tuple(value / top for value in row) for row in table)
    top = max(table)
    return tuple(value / top for value in table)


def utility_table(pairs: Sequence[Pair]) -> Tuple[float, ...]:
    """UTILITY_TABLE: need * gain per utility level, normalized."""
    levels = by_level(pairs, "utility")
    return normalized(tuple(n * g for n, g in zip(levels.need, levels.gain)))


def product_table(pairs: Sequence[Pair]) -> Tuple[Tuple[float, ...], ...]:
    """need(i) * gain(j), not normalized: it is also a prediction of the value."""
    need, gain = by_level(pairs, "need").need, by_level(pairs, "gain").gain
    return tuple(tuple(n * g for g in gain) for n in need)


def _cell(pair: Pair) -> Optional[Tuple[int, int]]:
    need, gain = pair.memory.get("need"), pair.memory.get("gain")
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
    """The mean of each cell, pulled toward the product and made monotone, not normalized."""
    prior = product_table(pairs)
    totals = [[0.0] * 3 for _ in range(3)]
    counts = [[0] * 3 for _ in range(3)]
    for pair in pairs:
        cell = _cell(pair)
        if cell is not None:
            totals[cell[0]][cell[1]] += pair.realized
            counts[cell[0]][cell[1]] += 1
    means = [[shrunk(totals[i][j], counts[i][j], prior[i][j]) for j in range(3)]
             for i in range(3)]
    weights = [[counts[i][j] + PRIOR_WEIGHT for j in range(3)] for i in range(3)]
    return monotone_table(means, weights)


def held_out_error(pairs: Sequence[Pair],
                   model: Callable[[Sequence[Pair]], Tuple[Tuple[float, ...], ...]]
                   ) -> Optional[float]:
    """Mean squared error on the pairs of each run, with the table of the others.

    None if no run left out can be predicted: with a single run the others are
    not there, and the table is not estimated.
    """
    runs = sorted({pair.run for pair in pairs})
    errors = []
    for run in runs:
        try:
            table = model([pair for pair in pairs if pair.run != run])
        except ValueError:
            continue
        for pair in pairs:
            cell = _cell(pair)
            if pair.run == run and cell is not None:
                errors.append((table[cell[0]][cell[1]] - pair.realized) ** 2)
    return sum(errors) / len(errors) if errors else None


def literal(name: str, table) -> List[str]:
    """The table as it is written in eviction.py."""
    if not isinstance(table[0], tuple):
        return [f"{name} = ({', '.join(f'{value:.4f}' for value in table)})"]
    lines = [f"{name} = (", "    # gain 1  gain 2  gain 3"]
    for level, row in enumerate(table, 1):
        lines.append(f"    ({', '.join(f'{value:.4f}' for value in row)}),  # need {level}")
    return lines + [")"]


def level_lines(levels: Levels, rating: str) -> List[str]:
    """Pairs, served, need and gain per level."""
    lines = [f"  {rating:<8}{'pairs':>8}{'served':>9}{'need':>8}{'gain':>8}"]
    for level in range(LEVELS[rating]):
        lines.append(f"  {level + 1:<8}{levels.pairs[level]:>8}{levels.served[level]:>9}"
                     f"{levels.need[level]:>8.3f}{levels.gain[level]:>8.3f}")
    return lines


def report(mode: str, runs: Sequence[dict]) -> List[str]:
    """The report of a consolidation_utility mode, with its table."""
    pairs = [pair for index, run in enumerate(runs) for pair in pairs_of(run, index)]
    harmful = sum(1 for pair in pairs if pair.value < 0)
    lines = [f"{mode}: {len(runs)} runs, {len(pairs)} pairs, "
             f"{sum(pair.served for pair in pairs)} served, {harmful} harmful"]
    changed = sum(changed_snapshots(run) for run in runs)
    if changed:
        lines.append(f"  WARNING: {changed} snapshots have a FAMA different from the "
                     f"recorded one: the criteria changed after the run")
    try:
        if mode == "single":
            lines += level_lines(by_level(pairs, "utility"), "utility")
            lines += [""] + literal("UTILITY_TABLE", utility_table(pairs))
        else:
            lines += level_lines(by_level(pairs, "need"), "need")
            lines += level_lines(by_level(pairs, "gain"), "gain")
            lines += [""] + literal("NEED_GAIN_TABLE", normalized(product_table(pairs)))
            lines += ["", "  Free table, for comparison:"]
            lines += ["  " + line for line in literal("free", normalized(free_table(pairs)))]
            product_error = held_out_error(pairs, product_table)
            free_error = held_out_error(pairs, free_table)
            if product_error is None or free_error is None:
                lines.append("  With a single run there is no run to leave out.")
            else:
                lines.append(f"  Error on the run left out: product "
                             f"{product_error:.5f}, free {free_error:.5f}")
    except ValueError as error:
        lines.append(f"  No table: {error}")
    return lines


def main(argv: Sequence[str]) -> int:
    """Calibrate the tables on the runs of the last saved commit and print them."""
    path = argv[1] if len(argv) > 1 else DEFAULT_TRACES
    runs = load_runs(path)
    if not runs:
        print(f"No trace saved in {path}")
        return 1

    commit = runs[-1].get("commit") or ""
    same = [run for run in runs if (run.get("commit") or "") == commit]
    print(f"{len(same)} runs of commit {commit or '(unknown)'} out of {len(runs)} saved")
    for mode in ("single", "need_gain"):
        mine = [run for run in same if run.get("consolidation_utility") == mode]
        print()
        if not mine:
            print(f"{mode}: no runs")
            continue
        for line in report(mode, mine):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
