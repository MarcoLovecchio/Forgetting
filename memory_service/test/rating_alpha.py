#!/usr/bin/env python3
"""How stable the model's ratings are: Krippendorff's alpha between calibration series.

Every --ratings file is one series of calibration_ratings.py: the memory points of the
calibration users, judged again with the same prompt. Each series is a rater and each
point a unit, so the points line up by construction. Of each file and mode only the
sessions of the last commit and model saved for that mode are read; with --model, of
the last commit saved with that model, so a file can hold the 4B and the 9B.

    safety       nominal, in both modes
    utility      ordinal 1-5, single
    need, gain   ordinal 1-3, need_gain
    need x gain  ordinal on need * gain, the order of NEED_GAIN_TABLE: the utility term
                 eviction reads in need_gain, to set beside utility

A rating a series did not give, because the point was safety or never judged, is a
missing value: alpha uses every point with at least two values. Next to alpha, the
95% interval over BOOTSTRAP resamples of the points and the exact agreement, the
points with every value the same, which favours the shorter scales.

    python3 memory_service/test/rating_alpha.py --ratings A.jsonl B.jsonl [C.jsonl ...]
                                                [--model qwen3.5-9b]
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import calibrate_utility

BOOTSTRAP = 1000
SEED = 0
RELIABLE = 0.800
TENTATIVE = 0.667

Key = Tuple[str, int, int]
Judgments = Dict[Key, Optional[Dict[str, Any]]]


def field(name: str) -> Callable[[Dict[str, Any]], Any]:
    return lambda judgment: judgment.get(name)


def product(judgment: Dict[str, Any]) -> Optional[int]:
    need, gain = judgment.get("need"), judgment.get("gain")
    return need * gain if need is not None and gain is not None else None


@dataclass(frozen=True)
class Rating:
    """A rating of one mode, its metric and its categories in order."""

    name: str
    mode: str
    metric: str
    categories: Tuple[Any, ...]
    value: Callable[[Dict[str, Any]], Any]


RATINGS = (
    Rating("safety", "need_gain", "nominal", (False, True), field("safety")),
    Rating("need", "need_gain", "ordinal", (1, 2, 3), field("need")),
    Rating("gain", "need_gain", "ordinal", (1, 2, 3), field("gain")),
    Rating("need x gain", "need_gain", "ordinal", (1, 2, 3, 4, 6, 9), product),
    Rating("safety", "single", "nominal", (False, True), field("safety")),
    Rating("utility", "single", "ordinal", (1, 2, 3, 4, 5), field("utility")),
)


def read_series(path: str, model: Optional[str] = None
                ) -> Dict[str, Tuple[List[Dict[str, Any]], Judgments]]:
    """mode -> (its sessions of the last commit and model, point -> judgment)."""
    records = calibrate_utility.load_ratings(path)
    series = {}
    for mode in calibrate_utility.MODES:
        same = calibrate_utility.series(records, mode, model)
        if same:
            series[mode] = (same, {(uuid, session, point): judgment
                                   for uuid, points in
                                   calibrate_utility.rated_points(same, mode).items()
                                   for (session, point), judgment in points.items()})
    return series


def units(series: Sequence[Judgments], rating: Rating) -> List[List[Any]]:
    """For every point, the values the series gave it, the missing ones left out."""
    found = []
    for key in sorted(set().union(*series)):
        values = []
        for judgments in series:
            judgment = judgments.get(key)
            value = rating.value(judgment) if judgment else None
            # `in` alone would take True for 1: bool only in a bool scale.
            if value in rating.categories and (isinstance(value, bool)
                                               == isinstance(rating.categories[0], bool)):
                values.append(value)
        found.append(values)
    return found


def coincidences(found: Sequence[Sequence[Any]], categories: Sequence[Any]) -> np.ndarray:
    """The coincidence matrix of each point with two values or more: (points, C, C)."""
    index = {category: position for position, category in enumerate(categories)}
    pairable = [values for values in found if len(values) > 1]
    matrices = np.zeros((len(pairable), len(categories), len(categories)))
    for row, values in enumerate(pairable):
        counts = np.bincount([index[value] for value in values], minlength=len(categories))
        matrices[row] = (np.outer(counts, counts) - np.diag(counts)) / (len(values) - 1)
    return matrices


def distances(metric: str, totals: np.ndarray) -> np.ndarray:
    """The squared distance between categories; the ordinal one counts the values between."""
    if metric == "nominal":
        return 1.0 - np.eye(len(totals))
    cumulative = np.concatenate(([0.0], np.cumsum(totals)))
    positions = np.arange(len(totals))
    low, high = np.minimum.outer(positions, positions), np.maximum.outer(positions, positions)
    between = cumulative[high + 1] - cumulative[low]
    return (between - np.add.outer(totals, totals) / 2) ** 2


def alpha(matrix: np.ndarray, metric: str) -> float:
    """1 - observed / expected disagreement; nan when every value is the same category."""
    totals = matrix.sum(axis=1)
    delta = distances(metric, totals)
    expected = (np.outer(totals, totals) * delta).sum()
    if not expected:
        return float("nan")
    return float(1.0 - (totals.sum() - 1) * (matrix * delta).sum() / expected)


def interval(matrices: np.ndarray, metric: str, draws: int = BOOTSTRAP,
             seed: int = SEED) -> Tuple[float, float]:
    """The 95% percentile interval of alpha, the points resampled with replacement."""
    generator = np.random.default_rng(seed)
    count = len(matrices)
    flat = matrices.reshape(count, -1)
    values = [alpha((np.bincount(generator.integers(0, count, count), minlength=count)
                     @ flat).reshape(matrices.shape[1:]), metric) for _ in range(draws)]
    values = [value for value in values if not np.isnan(value)]
    if not values:
        return float("nan"), float("nan")
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def rating_line(series: Sequence[Judgments], rating: Rating) -> str:
    found = [values for values in units(series, rating) if len(values) > 1]
    if not found:
        return f"  {rating.name:<12} no point rated in two series"
    matrices = coincidences(found, rating.categories)
    lower, upper = interval(matrices, rating.metric)
    value = alpha(matrices.sum(axis=0), rating.metric)
    same = sum(1 for values in found if len(set(values)) == 1) / len(found)
    return (f"  {rating.name:<12} {len(found):>6}  {value:6.3f}  {lower:6.3f} - {upper:.3f}"
            f"  {100 * same:6.1f}%")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Alpha of every rating between the series, mode by mode."""
    parser = argparse.ArgumentParser(description="Krippendorff's alpha of the ratings "
                                                 "between calibration series.")
    parser.add_argument("--ratings", nargs="+", required=True,
                        help="the files of calibration_ratings.py, one series each")
    parser.add_argument("--model", help="only the ratings of this model")
    arguments = parser.parse_args(argv)
    if len(arguments.ratings) < 2:
        parser.error("--ratings needs two series or more")

    loaded = [read_series(path, arguments.model) for path in arguments.ratings]
    for path, series in zip(arguments.ratings, loaded):
        print(f"{path}:" if series else f"{path}: no ratings")
        for mode, (records, _) in series.items():
            print(f"  {mode}: {len(records)} sessions, commit "
                  f"{records[-1].get('commit') or '(unknown)'}, model "
                  f"{records[-1].get('model') or '(unknown)'}")

    for mode in calibrate_utility.MODES:
        series = [modes[mode][1] for modes in loaded if mode in modes]
        if len(series) < 2:
            print(f"\n{mode}: in fewer than two series")
            continue
        keys = set().union(*series)
        complete = sum(1 for key in keys
                       if all(judgments.get(key) is not None for judgments in series))
        print(f"\n{mode}: {len(series)} series, {len(keys)} points, "
              f"{complete} judged in every series")
        print(f"  {'rating':<12} {'points':>6}  {'alpha':>6}  {'95% interval':^15}  "
              f"{'exact':>7}")
        for rating in RATINGS:
            if rating.mode == mode:
                print(rating_line(series, rating))
    print(f"\nKrippendorff: {RELIABLE:.3f} or more to rely on the ratings, "
          f"{TENTATIVE:.3f} for tentative conclusions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
