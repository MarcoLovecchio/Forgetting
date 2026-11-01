#!/usr/bin/env python3
"""Long test runs in a row for the calibration, each from an empty archive.

One run of test_long_term_interaction.py leaves one trace, and calibrate_utility.py
needs more of them. Before every run reset_archive.py empties the archive,
otherwise the run would start with the memories of the previous one. Every run is
a process of its own, so the core memory and the Chroma client die with it.

A run that leaves no trace does not stop the series: the next ones can still
succeed, and the end of the report says how many are missing. An archive that
cannot be cleaned does stop it, because a run on a dirty archive is worth nothing.
At the end calibrate_utility.py prints the tables.

The archive is the long test's: MEMORY_LONGRUN_CHROMA_PATH if set, otherwise
MEMORY_CHROMA_PATH. reset_archive.py deletes the whole folder, production
collection included.

    python memory_service/test/calibration_runs.py [--runs 5]
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, Mapping, Optional, Sequence

import calibrate_utility
from operation_matrix import load_runs

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from memory_service.config import load_environment  # noqa: E402

LONG_TEST = os.path.join(HERE, "test_long_term_interaction.py")
RESET_ARCHIVE = os.path.join(PACKAGE_ROOT, "reset_archive.py")
RUNS = 5


def run_environment(environment: Mapping[str, str]) -> Dict[str, str]:
    """The environment of a run: traces on, and the reset on the long test's folder."""
    prepared = dict(environment)
    prepared["MEMORY_LONGRUN_CALIBRATION"] = os.path.abspath(
        prepared.get("MEMORY_LONGRUN_CALIBRATION") or calibrate_utility.DEFAULT_TRACES)
    if prepared.get("MEMORY_LONGRUN_CHROMA_PATH"):
        prepared["MEMORY_CHROMA_PATH"] = prepared["MEMORY_LONGRUN_CHROMA_PATH"]
    return prepared


def run_series(runs: int, environment: Mapping[str, str]) -> Optional[int]:
    """Run the long test `runs` times: the traces saved, None if a reset failed."""
    path = environment["MEMORY_LONGRUN_CALIBRATION"]
    saved = 0
    for run in range(1, runs + 1):
        print(f"\n=== Calibration run {run} of {runs} ===", flush=True)
        reset = subprocess.run([sys.executable, RESET_ARCHIVE, "-y"], env=environment)
        if reset.returncode:
            print("Archive not cleaned: the series stops here.", flush=True)
            return None

        before = len(load_runs(path))
        test = subprocess.run([sys.executable, "-m", "pytest", LONG_TEST, "-v", "-s"],
                              env=environment)
        if len(load_runs(path)) > before:
            saved += 1
            if test.returncode:
                print(f"Run {run}: failed asserts, its trace stays in the file.", flush=True)
        else:
            print(f"Run {run} left no trace: see the pytest output above.", flush=True)
    return saved


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the series, then print the tables from the traces."""
    parser = argparse.ArgumentParser(description="Long test runs for the calibration.")
    parser.add_argument("--runs", type=int, default=RUNS, help="how many runs in a row")
    arguments = parser.parse_args(argv)

    load_environment()
    environment = run_environment(os.environ)
    saved = run_series(arguments.runs, environment)
    if saved is None:
        return 1

    print(f"\n{saved} traces saved out of {arguments.runs} runs\n", flush=True)
    if saved:
        calibrate_utility.main(["calibrate_utility.py",
                                environment["MEMORY_LONGRUN_CALIBRATION"]])
    return 0 if saved == arguments.runs else 1


if __name__ == "__main__":
    sys.exit(main())
