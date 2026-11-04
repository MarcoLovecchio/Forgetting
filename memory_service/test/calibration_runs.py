#!/usr/bin/env python3
"""Long test runs for the calibration, each from an empty archive.

One run of test_long_term_interaction.py leaves one trace, and calibrate_utility.py
needs more of them. Every run is a process of its own, so the core memory and the
Chroma client die with it, and before it reset_archive.py empties the archive it runs
on, otherwise it would start with the memories of the previous run.

With --workers N, N runs go at the same time. Each worker has an archive of its own,
WORK/worker1, WORK/worker2, ... where WORK is --work (test/calibration_work by
default, ignored by git): the production archive is never touched. Each run writes
its trace and its operations to files of its own in WORK, and when it ends the series
appends them to MEMORY_LONGRUN_CALIBRATION (test/longrun_calibration.jsonl by
default) and MEMORY_LONGRUN_RESULTS, so two runs never write the same file. With more
than one worker the output of each run goes to WORK/run<k>.log.

A run that leaves no trace does not stop the series: the next ones can still
succeed, and the end of the report says how many are missing. An archive that
cannot be cleaned does stop it: no run starts after it, because a run on a dirty
archive is worth nothing. At the end calibrate_utility.py prints the tables.

    python memory_service/test/calibration_runs.py [--runs 5] [--workers 1]
"""

import argparse
import contextlib
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Mapping, Optional, Sequence, Tuple

import calibrate_utility
import operation_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from memory_service.config import load_environment  # noqa: E402

LONG_TEST = os.path.join(HERE, "test_long_term_interaction.py")
RESET_ARCHIVE = os.path.join(PACKAGE_ROOT, "reset_archive.py")
RUNS = 5
WORK = os.path.join(HERE, "calibration_work")


def destinations(environment: Mapping[str, str]) -> Tuple[str, str]:
    """Where the series puts the traces and the operations of its runs."""
    return (os.path.abspath(environment.get("MEMORY_LONGRUN_CALIBRATION")
                            or calibrate_utility.DEFAULT_TRACES),
            os.path.abspath(environment.get("MEMORY_LONGRUN_RESULTS")
                            or operation_matrix.DEFAULT_RESULTS))


def run_environment(environment: Mapping[str, str], work: str, run: int,
                    worker: int) -> Dict[str, str]:
    """The environment of a run: its worker's archive, files of its own, traces on."""
    prepared = dict(environment)
    archive = os.path.join(work, f"worker{worker}")
    # The long test reads the first, reset_archive.py the second.
    prepared["MEMORY_LONGRUN_CHROMA_PATH"] = archive
    prepared["MEMORY_CHROMA_PATH"] = archive
    prepared["MEMORY_LONGRUN_CALIBRATION"] = os.path.join(work, f"run{run}.trace.jsonl")
    prepared["MEMORY_LONGRUN_RESULTS"] = os.path.join(work, f"run{run}.operations.jsonl")
    return prepared


def move_lines(source: str, destination: str) -> int:
    """Append the lines of source to destination, then delete source: how many."""
    if not os.path.exists(source):
        return 0
    with open(source, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if lines:
        with open(destination, "a", encoding="utf-8") as handle:
            handle.writelines(lines)
    os.remove(source)
    return len(lines)


def run_one(run: int, worker: int, environment: Mapping[str, str], work: str,
            log: Optional[str]) -> Optional[int]:
    """Reset the worker's archive, then the long test: its exit code, None if no reset."""
    prepared = run_environment(environment, work, run, worker)
    # Left by a series that was stopped: they would be merged with this run's.
    for leftover in (prepared["MEMORY_LONGRUN_CALIBRATION"], prepared["MEMORY_LONGRUN_RESULTS"]):
        if os.path.exists(leftover):
            os.remove(leftover)
    with (open(log, "w", encoding="utf-8") if log else contextlib.nullcontext()) as handle:
        output = {"stdout": handle, "stderr": subprocess.STDOUT} if handle else {}
        reset = subprocess.run([sys.executable, RESET_ARCHIVE, "-y"], env=prepared, **output)
        if reset.returncode:
            return None
        # No pytest cache: runs at the same time would all write it.
        test = subprocess.run([sys.executable, "-m", "pytest", LONG_TEST, "-v", "-s",
                               "-p", "no:cacheprovider"], env=prepared, **output)
        return test.returncode


def run_series(runs: int, environment: Mapping[str, str], workers: int = 1,
               work: str = WORK) -> Optional[int]:
    """Run the long test `runs` times, `workers` at a time: the traces saved, None if a
    reset failed."""
    work = os.path.abspath(work)
    os.makedirs(work, exist_ok=True)
    traces, operations = destinations(environment)
    free: "queue.Queue[int]" = queue.Queue()
    for worker in range(1, workers + 1):
        free.put(worker)
    stopped = threading.Event()

    def one(run: int) -> Tuple[int, Optional[int], Optional[int], Optional[str]]:
        if stopped.is_set():
            return run, None, None, None
        worker = free.get()
        log = os.path.join(work, f"run{run}.log") if workers > 1 else None
        try:
            print(f"\n=== Calibration run {run} of {runs}, worker {worker}"
                  f"{f', output in {log}' if log else ''} ===", flush=True)
            code = run_one(run, worker, environment, work, log)
            if code is None:
                stopped.set()
            return run, worker, code, log
        finally:
            free.put(worker)

    saved = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, run) for run in range(1, runs + 1)]):
            run, worker, code, log = future.result()
            if worker is None:
                continue
            if code is None:
                print(f"Run {run}: the archive of worker {worker} was not cleaned, no run "
                      "starts after this one.", flush=True)
                continue
            prepared = run_environment(environment, work, run, worker)
            move_lines(prepared["MEMORY_LONGRUN_RESULTS"], operations)
            if move_lines(prepared["MEMORY_LONGRUN_CALIBRATION"], traces):
                saved += 1
                if code:
                    print(f"Run {run}: failed asserts, its trace stays in the file.",
                          flush=True)
            else:
                print(f"Run {run} left no trace: see {log or 'the pytest output above'}.",
                      flush=True)
    return None if stopped.is_set() else saved


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the series, then print the tables from the traces."""
    parser = argparse.ArgumentParser(description="Long test runs for the calibration.")
    parser.add_argument("--runs", type=int, default=RUNS, help="how many runs")
    parser.add_argument("--workers", type=int, default=1,
                        help="runs at the same time, each on an archive of its own")
    parser.add_argument("--work", default=WORK,
                        help="the folder of the workers' archives and of the logs")
    arguments = parser.parse_args(argv)
    if arguments.runs < 1 or arguments.workers < 1:
        parser.error("--runs and --workers must be at least 1")

    load_environment()
    saved = run_series(arguments.runs, os.environ, arguments.workers, arguments.work)
    if saved is None:
        return 1

    print(f"\n{saved} traces saved out of {arguments.runs} runs\n", flush=True)
    if saved:
        calibrate_utility.main(["calibrate_utility.py", destinations(os.environ)[0]])
    return 0 if saved == arguments.runs else 1


if __name__ == "__main__":
    sys.exit(main())
