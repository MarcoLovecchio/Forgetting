#!/usr/bin/env python3
"""How many calls at once the model server takes before each call slows down.

Sends the consolidation model, with the service's settings, requests built from HaluMem
dialogues, at each concurrency level: a level sends level × --rounds requests, never more
than level at a time, each on a different stretch of dialogue so that no prompt is cached.
Per level it prints the median and 90th percentile of the seconds per call, the output
tokens per call, the output tokens per second of one call and of the whole server, and
both against the first level. More workers pay only while the server's throughput grows
with them: when it stays flat, each call gets slower by as much as the calls added, and
the extra workers only queue.

Launch it with nothing else on the server, or read each level as that many calls on top
of what is already running there.

    python3 memory_service/test/load_probe.py --halumem PATH [--model qwen3.5-9b]
                                              [--levels 1 5 10 15] [--rounds 2]
"""

import argparse
import dataclasses
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
for path in (PACKAGE_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import halumem_adapter as adapter  # noqa: E402
from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402

INSTRUCTION = ("\n\nList every fact about the user in these messages, one per line, "
               "written as a short sentence about the user.")


def turns(users: Sequence[Dict[str, Any]]) -> Iterator[str]:
    for user in users:
        for session in user["sessions"]:
            for message in session["dialogue"]:
                yield f"{message['role']}: {message['content']}"


def dialogue_prompts(users: Sequence[Dict[str, Any]], words: int, count: int) -> List[str]:
    """`count` prompts of about `words` words each, on consecutive stretches of dialogue."""
    prompts: List[str] = []
    stretch: List[str] = []
    size = 0
    for turn in turns(users):
        stretch.append(turn)
        size += len(turn.split())
        if size >= words:
            prompts.append("\n".join(stretch) + INSTRUCTION)
            stretch, size = [], 0
            if len(prompts) == count:
                return prompts
    if not prompts:
        raise ValueError("Not enough dialogue for one prompt")
    # A small dataset is reused: the same prompts again, later in the probe.
    return [prompts[index % len(prompts)] for index in range(count)]


def percentile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def level_stats(seconds: Sequence[float], outputs: Sequence[int], wall: float
                ) -> Dict[str, float]:
    return {
        "median s": statistics.median(seconds),
        "p90 s": percentile(seconds, 0.9),
        "tokens per call": sum(outputs) / len(outputs),
        "tokens/s per call": statistics.mean(out / took for out, took in zip(outputs, seconds)
                                             if took > 0),
        "tokens/s in all": sum(outputs) / wall if wall > 0 else 0.0,
    }


def probe_level(llm: Any, prompts: Sequence[str], level: int) -> Dict[str, Any]:
    """Every prompt sent, `level` at a time: the stats of the calls that answered."""

    def call(prompt: str):
        started = time.perf_counter()
        try:
            response = llm.invoke(prompt)
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"
        usage = getattr(response, "usage_metadata", None) or {}
        return time.perf_counter() - started, int(usage.get("output_tokens", 0))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=level) as pool:
        outcomes = list(pool.map(call, prompts))
    wall = time.perf_counter() - started
    answered = [(took, out) for took, out in outcomes if took is not None]
    errors = [out for took, out in outcomes if took is None]
    found: Dict[str, Any] = {"level": level, "calls": len(answered), "errors": errors,
                             "wall s": wall}
    if answered:
        found.update(level_stats([took for took, _ in answered],
                                 [out for _, out in answered], wall))
    return found


def print_levels(levels: Sequence[Dict[str, Any]]) -> None:
    first = next((found for found in levels if found["calls"]), None)
    print(f"\n{'level':>5} {'calls':>5} {'median s':>9} {'p90 s':>7} {'tok/call':>9} "
          f"{'tok/s 1 call':>13} {'tok/s all':>10} {'all vs first':>13} {'s vs first':>11}")
    for found in levels:
        if not found["calls"]:
            print(f"{found['level']:>5} {0:>5}  every call failed: {found['errors'][0]}")
            continue
        print(f"{found['level']:>5} {found['calls']:>5} {found['median s']:>9.1f} "
              f"{found['p90 s']:>7.1f} {found['tokens per call']:>9.0f} "
              f"{found['tokens/s per call']:>13.1f} {found['tokens/s in all']:>10.1f} "
              f"{found['tokens/s in all'] / first['tokens/s in all']:>12.2f}x "
              f"{found['median s'] / first['median s']:>10.2f}x")
        if found["errors"]:
            print(f"      {len(found['errors'])} failed, the first: {found['errors'][0]}")
    print("\nA level is worth it while 'all vs first' grows with it; when it stops, the "
          "calls added only queue and 's vs first' grows instead.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="How many calls at once the model server "
                                                 "takes before each call slows down.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--model", help="the Qwen model, instead of the one in LLM_CONFIG")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 5, 10, 15],
                        help="calls at the same time, one level after the other")
    parser.add_argument("--rounds", type=int, default=2,
                        help="requests per level, as a multiple of the level")
    parser.add_argument("--words", type=int, default=2500,
                        help="words of dialogue in each prompt, about a consolidation call")
    arguments = parser.parse_args(argv)
    if min(arguments.levels) < 1 or arguments.rounds < 1:
        parser.error("--levels and --rounds must be at least 1")

    # Here and not on top: it loads the workspace's .env.
    from live_model import chat_model_unavailable

    config = MemoryConfig.from_environment()
    if arguments.model:
        config = dataclasses.replace(
            config, llm_config={**config.llm_config, "model_name": arguments.model})
    backends.reset()
    backends.configure(config=config)
    reason = chat_model_unavailable()
    if reason:
        raise SystemExit(f"The model does not answer: {reason}")

    root = os.path.abspath(os.path.expanduser(arguments.halumem))
    needed = sum(level * arguments.rounds for level in arguments.levels)
    prompts = dialogue_prompts(adapter.dataset_users(root), arguments.words, needed)
    llm = backends.get_llm("consolidation")
    print(f"Model {config.llm_config.get('model_name')}, consolidation settings, "
          f"{needed} requests of about {arguments.words} words of dialogue.")

    levels, offset = [], 0
    for level in arguments.levels:
        count = level * arguments.rounds
        print(f"Level {level}: {count} requests...", flush=True)
        levels.append(probe_level(llm, prompts[offset:offset + count], level))
        offset += count
    print_levels(levels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
