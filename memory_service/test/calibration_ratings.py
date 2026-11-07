#!/usr/bin/env python3
"""The model's judgment of the calibration users' memory points, for calibrate_utility.py.

halumem_adapter.py never replays the first CALIBRATION_USERS users of HaluMem-Medium:
their memory points are judged here as consolidation judges a new fact, with the same
descriptions of safety and of the ratings of the mode (--utility) and the settings of
the consolidation node. One call per session, with every point of the session; the
points the model skipped are asked again. The interference points stay out: they are
false facts the assistant said, never a memory to keep. The person is "the user", as
in the adapter.

Every session leaves one line in the ratings file (--ratings, by default
calibration_work/halumem_ratings.jsonl, ignored by git) with the mode, the model and
the commit. Launched again, a session already judged with the same mode, model and
commit is skipped, so a stopped series goes on where it was. At the end
calibrate_utility.py prints the tables.

    python3 memory_service/test/calibration_ratings.py --halumem PATH --utility need_gain
"""

import argparse
import dataclasses
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

import calibrate_utility
import halumem_adapter as adapter

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from memory_service import backends  # noqa: E402
from memory_service.config import MemoryConfig  # noqa: E402
from memory_service.consolidation import (  # noqa: E402
    MODE_RATINGS,
    SAFETY_DESCRIPTION,
    NeedGainJudgedOperation,
    UtilityJudgedOperation,
    read_ratings,
)
from memory_service.memory_manager_llm import (  # noqa: E402
    NEED_GAIN_STEP,
    REQUIRED,
    SAFETY_STEP,
    UTILITY_STEP,
)

ATTEMPTS = 3
NUMBER = "The number of the fact in the list."


class SafetyFact(BaseModel):
    """A fact whose loss could put the user at risk: protected, not rated."""

    number: int = Field(description=NUMBER)
    safety: Literal[True] = Field(description=SAFETY_DESCRIPTION)


class NeedGainFact(BaseModel):
    """A fact that is not about safety, with need and gain, both required."""

    number: int = Field(description=NUMBER)
    safety: Literal[False] = Field(description=SAFETY_DESCRIPTION)
    need: Literal[1, 2, 3] = Field(
        description=NeedGainJudgedOperation.model_fields["need"].description)
    gain: Literal[1, 2, 3] = Field(
        description=NeedGainJudgedOperation.model_fields["gain"].description)


class UtilityFact(BaseModel):
    """A fact with its safety flag and, unless it is about safety, its utility."""

    number: int = Field(description=NUMBER)
    safety: bool = Field(description=SAFETY_DESCRIPTION)
    utility: Optional[Literal[1, 2, 3, 4, 5]] = Field(
        default=None, description=UtilityJudgedOperation.model_fields["utility"].description)


class JudgeNeedGain(BaseModel):
    """Judge every fact you are given, by its number."""

    model_config = ConfigDict(title="JudgeFacts")

    facts: List[Union[NeedGainFact, SafetyFact]]


class JudgeUtility(BaseModel):
    """Judge every fact you are given, by its number."""

    model_config = ConfigDict(title="JudgeFacts")

    facts: List[UtilityFact]


TOOLS = {"single": JudgeUtility, "need_gain": JudgeNeedGain}

JUDGE_SYSTEM = """You judge facts about the user that the assistant has just stored: each one
        is a new memory, given with its number. Return one judgment for every fact."""

PROMPTS = {
    mode: ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM + SAFETY_STEP + step),
        ("human", "{facts}\n\nFlag every fact for safety and, unless safety is true, rate "
                  + rated + ".")])
    for mode, step, rated in (("single", UTILITY_STEP, "its utility"),
                              ("need_gain", NEED_GAIN_STEP, "its need and its gain"))
}


def _number(value: Any, top: int) -> Optional[int]:
    """A position from 1 to `top`, also written as a string; anything else is none."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() and 1 <= number <= top else None


def judge(llm: Any, mode: str, facts: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    """The judgment of each fact, by its position from 0; a fact the model skipped is not there."""
    chain = PROMPTS[mode] | llm.bind_tools([TOOLS[mode]], tool_choice=REQUIRED)
    listed = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, 1))
    response = adapter.with_retries(lambda: chain.invoke({"facts": listed}))
    judged: Dict[int, Dict[str, Any]] = {}
    for call in (getattr(response, "tool_calls", None) or [])[:1]:
        for item in call["args"].get("facts") or []:
            number = _number(item.get("number"), len(facts)) if isinstance(item, dict) else None
            if number is not None and number - 1 not in judged:
                ratings = read_ratings(item)
                judged[number - 1] = {"safety": ratings["safety"],
                                      **{name: ratings[name] for name in MODE_RATINGS[mode]}}
    return judged


def judge_session(llm: Any, mode: str, facts: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
    """Every fact judged, the skipped ones asked again; None for a fact never judged."""
    judgments: List[Optional[Dict[str, Any]]] = [None] * len(facts)
    waiting = list(range(len(facts)))
    for _ in range(ATTEMPTS):
        if not waiting:
            break
        judged = judge(llm, mode, [facts[index] for index in waiting])
        for position, judgment in judged.items():
            judgments[waiting[position]] = judgment
        waiting = [index for position, index in enumerate(waiting) if position not in judged]
    return judgments


def rate_session(llm: Any, user: Dict[str, Any], session_index: int,
                 stamp: Dict[str, Any]) -> Dict[str, Any]:
    """The line of a session: the judgment of each point but the interference ones."""
    name = adapter.user_name(user["persona_info"])
    points = [(index, adapter.neutral(point["memory_content"], name))
              for index, point in enumerate(user["sessions"][session_index]["memory_points"])
              if point.get("memory_source") != "interference"]
    judgments = judge_session(llm, stamp["mode"], [text for _, text in points]) if points else []
    return {"uuid": user["uuid"], "session": session_index, **stamp,
            "ratings": {str(index): judgment for (index, _), judgment in zip(points, judgments)}}


def rate_users(users: Sequence[Dict[str, Any]], llm: Any, stamp: Dict[str, Any], path: str,
               workers: int = 1) -> int:
    """Judge the sessions not judged yet with the same stamp: how many were judged now."""
    done = {(record["uuid"], record["session"])
            for record in calibrate_utility.load_ratings(path)
            if all(record.get(key) == value for key, value in stamp.items())}
    pending = [(user, index) for user in users for index in range(len(user["sessions"]))
               if (user["uuid"], index) not in done]
    print(f"{len(pending)} sessions to judge, {len(done)} already judged", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lock = threading.Lock()

    def one(user: Dict[str, Any], index: int) -> Dict[str, Any]:
        record = rate_session(llm, user, index, stamp)
        with lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    judged = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, *item) for item in pending]):
            record = future.result()
            judged += 1
            skipped = sum(1 for judgment in record["ratings"].values() if judgment is None)
            print(f"[{judged}/{len(pending)}] user {record['uuid'][:8]} session "
                  f"{record['session']}: {len(record['ratings'])} points"
                  f"{f', {skipped} never judged' if skipped else ''}", flush=True)
    return judged


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Judge the calibration users' points, then print the tables."""
    parser = argparse.ArgumentParser(description="The model's judgment of the memory points "
                                                 "of HaluMem's calibration users.")
    parser.add_argument("--halumem", required=True, help="the folder of the HaluMem clone")
    parser.add_argument("--utility", required=True, choices=("single", "need_gain"))
    parser.add_argument("--model", help="the Qwen model, instead of the one in LLM_CONFIG")
    parser.add_argument("--workers", type=int, default=1, help="sessions at the same time")
    parser.add_argument("--ratings", default=calibrate_utility.DEFAULT_RATINGS,
                        help="the file the judgments are appended to")
    arguments = parser.parse_args(argv)
    if arguments.workers < 1:
        parser.error("--workers must be at least 1")

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
    stamp = {"mode": arguments.utility, "model": config.llm_config.get("model_name"),
             "commit": adapter.git_commit()}
    rate_users(adapter.calibration_users(root), backends.get_llm("consolidation"), stamp,
               arguments.ratings, arguments.workers)
    shown = ["--model", stamp["model"]] if stamp["model"] else []
    return calibrate_utility.main(["--halumem", root, "--ratings", arguments.ratings, *shown])


if __name__ == "__main__":
    sys.exit(main())
