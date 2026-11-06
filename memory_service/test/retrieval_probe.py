#!/usr/bin/env python3
"""Where the archive search of a finished HaluMem user loses what a question needs.

The result file holds every operation, so the archive each question searched can be
rebuilt. Its memories are embedded again with the service's embedding model and ranked
by exact cosine for five queries: the question, the query the model wrote (from the
user's log), that query without dates, that query with Qwen3-Embedding's instruction,
and the question in Italian, as the robot's users ask it while the facts are in English;
and for the merge of each question with the model's query (max cosine). On runs where the
service already searched with the user's words the model query is the question.
The Italian questions are translated once by the service's model and kept in --italian.
It prints:

- evidence covered: for every evidence point of HaluMem that has a memory at least this
  close in the archive (cosine, at three thresholds), whether one such memory is in the
  top 20 of each query: the comparison over all the questions;
- agreement: how much of what the run retrieved is the exact top for the model's query.
  Low points at the vector index, high at the embeddings themselves;
- the rank of each memory given with --target, under each query;
- how many short memories (the persona facts) fill the exact top 20 of each query;
- the persisted archive: Chroma's top 20 against the exact top 20 of each memory;
- three pairs of sentences, as a sanity check of the embedding server.

    python3 memory_service/test/retrieval_probe.py --run RUN_DIR --uuid UUID \
        --target 38=38fa36a2 --target 126=86910d6c,16e944e6
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import halumem_adapter as adapter

K = 20
SHORT = 50
THRESHOLDS = (0.70, 0.75, 0.80)
COLLECTION = "halumem"  # as in halumem_adapter.process_user
INSTRUCTION = ("Instruct: Given a question about the user, retrieve the stored facts about "
               "the user that answer it\nQuery: ")
TRANSLATION = ("Translate the question into Italian, as an Italian speaker would ask it. Keep "
               "names, dates and numbers as they are. Reply with the translation only.")
MONTH = (r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?"
         r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
DATES = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?\b,?", re.IGNORECASE),
    re.compile(MONTH + r"\.?(?=\s+(?:19|20)\d{2}\b)", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\b"),
)
MERGED = {"merged": "question", "merged it": "question it"}
SANITY = (("The user likes dogs", "The user loves puppies"),
          ("The user likes dogs", "The user lives in Columbus"),
          ("The user likes dogs", "The quarterly report is due on Friday"))


def without_dates(text: str) -> str:
    for pattern in DATES:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def archives_at_questions(result: Dict[str, Any]) -> Tuple[Dict[str, str], List[Tuple]]:
    """The text of every memory, and each question with the ids archived when it was asked."""
    content: Dict[str, str] = {}
    where: Dict[str, str] = {}
    asked = []
    for session in result["sessions"]:
        for operation in session["operations"]:
            kind, item = operation["op_type"], operation["item_id"]
            if kind in ("create", "update"):
                content[item] = operation["content"]
                where[item] = "core"
                if kind == "update":
                    where.pop(operation.get("related_item_id"), None)
            elif kind == "archive" and item in where:
                where[item] = "archive"
            elif kind in ("prune", "evict", "delete"):
                where.pop(item, None)
        for question in session.get("questions", []):
            asked.append((question, [item for item, place in where.items() if place == "archive"]))
    return content, asked


def logged_queries(path: str) -> List[str]:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return [ast.literal_eval(args)["query"]
            for args in re.findall(r"Invoking retrieve_memory with args: (\{.*\})", text)]


def italian_questions(llm: Any, questions: List[str], path: str) -> List[str]:
    """The questions in Italian, translated once by the service's model and kept in `path`."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
        if len(saved) == len(questions):
            return saved
    print(f"Translating {len(questions)} questions into Italian", flush=True)
    replies = adapter.with_retries(lambda: llm.batch(
        [[("system", TRANSLATION), ("human", question)] for question in questions],
        config={"max_concurrency": 4}))
    translated = [str(reply.content).strip() for reply in replies]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(translated, handle, ensure_ascii=False, indent=1)
    return translated


def unit(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def parse_targets(values: Sequence[str]) -> Dict[int, List[str]]:
    targets: Dict[int, List[str]] = {}
    for value in values:
        number, ids = value.split("=", 1)
        targets.setdefault(int(number), []).extend(ids.split(","))
    return targets


def probe_questions(embeddings: Any, content: Dict[str, str], asked: List[Tuple],
                    queries: List[str], italian: List[str], person: str,
                    targets: Dict[int, List[str]]) -> None:
    texts = sorted({content[item] for _, archived in asked for item in archived})
    evidence = [[adapter.neutral(point["memory_content"], person)
                 for point in question.get("evidence") or []] for question, _ in asked]
    points = sorted({point for listed in evidence for point in listed})
    print(f"Embedding {len(texts)} memories, {len(points)} evidence points and the queries",
          flush=True)
    memory = dict(zip(texts, unit(embeddings.embed_documents(texts))))
    gold = dict(zip(points, unit(embeddings.embed_documents(points)))) if points else {}
    variants = {
        "question": [question["asked"] for question, _ in asked],
        "model query": queries,
        "no dates": [without_dates(query) for query in queries],
        "instruction": [INSTRUCTION + query for query in queries],
        "question it": italian,
    }
    vectors = {name: unit(embeddings.embed_documents(batch)) for name, batch in variants.items()}
    names = list(variants) + list(MERGED)

    agreement, short = [], {name: [] for name in names}
    matchable, covered = Counter(), {name: Counter() for name in names}
    for number, (question, archived) in enumerate(asked):
        if not archived:
            continue
        matrix = np.array([memory[content[item]] for item in archived])
        similarity = {name: matrix @ vectors[name][number] for name in variants}
        # A merged search: the closest to either text is the top of the larger cosine.
        for name, source in MERGED.items():
            similarity[name] = np.maximum(similarity[source], similarity["model query"])
        ranked, top = {}, {}
        for name in names:
            order = np.argsort(-similarity[name], kind="stable")
            ranked[name] = [archived[index] for index in order]
            top[name] = set(order[:K].tolist())
            short[name].append(sum(len(content[item]) < SHORT for item in ranked[name][:K]))
        if evidence[number]:
            closeness = np.array([gold[point] for point in evidence[number]]) @ matrix.T
            for threshold in THRESHOLDS:
                for row in closeness:
                    relevant = set(np.flatnonzero(row >= threshold).tolist())
                    if relevant:
                        matchable[threshold] += 1
                        for name in names:
                            covered[name][threshold] += bool(relevant & top[name])
        retrieved = re.findall(r"ID: ([0-9a-f]{8})", question["context"])
        if retrieved:
            exact = set(ranked["model query"][:len(retrieved)])
            agreement.append(len(exact & set(retrieved)) / len(retrieved))
        if number in targets:
            print(f"\n#{number}: {question['asked']}\n    model query: {queries[number]}"
                  f"\n    in Italian:  {italian[number]}")
            for name, order in ranked.items():
                ranks = [order.index(item) + 1 if item in order else None
                         for item in targets[number]]
                print(f"    {name:12} rank of the targets {ranks} of {len(order)}")

    print(f"\nEvidence points with a memory at least this close in the archive, and the share "
          f"of them with one such memory in the top {K}:")
    print("    " + " " * 12 + "".join(f"{threshold:>8.2f}" for threshold in THRESHOLDS))
    print("    " + f"{'points':12}" + "".join(f"{matchable[t]:>8}" for t in THRESHOLDS))
    for name in names:
        shares = "".join(f"{100 * covered[name][t] / matchable[t]:>7.0f}%" if matchable[t]
                         else f"{'-':>8}" for t in THRESHOLDS)
        print(f"    {name:12}{shares}")
    shares = [sum(len(content[item]) < SHORT for item in archived) / len(archived)
              for _, archived in asked if archived]
    print(f"\nRetrieved by the run that is also the exact top for the model's query: "
          f"{100 * sum(agreement) / len(agreement):.0f}%")
    print(f"Memories shorter than {SHORT} characters: "
          f"{100 * sum(shares) / len(shares):.0f}% of the archive, in the exact top {K}:")
    for name, counts in short.items():
        print(f"    {name:12} {sum(counts) / len(counts):.1f} of {K}")


def probe_index(archive: str) -> None:
    import chromadb

    collection = chromadb.PersistentClient(path=archive).get_collection(COLLECTION)
    stored = collection.get(where={"status": "active"}, include=["embeddings"])
    ids, matrix = stored["ids"], unit(stored["embeddings"])
    size = min(K, len(ids))
    recall = []
    for vector in matrix:
        found = collection.query(query_embeddings=[vector.tolist()], n_results=size,
                                 where={"status": "active"})["ids"][0]
        exact = {ids[index] for index in np.argsort(-(matrix @ vector))[:size]}
        recall.append(len(exact & set(found)) / size)
    print(f"\nPersisted archive: {len(ids)} active memories, Chroma's top {size} holds "
          f"{100 * sum(recall) / len(recall):.0f}% of the exact top {size} "
          f"(worst {100 * min(recall):.0f}%)")


def probe_server(embeddings: Any) -> None:
    raw = embeddings.embed_documents([text for pair in SANITY for text in pair])
    print(f"\nEmbedding server: dimension {len(raw[0])}, norm {np.linalg.norm(raw[0]):.3f}")
    vectors = unit(raw)
    for index, (first, second) in enumerate(SANITY):
        print(f"    {vectors[2 * index] @ vectors[2 * index + 1]:.3f}  {first} / {second}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="the folder of the run, with tmp and logs")
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--target", action="append", default=[],
                        help="QUESTION=ID[,ID]: question number from 0 and the memories it missed")
    parser.add_argument("--italian", help="the file of the translated questions, by default "
                                          "probe/UUID_questions_it.json in the run")
    arguments = parser.parse_args(argv)

    from memory_service import backends
    from memory_service.config import MemoryConfig

    run = os.path.abspath(os.path.expanduser(arguments.run))
    with open(os.path.join(run, "tmp", f"{arguments.uuid}.json"), encoding="utf-8") as handle:
        result = json.load(handle)
    content, asked = archives_at_questions(result)
    queries = logged_queries(os.path.join(run, "logs", f"{arguments.uuid}.log"))
    if len(queries) != len(asked):
        raise SystemExit(f"{len(queries)} searches in the log for {len(asked)} questions")

    config = MemoryConfig.from_environment()
    backends.reset()
    backends.configure(config=config)
    italian = italian_questions(
        backends.get_llm("retrieval"), [question["asked"] for question, _ in asked],
        arguments.italian or os.path.join(run, "probe", f"{arguments.uuid}_questions_it.json"))
    for question, translated in list(zip(asked, italian))[:3]:
        print(f"    {question[0]['asked']}  ->  {translated}")

    embeddings = backends._build_embeddings(config)
    probe_server(embeddings)
    probe_questions(embeddings, content, asked, queries, italian, result["user_name"],
                    parse_targets(arguments.target))
    archive = os.path.join(run, "archives", arguments.uuid)
    if os.path.isdir(archive):
        probe_index(archive)
    else:
        print(f"\nNo archive at {archive}: the index check is skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
