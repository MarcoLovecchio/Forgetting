"""Confusion matrix of the consolidation operations.

For each message of the long test's conversation: the operation it should
trigger, from the comment next to it in CONVERSATION, against the one
consolidation really made. FAMA looks at the state of the memory on 56
questions; this one looks at every single decision.

The unit is the message, not the operation. A message that produces two `new`
stays a new: how many memories come out of it is granularity, and it is counted
separately. Where a message carries two facts ("mia sorella si chiama Chiara e
vive a Milano") the repetitions are expected, elsewhere they are fragmentation.
A message that produces different types ends up in "mixed".

Where more readings are defensible there are accepted combinations, and the
accuracies are two: strict on the label, lenient on the combinations. The types
produced must coincide with a combination, being a part of one is not enough:
at msg 81 a redundant alone loses the guitar. A `new` instead of an `update` is
acceptable only if the old memory stays true (Wednesday is added to Saturday),
never if the value changes: 2000 and 2200 active together are a contradiction
in memory.

It measures the type, not the target: a right update on the wrong memory here
turns out correct.

    python memory_service/test/operation_matrix.py [file.jsonl]

sums the runs saved by the long test, only those of the last commit.
"""

import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

CLASSES = ("new", "redundant", "update", "delete", "none")
MIXED = "mixed"
COLUMNS = CLASSES + (MIXED,)

DEFAULT_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "longrun_operations.jsonl")

_DISPLAY = {"none": "none", MIXED: "mixed"}

# Log entries that are not a classification: the core split's move and
# eviction's removals. The log writes "create" where the tool says "new".
_NOT_CLASSIFIED = ("archive", "evict", "prune")
_LOG_TO_CLASS = {"create": "new"}

# From the comments of CONVERSATION: contradict counts as update, questions and
# chit-chat as no operation, everything else - presentation included - as new.
# test_operation_matrix checks that they still match.
_EXPECTED = {
    "update": (38, 48, 53, 56, 59, 64, 69, 72, 73, 81, 86, 94, 102, 105, 110, 114, 117, 119,
               123, 124, 132, 134, 140, 142, 146, 150, 155, 156, 159, 162, 166, 168, 170,
               172, 175, 180, 186, 187, 191, 193, 196, 204, 210, 216, 217, 228, 241),
    "redundant": (37, 41, 46, 54, 65, 80, 84, 92, 104, 161, 184, 185, 200, 202, 219, 221,
                  223, 226, 237, 238),
    "delete": (66, 87, 111, 129, 147, 169, 178, 195, 197, 198, 206, 208, 213, 224, 234, 236),
    "none": (35, 36, 39, 40, 42, 47, 51, 60, 61, 62, 67, 70, 74, 76, 83, 85, 88, 95, 97, 103,
             107, 112, 113, 115, 118, 121, 125, 126, 131, 133, 135, 139, 141, 145, 148, 157,
             160, 163, 165, 167, 177, 179, 181, 182, 188, 190, 192, 194, 199, 201, 203, 205,
             209, 211, 212, 214, 218, 220, 222, 225, 229, 231, 242, 243, 244),
}
_PRIMARY = {index: label for label, indices in _EXPECTED.items() for index in indices}

ACCEPTED_ALSO: Dict[int, Tuple[Tuple[str, ...], ...]] = {
    3: (("update",),),                    # the birthday next to the age, in the same memory
    17: (("update",), ("new", "update")), # "Argo e' un labrador": detail on the dog of msg 16
    18: (("update",),),                   # detail on the dog, in the same memory
    20: (("update",),),                   # "corro la mattina presto": detail on running
    35: (("new",),),                      # "sette pazienti" read as a fact about work
    41: (("update",),),                   # "in borsa" can be a new detail
    44: (("update",),),                   # the bike is already in the commute of msg 6
    69: (("new",),),                      # levothyroxine stays true, the dose is added
    73: (("new", "update"),),             # no coffee, and the reason: it made them jittery
    75: (("update",),),                   # the peanut allergy is already in memory
    81: (("new",), ("new", "redundant")), # piano confirmed, guitar new
    85: (("redundant",),),                # the question confirms the trip to Naxos
    86: (("new", "update"),),             # no running, and the knee that msg 206 deletes
    87: (("update",),),                   # if msg 3 is an update: the age goes, not the birthday
    93: (("update",),),                   # swimming instead of running
    96: (("update",),),                   # detail on the photography course
    101: (("update",), ("new", "update")), # date and witness about Giulia
    110: (("new",),),                     # Wednesday is added, Saturday stays true
    111: (("update",), ("delete", "new")),  # the name goes, the sister in Turin stays true
    114: (("new",),),                     # the part-time is added, the clinic stays true
    117: (("new",),),                     # the oil is added, the allergy stays true
    128: (("update",),),                  # detail on the dog
    131: (("new",),),                     # the question implies a hike
    132: (("new",),),                     # the essays are added, the historical novels stay
    138: (("update",),),                  # the budget is a detail of the shopping of msg 137
    139: (("redundant",),),               # the question confirms the Thursday shopping
    140: (("new",),),                     # the snack is added, the afternoon stays true
    144: (("update",),),                  # detail on the dog
    151: (("update",),),                  # detail on the dog
    155: (("new",),),                     # the recipe changes, the favourite dish stays
    158: (("update",),),                  # the guitar is already in memory
    159: (("new",),),                     # the wish stays true, the booking is added
    160: (("redundant",),),               # the question confirms the departure in August
    161: (("update",),),                  # "ma non carne" adds a detail to the diet
    162: (("new", "update"),),            # 1800 calories, and the reason: losing weight
    168: (("new",),),                     # Spanish stays true, the level is added
    171: (("update",),),                  # detail on the crime novels
    176: (("update",), ("new", "update")),  # "Milo e' un soriano": detail on the cat, msg 173
    178: (("update",), ("delete", "update")),  # the gluten-free cooking can stay
    187: (("new", "update"),),            # if msg 93 put swimming in place of running
    189: (("update",),),                  # if the sport is in a single memory
    191: (("new",),),                     # tomatoes and basil stay true
    195: (("update",),),                  # if the reason of msg 162 is in the same memory
    200: (("update",), ("new",)),         # the early morning lost among msgs 86, 93 and 187
    203: (("new",),),                     # the question implies a trip to the lake
    206: (("update",),),                  # the knee can sit in the memory about running
    208: (("none",),),                    # relatives already deleted at msgs 111 and 178
    231: (("new",),),                     # the question implies a dinner out
    233: (("update",),),                  # detail on the fish-based diet
}

# Where several operations of the same type are right, and not fragmentation.
MULTIPLE_EXPECTED = (
    9,    # the asthma and the inhaler
    23,   # sister: the name and the city
    24,   # coeliac mother, and gluten-free at home
    31,   # the tomatoes and the basil
    79,   # the fear and the preferred means of transport
    86,   # running in two memories, if msg 20 became a new
    101,   # the date and the witness
    134,  # Saturday and Wednesday yoga, if msg 110 became a new
    142,  # the clinic and the part-time, if msg 114 became a new
    147,  # the same two memories about work, to be deleted
    169,  # the bedtime and the alarm time
    175,  # historical novels and essays, if msg 132 became a new
    178,  # coeliac disease and gluten-free cooking, if msg 24 made two of them
    180,  # the date and the witness, if msg 101 made two of them
    187,  # running taken up again and the knee getting better
    204,  # the fear and the preferred means of transport, if msg 79 made two of them
    208,  # all the relatives
    224,  # the vegetable garden and the chili peppers, if msg 191 became a new
    228,  # the wish and the booking, if msg 159 became a new
)


def display(name: str) -> str:
    """The name of a class as it appears in the report."""
    return _DISPLAY.get(name, name)


def expected(index: int) -> Tuple[str, Tuple[Tuple[str, ...], ...]]:
    """Main label and accepted type combinations of a message."""
    primary = _PRIMARY.get(index, "new")
    also = tuple(tuple(sorted(combination)) for combination in ACCEPTED_ALSO.get(index, ()))
    return primary, ((primary,),) + also


def consolidated_message(turn: int, maximum_historical_messages: int) -> int:
    """The message consolidated by this turn's insert.

    Two messages per turn and a window of maximum_historical_messages: message
    m leaves the window, and is consolidated, at turn m + lag. It is the same
    lag as fama.memory_snapshot_turn, read from the other side.
    """
    return turn - max(1, maximum_historical_messages // 2)


def classify(op_types: Iterable[str]) -> Tuple[str, Tuple[str, ...], int]:
    """Label, distinct types and number of the operations produced by a message."""
    ops = [_LOG_TO_CLASS.get(op, op) for op in op_types if op not in _NOT_CLASSIFIED]
    types = tuple(sorted(set(ops)))
    if not types:
        return "none", (), 0
    return (types[0] if len(types) == 1 else MIXED), types, len(ops)


def row(index: int, op_types: Iterable[str]) -> Dict:
    """The comparison of a message, in the form that is saved and summed."""
    primary, accepted = expected(index)
    predicted, types, count = classify(op_types)
    return {"index": index, "expected": primary,
            "accepted": [list(combination) for combination in accepted],
            "predicted": predicted, "types": list(types), "count": count}


def is_strict(record: Dict) -> bool:
    return record["predicted"] == record["expected"]


def is_lenient(record: Dict) -> bool:
    return (list(record["types"]) or ["none"]) in record["accepted"]


def matrix(records: Iterable[Dict]) -> Dict[str, Counter]:
    """Rows: expected operation. Columns: executed operation."""
    counts = {label: Counter() for label in CLASSES}
    for record in records:
        counts[record["expected"]][record["predicted"]] += 1
    return counts


def format_report(records: Sequence[Dict]) -> List[str]:
    """The matrix with recall and precision per class, and the two accuracies."""
    records = list(records)
    counts = matrix(records)
    lines = ["  " + f"{'expected / actual':<18}"
             + "".join(f"{display(column):>10}" for column in COLUMNS)
             + f"{'tot':>7}{'recall':>10}"]
    for label in CLASSES:
        total = sum(counts[label].values())
        recall = f"{counts[label][label] / total:.0%}" if total else "-"
        lines.append("  " + f"{display(label):<18}"
                     + "".join(f"{counts[label][column]:>10}" for column in COLUMNS)
                     + f"{total:>7}{recall:>10}")

    precision = []
    for column in COLUMNS:
        predicted = sum(counts[label][column] for label in CLASSES)
        right = counts[column][column] if column in counts else 0
        precision.append(f"{right / predicted:.0%}" if predicted and column != MIXED else "-")
    lines.append("  " + f"{'precision':<18}" + "".join(f"{value:>10}" for value in precision))

    total = len(records)
    strict = sum(is_strict(record) for record in records)
    lenient = sum(is_lenient(record) for record in records)
    repeated = [record for record in records if record["count"] > len(record["types"])]
    fragmented = [record for record in repeated if record["index"] not in MULTIPLE_EXPECTED]
    where = sorted({record["index"] for record in fragmented})
    lines.append("")
    if total:
        lines.append(f"  Strict accuracy {strict}/{total} ({strict / total:.1%}), "
                     f"lenient {lenient}/{total} ({lenient / total:.1%})")
    lines.append(f"  Fragmentation: {len(fragmented)} messages with several operations of the "
                 f"same type" + (f" (msg {', '.join(map(str, where))})" if where else "")
                 + f", plus {len(repeated) - len(fragmented)} where they are expected")
    return lines


def save_run(path: str, run: Dict) -> None:
    """Append a run to the file, one JSON line per run."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, ensure_ascii=False) + "\n")


def load_runs(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str]) -> int:
    """Sum the runs of the last saved commit and print the matrix."""
    path = argv[1] if len(argv) > 1 else DEFAULT_RESULTS
    runs = load_runs(path)
    if not runs:
        print(f"No run saved in {path}")
        return 1

    commit = runs[-1].get("commit") or ""
    same = [run for run in runs if (run.get("commit") or "") == commit]
    records = [record for run in same for record in run.get("messages", [])]
    print(f"{len(same)} runs of commit {commit or '(unknown)'} out of {len(runs)} saved, "
          f"{len(records)} consolidated messages\n")
    for line in format_report(records):
        print(line)

    wrong = Counter(record["index"] for record in records if not is_strict(record))
    if wrong:
        print("\n  Messages wrong in at least one run (times / runs they were in):")
    for index, times in sorted(wrong.items(), key=lambda item: (-item[1], item[0])):
        mine = [record for record in records if record["index"] == index]
        seen = Counter(display(record["predicted"]) for record in mine if not is_strict(record))
        detail = ", ".join(f"{name} x{count}" for name, count in seen.most_common())
        print(f"    msg {index:>3}  expected {display(mine[0]['expected']):<9} "
              f"{times}/{len(mine)}  actual: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
