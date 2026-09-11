"""Matrice di confusione delle operazioni di consolidamento.

Per ogni messaggio della conversazione del test lungo: l'operazione che dovrebbe
provocare, dal commento accanto in CONVERSATION, contro quella che il
consolidamento ha fatto davvero. FAMA guarda lo stato della memoria su
diciassette domande; questa guarda ogni singola decisione.

L'unita' e' il messaggio, non l'operazione. Un messaggio che produce due `new`
resta un new: quante memorie ne escono e' granularita', e si conta a parte. Dove
un messaggio porta due fatti ("mia sorella si chiama Chiara e vive a Milano") le
ripetizioni sono previste, altrove sono frammentazione. Un messaggio che produce
tipi diversi finisce in "misto".

Dove piu' letture sono difendibili ci sono combinazioni accettate, e le
accuratezze sono due: stretta sull'etichetta, tollerante sulle combinazioni. I
tipi prodotti devono coincidere con una combinazione, non basta esserne una
parte: al msg 52 un redundant da solo perde la chitarra. Un `new` al posto di un
`update` e' accettabile solo se la memoria vecchia resta vera (il mercoledi' si
aggiunge al sabato), mai se il valore cambia: 2000 e 2200 attivi insieme sono una
contraddizione in memoria.

Misura il tipo, non il bersaglio: un update giusto sulla memoria sbagliata qui
risulta corretto.

    python memory_service/test/operation_matrix.py [file.jsonl]

somma le run salvate dal test lungo, solo quelle dell'ultimo commit.
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

_DISPLAY = {"none": "nessuna", MIXED: "misto"}

# Voci del log che non sono una classificazione: lo spostamento del core split e
# la rimozione dei tombstone. Il log scrive "create" dove lo strumento dice "new".
_NOT_CLASSIFIED = ("archive", "evict")
_LOG_TO_CLASS = {"create": "new"}

# Dai commenti di CONVERSATION: contradict conta come update, domande e
# chiacchiere come nessuna operazione, tutto il resto - presentazione compresa -
# come new. test_operation_matrix controlla che coincidano ancora.
_EXPECTED = {
    "update": (29, 33, 35, 37, 39, 42, 47, 48, 52, 54, 59, 62, 64, 66, 69, 71, 72, 74,
               75, 79, 80, 83, 84, 86, 90, 91, 94, 97, 98, 100, 105, 106),
    "redundant": (28, 32, 36, 43, 51, 57, 63, 93, 104, 110, 114, 115, 116),
    "delete": (44, 55, 67, 77, 87, 96, 102, 108, 109, 111, 112),
    "none": (26, 27, 30, 40, 45, 49, 56, 60, 68, 76, 81, 88, 92, 95, 103, 107, 113, 117),
}
_PRIMARY = {index: label for label, indices in _EXPECTED.items() for index in indices}

ACCEPTED_ALSO: Dict[int, Tuple[Tuple[str, ...], ...]] = {
    13: (("update",), ("new", "update")),   # "Argo e' un labrador": dettaglio sul cane del msg 12
    15: (("update",),),                     # "corro la mattina presto": dettaglio sulla corsa
    26: (("new",),),                        # "sette pazienti" letto come fatto sul lavoro
    48: (("new", "update"),),               # niente caffe', e il perche': mi agitava
    52: (("new",), ("new", "redundant")),   # pianoforte confermato, chitarra nuova
    54: (("new", "update"),),               # niente corsa, e il ginocchio che cancella il 111
    58: (("update",),),                     # il nuoto al posto della corsa
    66: (("new",),),                        # il mercoledi' si aggiunge, il sabato resta vero
    69: (("new",),),                        # il part-time si aggiunge, la clinica resta vera
    71: (("new",),),                        # l'olio si aggiunge, l'allergia resta vera
    79: (("new",),),                        # i saggi si aggiungono, i romanzi storici restano
    83: (("new",),),                        # la merenda si aggiunge, il pomeriggio resta vero
    90: (("new",),),                        # cambia la ricetta, il piatto preferito resta
    94: (("new", "update"),),               # 1800 calorie, e il perche': dimagrire
    101: (("update",), ("new", "update")),  # "Milo e' un soriano": dettaglio sul gatto del msg 99
}

# Dove piu' operazioni dello stesso tipo sono giuste, e non frammentazione.
MULTIPLE_EXPECTED = (
    18,   # sorella: il nome e la citta'
    19,   # madre celiaca, e in casa senza glutine
    106,  # la corsa ripresa e il ginocchio che sta meglio
    112,  # tutti i familiari
)


def display(name: str) -> str:
    """Il nome di una classe come compare nel resoconto."""
    return _DISPLAY.get(name, name)


def expected(index: int) -> Tuple[str, Tuple[Tuple[str, ...], ...]]:
    """Etichetta principale e combinazioni di tipi accettate di un messaggio."""
    primary = _PRIMARY.get(index, "new")
    also = tuple(tuple(sorted(combination)) for combination in ACCEPTED_ALSO.get(index, ()))
    return primary, ((primary,),) + also


def consolidated_message(turn: int, maximum_historical_messages: int) -> int:
    """Il messaggio consolidato dall'insert di questo turno.

    Due messaggi per turno e una finestra di maximum_historical_messages: il
    messaggio m esce dalla finestra, e viene consolidato, al turno m + lag. E' lo
    stesso ritardo di fama.memory_snapshot_turn, letto dall'altra parte.
    """
    return turn - max(1, maximum_historical_messages // 2)


def classify(op_types: Iterable[str]) -> Tuple[str, Tuple[str, ...], int]:
    """Etichetta, tipi distinti e numero delle operazioni prodotte da un messaggio."""
    ops = [_LOG_TO_CLASS.get(op, op) for op in op_types if op not in _NOT_CLASSIFIED]
    types = tuple(sorted(set(ops)))
    if not types:
        return "none", (), 0
    return (types[0] if len(types) == 1 else MIXED), types, len(ops)


def row(index: int, op_types: Iterable[str]) -> Dict:
    """Il confronto di un messaggio, nella forma che si salva e si somma."""
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
    """Righe: operazione attesa. Colonne: operazione eseguita."""
    counts = {label: Counter() for label in CLASSES}
    for record in records:
        counts[record["expected"]][record["predicted"]] += 1
    return counts


def format_report(records: Sequence[Dict]) -> List[str]:
    """La matrice con richiamo e precisione per classe, e le due accuratezze."""
    records = list(records)
    counts = matrix(records)
    lines = ["  " + f"{'attesa / eseguita':<18}"
             + "".join(f"{display(column):>10}" for column in COLUMNS)
             + f"{'tot':>7}{'richiamo':>10}"]
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
    lines.append("  " + f"{'precisione':<18}" + "".join(f"{value:>10}" for value in precision))

    total = len(records)
    strict = sum(is_strict(record) for record in records)
    lenient = sum(is_lenient(record) for record in records)
    repeated = [record for record in records if record["count"] > len(record["types"])]
    fragmented = [record for record in repeated if record["index"] not in MULTIPLE_EXPECTED]
    where = sorted({record["index"] for record in fragmented})
    lines.append("")
    if total:
        lines.append(f"  Accuratezza stretta {strict}/{total} ({strict / total:.1%}), "
                     f"tollerante {lenient}/{total} ({lenient / total:.1%})")
    lines.append(f"  Frammentazione: {len(fragmented)} messaggi con piu' operazioni dello "
                 f"stesso tipo" + (f" (msg {', '.join(map(str, where))})" if where else "")
                 + f", piu' {len(repeated) - len(fragmented)} dove sono previste")
    return lines


def save_run(path: str, run: Dict) -> None:
    """Aggiunge una run al file, una riga JSON per run."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, ensure_ascii=False) + "\n")


def load_runs(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str]) -> int:
    """Somma le run dell'ultimo commit salvato e stampa la matrice."""
    path = argv[1] if len(argv) > 1 else DEFAULT_RESULTS
    runs = load_runs(path)
    if not runs:
        print(f"Nessuna run salvata in {path}")
        return 1

    commit = runs[-1].get("commit") or ""
    same = [run for run in runs if (run.get("commit") or "") == commit]
    records = [record for run in same for record in run.get("messages", [])]
    print(f"{len(same)} run del commit {commit or '(sconosciuto)'} su {len(runs)} salvate, "
          f"{len(records)} messaggi consolidati\n")
    for line in format_report(records):
        print(line)

    wrong = Counter(record["index"] for record in records if not is_strict(record))
    if wrong:
        print("\n  Messaggi sbagliati in almeno una run (volte / run in cui c'erano):")
    for index, times in sorted(wrong.items(), key=lambda item: (-item[1], item[0])):
        mine = [record for record in records if record["index"] == index]
        seen = Counter(display(record["predicted"]) for record in mine if not is_strict(record))
        detail = ", ".join(f"{name} x{count}" for name, count in seen.most_common())
        print(f"    msg {index:>3}  attesa {display(mine[0]['expected']):<9} "
              f"{times}/{len(mine)}  eseguita: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
