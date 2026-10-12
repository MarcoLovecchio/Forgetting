"""Matrice di confusione delle operazioni di consolidamento.

Per ogni messaggio della conversazione del test lungo: l'operazione che dovrebbe
provocare, dal commento accanto in CONVERSATION, contro quella che il
consolidamento ha fatto davvero. FAMA guarda lo stato della memoria su
56 domande; questa guarda ogni singola decisione.

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
# le rimozioni dell'eviction. Il log scrive "create" dove lo strumento dice "new".
_NOT_CLASSIFIED = ("archive", "evict", "prune")
_LOG_TO_CLASS = {"create": "new"}

# Dai commenti di CONVERSATION: contradict conta come update, domande e
# chiacchiere come nessuna operazione, tutto il resto - presentazione compresa -
# come new. test_operation_matrix controlla che coincidano ancora.
_EXPECTED = {
    "update": (38, 47, 51, 54, 57, 62, 66, 69, 70, 77, 82, 89, 96, 99, 103, 107, 110, 112,
               116, 117, 124, 126, 132, 134, 138, 141, 146, 147, 150, 153, 156, 158, 160,
               162, 165, 170, 175, 176, 180, 182, 185, 193, 199, 204, 205, 216, 229),
    "redundant": (37, 41, 45, 52, 63, 76, 80, 87, 98, 152, 173, 174, 189, 191, 207, 209,
                  211, 214, 225, 226),
    "delete": (64, 83, 104, 121, 139, 159, 168, 184, 186, 187, 195, 197, 202, 212, 222, 224),
    "none": (35, 36, 39, 40, 42, 46, 50, 58, 59, 60, 65, 67, 71, 73, 79, 81, 84, 90, 92, 97,
             101, 105, 106, 108, 111, 114, 118, 119, 123, 125, 127, 131, 133, 137, 140, 148,
             151, 154, 155, 157, 167, 169, 171, 172, 177, 179, 181, 183, 188, 190, 192, 194,
             198, 200, 201, 203, 206, 208, 210, 213, 217, 219, 230, 231, 232),
}
_PRIMARY = {index: label for label, indices in _EXPECTED.items() for index in indices}

ACCEPTED_ALSO: Dict[int, Tuple[Tuple[str, ...], ...]] = {
    3: (("update",),),                    # il compleanno accanto all'eta', nella stessa memoria
    17: (("update",), ("new", "update")), # "Argo e' un labrador": dettaglio sul cane del msg 16
    18: (("update",),),                   # dettaglio sul cane, nella stessa memoria
    20: (("update",),),                   # "corro la mattina presto": dettaglio sulla corsa
    35: (("new",),),                      # "sette pazienti" letto come fatto sul lavoro
    41: (("update",),),                   # "in borsa" puo' essere un dettaglio nuovo
    43: (("update",),),                   # la bici e' gia' negli spostamenti del msg 6
    66: (("new",),),                      # la levotiroxina resta vera, la dose si aggiunge
    70: (("new", "update"),),             # niente caffe', e il perche': mi agitava
    72: (("update",),),                   # l'allergia alle arachidi e' gia' in memoria
    77: (("new",), ("new", "redundant")), # pianoforte confermato, chitarra nuova
    81: (("redundant",),),                # la domanda conferma il viaggio a Naxos
    82: (("new", "update"),),             # niente corsa, e il ginocchio che cancella il 195
    83: (("update",),),                   # se il msg 3 e' un update: via l'eta', non il compleanno
    88: (("update",),),                   # il nuoto al posto della corsa
    91: (("update",),),                   # dettaglio sul corso di fotografia
    95: (("update",), ("new", "update")), # data e testimone su Giulia
    103: (("new",),),                     # il mercoledi' si aggiunge, il sabato resta vero
    104: (("update",), ("delete", "new")),  # via il nome, la sorella a Torino resta vera
    107: (("new",),),                     # il part-time si aggiunge, la clinica resta vera
    110: (("new",),),                     # l'olio si aggiunge, l'allergia resta vera
    120: (("update",),),                  # dettaglio sul cane
    123: (("new",),),                     # la domanda lascia intendere un'escursione
    124: (("new",),),                     # i saggi si aggiungono, i romanzi storici restano
    130: (("update",),),                  # il budget e' un dettaglio della spesa del msg 129
    131: (("redundant",),),               # la domanda conferma la spesa del giovedi'
    132: (("new",),),                     # la merenda si aggiunge, il pomeriggio resta vero
    136: (("update",),),                  # dettaglio sul cane
    142: (("update",),),                  # dettaglio sul cane
    146: (("new",),),                     # cambia la ricetta, il piatto preferito resta
    149: (("update",),),                  # la chitarra e' gia' in memoria
    150: (("new",),),                     # il desiderio resta vero, la prenotazione si aggiunge
    151: (("redundant",),),               # la domanda conferma la partenza ad agosto
    152: (("update",),),                  # "ma non carne" aggiunge un dettaglio alla dieta
    153: (("new", "update"),),            # 1800 calorie, e il perche': dimagrire
    158: (("new",),),                     # lo spagnolo resta vero, il livello si aggiunge
    161: (("update",),),                  # dettaglio sui gialli
    166: (("update",), ("new", "update")),  # "Milo e' un soriano": dettaglio sul gatto, msg 163
    168: (("update",), ("delete", "update")),  # la cucina senza glutine puo' restare
    176: (("new", "update"),),            # se il msg 88 ha messo il nuoto al posto della corsa
    178: (("update",),),                  # se lo sport e' in una memoria sola
    180: (("new",),),                     # pomodori e basilico restano veri
    184: (("update",),),                  # se il perche' del msg 153 e' nella stessa memoria
    189: (("update",), ("new",)),         # la mattina presto persa fra i msg 82, 88 e 176
    192: (("new",),),                     # la domanda lascia intendere una gita al lago
    195: (("update",),),                  # il ginocchio puo' stare nella memoria sulla corsa
    197: (("none",),),                    # familiari gia' cancellati ai msg 104 e 168
    219: (("new",),),                     # la domanda lascia intendere una cena fuori
    221: (("update",),),                  # dettaglio sulla dieta a base di pesce
}

# Dove piu' operazioni dello stesso tipo sono giuste, e non frammentazione.
MULTIPLE_EXPECTED = (
    9,    # l'asma e l'inalatore
    23,   # sorella: il nome e la citta'
    24,   # madre celiaca, e in casa senza glutine
    31,   # i pomodori e il basilico
    75,   # la paura e i mezzi preferiti
    82,   # la corsa in due memorie, se il msg 20 e' diventato un new
    95,   # la data e la testimone
    126,  # lo yoga del sabato e del mercoledi', se il 103 e' diventato un new
    134,  # la clinica e il part-time, se il 107 e' diventato un new
    139,  # le stesse due memorie sul lavoro, da cancellare
    159,  # l'ora in cui va a dormire e quella della sveglia
    165,  # romanzi storici e saggi, se il msg 124 e' diventato un new
    168,  # la celiachia e la cucina senza glutine, se il msg 24 ne ha fatte due
    170,  # la data e la testimone, se il msg 95 ne ha fatte due
    176,  # la corsa ripresa e il ginocchio che sta meglio
    193,  # la paura e i mezzi preferiti, se il msg 75 ne ha fatte due
    197,  # tutti i familiari
    212,  # l'orto e i peperoncini, se il msg 180 e' diventato un new
    216,  # il desiderio e la prenotazione, se il msg 150 e' diventato un new
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
