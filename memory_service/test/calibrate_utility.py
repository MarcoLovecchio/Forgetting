"""Calibrazione di UTILITY_TABLE e NEED_GAIN_TABLE dalle tracce del test lungo.

Le due tabelle, in eviction.py, traducono i voti del classificatore nel termine
utility dello score. Finche' non sono calibrate valgono voto / 5 e
(need / 3) * (gain / 3): prodotti di codici ordinali, con i pari merito che la
letteratura sulle matrici di rischio rimprovera. Questo script le ricava da
quello che le memorie hanno fatto davvero.

L'oracolo: FAMA-memoria senza la memoria
----------------------------------------
Con MEMORY_LONGRUN_CALIBRATION il test lungo salva, per ogni domanda FAMA, le
memorie attive alla sua fotografia, con i voti. Per ogni coppia (memoria,
domanda) lo script toglie la memoria e ricalcola FAMA-memoria:

    valore = FAMA con la memoria - FAMA senza

Una memoria e' servita alla domanda se il valore e' positivo, e il valore e' il
suo gain. E' need x gain di Mattar & Daw misurato a posteriori: quanto e'
probabile che serva, per quanto vale quando serve. Nessuna chiamata al modello:
FAMA-memoria e' deterministica.

Ogni coppia e' un'occasione di servire: una memoria attiva a cinque domande
conta cinque volte, una scritta dopo l'ultima domanda non conta. Le memorie con
safety vera restano fuori, come dall'eviction, e cosi' quelle senza il voto da
calibrare.

Un valore negativo vuol dire che la memoria fa danno: di solito un valore che
consolidation doveva superare o cancellare. E' un errore di classificazione,
non una misura di utilita': conta come non servita, e il resoconto dice quante
sono.

Dalle coppie alle tabelle
-------------------------
Per ogni livello di un voto:

    need   la frazione di coppie servite
    gain   la media dei valori delle coppie servite

entrambi avvicinati al valore di tutti i livelli insieme con il peso di
PRIOR_WEIGHT coppie, cosi' un livello con pochi dati non inventa niente, e poi
resi non decrescenti con la regressione isotonica pesata sulle coppie.

    UTILITY_TABLE      need(u) * gain(u) per i cinque livelli di utility
    NEED_GAIN_TABLE    need(i) * gain(j): il prodotto di Mattar & Daw, con need
                       stimato sui livelli di need e gain su quelli di gain

Tutte e due divise per la cella piu' alta, cosi' stanno in (0, 1] come le
provvisorie e si incollano in eviction.py cosi' come sono stampate.

Accanto al prodotto c'e' la tabella libera: per ogni cella la media dei valori,
con 0 per le coppie non servite, avvicinata alla cella del prodotto con il peso
di PRIOR_WEIGHT coppie e resa non decrescente lungo righe e colonne. Con almeno
due run il resoconto confronta le due su ogni run lasciata fuori: se la libera
predice meglio, l'ipotesi moltiplicativa non regge sui nostri dati.

Limiti
------
- Le domande FAMA sono 56, fra richiami, compiti, update e delete: molte
  coppie, poche servite. Servono piu' run, e il resoconto dice quante coppie
  regge ogni livello.
- Togliere una memoria misura un valore marginale: se due memorie dicono la
  stessa cosa, nessuna delle due risulta servita.
- La calibrazione vale per la conversazione del test lungo e le sue domande.

    python memory_service/test/calibrate_utility.py [traccia.jsonl]

usa le run dell'ultimo commit salvato, come operation_matrix.py.
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
    """Una memoria attiva alla fotografia di una domanda, e quanto le e' servita."""

    run: int
    question: int
    memory: dict
    value: float

    @property
    def served(self) -> bool:
        """Toglierla abbassa FAMA-memoria."""
        return self.value > 0

    @property
    def realized(self) -> float:
        """Il valore come lo conta la calibrazione: 0 se non e' servita."""
        return max(0.0, self.value)


@dataclass(frozen=True)
class Levels:
    """need e gain di ogni livello di un voto, gia' monotoni, e le coppie dietro."""

    pairs: Tuple[int, ...]
    served: Tuple[int, ...]
    need: Tuple[float, ...]
    gain: Tuple[float, ...]


def memory_text(memories: Sequence[dict]) -> str:
    """Le memorie come le legge FAMA-memoria nel test lungo: una per riga."""
    return "\n".join(memory["content"] for memory in memories)


def pairs_of(run: dict, run_index: int) -> List[Pair]:
    """Le coppie (memoria, domanda) di una run, togliendo una memoria alla volta."""
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
    """Le fotografie il cui FAMA non torna: i criteri sono cambiati dopo la run."""
    return sum(1 for snapshot in run["questions"]
               if abs(fama.score(fama.QUESTIONS[snapshot["index"]],
                                 memory_text(snapshot["memories"])).fama
                      - snapshot["fama"]) > 1e-9)


def isotonic(values: Sequence[float], weights: Sequence[float]) -> List[float]:
    """La successione non decrescente piu' vicina, nei minimi quadrati pesati (PAV)."""
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
    """Una media avvicinata a `prior` con il peso di PRIOR_WEIGHT osservazioni."""
    return (total + PRIOR_WEIGHT * prior) / (count + PRIOR_WEIGHT)


def by_level(pairs: Sequence[Pair], rating: str) -> Levels:
    """need e gain per livello di `rating`, sulle coppie che hanno quel voto."""
    rated = [pair for pair in pairs if pair.memory.get(rating) is not None]
    gains = [pair.value for pair in rated if pair.served]
    if not gains:
        raise ValueError(f"nessuna coppia servita con il voto {rating}: servono altre run")
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
    """La tabella divisa per la sua cella piu' alta, in (0, 1]."""
    if isinstance(table[0], tuple):
        top = max(max(row) for row in table)
        return tuple(tuple(value / top for value in row) for row in table)
    top = max(table)
    return tuple(value / top for value in table)


def utility_table(pairs: Sequence[Pair]) -> Tuple[float, ...]:
    """UTILITY_TABLE: need * gain per livello di utility, normalizzata."""
    levels = by_level(pairs, "utility")
    return normalized(tuple(n * g for n, g in zip(levels.need, levels.gain)))


def product_table(pairs: Sequence[Pair]) -> Tuple[Tuple[float, ...], ...]:
    """need(i) * gain(j), non normalizzata: e' anche una predizione del valore."""
    need, gain = by_level(pairs, "need").need, by_level(pairs, "gain").gain
    return tuple(tuple(n * g for g in gain) for n in need)


def _cell(pair: Pair) -> Optional[Tuple[int, int]]:
    need, gain = pair.memory.get("need"), pair.memory.get("gain")
    return None if need is None or gain is None else (need - 1, gain - 1)


def monotone_table(table, weights, rounds: int = 1000):
    """Righe e colonne non decrescenti, alternando la regressione isotonica sulle due."""
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
    # Le proiezioni si fermano a qualche 1e-13 dall'ordine: il massimo cumulativo,
    # prima sulle righe e poi sulle colonne, lo rende esatto.
    table = [list(accumulate(row, max)) for row in table]
    columns = [list(accumulate(column, max)) for column in zip(*table)]
    return tuple(tuple(row) for row in zip(*columns))


def free_table(pairs: Sequence[Pair]) -> Tuple[Tuple[float, ...], ...]:
    """La media di ogni cella, avvicinata al prodotto e resa monotona, non normalizzata."""
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
    """Errore quadratico medio sulle coppie di ogni run, con la tabella delle altre.

    None se nessuna run lasciata fuori si puo' predire: con una run sola le altre
    non ci sono, e la tabella non si stima.
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
    """La tabella come si scrive in eviction.py."""
    if not isinstance(table[0], tuple):
        return [f"{name} = ({', '.join(f'{value:.4f}' for value in table)})"]
    lines = [f"{name} = (", "    # gain 1  gain 2  gain 3"]
    for level, row in enumerate(table, 1):
        lines.append(f"    ({', '.join(f'{value:.4f}' for value in row)}),  # need {level}")
    return lines + [")"]


def level_lines(levels: Levels, rating: str) -> List[str]:
    """Coppie, servite, need e gain per livello."""
    lines = [f"  {rating:<8}{'coppie':>8}{'servite':>9}{'need':>8}{'gain':>8}"]
    for level in range(LEVELS[rating]):
        lines.append(f"  {level + 1:<8}{levels.pairs[level]:>8}{levels.served[level]:>9}"
                     f"{levels.need[level]:>8.3f}{levels.gain[level]:>8.3f}")
    return lines


def report(mode: str, runs: Sequence[dict]) -> List[str]:
    """Il resoconto di una modalita' di consolidation_utility, con la sua tabella."""
    pairs = [pair for index, run in enumerate(runs) for pair in pairs_of(run, index)]
    harmful = sum(1 for pair in pairs if pair.value < 0)
    lines = [f"{mode}: {len(runs)} run, {len(pairs)} coppie, "
             f"{sum(pair.served for pair in pairs)} servite, {harmful} dannose"]
    changed = sum(changed_snapshots(run) for run in runs)
    if changed:
        lines.append(f"  ATTENZIONE: {changed} fotografie hanno un FAMA diverso da quello "
                     f"registrato: i criteri sono cambiati dopo la run")
    try:
        if mode == "single":
            lines += level_lines(by_level(pairs, "utility"), "utility")
            lines += [""] + literal("UTILITY_TABLE", utility_table(pairs))
        else:
            lines += level_lines(by_level(pairs, "need"), "need")
            lines += level_lines(by_level(pairs, "gain"), "gain")
            lines += [""] + literal("NEED_GAIN_TABLE", normalized(product_table(pairs)))
            lines += ["", "  Tabella libera, per confronto:"]
            lines += ["  " + line for line in literal("free", normalized(free_table(pairs)))]
            product_error = held_out_error(pairs, product_table)
            free_error = held_out_error(pairs, free_table)
            if product_error is None or free_error is None:
                lines.append("  Con una run sola non c'e' una run da lasciare fuori.")
            else:
                lines.append(f"  Errore sulla run lasciata fuori: prodotto "
                             f"{product_error:.5f}, libera {free_error:.5f}")
    except ValueError as error:
        lines.append(f"  Niente tabella: {error}")
    return lines


def main(argv: Sequence[str]) -> int:
    """Calibra le tabelle sulle run dell'ultimo commit salvato e le stampa."""
    path = argv[1] if len(argv) > 1 else DEFAULT_TRACES
    runs = load_runs(path)
    if not runs:
        print(f"Nessuna traccia salvata in {path}")
        return 1

    commit = runs[-1].get("commit") or ""
    same = [run for run in runs if (run.get("commit") or "") == commit]
    print(f"{len(same)} run del commit {commit or '(sconosciuto)'} su {len(runs)} salvate")
    for mode in ("single", "need_gain"):
        mine = [run for run in same if run.get("consolidation_utility") == mode]
        print()
        if not mine:
            print(f"{mode}: nessuna run")
            continue
        for line in report(mode, mine):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
