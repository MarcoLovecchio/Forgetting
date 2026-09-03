#!/usr/bin/env python3
"""Cancella l'archivio vettoriale, per ripartire da una memoria vuota.

Serve fra un test sullo stack reale e l'altro: l'archivio e' su disco e
sopravvive al processo, quindi senza ripulirlo il test successivo parte con le
memorie del precedente e le classificazioni cambiano senso.

La core memory invece non serve azzerarla: vive in RAM e muore con il processo.

Il percorso non e' scritto qui dentro, viene chiesto alla stessa configurazione
che usa il servizio - cosi' cancella sempre l'archivio giusto anche se
MEMORY_CHROMA_PATH punta altrove.

    python3 memory_service/reset_archive.py          # chiede conferma
    python3 memory_service/reset_archive.py -y       # senza chiedere
    python3 memory_service/reset_archive.py --show   # guarda e basta
"""

import argparse
import os
import shutil
import sys

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from memory_service.config import MemoryConfig  # noqa: E402

# Un archivio Chroma contiene sempre almeno uno di questi.
CHROMA_MARKERS = ("chroma.sqlite3", "chroma-embeddings.parquet", "index")


def directory_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def looks_like_an_archive(path):
    """La cartella e' un archivio Chroma, o almeno e' vuota?

    Il controllo esiste perche' MEMORY_CHROMA_PATH e' una variabile d'ambiente:
    un valore sbagliato trasformerebbe questo script in un rm -rf su una
    cartella qualsiasi.
    """
    entries = os.listdir(path)
    if not entries:
        return True
    return any(marker in entries for marker in CHROMA_MARKERS)


def refuse_dangerous_paths(path):
    """Percorsi che non possono essere un archivio, qualunque cosa dica la config."""
    forbidden = {
        os.path.abspath(os.sep),
        os.path.abspath(os.path.expanduser("~")),
        os.path.abspath(os.getcwd()),
        os.path.abspath(os.path.dirname(PACKAGE_ROOT)),
        PACKAGE_ROOT,
    }
    if path in forbidden:
        raise SystemExit(f"Rifiuto di cancellare {path}: non e' un archivio, e' una cartella viva.")


def main():
    parser = argparse.ArgumentParser(description="Cancella l'archivio vettoriale del memory service.")
    parser.add_argument("-y", "--yes", action="store_true", help="non chiedere conferma")
    parser.add_argument("--show", action="store_true", help="mostra il percorso senza cancellare")
    arguments = parser.parse_args()

    config = MemoryConfig.from_environment()
    path = config.chroma_path

    print(f"Archivio    : {path}")
    print(f"Collezione  : {config.collection_name}")

    if not os.path.exists(path):
        print("Stato       : non esiste, niente da cancellare.")
        return 0

    if not os.path.isdir(path):
        raise SystemExit(f"{path} non e' una cartella.")

    refuse_dangerous_paths(path)

    entries = sorted(os.listdir(path))
    print(f"Stato       : {len(entries)} elementi, {directory_size(path) / 1024:.0f} KiB")
    print(f"Contenuto   : {', '.join(entries[:6])}{' ...' if len(entries) > 6 else ''}")

    if not looks_like_an_archive(path):
        raise SystemExit(
            "Questa cartella non sembra un archivio Chroma: nessuno dei file attesi "
            f"({', '.join(CHROMA_MARKERS)}) e non e' vuota. Controlla MEMORY_CHROMA_PATH.")

    if arguments.show:
        return 0

    if not arguments.yes:
        if input("\nCancello? [s/N] ").strip().lower() not in ("s", "si", "y", "yes"):
            print("Annullato.")
            return 1

    shutil.rmtree(path)
    print("\nCancellato. Il prossimo test partira' con l'archivio vuoto.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
