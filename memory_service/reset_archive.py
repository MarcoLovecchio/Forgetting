#!/usr/bin/env python3
"""Delete the vector archive, to start again from an empty memory.

Needed between one test on the real stack and the next: the archive is on disk
and outlives the process, so without cleaning it the next test starts with the
memories of the previous one and the classifications change meaning.

The core memory instead needs no reset: it lives in RAM and dies with the
process.

The path is not written in here, it is asked of the same configuration the
service uses - so it always deletes the right archive even when
MEMORY_CHROMA_PATH points elsewhere.

    python3 memory_service/reset_archive.py          # asks for confirmation
    python3 memory_service/reset_archive.py -y       # without asking
    python3 memory_service/reset_archive.py --show   # only looks
"""

import argparse
import os
import shutil
import sys

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from memory_service.config import MemoryConfig  # noqa: E402

# A Chroma archive always contains at least one of these.
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
    """Is the folder a Chroma archive, or at least empty?

    The check exists because MEMORY_CHROMA_PATH is an environment variable:
    a wrong value would turn this script into an rm -rf on any folder.
    """
    entries = os.listdir(path)
    if not entries:
        return True
    return any(marker in entries for marker in CHROMA_MARKERS)


def refuse_dangerous_paths(path):
    """Paths that cannot be an archive, whatever the config says."""
    forbidden = {
        os.path.abspath(os.sep),
        os.path.abspath(os.path.expanduser("~")),
        os.path.abspath(os.getcwd()),
        os.path.abspath(os.path.dirname(PACKAGE_ROOT)),
        PACKAGE_ROOT,
    }
    if path in forbidden:
        raise SystemExit(f"Refusing to delete {path}: it is not an archive, it is a live folder.")


def main():
    parser = argparse.ArgumentParser(description="Delete the memory service's vector archive.")
    parser.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument("--show", action="store_true", help="show the path without deleting")
    arguments = parser.parse_args()

    config = MemoryConfig.from_environment()
    path = config.chroma_path

    print(f"Archive     : {path}")
    # The deletion takes away the whole folder, so ALL the collections:
    # the production one and the long test's one, which live there together.
    print(f"Collection  : {config.collection_name} (and every other one in the same folder)")

    if not os.path.exists(path):
        print("State       : does not exist, nothing to delete.")
        return 0

    if not os.path.isdir(path):
        raise SystemExit(f"{path} is not a folder.")

    refuse_dangerous_paths(path)

    entries = sorted(os.listdir(path))
    print(f"State       : {len(entries)} entries, {directory_size(path) / 1024:.0f} KiB")
    print(f"Contents    : {', '.join(entries[:6])}{' ...' if len(entries) > 6 else ''}")

    if not looks_like_an_archive(path):
        raise SystemExit(
            "This folder does not look like a Chroma archive: none of the expected files "
            f"({', '.join(CHROMA_MARKERS)}) and it is not empty. Check MEMORY_CHROMA_PATH.")

    if arguments.show:
        return 0

    if not arguments.yes:
        if input("\nDelete? [y/N] ").strip().lower() not in ("s", "si", "y", "yes"):
            print("Cancelled.")
            return 1

    shutil.rmtree(path)
    print("\nDeleted. The next test will start with an empty archive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
