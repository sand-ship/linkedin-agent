"""
Build (or rebuild) the semantic search index over cached connections.

Usage:
    python embed.py

Run this after import_csv.py to enable semantic search in the bot.
First run downloads the ~80MB all-MiniLM-L6-v2 model.
"""

import sys
import db
import embeddings


def progress(text: str):
    print(f"\r{text}", end="", flush=True)


def main():
    count = db.get_connection_count()
    if count == 0:
        print("No connections in DB. Run import_csv.py first.")
        sys.exit(1)

    print(f"Building semantic index for {count} connections...")
    print("(First run downloads ~80MB model — give it a minute)\n")

    indexed = embeddings.build_index(progress_cb=progress)
    print(f"\nDone — {indexed} connections indexed.")


if __name__ == "__main__":
    main()
