"""
Import LinkedIn connections from a data export CSV into the local SQLite DB.

Usage:
    python import_csv.py                        # auto-discovers export in data/
    python import_csv.py path/to/export.zip     # explicit zip file
    python import_csv.py path/to/Connections.csv  # explicit CSV file

Auto-discovery checks (in order):
  1. *.zip files in data/ that contain Connections.csv
  2. Extracted directories in data/ named *.zip containing Connections.csv
"""

import csv
import glob
import io
import sys
import zipfile
from pathlib import Path

import db

DATA_DIR = Path(__file__).parent / "data"


def find_connections_source():
    """Return either a Path to a Connections.csv or a zipfile.Path inside a zip.

    Searches data/ for zip files first, then pre-extracted directories.
    Returns the most recently modified match.
    """
    # 1. Actual zip files in data/
    zips = sorted(DATA_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    for z in reversed(zips):
        if not z.is_file():
            continue
        try:
            zf = zipfile.ZipFile(z)
            if "Connections.csv" in zf.namelist():
                return zf, z
        except zipfile.BadZipFile:
            continue

    # 2. Extracted directories whose names end in .zip (macOS double-click behaviour)
    candidates = sorted(
        glob.glob(str(DATA_DIR / "*.zip" / "Connections.csv"))
    )
    if candidates:
        return None, Path(candidates[-1])

    raise FileNotFoundError(
        "Could not find Connections.csv in data/. "
        "Put your LinkedIn export zip in data/ and re-run."
    )


def profile_id_from_url(url: str) -> str:
    url = url.rstrip("/")
    return url.split("/")[-1] if url else ""


def _parse_csv_lines(lines: list) -> tuple[int, int]:
    imported = skipped = 0
    header_idx = next(
        (i for i, line in enumerate(lines) if "First Name" in line), None
    )
    if header_idx is None:
        raise ValueError("Could not find header row in Connections.csv")

    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        url = row.get("URL", "").strip()
        pid = profile_id_from_url(url)
        if not pid:
            skipped += 1
            continue

        first = row.get("First Name", "").strip()
        last = row.get("Last Name", "").strip()
        name = f"{first} {last}".strip() or "Unknown"
        position = row.get("Position", "").strip()
        company = row.get("Company", "").strip()

        db.upsert_connection({
            "profile_id": pid,
            "name": name,
            "headline": position,
            "current_title": position,
            "current_company": company,
            "location": "",
            "industry": "",
            "profile_url": url,
            "degree": 1,
        })
        imported += 1

    return imported, skipped


def import_connections(source) -> tuple[int, int]:
    """Import from a Path (csv or zip) or a (zipfile.ZipFile, zip_path) tuple."""
    db.init_db()

    if isinstance(source, tuple):
        zf, zip_path = source
        print(f"Importing from: {zip_path} (Connections.csv inside zip)")
        raw = zf.read("Connections.csv").decode("utf-8-sig")
        lines = io.StringIO(raw).readlines()
    else:
        path = Path(source)
        if path.suffix == ".zip" and path.is_file():
            with zipfile.ZipFile(path) as zf:
                if "Connections.csv" not in zf.namelist():
                    raise FileNotFoundError(f"No Connections.csv inside {path}")
                print(f"Importing from: {path} (Connections.csv inside zip)")
                raw = zf.read("Connections.csv").decode("utf-8-sig")
                lines = io.StringIO(raw).readlines()
        else:
            print(f"Importing from: {path}")
            with open(path, newline="", encoding="utf-8-sig") as f:
                lines = f.readlines()

    return _parse_csv_lines(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
    else:
        source = find_connections_source()

    imported, skipped = import_connections(source)
    total = db.get_connection_count()
    print(f"Done — {imported} imported, {skipped} skipped (no URL).")
    print(f"Total connections in DB: {total}")
