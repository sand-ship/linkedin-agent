import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "linkedin.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS connections (
                profile_id   TEXT PRIMARY KEY,
                name         TEXT,
                headline     TEXT,
                current_title    TEXT,
                current_company  TEXT,
                location     TEXT,
                industry     TEXT,
                profile_url  TEXT,
                degree       INTEGER DEFAULT 1,
                last_synced  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS enrichment (
                company_name    TEXT PRIMARY KEY,
                funding_amount  INTEGER,
                funding_round   TEXT,
                funding_date    TEXT,
                last_fetched    TEXT
            )
        """)
        conn.commit()


def upsert_connection(profile: dict):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO connections
            (profile_id, name, headline, current_title, current_company,
             location, industry, profile_url, degree, last_synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            profile.get("profile_id"),
            profile.get("name"),
            profile.get("headline"),
            profile.get("current_title"),
            profile.get("current_company"),
            profile.get("location"),
            profile.get("industry"),
            profile.get("profile_url"),
            profile.get("degree", 1),
        ))
        conn.commit()


def search_connections(
    title_keywords: list = None,
    location: str = None,
    company: str = None,
    keywords: list = None,
    limit: int = 20,
):
    """Search cached connections.

    - title_keywords: matched against headline + current_title
    - company: matched against current_company AND headline (since current_company is often empty)
    - keywords: broad match across all text fields
    - location: matched against location field
    """
    query = "SELECT * FROM connections WHERE 1=1"
    params = []

    if title_keywords:
        clauses = " OR ".join(
            ["(headline LIKE ? OR current_title LIKE ?)"] * len(title_keywords)
        )
        query += f" AND ({clauses})"
        for kw in title_keywords:
            params.extend([f"%{kw}%", f"%{kw}%"])

    if company:
        # Search both current_company and headline since current_company is often unpopulated
        query += " AND (current_company LIKE ? OR headline LIKE ?)"
        params.extend([f"%{company}%", f"%{company}%"])

    if keywords:
        for kw in keywords:
            query += " AND (headline LIKE ? OR current_title LIKE ? OR current_company LIKE ? OR name LIKE ?)"
            params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%", f"%{kw}%"])

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")

    query += " ORDER BY degree ASC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_connection_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]


def upsert_enrichment(company_name: str, funding_amount: int, funding_round: str, funding_date: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO enrichment
            (company_name, funding_amount, funding_round, funding_date, last_fetched)
            VALUES (?, ?, ?, ?, datetime('now'))
        """, (company_name, funding_amount, funding_round, funding_date))
        conn.commit()


def get_enrichment(company_name: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM enrichment WHERE company_name = ?", (company_name,)
        ).fetchone()
        return dict(row) if row else None
