"""SQLite store for observation reports (stdlib only)."""
import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.observations.schema import ObservationIn

_DDL = """CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    crop TEXT NOT NULL,
    problem TEXT NOT NULL,
    observed_date TEXT NOT NULL,
    payload TEXT NOT NULL)"""


def _connect(path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(_DDL)
    return conn


def add(path, obs: ObservationIn) -> dict:
    rec = obs.model_dump(mode="json")
    rec["id"] = uuid.uuid4().hex
    rec["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    with closing(_connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO observations VALUES (?,?,?,?,?,?)",
            (rec["id"], rec["created_at_utc"], rec["crop"], rec["problem"],
             rec["observed_date"], json.dumps(rec)))
    return rec


def list_observations(path, crop=None, problem=None, limit=100) -> list[dict]:
    sql, args = "SELECT payload FROM observations WHERE 1=1", []
    if crop:
        sql += " AND crop=?"; args.append(crop)
    if problem:
        sql += " AND problem=?"; args.append(problem)
    sql += " ORDER BY observed_date DESC, created_at_utc DESC LIMIT ?"
    args.append(limit)
    with closing(_connect(path)) as conn:
        return [json.loads(r["payload"]) for r in conn.execute(sql, args)]
