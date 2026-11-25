"""Persistent invocation log for the ARC gateway.

Every /query request appends one row to a SQLite DB (WAL mode, thread-safe
with a single process lock; safe across processes too). Rows carry the full
serving decision — status (hit/miss/error), phase, matched entry + family +
score, params, reason, latency, token delta — so the log doubles as the
dataset for ongoing threshold monitoring and audit.

Schema
------
invocations(id INTEGER PK, ts REAL, query TEXT, status TEXT, phase TEXT,
            source TEXT, family TEXT, entry TEXT, score REAL, params TEXT,
            reason TEXT, serve_ms REAL, tokens_in INTEGER, tokens_out INTEGER)

CLI
---
  python3 -m arc.invocations summary --db <invocations.db> [--since-hours 24]
  python3 -m arc.invocations recent  --db <invocations.db> [--n 10]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS invocations (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    phase TEXT,
    source TEXT,
    family TEXT,
    entry TEXT,
    score REAL,
    params TEXT,
    reason TEXT,
    serve_ms REAL,
    tokens_in INTEGER,
    tokens_out INTEGER
)
"""


class InvocationLog:
    def __init__(self, path: str | Path, retention_days: float | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inv_ts ON invocations(ts)")
        self._conn.commit()
        if retention_days:
            self.prune(retention_days)

    # -- write --------------------------------------------------------------
    def record(self, *, ts: float | None = None, query: str, status: str,
               phase: str | None = None, source: str | None = None,
               family: str | None = None, entry: str | None = None,
               score: float | None = None, params: dict | None = None,
               reason: str | None = None, serve_ms: float | None = None,
               tokens_in: int | None = None,
               tokens_out: int | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO invocations (ts, query, status, phase, source, "
                "family, entry, score, params, reason, serve_ms, tokens_in, "
                "tokens_out) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts or time.time(), query, status, phase, source, family, entry,
                 score, json.dumps(params) if params is not None else None,
                 reason, serve_ms, tokens_in, tokens_out))
            self._conn.commit()

    # -- read ---------------------------------------------------------------
    def count(self, since_s: float | None = None) -> int:
        with self._lock:
            if since_s is None:
                n = self._conn.execute("SELECT COUNT(*) FROM invocations").fetchone()[0]
            else:
                n = self._conn.execute(
                    "SELECT COUNT(*) FROM invocations WHERE ts >= ?", (since_s,)).fetchone()[0]
        return int(n)

    def recent(self, n: int = 10) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT ts, query, status, phase, family, entry, score, params, "
                "reason, serve_ms, tokens_in, tokens_out FROM invocations "
                "ORDER BY id DESC LIMIT ?", (n,))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def summary(self, since_s: float | None = None) -> dict:
        tail = "WHERE ts >= ?" if since_s is not None else ""
        args = (since_s,) if since_s is not None else ()
        with self._lock:
            by_status = dict(self._conn.execute(
                f"SELECT status, COUNT(*) FROM invocations {tail} "
                "GROUP BY status", args).fetchall())
            lat = self._conn.execute(
                f"SELECT status, MIN(serve_ms), MAX(serve_ms), AVG(serve_ms) "
                f"FROM invocations {tail} GROUP BY status", args).fetchall()
            fam_where = (
                "WHERE status='hit' AND family IS NOT NULL"
                + (" AND ts >= ?" if since_s is not None else ""))
            fams = self._conn.execute(
                "SELECT family, COUNT(*), MIN(score), MAX(score) "
                f"FROM invocations {fam_where} GROUP BY family", args).fetchall()
        total = sum(by_status.values())
        return {
            "total": total,
            "by_status": by_status,
            "hit_rate": (by_status.get("hit", 0) / total) if total else None,
            "latency_ms": {s: {"min": mn, "max": mx, "avg": av}
                           for s, mn, mx, av in lat},
            "hit_score_range_by_family": {
                f: {"n": n, "min_score": mn, "max_score": mx} for f, n, mn, mx in fams},
        }

    # -- maintenance ---------------------------------------------------------
    def prune(self, retention_days: float) -> int:
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            cur = self._conn.execute("DELETE FROM invocations WHERE ts < ?",
                                     (cutoff,))
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="ARC invocation log inspector")
    ap.add_argument("cmd", choices=["summary", "recent"])
    ap.add_argument("--db", required=True, help="path to invocations.db")
    ap.add_argument("--since-hours", type=float, default=None)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    log = InvocationLog(args.db)
    since = time.time() - args.since_hours * 3600 if args.since_hours else None
    out = log.recent(args.n) if args.cmd == "recent" else log.summary(since)
    print(json.dumps(out, indent=2))
    log.close()


if __name__ == "__main__":
    main()
