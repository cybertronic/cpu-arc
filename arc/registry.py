"""Function + entry-point registry with SQLite persistence (P1).

The paper's master cache is a vector-indexed registry of semantic functions and
entry points. Here we persist: function source (not closures — P0's in-memory
closures don't survive restarts), function descriptions, and entry points
(generic template + original input + param descriptions).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .sandbox import exec_function_source


class Registry:
    """In-memory registry + optional SQLite persistence."""

    def __init__(self, embedder, path: str | Path | None = None,
                 function_threshold: float = 0.55):
        self.embedder = embedder
        self.path = Path(path) if path else None
        self.function_threshold = function_threshold
        self.functions: list[dict] = []  # {id, description, vec, source, fn, meta}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            self._load()

    # -- registration --------------------------------------------------------
    def register(self, description: str, fn: Callable[[Any], dict],
                 source: str | None = None, meta: dict | None = None) -> str:
        fid = f"fn_{len(self.functions):04d}"
        with self._lock:
            self.functions.append({
                "id": fid,
                "description": description,
                "vec": self.embedder.embed(description),
                "source": source,
                "fn": fn,
                "meta": meta or {},
            })
            self._save_row(fid, description, source, meta or {})
        return fid

    def search(self, query: str, top_k: int = 1) -> list[tuple[dict, float]]:
        if not self.functions:
            return []
        qv = self.embedder.embed(query)
        vecs = np.stack([f["vec"] for f in self.functions])
        scores = vecs @ qv
        order = np.argsort(-scores)[:top_k]
        return [(self.functions[i], float(scores[i])) for i in order]

    # -- persistence -----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self.path is None:
            raise RuntimeError("registry has no persistence path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS functions (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                source TEXT,
                meta TEXT,
                dim INTEGER
            )""")
        return conn

    def _save_row(self, fid: str, description: str, source: str | None, meta: dict) -> None:
        if self.path is None:
            return
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO functions (id, description, source, meta, dim) VALUES (?,?,?,?,?)",
                (fid, description, source, json.dumps(meta), self.embedder.dim))
            conn.commit()
        finally:
            conn.close()

    def _load(self) -> None:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, description, source, meta FROM functions").fetchall()
        finally:
            conn.close()
        for fid, description, source, meta in rows:
            if not source:
                continue  # closures can't be restored; skip
            meta = json.loads(meta) if meta else {}
            fn = exec_function_source(source, primitives=_lazy_primitives(),
                                      timeout_s=meta.get("timeout_s", 30.0))
            self.functions.append({
                "id": fid, "description": description,
                "vec": self.embedder.embed(description),
                "source": source, "fn": fn, "meta": meta,
            })


def _lazy_primitives() -> dict:
    from .primitives import _primitives
    return _primitives()


class EntryStore:
    """Entry points: (description, root_function_id, param_descriptions) for IS1."""

    def __init__(self, embedder, path: str | Path | None = None,
                 entry_threshold: float = 0.55):
        self.embedder = embedder
        self.path = Path(path) if path else None
        self.entry_threshold = entry_threshold
        self.entries: list[dict] = []
        if self.path and self.path.exists():
            self._load()

    def register(self, description: str, root_function_id: str,
                 param_descriptions: list[str]) -> None:
        self.entries.append({
            "description": description,
            "vec": self.embedder.embed(description),
            "root_function_id": root_function_id,
            "params": param_descriptions,
        })
        if self.path:
            self._save()

    def search(self, query: str) -> tuple[dict | None, float]:
        if not self.entries:
            return None, 0.0
        qv = self.embedder.embed(query)
        vecs = np.stack([e["vec"] for e in self.entries])
        scores = vecs @ qv
        i = int(np.argmax(scores))
        return self.entries[i], float(scores[i])

    def _save(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS entries ("
                         "rowid INTEGER PRIMARY KEY, description TEXT, root TEXT, "
                         "params TEXT, vec BLOB)")
            # Migrate pre-vector schema in place (old table lacks the vec column;
            # CREATE TABLE IF NOT EXISTS above is a no-op on an existing table).
            cols = {r[1] for r in conn.execute("PRAGMA table_info(entries)")}
            if "vec" not in cols:
                conn.execute("ALTER TABLE entries RENAME TO entries_old")
                conn.execute("CREATE TABLE entries ("
                             "rowid INTEGER PRIMARY KEY, description TEXT, "
                             "root TEXT, params TEXT, vec BLOB)")
                conn.execute("INSERT INTO entries (description, root, params) "
                             "SELECT description, root, params FROM entries_old")
                conn.execute("DROP TABLE entries_old")
            conn.execute("DELETE FROM entries")
            for e in self.entries:
                vec = e.get("vec")
                conn.execute("INSERT INTO entries (description, root, params, vec) "
                             "VALUES (?,?,?,?)",
                             (e["description"], e["root_function_id"],
                              json.dumps(e["params"]),
                              vec.tobytes() if vec is not None else None))
            conn.commit()
        finally:
            conn.close()

    def _load(self) -> None:
        """Load entries; uses persisted vectors when present so cold start
        needs no embedder round-trips (old DBs without the vec column fall
        back to embedding and are upgraded in place)."""
        assert self.path is not None
        conn = sqlite3.connect(self.path)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(entries)")}
            sel = "vec" if "vec" in cols else "NULL"
            rows = conn.execute(
                f"SELECT description, root, params, {sel} FROM entries").fetchall()
        finally:
            conn.close()
        need_upgrade = "vec" not in cols
        for desc, root, params, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32) if blob else None
            if vec is None:
                vec = self.embedder.embed(desc)
            self.entries.append({
                "description": desc,
                "vec": vec,
                "root_function_id": root,
                "params": json.loads(params),
            })
        if need_upgrade and self.entries:
            self._save()  # persist vectors for next cold start
