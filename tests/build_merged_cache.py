"""Merge the divisors + gcd + lcm family caches into ONE multi-family cache
(tests/.merged_cache) so a single gateway can serve all three — the scenario
where per-family thresholds + the cross-family separation guard matter.

Vectors are copied verbatim (scores match eval/threshold_matrix.py exactly).
Function ids are prefixed per family to avoid collisions; entries' root ids
are remapped to match. No compiled module (runtime dispatch path).
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAMILIES = [
    ("divisors", "tests/.p2_live_cache"),
    ("gcd", "tests/.gcd_cache"),
    ("lcm", "tests/.lcm_cache"),
]
OUT = "tests/.merged_cache"


def main() -> None:
    out = Path(OUT)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    fconn = sqlite3.connect(out / "functions.db")
    fconn.execute("CREATE TABLE functions (id TEXT PRIMARY KEY, description TEXT NOT NULL, "
                  "source TEXT, meta TEXT, dim INTEGER)")

    econn = sqlite3.connect(out / "entries.db")
    econn.execute("CREATE TABLE entries (rowid INTEGER PRIMARY KEY, description TEXT, "
                  "root TEXT, params TEXT, vec BLOB, family TEXT)")

    id_map: dict[str, str] = {}
    for fam, d in FAMILIES:
        fsrc = sqlite3.connect(Path(d) / "functions.db")
        for fid, desc, source, meta, dim in fsrc.execute(
                "SELECT id, description, source, meta, dim FROM functions"):
            new_id = f"{fam}_{fid}"
            id_map[f"{d}:{fid}"] = new_id
            fconn.execute("INSERT INTO functions VALUES (?,?,?,?,?)",
                          (new_id, desc, source, meta, dim))
        esrc = sqlite3.connect(Path(d) / "entries.db")
        cols = {r[1] for r in esrc.execute("PRAGMA table_info(entries)")}
        sel_vec = "vec" if "vec" in cols else "NULL"
        sel_fam = "family" if "family" in cols else f"'{fam}'"
        for desc, root, params, vec, fam2 in esrc.execute(
                f"SELECT description, root, params, {sel_vec}, {sel_fam} FROM entries"):
            new_root = id_map[f"{d}:{root}"]
            econn.execute("INSERT INTO entries (description, root, params, vec, family) "
                          "VALUES (?,?,?,?,?)", (desc, new_root, params, vec, fam2))
        fsrc.close()
        esrc.close()
    fconn.commit()
    econn.commit()
    fconn.close()
    econn.close()

    n_f = sum(1 for _, d in FAMILIES
              for _ in sqlite3.connect(Path(d) / "functions.db").execute(
                  "SELECT 1 FROM functions"))
    n_e = sum(1 for _, d in FAMILIES
              for _ in sqlite3.connect(Path(d) / "entries.db").execute(
                  "SELECT 1 FROM entries"))
    print(f"merged {n_f} functions, {n_e} entries -> {OUT}")


if __name__ == "__main__":
    main()
