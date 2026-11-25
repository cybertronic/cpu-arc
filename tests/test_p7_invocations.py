"""P7: persistent invocation log.

Offline (no servers needed):
  1-7  InvocationLog unit: record/count/recent/summary/prune/retention/CLI
Live (needs embed :8081; LLM optional, disabled here):
  8-12 ArcGateway integration: every request path logs exactly one row with
      the right status/phase/family/score; /health exposes the count;
      --no-invocation-log disables cleanly; token deltas record when an
      LM is present (mocked via MockLM).

Run: python3 tests/test_p7_invocations.py
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc.invocations import InvocationLog  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def test_offline() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="p7_inv_"))
    db = tmp / "invocations.db"
    log = InvocationLog(db)

    # 1. record + count
    log.record(query="q1", status="hit", phase="execute", family="divisors",
               entry="How many divisors...", score=0.98,
               params={"the input number": "196"}, serve_ms=5.2)
    log.record(query="q2", status="miss", phase="threshold", family="gcd",
               entry="Find the gcd...", score=0.40,
               reason="score 0.400 < 0.720 (family=gcd)", serve_ms=12.0)
    log.record(query="q3", status="error", phase="execute", family="lcm",
               entry="lcm...", score=0.90, reason="execution error: boom",
               serve_ms=5001.0, tokens_in=2100, tokens_out=320)
    check("count after 3 records", log.count() == 3, f"got {log.count()}")

    # 2. recent order + fields
    recs = log.recent(10)
    check("recent returns newest first",
          [r["query"] for r in recs] == ["q3", "q2", "q1"],
          str([r["query"] for r in recs]))
    check("recent row fields",
          recs[0]["status"] == "error" and recs[0]["reason"] == "execution error: boom"
          and recs[0]["tokens_in"] == 2100 and recs[0]["tokens_out"] == 320,
          str(recs[0]))

    # 3. summary aggregates
    s = log.summary()
    check("summary total/by_status",
          s["total"] == 3 and s["by_status"] == {"hit": 1, "miss": 1, "error": 1},
          str(s["by_status"]))
    check("summary hit_rate", abs((s["hit_rate"] or 0) - 1 / 3) < 1e-9,
          str(s["hit_rate"]))
    check("summary latency per status",
          s["latency_ms"]["hit"]["min"] == 5.2 and s["latency_ms"]["error"]["max"] == 5001.0,
          str(s["latency_ms"]))
    check("summary hit score range by family",
          s["hit_score_range_by_family"] == {
              "divisors": {"n": 1, "min_score": 0.98, "max_score": 0.98}},
          str(s["hit_score_range_by_family"]))

    # 4. since filter
    now = time.time()
    s2 = log.summary(since_s=now - 60)
    check("summary since filter counts all (fresh rows)", s2["total"] == 3,
          str(s2["total"]))
    s3 = log.summary(since_s=now + 60)
    check("summary since filter excludes old", s3["total"] == 0, str(s3["total"]))

    # 5. prune old rows
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO invocations (ts, query, status) VALUES (?,?,?)",
                  (time.time() - 90 * 86400, "ancient", "hit"))
    pruned = log.prune(retention_days=30)
    check("prune removes only old rows", pruned == 1 and log.count() == 3,
          f"pruned={pruned} count={log.count()}")

    # 6. constructor retention + persistence across reopen
    log.close()
    log2 = InvocationLog(db)  # reopen
    check("rows persist across reopen", log2.count() == 3, str(log2.count()))
    log2.close()
    # retention prunes pre-existing old rows at open time (like a startup job)
    tmpdb = tmp / "ret" / "invocations.db"
    seed = InvocationLog(tmpdb)
    seed.record(query="old", status="hit", ts=time.time() - 2 * 86400)
    seed.record(query="new", status="hit")
    seed.close()
    logr = InvocationLog(tmpdb, retention_days=1)
    check("constructor retention prunes on open", logr.count() == 1
          and logr.recent(1)[0]["query"] == "new",
          str(logr.count()))
    logr.close()

    # 7. CLI inspector
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "arc.invocations", "summary", "--db", str(db)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    out = json.loads(r.stdout)
    check("CLI summary works", out["total"] == 3 and out["by_status"]["miss"] == 1,
          r.stdout[:200] + r.stderr[:200])
    r2 = subprocess.run(
        [sys.executable, "-m", "arc.invocations", "recent", "--db", str(db), "--n", "2"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    rows = json.loads(r2.stdout)
    check("CLI recent works", len(rows) == 2, r2.stdout[:200] + r2.stderr[:200])
    return tmp


def test_live() -> None:
    from arc import Embedder
    from arc.gateway import ArcGateway, GatewayConfig

    tmp = Path(tempfile.mkdtemp(prefix="p7_gw_"))
    cache = tmp / "cache"
    shutil.copytree("tests/gateway_cache", cache)
    (cache / "invocations.db").unlink(missing_ok=True)
    (cache / "invocations.db-wal").unlink(missing_ok=True)
    (cache / "invocations.db-shm").unlink(missing_ok=True)

    # model=None -> no LLM fallback proxy (fast miss), no token deltas
    gw = ArcGateway(GatewayConfig(
        port=0, cache_dir=str(cache), chat_url="http://127.0.0.1:8080/v1",
        embed_url="http://127.0.0.1:8081", model=None,
        family_thresholds={"divisors": 0.725}, min_separation=0.05))

    # 8. hit logs one row with family/score/params
    r = gw.handle_query("How many positive whole-number divisors does 196 have?")
    check("live hit answers 9", r["source"] == "cache"
          and r.get("answer", {}).get("answer") == 9, str(r)[:200])
    rows = gw.invocation_log.recent(5)
    check("live hit logged", len(rows) == 1 and rows[0]["status"] == "hit"
          and rows[0]["family"] == "divisors" and rows[0]["score"] is not None
          and rows[0]["phase"] == "execute" and rows[0]["serve_ms"] is not None,
          str(rows[0]) if rows else "no rows")
    check("live hit params logged",
          rows[0]["params"] is not None
          and "196" in json.dumps(rows[0]["params"]), str(rows[0]["params"]))

    # 9. threshold miss logs the reason + family
    r = gw.handle_query("How many divisors does 7 have?")  # short style ~0.6-0.7
    if r["source"] == "miss" or "reason" in r:
        rows = gw.invocation_log.recent(5)
        top = rows[0]
        check("live miss logged with phase+reason",
              top["status"] in ("miss", "error") and top["phase"] in
              ("threshold", "separation", "extract", "execute")
              and top["reason"], str(top))
    else:
        # 7 may score >= 0.725 (small number, template-ish) -> still must be logged
        rows = gw.invocation_log.recent(5)
        check("live 7-query logged (hit or miss)",
              rows[0]["query"].endswith("7 have?"), str(rows[0]))

    # 10. count matches
    check("invocation count == 2", gw.invocation_log.count() == 2,
          str(gw.invocation_log.count()))

    # 11. disable flag
    cache2 = tmp / "cache2"
    shutil.copytree("tests/gateway_cache", cache2)
    for f in ("invocations.db", "invocations.db-wal", "invocations.db-shm"):
        (cache2 / f).unlink(missing_ok=True)
    gw2 = ArcGateway(GatewayConfig(
        port=0, cache_dir=str(cache2), chat_url="http://127.0.0.1:8080/v1",
        embed_url="http://127.0.0.1:8081", model=None,
        invocation_log=False))
    r = gw2.handle_query("How many positive whole-number divisors does 196 have?")
    check("no-invocation-log still serves", r["source"] == "cache", str(r)[:120])
    check("no-invocation-log creates no DB", gw2.invocation_log is None
          and not (cache2 / "invocations.db").exists(), "")

    # 12. token deltas with a mock LM (records every call into its ledger)
    from arc.backends import MockLM
    cache3 = tmp / "cache3"
    shutil.copytree("tests/gateway_cache", cache3)
    for f in ("invocations.db", "invocations.db-wal", "invocations.db-shm"):
        (cache3 / f).unlink(missing_ok=True)
    gw3 = ArcGateway(GatewayConfig(
        port=0, cache_dir=str(cache3), chat_url="http://127.0.0.1:8080/v1",
        embed_url="http://127.0.0.1:8081", model="mock",
        family_thresholds={"divisors": 0.99}))
    gw3.lm = MockLM(routes={"": "Answer: 42"})

    # hit: regex extraction, no LM call -> zero token delta
    r = gw3.handle_query("How many positive whole-number divisors does 196 have?")
    rows = gw3.invocation_log.recent(3)
    check("hit row records zero token delta (no LM calls)",
          r["source"] == "cache"
          and (rows[0]["tokens_in"] or 0) == 0
          and (rows[0]["tokens_out"] or 0) == 0, str(rows[0]))

    # threshold miss: _miss proxies to the LM -> token delta > 0
    r = gw3.handle_query("What is the meaning of life in mathematics?")
    rows = gw3.invocation_log.recent(3)
    check("miss row records token delta from LM fallback",
          r["source"] == "llm-fallback" and r.get("answer") == "Answer: 42"
          and rows[0]["status"] == "miss" and rows[0]["phase"] == "threshold"
          and (rows[0]["tokens_in"] or 0) > 0, str(rows[0]))

    shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    print("[offline] InvocationLog unit tests")
    tmp = test_offline()
    print("[live] ArcGateway integration (needs embed server :8081)")
    test_live()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nP7 INVOCATIONS RESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
