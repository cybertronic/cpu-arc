"""Nightly production feedback loop: mine misses -> gated auto-promote -> report.

The "close" half of the loop, invoked by deploy/nightly_curate.sh (cron):

    1. mine the invocation log (curate_misses.mine + attach_queue)
    2. select auto-promotable candidates:
         - known family + repeated misses (count >= CURATOR_MIN_REPEATS)
         - a golden expression template exists for the family
           (e.g. gcd -> "math.gcd(*params)")
         - not cross-family-ambiguous, not below every family's band
    3. for each (capped per night): run eval/promote.py
       (build -> recompile -> validate vs golden -> rollback on failure)
    4. write machine-readable report + human summary

Anything not provable (no golden template, below repeat floor, ambiguous)
goes to the report for human review. The loop only auto-promotes what it can
*prove* correct on the production compiled path.

Exit codes: 0 = loop completed (0+ promotions, rollbacks are expected
outcomes), 1 = operational failure (missing DB, driver error).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from curate_misses import attach_queue, mine  # noqa: E402


def get_embedder(backend: str, url: str, fallback: str):
    from arc import Embedder
    if backend == "hash":
        return Embedder(backend="hash"), "hash"
    try:
        e = Embedder(backend="server", url=url)
        e.embed("probe")  # fail fast if the embed server is down
        return e, "server"
    except Exception as exc:  # noqa: BLE001
        if fallback == "none":
            raise
        print(f"[warn] embed server unreachable ({exc}); falling back to hash")
        return Embedder(backend="hash"), "hash"


def auto_promotable(c: dict, golden_map: dict, min_repeats: int) -> tuple[str | None, str]:
    """Why (not) to auto-promote this cluster. Returns (golden_expr, reason)."""
    fam = c.get("family")
    if fam not in golden_map:
        return None, f"no golden template for family {fam!r}"
    if c["count"] < min_repeats:
        return None, f"below repeat floor ({c['count']} < {min_repeats})"
    if "separation" in c["phases"]:
        return None, "cross-family ambiguity — guard work, not an entry build"
    if c["min_score"] is not None and c["min_score"] < 0.60:
        return None, "below every family's band — new-family review"
    if not re.search(r"\b\d{1,15}\b", c["query"]):
        return None, "no parameters in query"
    return golden_map[fam], "ok"


def run_promote(repo: Path, fam: str, cache_dir: str, seed: str, golden: str,
                cases: int, threshold: float, chat_url: str, embed_url: str,
                model: str, queue_job: str | None) -> tuple[int, str, bool]:
    cmd = [sys.executable, str(repo / "eval" / "promote.py"),
           "--family", fam, "--cache-dir", cache_dir,
           "--golden", golden, "--cases", str(cases),
           "--threshold", f"{threshold}",
           "--chat-url", chat_url, "--embed-url", embed_url, "--model", model]
    if queue_job:
        cmd += ["--job", queue_job]
    else:
        cmd += ["--seed", seed]
    print(f"[promote] {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-2000:])  # tail: the gate table + verdict
    return r.returncode, out, r.returncode == 0 and "PROMOTED:" in out


def main() -> None:
    ap = argparse.ArgumentParser(description="Nightly curator: mine -> gated auto-promote")
    ap.add_argument("--db", required=True, help="invocations.db")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--build-queue-dir", default=None)
    ap.add_argument("--embed-url", default="http://127.0.0.1:8081")
    ap.add_argument("--embed", choices=["server", "hash"], default="server")
    ap.add_argument("--embed-fallback", choices=["auto", "none"], default="auto")
    ap.add_argument("--golden-map", required=True,
                    help="JSON: family -> golden expr, e.g. {\"gcd\": \"math.gcd(*params)\"}")
    ap.add_argument("--auto-promote", default="1",
                    help="1 = run gated promotes; 0 = report only")
    ap.add_argument("--min-repeats", type=int, default=3)
    ap.add_argument("--max-promotions", type=int, default=2,
                    help="cap on gated builds per night (each ~2-3 min model time)")
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--since-hours", type=float, default=168.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--family-thresholds", default="",
                    help="'fam=th,fam=th' (same format as gateway --family-threshold)")
    ap.add_argument("--chat-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"[fail] invocation log not found: {args.db}")
        sys.exit(1)

    golden_map = json.loads(Path(args.golden_map).read_text())
    fam_th = dict(p.split("=", 1) for p in args.family_thresholds.split(",")
                  if "=" in p)
    repo = Path(__file__).resolve().parent.parent
    auto = args.auto_promote == "1"

    from arc.invocations import InvocationLog
    log = InvocationLog(args.db)
    embedder, embed_used = get_embedder(args.embed, args.embed_url, args.embed_fallback)
    since = time.time() - args.since_hours * 3600
    clusters = mine(log, embedder, since_s=since)
    if args.build_queue_dir and Path(args.build_queue_dir).exists():
        attach_queue(clusters, Path(args.build_queue_dir), embedder)
    log.close()

    print(f"[curate] {sum(c['count'] for c in clusters)} missed rows in the "
          f"last {args.since_hours:.0f}h -> {len(clusters)} clusters "
          f"(embed={embed_used}, auto_promote={auto}, min_repeats={args.min_repeats})")

    actions = []
    promotions = 0
    for c in clusters[:args.top]:
        label = f"x{c['count']} {c['query'][:60]!r}"
        expr, why = auto_promotable(c, golden_map, args.min_repeats)
        rec = {"candidate": c, "golden": expr, "decision": None, "rc": None}
        if not auto:
            rec["decision"] = "report-only (auto-promote disabled)"
        elif expr is None:
            rec["decision"] = f"review: {why}"
        elif promotions >= args.max_promotions:
            rec["decision"] = "deferred (nightly promotion cap reached)"
        else:
            fam = c["family"]
            job = None
            if c.get("queued_job", {}).get("file"):
                job = str(Path(args.build_queue_dir) / c["queued_job"]["file"])
            try:
                rc, _out, ok = run_promote(
                    repo, fam, args.cache_dir, c["query"], expr,
                    args.cases, fam_th.get(fam, 0.72),
                    args.chat_url, args.embed_url, args.model, job)
                rec["rc"] = rc
                rec["decision"] = ("promoted (gate passed)" if ok
                                   else "rolled back (gate failed)" if rc == 2
                                   else f"FAILED rc={rc} — inspect report")
                if ok:
                    promotions += 1
            except subprocess.TimeoutExpired:
                rec["decision"] = "FAILED: promote timed out (30 min)"
            except Exception as exc:  # noqa: BLE001
                rec["decision"] = f"FAILED: {exc}"
        actions.append(rec)
        mark = {"promoted (gate passed)": "PROMOTED",
                "rolled back (gate failed)": "rolled back"}.get(rec["decision"], "skip")
        print(f"[{mark:12s}] {label} :: {rec['decision']}")

    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report = {
        "run_at": time.time(), "date": stamp,
        "db": args.db, "cache_dir": args.cache_dir,
        "embed": embed_used, "auto_promote": auto,
        "total_miss_rows": sum(c["count"] for c in clusters),
        "candidates": clusters,
        "actions": [
            {"query": a["candidate"]["query"], "count": a["candidate"]["count"],
             "family": a["candidate"]["family"], "golden": a["golden"],
             "decision": a["decision"], "rc": a["rc"]}
            for a in actions],
        "promoted": promotions,
    }
    rp = Path(args.report_dir) / f"curate-{stamp}.json"
    rp.write_text(json.dumps(report, indent=2, default=str))
    print(f"[done] {promotions} promoted, report -> {rp}")


if __name__ == "__main__":
    main()
