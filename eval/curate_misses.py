"""Mine the invocation log into ranked cache-build candidates.

The "mine" half of the production feedback loop:

    production miss  ->  invocations.db (P7)  ->  THIS (rank candidates)
                       ->  eval/promote.py (gated build + validate + promote)

Usage
-----
  python3 eval/curate_misses.py --db tests/gateway_cache/invocations.db \
      [--since-hours 168] [--top 10] [--out candidates.json] \
      [--build-queue-dir tests/build_queue] \
      [--embed-url http://127.0.0.1:8081]     # or --embed hash for offline

Each candidate is a cluster of near-duplicate missed queries (cosine >=
0.92), ranked by repeat count then recency, with a suggested action.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def mine(invocation_log, embedder, since_s: float | None = None,
         dedup_cosine: float = 0.92) -> list[dict]:
    """Cluster near-duplicate missed queries; rank by count, then recency."""
    with invocation_log._lock:  # read-only scan
        rows = invocation_log._conn.execute(
            "SELECT ts, query, status, phase, family, entry, score, reason "
            "FROM invocations WHERE status IN ('miss', 'error')"
            + (" AND ts >= ?" if since_s is not None else "")
            + " ORDER BY ts",
            (since_s,) if since_s is not None else ()).fetchall()

    # group by exact text first (cheap)
    groups: dict[str, dict] = {}
    for ts, query, status, phase, family, entry, score, reason in rows:
        g = groups.setdefault(query, {"count": 0, "first_ts": ts, "last_ts": ts,
                                      "phases": set(), "family": family,
                                      "entry": entry, "scores": [],
                                      "reasons": {}})
        g["count"] += 1
        g["first_ts"] = min(g["first_ts"], ts)
        g["last_ts"] = max(g["last_ts"], ts)
        if phase:
            g["phases"].add(phase)
        if family and not g["family"]:
            g["family"] = family
        if score is not None:
            g["scores"].append(score)
        if reason:
            g["reasons"][reason] = g["reasons"].get(reason, 0) + 1

    if not groups:
        return []

    # embed unique texts, greedy cluster
    texts = list(groups.keys())
    vecs = np.array([embedder.embed(t) for t in texts], dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    assigned: list[int | None] = [None] * len(texts)
    clusters: list[dict] = []
    for i, t in enumerate(texts):
        if assigned[i] is not None:
            continue
        members = [i]
        assigned[i] = len(clusters)
        if len(texts) > 1:
            sims = vecs @ vecs[i]
            for j in range(len(texts)):
                if j != i and assigned[j] is None and sims[j] >= dedup_cosine:
                    members.append(j)
                    assigned[j] = len(clusters)
        rep = max(members, key=lambda m: groups[texts[m]]["count"])
        g = groups[texts[rep]]
        total = sum(groups[texts[m]]["count"] for m in members)
        all_scores = [s for m in members for s in groups[texts[m]]["scores"]]
        clusters.append({
            "query": texts[rep],
            "variants": [texts[m] for m in members if m != rep][:3],
            "count": total,
            "first_ts": min(groups[texts[m]]["first_ts"] for m in members),
            "last_ts": max(groups[texts[m]]["last_ts"] for m in members),
            "phases": sorted(set().union(*(groups[texts[m]]["phases"]
                                           for m in members))),
            "family": g["family"],
            "entry": g["entry"],
            "min_score": min(all_scores) if all_scores else None,
            "reasons": dict(sorted(g["reasons"].items(),
                                   key=lambda kv: -kv[1])[:2]),
        })
    clusters.sort(key=lambda c: (-c["count"], -c["last_ts"]))
    for c in clusters:
        c["suggestion"] = suggest(c)
    return clusters


def suggest(c: dict) -> str:
    fam = c["family"]
    if "separation" in c["phases"]:
        return (f"cross-family ambiguity (top family={fam}) — needs an "
                f"operation-signature guard or a concrete entry in '{fam}'")
    score = c["min_score"]
    if score is not None and score < 0.60:
        return (f"far below every family's band (min score {score:.3f}) — "
                f"new-family candidate; review before building")
    if fam and c["count"] >= 2:
        return (f"repeated miss near family '{fam}' — build a concrete "
                f"entry: eval/promote.py --family {fam} --seed {c['query']!r}")
    if fam:
        return f"one-off miss near family '{fam}' — monitor; build if it repeats"
    return "no family context — new-family candidate"


def attach_queue(clusters: list[dict], queue_dir: Path,
                 embedder, dedup_cosine: float = 0.92) -> None:
    """Attach pending lazy-build queue jobs (with transcripts) to clusters."""
    if not queue_dir.exists():
        return
    jobs = []
    for f in sorted(queue_dir.glob("build_*.json")):
        try:
            j = json.loads(f.read_text())
            if j.get("status") == "pending" and j.get("query"):
                jobs.append((j["query"], f.name, bool(j.get("transcript"))))
        except (json.JSONDecodeError, OSError):
            continue
    if not jobs:
        return
    jvecs = np.array([embedder.embed(q) for q, _, _ in jobs], dtype=np.float32)
    jvecs /= (np.linalg.norm(jvecs, axis=1, keepdims=True) + 1e-12)
    for c in clusters:
        v = embedder.embed(c["query"])
        v = v / (np.linalg.norm(v) + 1e-12)
        sims = jvecs @ v
        for j in range(len(jobs)):
            if sims[j] >= dedup_cosine:
                c["queued_job"] = {"file": jobs[j][1],
                                   "has_transcript": jobs[j][2]}
                break


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine invocation log for candidates")
    ap.add_argument("--db", required=True)
    ap.add_argument("--since-hours", type=float, default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--build-queue-dir", default=None)
    from arc import endpoints
    ap.add_argument("--embed-url", default=endpoints.embed_url())
    ap.add_argument("--embed", choices=["server", "hash"], default="server")
    args = ap.parse_args()

    from arc import Embedder
    from arc.invocations import InvocationLog
    log = InvocationLog(args.db)
    embedder = (Embedder(backend="server", url=args.embed_url)
                if args.embed == "server" else Embedder(backend="hash"))
    since = time.time() - args.since_hours * 3600 if args.since_hours else None
    clusters = mine(log, embedder, since_s=since)
    if args.build_queue_dir:
        attach_queue(clusters, Path(args.build_queue_dir), embedder)
    top = clusters[:args.top]
    for i, c in enumerate(top, 1):
        print(f"{i:2d}. [x{c['count']}] {c['query'][:72]}")
        print(f"    family={c['family']} phases={c['phases']} "
              f"min_score={c['min_score']}")
        if c["variants"]:
            print(f"    variants: {c['variants'][0][:60]}")
        print(f"    -> {c['suggestion']}")
        if c.get("queued_job"):
            print(f"    queued job: {c['queued_job']['file']} "
                  f"(transcript={c['queued_job']['has_transcript']})")
    if args.out:
        out = {"mined_at": time.time(), "total_miss_rows": sum(c["count"] for c in clusters),
               "candidates": clusters}
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n{len(clusters)} candidates -> {args.out}")
    log.close()


if __name__ == "__main__":
    main()
