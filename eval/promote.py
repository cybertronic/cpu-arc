"""Gated cache promotion: build a new entry from a seed, validate against a
golden function, and only promote if validation passes (else roll back).

This is the "build" half of the production feedback loop:

    production miss  ->  invocations.db (P7 log)  [+ build_queue job if
    enable_lazy_build]  ->  eval/curate_misses.py ranks candidates
                          ->  eval/promote.py builds + validates + promotes

Usage
-----
  # build from an explicit seed (source-solved on the fly):
  python3 eval/promote.py --family gcd --cache-dir tests/.gcd_cache \
      --seed "What is the greatest common divisor of 315 and 108?" \
      --golden "math.gcd(p0, p1)"

  # build from a lazy-build queue job (has query + transcript already):
  python3 eval/promote.py --family gcd --cache-dir tests/.gcd_cache \
      --job tests/build_queue/build_123.json --golden "math.gcd(p0, p1)"

Validation gate: `--cases` random parameter tuples are generated, plugged
into the seed (numbers substituted in order), served by a throwaway gateway
(no LLM). Criteria: ZERO wrong answers (a hit that disagrees with the
golden is a hard fail), and hit rate >= 60% (the first entry of a new
parameter shape sits at the calibration band edge; misses are safe — they
fall back to the LLM and get logged as candidates for the next batch
entry). Any gate failure rolls the cache back to its pre-build state.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SRC_SYSTEM = ("You are a careful mathematician. Solve the problem with clear "
              "step-by-step reasoning. End with 'Answer: <value>'.")


def extract_answer(ans) -> float | None:
    """Pull a numeric answer out of an entry answer dict (or pass through)."""
    if isinstance(ans, dict):
        for v in ans.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        vals = list(ans.values())
        if len(vals) == 1:
            return extract_answer(vals[0])
        return None
    if isinstance(ans, (int, float)) and not isinstance(ans, bool):
        return float(ans)
    if isinstance(ans, str):
        m = re.search(r"-?\d+(?:\.\d+)?", ans)
        return float(m.group()) if m else None
    return None


def make_cases(seed: str, n: int, rng: random.Random) -> list[list[int]]:
    """Generate n random numeric tuples by substituting the seed's numbers
    (in order of appearance) with fresh random integers."""
    orig = [int(x) for x in re.findall(r"\b\d{1,15}\b", seed)]
    k = max(len(orig), 1)
    cases = []
    for _ in range(n):
        nums = [rng.randint(10, 99999) for _ in range(k)]
        # ensure a non-trivial shared factor in ~half the cases (real workloads
        # often do; gcd/lcm answers stay interesting)
        if rng.random() < 0.5 and k >= 2:
            a, b = rng.randint(2, 12), rng.randint(2, 999)
            nums[0], nums[1] = a * b, a * rng.randint(2, 999)
        cases.append(nums)
    return cases


def case_query(seed: str, nums: list[int]) -> str:
    """Substitute the seed's numbers, left to right, with `nums`."""
    it = iter(nums)
    return re.sub(r"\b\d{1,15}\b", lambda _: str(next(it)), seed)


MIN_HIT_RATE = 0.6  # floor for the FIRST entry of a new parameter shape


def _numdiv(n: int) -> int:
    """Number of positive divisors (golden helper for the 'divisors' family)."""
    c, i = 0, 1
    while i * i <= n:
        if n % i == 0:
            c += 1
            if i != n // i:
                c += 1
        i += 1
    return c


def run_gate(cache_dir: Path, family: str, seed: str, golden_expr: str,
             n_cases: int, threshold: float,
             embed_url: str) -> tuple[bool, bool, float, list[dict]]:
    """Serve generated cases through a throwaway gateway; all must hit."""
    from arc.gateway import ArcGateway, GatewayConfig
    gw = ArcGateway(GatewayConfig(
        port=0, cache_dir=str(cache_dir),
        chat_url="http://127.0.0.1:8080/v1", embed_url=embed_url,
        model=None,  # no LLM: a miss is a miss
        family_thresholds={family: threshold}, min_separation=0.05))
    rng = random.Random(20261125)
    results = []
    for nums in make_cases(seed, n_cases, rng):
        q = case_query(seed, nums)
        # trusted CLI arg; namespace = stdlib math + p0..pN + params list
        # (family-level templates in deploy/golden-map.json use ``*params``)
        ns = {f"p{i}": v for i, v in enumerate(nums)}
        ns["params"] = nums  # family templates use ``math.gcd(*params)``
        golden = float(eval(golden_expr,
                            {"math": math, "numdiv": _numdiv, **ns}))
        t0 = time.perf_counter()
        r = gw.handle_query(q)
        ms = (time.perf_counter() - t0) * 1000
        got = extract_answer(r.get("answer")) if r.get("source") == "cache" else None
        # correctness: if it HIT, it must be right. Misses are allowed (they
        # fall back to the LLM in production); WRONG answers are not.
        correct = got is None or abs(got - golden) < 1e-9
        results.append({"query": q, "golden": golden, "got": got,
                        "source": r.get("source"), "ok": correct,
                        "hit": r.get("source") == "cache", "ms": round(ms, 1)})
    all_correct = all(x["ok"] for x in results)
    hit_rate = sum(1 for x in results if x["hit"]) / len(results)
    return all_correct, hit_rate >= MIN_HIT_RATE, hit_rate, results


def main() -> None:
    ap = argparse.ArgumentParser(description="Gated cache promotion")
    ap.add_argument("--family", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--seed", default=None, help="seed problem text")
    ap.add_argument("--job", default=None,
                    help="lazy-build queue job JSON (query + transcript)")
    ap.add_argument("--golden", required=True,
                    help="python expr over p0..pN-1, e.g. 'math.gcd(p0, p1)'")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.72,
                    help="per-family entry threshold applied during the gate")
    ap.add_argument("--chat-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M")
    ap.add_argument("--no-recompile", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    seed = args.seed
    transcript = None
    if args.job:
        job = json.loads(Path(args.job).read_text())
        seed = seed or job["query"]
        transcript = job.get("transcript")
    if not seed:
        ap.error("need --seed or --job")

    from arc import ARC, Thresholds, Embedder, LMBackend, Ledger
    from arc.builder import Builder
    from arc.compiler import Compiler

    # 0. backup for rollback
    backup = Path(tempfile.mkdtemp(prefix="promote_bak_"))
    for name in ("functions.db", "entries.db"):
        src = cache_dir / name
        if src.exists():
            shutil.copy2(src, backup / name)
    if (cache_dir / "compiled").exists():
        shutil.copytree(cache_dir / "compiled", backup / "compiled")
    print(f"[0] cache backup -> {backup}")

    embedder = Embedder(backend="server", url=args.embed_url)
    b27 = LMBackend(url=args.chat_url, model=args.model)
    ledger = Ledger()
    b27.ledger = ledger

    # 1. source solve (if not provided by a queue job)
    if transcript is None:
        print(f"[1] source-solving seed ...")
        t0 = time.perf_counter()
        transcript = b27.complete(seed, system=SRC_SYSTEM, role="source",
                                  max_tokens=1500).text
        ans_line = [l for l in transcript.splitlines() if "Answer" in l]
        print(f"    {time.perf_counter()-t0:.0f}s; "
              f"{ans_line[-1][:90] if ans_line else '(no Answer line)'}")

    # 2. build into the EXISTING family cache
    arc_fam = ARC(embedder, b27,
                  Thresholds(entry=0.72, function=0.55, load=0.45),
                  db_path=str(cache_dir), ledger=ledger)
    builder = Builder(arc_fam, builder_lm=b27, judge_lm=b27,
                      max_rewrites=3, roundtrips=2)
    print(f"[2] building entry into {cache_dir} (family={args.family}) ...")
    t0 = time.perf_counter()
    result = builder.build(seed, transcript, family=args.family)
    build_s = time.perf_counter() - t0
    print(f"    ok={result.ok} in {build_s:.0f}s rewrites={result.rewrites}")
    if not result.ok:
        print(f"    errors: {result.errors[:3]}")
        _rollback(backup, cache_dir, ap)
    print(f"    template: {result.template}")
    print(f"    params: {[f'{p.description}={p.value}' for p in result.params]}")

    # 3. recompile BEFORE the gate: the gate exercises the compiled (worker
    # pool) path, which is what production serves — the module must already
    # contain the new entry or dispatch will KeyError on it.
    if args.no_recompile:
        print("[3] --no-recompile: gate will run against the existing module")
    else:
        compiler = Compiler(arc_fam.registry, arc_fam.entry_store)
        crec = compiler.compile_all()
        compiler.write(crec, cache_dir / "compiled")
        print(f"[3] recompiled: fully_static="
              f"{crec.manifest['stats']['fully_static']} all_resolved="
              f"{crec.all_resolved} entries={len(arc_fam.entry_store.entries)} "
              f"({len(crec.entries)} unique)")

    # 4. validation gate
    print(f"[4] validation gate: {args.cases} random cases vs golden "
          f"({args.golden}), threshold {args.threshold} ...")
    all_correct, hit_ok, hit_rate, rows = run_gate(
        cache_dir, args.family, seed, args.golden,
        args.cases, args.threshold, args.embed_url)
    for r in rows:
        flag = "ok  " if r["ok"] else "WRONG"
        tag = "hit " if r["hit"] else "miss"
        print(f"    {tag} {flag} {r['query'][:60]:60s} golden={r['golden']:.0f} "
              f"got={r['got']} {r['ms']}ms")
    wrong = sum(1 for r in rows if not r["ok"])
    print(f"    hit rate {hit_rate:.0%} (floor {MIN_HIT_RATE:.0%}), "
          f"wrong answers {wrong}/{len(rows)}")
    if wrong:
        print(f"    GATE FAILED: {wrong} WRONG answer(s) -> rolling back")
        _rollback(backup, cache_dir, ap)
    if not hit_ok:
        print(f"    GATE FAILED: coverage {hit_rate:.0%} < {MIN_HIT_RATE:.0%} "
              f"-> rolling back")
        _rollback(backup, cache_dir, ap)

    if args.no_recompile:
        print("[!] --no-recompile: module left as rebuilt in step 3")
    print(f"    ledger: {ledger.snapshot()}")
    print(f"PROMOTED: family={args.family} seed={seed[:60]!r} "
          f"coverage={hit_rate:.0%} wrong=0")


def _rollback(backup: Path, cache_dir: Path, ap: argparse.ArgumentParser) -> None:
    print(f"[!] rolling back {cache_dir} from {backup}")
    for name in ("functions.db", "entries.db"):
        src = backup / name
        if src.exists():
            shutil.copy2(src, cache_dir / name)
    comp = backup / "compiled"
    if comp.exists():
        dst = cache_dir / "compiled"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(comp, dst)
    ap.error("promotion aborted (cache restored to pre-build state)")


if __name__ == "__main__":
    main()
