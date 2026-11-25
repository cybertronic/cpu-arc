"""P4 eval: build a SECOND cache entry (GCD) from a real MATH problem, then
run the multi-family live eval (divisors + GCD) with C vs S accuracy.

Steps:
  1. Source model solves a real MATH number-theory GCD problem (transcript)
  2. Builder runs the 5 phases on (query, transcript) -> fresh cache
  3. Compiler emits standalone module
  4. Gateway serves both families; S baseline answers raw
  5. Report: per-family C Acc vs S Acc, hit rates, latency

Run: python3 tests/test_p4_multi_live.py   (~15-25 min of 27B inference)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc import ARC, Thresholds, Embedder, LMBackend, Ledger
from arc.builder import Builder
from arc.compiler import Compiler
from arc.gateway import ArcGateway, GatewayConfig
from eval.harness import (
    EvalRunner, EvalReport, CaseResult, Problem, EvalCase,
    perturb_original, load_math,
)

SRC_SYSTEM = ("You are a careful mathematician. Solve the problem with clear "
              "step-by-step reasoning. End with 'Answer: <value>'.")


def gcd_golden(*nums: int) -> int:
    return math.gcd(*nums)


def make_gcd_problems(limit: int) -> list[Problem]:
    """Synthetic GCD problems with locally-computed goldens."""
    pairs = [
        (3339, 2961), (1491, 2205), (48, 18), (100, 75), (196, 42),
        (12, 8), (1001, 455), (65536, 4096), (97, 101), (210, 132),
        (123456, 54321), (720, 96),
    ]
    problems = []
    for i, (a, b) in enumerate(pairs[:limit]):
        g = gcd_golden(a, b)
        problems.append(Problem(
            id=f"gcd_{a}_{b}",
            question=f"What is the greatest common divisor of {a} and {b}?",
            golden=str(g),
            source="synthetic",
            subject="number-theory/gcd",
            solution=f"gcd({a}, {b}) = {g}.",
        ))
    return problems


def make_gcd3_problems(limit: int) -> list[Problem]:
    """Three-integer GCD problems — in the family of the 3-param cache entry
    (2-integer queries legitimately are not, and must fall back)."""
    triples = [
        (63, 147, 105), (120, 80, 60), (252, 180, 330),
        (49, 147, 245), (1000, 750, 500), (91, 143, 187),
    ]
    problems = []
    for (a, b, c) in triples[:limit]:
        g = gcd_golden(a, b, c)
        problems.append(Problem(
            id=f"gcd_{a}_{b}_{c}",
            question=f"What is the greatest common divisor of {a}, {b}, and {c}?",
            golden=str(g),
            source="synthetic",
            subject="number-theory/gcd",
            solution=f"gcd({a}, {b}, {c}) = {g}.",
        ))
    return problems


def main():
    chat_url = "http://127.0.0.1:8080/v1"
    embed_url = "http://127.0.0.1:8081"
    model = "unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M"

    # --- Pick a real MATH GCD problem as the build seed -------------------
    print("[1] picking a real MATH GCD problem as build seed ...")
    maths = load_math(subjects=["number_theory"])
    seed = next(p for p in maths
                if "greatest common divisor of $3339$" in p.question.replace(" ", "")
                or ("Find the greatest common divisor" in p.question and "$3339$" in p.question))
    if seed is None:
        seed = next(p for p in maths if "greatest common divisor" in p.question.lower()
                    and "two positive integers less than" not in p.question)
    print(f"    seed: {seed.id}")
    print(f"    Q: {seed.question[:120]}")

    embedder = Embedder(backend="server", url=embed_url)
    b27 = LMBackend(url=chat_url, model=model)
    ledger = Ledger()
    b27.ledger = ledger

    # --- Source solve ------------------------------------------------------
    print("[2] source model solves the seed problem ...")
    t0 = time.perf_counter()
    transcript = b27.complete(seed.question, system=SRC_SYSTEM, role="source",
                              max_tokens=1500).text
    print(f"    done in {time.perf_counter()-t0:.0f}s")
    print(f"    answer line: {[l for l in transcript.splitlines() if 'Answer' in l][-1][:100] if any('Answer' in l for l in transcript.splitlines()) else '(none)'}")

    # --- Build into a FRESH cache ------------------------------------------
    print("[3] builder 5-phase construction -> fresh GCD cache ...")
    gcd_cache_dir = "tests/.gcd_cache"
    Path(gcd_cache_dir).mkdir(exist_ok=True)
    arc_gcd = ARC(embedder, b27, Thresholds(entry=0.72, function=0.55, load=0.45),
                  db_path=gcd_cache_dir, ledger=ledger)
    builder = Builder(arc_gcd, builder_lm=b27, judge_lm=b27, max_rewrites=3, roundtrips=2)
    t0 = time.perf_counter()
    result = builder.build(seed.question, transcript)
    build_s = time.perf_counter() - t0
    print(f"    ok={result.ok} in {build_s:.0f}s rewrites={result.rewrites}")
    if not result.ok:
        print(f"    errors: {result.errors[:3]}")
        sys.exit("GCD build failed — aborting multi-family eval")
    print(f"    template: {result.template}")
    print(f"    params: {[f'{p.description}={p.value}' for p in result.params]}")
    print(f"    ledger: {ledger.snapshot()}")

    # --- Compile -------------------------------------------------------------
    print("[4] compiling GCD cache ...")
    compiler = Compiler(arc_gcd.registry, arc_gcd.entry_store)
    crec = compiler.compile_entry()
    compiler.write(crec, Path(gcd_cache_dir) / "compiled")
    print(f"    functions={list(crec.functions.keys())} fully_static={crec.all_resolved}")

    # --- Multi-family gateway -------------------------------------------------
    # The divisors cache is in tests/.p2_live_cache; GCD in tests/.gcd_cache.
    # For a true multi-entry cache we'd merge both into one DB. Instead, run two
    # gateways (one per family) and report side by side — the harness supports it.
    print("[5] gateway on GCD cache ...")
    gw_gcd = ArcGateway(GatewayConfig(
        port=0, cache_dir=gcd_cache_dir, chat_url=chat_url, embed_url=embed_url,
        model=model, entry_threshold=0.72))
    print(f"    entries={len(gw_gcd.entry_store.entries)} compiled={'yes' if gw_gcd._compiled_mod else 'no'}")

    # Divisors gateway (existing P2 cache)
    gw_div = ArcGateway(GatewayConfig(
        port=0, cache_dir="tests/.p2_live_cache", chat_url=chat_url, embed_url=embed_url,
        model=model, entry_threshold=0.72))

    # --- Eval problems ---------------------------------------------------------
    div_problems = [
        Problem(id=f"div_{n:06d}", question=f"How many positive whole-number divisors does {n} have?",
                golden=str(_num_divisors(n)), source="synthetic", subject="number-theory/divisors")
        for n in [196, 144, 360, 1000, 1, 2, 7, 100, 65536, 999983, 12, 84]
    ]
    gcd_problems = make_gcd_problems(12) + make_gcd3_problems(6)

    # --- Run C (cache) for both families ----------------------------------------
    print(f"[6] running cache-serve cases: {len(div_problems)} divisors + {len(gcd_problems)} GCD "
          f"({sum(1 for p in gcd_problems if p.id.count('_') == 2)} two-int + "
          f"{sum(1 for p in gcd_problems if p.id.count('_') == 3)} three-int) ...")
    runner_div = EvalRunner(gateway=gw_div, source_lm=b27)
    runner_gcd = EvalRunner(gateway=gw_gcd, source_lm=b27)

    t0 = time.perf_counter()
    rep_div_c = runner_div.run_cases([perturb_original(p) for p in div_problems])
    rep_gcd_c = runner_gcd.run_cases([perturb_original(p) for p in gcd_problems])
    serve_s = time.perf_counter() - t0

    # --- S baseline (raw 27B, no cache) -------------------------------------------
    print(f"[7] running S-baseline cases (raw 27B): {len(div_problems)} + {len(gcd_problems)} ...")
    runner_s = EvalRunner(source_lm=b27)
    all_cases = [perturb_original(p) for p in div_problems] + [perturb_original(p) for p in gcd_problems]
    rep_s = runner_s.run_cases(all_cases)

    # --- Report --------------------------------------------------------------------
    print()
    print("=" * 66)
    print("P4 MULTI-FAMILY LIVE — divisors (P2 cache) + GCD (fresh build)")
    print("=" * 66)
    for label, rep in [("divisors", rep_div_c), ("GCD", rep_gcd_c)]:
        correct = sum(1 for r in rep.results if r.correct)
        hits = sum(1 for r in rep.results if r.source == "cache")
        print(f"\n  {label} (C): {correct}/{len(rep.results)} correct, {hits} cache hits")
        for r in rep.results:
            mark = "ok" if r.correct else "FAIL"
            print(f"    {mark}  {r.problem_id} [{r.source}] ans={str(r.answer)[:25]} golden={r.golden} score={r.entry_score} {r.ms:.0f}ms")

    s_div = [r for r in rep_s.results if r.mode == "original" and r.problem_id.startswith("div_")]
    s_gcd = [r for r in rep_s.results if r.mode == "original" and r.problem_id.startswith("gcd_")]
    sd = sum(1 for r in s_div if r.correct)
    sg = sum(1 for r in s_gcd if r.correct)

    print("-" * 66)
    print(f"  C Acc divisors: {sum(1 for r in rep_div_c.results if r.correct)/len(rep_div_c.results):.3f}   S Acc: {sd/len(s_div):.3f}")
    print(f"  C Acc GCD:      {sum(1 for r in rep_gcd_c.results if r.correct)/len(rep_gcd_c.results):.3f}   S Acc: {sg/len(s_gcd):.3f}")
    c_total = sum(1 for r in list(rep_div_c.results) + list(rep_gcd_c.results) if r.correct)
    s_total = sd + sg
    n_total = len(all_cases)
    print(f"  C Acc overall:  {c_total/n_total:.3f} ({c_total}/{n_total})   S Acc: {s_total/n_total:.3f} ({s_total}/{n_total})")
    print(f"  GCD build time: {build_s:.0f}s | serve wall: {serve_s:.1f}s for {len(all_cases)} cache cases (avg {serve_s/len(all_cases)*1000:.0f}ms)")
    print(f"  total ledger tokens: {ledger.total_tokens()}")

    out = Path("tests/p4_multi_live_report.json")
    out.write_text(json.dumps({
        "divisors_C": [r.__dict__ for r in rep_div_c.results],
        "gcd_C": [r.__dict__ for r in rep_gcd_c.results],
        "S_baseline": [r.__dict__ for r in rep_s.results],
        "summary": {
            "c_acc_divisors": round(sum(1 for r in rep_div_c.results if r.correct)/len(rep_div_c.results), 4),
            "s_acc_divisors": round(sd/len(s_div), 4),
            "c_acc_gcd": round(sum(1 for r in rep_gcd_c.results if r.correct)/len(rep_gcd_c.results), 4),
            "s_acc_gcd": round(sg/len(s_gcd), 4),
            "c_acc_overall": round(c_total/n_total, 4),
            "s_acc_overall": round(s_total/n_total, 4),
            "gcd_build_s": round(build_s, 1),
            "serve_avg_ms": round(serve_s/len(all_cases)*1000, 1),
            "total_tokens": ledger.total_tokens(),
        },
    }, indent=2))
    print(f"\n  report -> {out}")


def _num_divisors(n: int) -> int:
    c, i = 0, 1
    while i * i <= n:
        if n % i == 0:
            c += 1
            if i * i != n:
                c += 1
        i += 1
    return c


if __name__ == "__main__":
    main()
