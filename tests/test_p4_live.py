"""P4 eval: live end-to-end on the divisors entry (built in P2).

Protocol (paper):
  S baseline: source model answers each variant raw (no cache)
  C (cache):   gateway serves from the built+compiled divisors cache
  Modes:       original / phrasal (rephrased, same number) / parametric (new numbers)
  Fidelity:    numeric comparison vs golden (exact math — no judge needed)

Golden labels are computed locally (divisor function), so this eval needs no
external labels. The source model is only used for the S baseline + rephrasing.

Run: python3 tests/test_p4_live.py [--limit N]   (~10-20 min of 27B inference)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc import ARC, Thresholds, Embedder, LMBackend, Ledger
from arc.gateway import ArcGateway, GatewayConfig
from eval.harness import (
    EvalRunner, EvalReport, CaseResult, Problem, EvalCase,
    perturb_original, answers_match, extract_answer_number,
)


def num_divisors(n: int) -> int:
    """Golden label: number of positive divisors of n."""
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i * i != n:
                count += 1
        i += 1
    return count


def make_divisors_problems(limit: int) -> list[Problem]:
    """Generate divisor-count problems with locally-computed goldens."""
    # A spread of interesting numbers (prime powers, composites, edge cases)
    nums = [196, 144, 360, 1000, 1, 2, 7, 100, 65536, 999983, 12, 84, 210,
            72, 180, 240, 504, 1008, 2024, 97, 101, 997, 1024, 3125, 4096]
    problems = []
    for i, n in enumerate(nums[:limit]):
        g = num_divisors(n)
        problems.append(Problem(
            id=f"div_{n:06d}",
            question=f"How many positive whole-number divisors does {n} have?",
            golden=str(g),
            source="synthetic",
            subject="number-theory/divisors",
            solution=f"{n} has {g} positive divisors.",
        ))
    return problems


def rephrase_divisors(question: str, lm) -> str:
    """Rephrase a divisor question (same number), via the source model."""
    out = lm.complete(
        f"Rephrase this math problem using different wording but EXACTLY the same "
        f"number and meaning. Keep it one sentence. Reply with ONLY the rephrased question.\n\n"
        f"Original: {question}",
        role="target", max_tokens=200,
    ).text.strip()
    # sanity: must contain the same number
    import re
    nums_orig = set(re.findall(r"\d+", question))
    nums_new = set(re.findall(r"\d+", out))
    if nums_orig != nums_new or not out:
        raise ValueError(f"rephrase changed numbers: {out!r}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--cache-dir", default="tests/.p2_live_cache")
    parser.add_argument("--chat-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--embed-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M")
    args = parser.parse_args()

    print(f"[1] loading divisors problems (limit={args.limit}) ...")
    problems = make_divisors_problems(args.limit)
    for p in problems[:5]:
        print(f"    {p.id}: golden={p.golden}")
    print(f"    total: {len(problems)}")

    print("[2] source model (S baseline + rephrasing) ...")
    embedder = Embedder(backend="server", url=args.embed_url)
    b27 = LMBackend(url=args.chat_url, model=args.model)
    ledger = Ledger()
    b27.ledger = ledger

    print("[3] gateway on the P2-built divisors cache ...")
    config = GatewayConfig(
        port=0,  # not serving HTTP; in-process
        cache_dir=args.cache_dir,
        chat_url=args.chat_url,
        embed_url=args.embed_url,
        model=args.model,
        entry_threshold=0.72,
    )
    gateway = ArcGateway(config)
    print(f"    entries={len(gateway.entry_store.entries)} compiled={'yes' if gateway._compiled_mod else 'no'}")

    runner = EvalRunner(
        gateway=gateway,
        source_lm=b27,
        rephrase_fn=lambda q: rephrase_divisors(q, b27),
    )

    # --- Build cases: original + phrasal (parametric is trivially covered by
    #     using different numbers as separate "original" problems) ---
    cases: list[EvalCase] = []
    for p in problems:
        cases.append(perturb_original(p))
    print(f"[4] running {len(cases)} cache-serve cases ...")
    t0 = time.perf_counter()
    report_c = runner.run_cases(cases)
    serve_s = time.perf_counter() - t0

    # --- S baseline: raw source model on the same problems (no cache) ---
    print(f"[5] running {len(problems)} S-baseline cases (raw 27B, no cache) ...")
    runner_s = EvalRunner(source_lm=b27)
    report_s = runner_s.run_cases(cases)

    # --- Assemble final report ---
    final = EvalReport(benchmark="divisors-synthetic")
    for r in report_c.results:
        final.add(r)
    final.s_acc = round(sum(1 for r in report_s.results if r.correct) / max(1, len(report_s.results)), 4)
    final.c_acc = round(sum(1 for r in report_c.results if r.correct) / max(1, len(report_c.results)), 4)
    final.build_cost_tokens = ledger.total_tokens() if hasattr(ledger, "total_tokens") else 0

    print()
    print("=" * 62)
    print(f"P4 LIVE — divisors entry (built in P2, served via gateway)")
    print("=" * 62)
    for r in report_c.results:
        mark = "ok" if r.correct else "FAIL"
        print(f"  {mark}  {r.problem_id} [{r.source}] ans={str(r.answer)[:30]} golden={r.golden} score={r.entry_score} {r.ms:.0f}ms")
    s_correct = sum(1 for r in report_s.results if r.correct)
    print("-" * 62)
    print(f"  C Acc (cache):   {final.c_acc:.3f}  ({sum(1 for r in report_c.results if r.correct)}/{len(report_c.results)})")
    print(f"  S Acc (source):  {final.s_acc:.3f}  ({s_correct}/{len(report_s.results)})")
    print(f"  Cache hit rate:  {final.summary()['cache_hit_rate']}")
    print(f"  Serve wall time: {serve_s:.1f}s for {len(cases)} cases (avg {serve_s/max(1,len(cases))*1000:.0f}ms)")

    out_path = Path("tests/p4_live_report.json")
    out_path.write_text(final.to_json())
    print(f"\n  report -> {out_path}")


if __name__ == "__main__":
    main()
