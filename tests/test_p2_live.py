"""P2 live demo: real-model 5-phase builder, end to end.

Flow (all against system servers; 27B plays S and B+J, extraction at serve time):
  1. Source solve: 27B answers a fresh problem class (divisors of N) with full
     reasoning -> transcript
  2. Builder (27B + 27B judge) runs the 5 phases -> registers entry in a NEW cache
  3. Fresh ARC reloaded from disk serves: original, rephrased, reparameterized,
     and an unrelated question (must miss)
  4. Cost report: build $-equivalent vs per-invocation (paper's Build x / Savings x)

Run: python3 tests/test_p2_live.py   (takes a while: ~10-20 min of 27B inference)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc import ARC, Thresholds, Embedder, LMBackend, Ledger
from arc.builder import Builder

QUERY = "How many positive whole-number divisors does 196 have?"

SRC_SYSTEM = ("You are a careful mathematician. Solve the problem with clear step-by-step "
              "reasoning. End with 'Answer: <value>'.")


def main():
    embedder = Embedder(backend="server", url="http://127.0.0.1:8081")
    b27 = LMBackend(url="http://127.0.0.1:8080/v1",
                    model="unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M")

    cache_dir = Path("tests/.p2_live_cache")
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)

    ledger = Ledger()
    b27.ledger = ledger
    arc = ARC(embedder, b27,
              Thresholds(entry=0.72, function=0.55, load=0.45),
              db_path=cache_dir, ledger=ledger,
              invocation_log=Path("tests/.p2_live_invocations.jsonl"))
    builder = Builder(arc, builder_lm=b27, judge_lm=b27,
                      max_rewrites=3, roundtrips=2)

    print("[1] source solve (27B as S) ...", flush=True)
    t0 = time.time()
    transcript = b27.complete(QUERY, system=SRC_SYSTEM,
                              role="source", max_tokens=1500).text
    src_ms = time.time() - t0
    print(f"    done in {src_ms:.0f}s")
    print("    " + transcript.replace("\n", "\n    ")[:600])

    print(f"[2] builder 5-phase construction (27B as B, 27B as J) ...", flush=True)
    t0 = time.time()
    res = builder.build(QUERY, transcript, golden=None)
    build_ms = time.time() - t0
    print(f"    ok={res.ok} ms={build_ms:.0f} rewrites={res.rewrites}")
    print(f"    template: {res.template!r}")
    print(f"    params: {[p.description + '=' + p.value for p in res.params]}")
    for r in res.roundtrips:
        print(f"    round {r['round']}: {r['result']} pass={r['pass']} ({r['critique']})")
    if res.errors:
        print(f"    errors: {res.errors}")
    print(f"    build ledger roles: "
          f"{ {k: v['calls'] for k, v in ledger.snapshot()['by_role'].items()} }")
    if not res.ok:
        print("BUILD NOT OK — aborting (see errors above)")
        return 1

    print(f"[3] fresh ARC from disk -> serve 4 queries", flush=True)
    fresh = ARC(embedder,
                LMBackend(url="http://127.0.0.1:8080/v1",
                          model="unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M"),
                Thresholds(entry=0.72, function=0.55, load=0.45),
                db_path=cache_dir, ledger=Ledger())
    cases = [
        ("original", QUERY, 9),
        ("rephrased", "Find how many positive integers evenly divide 196.", 9),
        ("reparameterized", "How many positive whole-number divisors does 144 have?", 15),
        ("unrelated (should miss)", "How many primes are there less than 100?", None),
    ]
    passed = 0
    for name, q, expected in cases:
        t0 = time.time()
        try:
            out = fresh.use(q)
        except Exception as e:
            out = f"EXC: {e}"
        ms = (time.time() - t0) * 1000
        if expected is None:
            ok = out is False or (isinstance(out, str) and out.startswith("EXC"))
            passed += bool(ok)
            print(f"  {'ok ' if ok else 'FAIL'} {name}: {out if out is False else 'unexpected hit'} "
                  f"({ms:.0f}ms)")
            continue
        if out is False or isinstance(out, str):
            print(f"  FAIL {name}: {out}")
            continue
        ans = None
        for k, v in out["result"].items():
            if isinstance(v, int):
                ans = v
                break
        ok = ans == expected
        passed += bool(ok)
        print(f"  {'ok ' if ok else 'FAIL'} {name}: answer={ans} (exp {expected}) "
              f"score={out['entry_score']:.3f} {ms:.0f}ms")
        if ok:
            for line in out["narration"]:
                print(f"        {line}")

    print(f"\nLIVE P2 RESULT: {passed}/{len(cases)} serve cases passed, "
          f"build_ok={res.ok}")
    print(f"build wall time: {build_ms / 60:.1f} min")
    print(f"total ledger (S + B + J + serve): {ledger.snapshot()}")
    return 0 if passed == len(cases) and res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
