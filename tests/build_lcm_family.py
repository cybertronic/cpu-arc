"""Build an LCM cache family (tests/.lcm_cache) with the real 27B.

LCM is the classic GCD false-friend: same numbers, different operation —
the most dangerous cross-family confusion for threshold calibration.

Run: python3 tests/build_lcm_family.py  (~1-2 min of 27B inference)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc import ARC, Thresholds, Embedder, LMBackend, Ledger
from arc.builder import Builder
from arc.compiler import Compiler

SRC_SYSTEM = ("You are a careful mathematician. Solve the problem with clear "
              "step-by-step reasoning. End with 'Answer: <value>'.")

SEED = "What is the least common multiple of 48 and 180?"  # = 720


def main() -> None:
    chat_url = "http://127.0.0.1:8080/v1"
    embed_url = "http://127.0.0.1:8081"
    model = "unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M"

    embedder = Embedder(backend="server", url=embed_url)
    b27 = LMBackend(url=chat_url, model=model)
    ledger = Ledger()
    b27.ledger = ledger

    print("[1] source model solves the LCM seed ...")
    t0 = time.perf_counter()
    transcript = b27.complete(SEED, system=SRC_SYSTEM, role="source",
                              max_tokens=1500).text
    ans = [l for l in transcript.splitlines() if "Answer" in l]
    print(f"    done in {time.perf_counter()-t0:.0f}s")
    print(f"    answer line: {ans[-1][:100] if ans else '(none)'}")

    print("[2] builder 5-phase construction -> fresh LCM cache ...")
    lcm_dir = "tests/.lcm_cache"
    Path(lcm_dir).mkdir(exist_ok=True)
    arc_lcm = ARC(embedder, b27, Thresholds(entry=0.72, function=0.55, load=0.45),
                  db_path=lcm_dir, ledger=ledger)
    builder = Builder(arc_lcm, builder_lm=b27, judge_lm=b27, max_rewrites=3,
                      roundtrips=2)
    t0 = time.perf_counter()
    result = builder.build(SEED, transcript, family="lcm")
    build_s = time.perf_counter() - t0
    print(f"    ok={result.ok} in {build_s:.0f}s rewrites={result.rewrites}")
    if not result.ok:
        print(f"    errors: {result.errors[:3]}")
        sys.exit(1)
    print(f"    template: {result.template}")
    print(f"    params: {[f'{p.description}={p.value}' for p in result.params]}")

    print("[3] compiling LCM cache ...")
    compiler = Compiler(arc_lcm.registry, arc_lcm.entry_store)
    crec = compiler.compile_entry()
    compiler.write(crec, Path(lcm_dir) / "compiled")
    print(f"    fully_static={crec.manifest['stats']['fully_static']} all_resolved={crec.all_resolved} -> {lcm_dir}/compiled/arc_compiled.py")
    print(f"    ledger: {ledger.snapshot()}")
    print("LCM family build complete.")


if __name__ == "__main__":
    main()
