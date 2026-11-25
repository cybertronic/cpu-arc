"""P1 offline tests: MockLM + hash embedder. Zero GPU required.

Covers: dispatch/scoping, extraction + smart_convert fallback, miss fall-through,
persistence round-trip, sandbox rejection, per-role cost ledger, per-scope
thresholds + exact-match fast path.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc import (ARC, serve, Thresholds, CacheInvocationError,
                 Embedder, MockLM, Ledger, exec_function_source, SandboxError)

PASS, FAIL = 0, []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name} {detail}")


GAS_ROOT = '''
def compute(ctx):
    n = load(ctx, "the molar amount of gas with unit", d_type=float)
    v = load(ctx, "the volume of the container with unit", d_type=float)
    t = load(ctx, "the temperature with unit", d_type=float)
    narrate(ctx, f"P = {n}*8.314*{t+273.15}/{v}")
    p = n * 8.314 * (t + 273.15) / v
    return {"the pressure in kPa": p}
'''

GAS_PARAMS = ["the molar amount of gas with unit",
              "the volume of the container with unit",
              "the temperature with unit"]

MOCK = MockLM(routes={
    "the molar amount": "10.0 mol",
    "the volume": "4.860 dm3",
    "the temperature": "27 degrees Celsius",
    "molar amount of gas.*look carefully": "10.0 mol",  # hinted retry
    "numeric temperature value": "27.0",
    "temperature unit": "C",
})


def make_arc(tmp, db=True):
    embedder = Embedder(backend="hash")
    ledger = Ledger()
    arc = ARC(embedder, MOCK, Thresholds(entry=0.3, function=0.3, load=0.3),
              db_path=tmp / "cache" if db else None, ledger=ledger,
              invocation_log=tmp / "invocations.jsonl")
    arc.register_entry(
        "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27C. Predict the pressure.",
        GAS_ROOT, "Calculate gas pressure using PV = nRT rearranged to P = nRT/V",
        GAS_PARAMS)
    return arc


def main():
    tmp = Path(tempfile.mkdtemp(prefix="arc_p1_"))
    print(f"[1] dispatch + extraction + narration")
    arc = make_arc(tmp)
    out = arc.use("Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27C. Predict the pressure.")
    p = out["result"]["the pressure in kPa"]
    check("correct answer", abs(p - 10.0 * 8.314 * 300.15 / 4.860) < 1e-6, f"p={p}")
    check("narration recorded", any("P =" in n for n in out["narration"]))
    check("entry score high", out["entry_score"] > 0.8, f"score={out['entry_score']:.3f}")

    print(f"[2] per-role cost ledger")
    roles = out["lm_stats"]["by_role"]
    check("extractor role tracked", roles.get("extractor", {}).get("calls", 0) == 3, f"roles={roles}")
    check("ledger total tokens > 0", out["lm_stats"]["total_tokens"] > 0)

    print(f"[3] miss -> fall-through to source LLM")
    r = serve(arc, "What is the capital of France?", source_llm=lambda q: "Paris")
    check("fallback used", r["source"] == "llm-fallback" and r["answer"] == "Paris")
    check("miss recorded", True)

    print(f"[4] kill-switch: cache error -> fall-through")
    # query matching the entry but with missing param -> NOT_FOUND -> retry -> still fail
    mock2 = MockLM(routes={"the molar amount": "10.0 mol"}, default="NOT_FOUND")
    arc2 = ARC(Embedder(backend="hash"), mock2, Thresholds(entry=0.0, function=0.0, load=0.0),
               ledger=Ledger())
    arc2.register_entry("Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27C.",
                        GAS_ROOT, "Calculate gas pressure using PV = nRT rearranged to P = nRT/V",
                        GAS_PARAMS)
    r = serve(arc2, "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27C.",
              source_llm=lambda q: "fallback-answer")
    check("error fell through", r["source"] == "llm-fallback" and r["answer"] == "fallback-answer")
    check("cache_error captured", "cache_error" in r)

    print(f"[5] persistence round-trip")
    (tmp / "cache").mkdir(parents=True, exist_ok=True)
    arc3 = make_arc(tmp, db=True)
    arc3b = ARC(Embedder(backend="hash"), MockLM(routes=MOCK.routes, default="NOT_FOUND"),
                Thresholds(entry=0.3, function=0.3, load=0.3), db_path=tmp / "cache")
    check("function restored from source", len(arc3b.registry.functions) >= 1,
          f"n={len(arc3b.registry.functions)}")
    check("entry restored", len(arc3b.entry_store.entries) >= 1)

    print(f"[6] sandbox: forbidden constructs rejected")
    for bad, label in [
        ("import os\ndef f(ctx):\n    return {}\n", "import os"),
        ("def f(ctx):\n    return eval('1+1')\n", "eval()"),
        ("def f(ctx):\n    return ctx.__class__\n", "dunder access"),
        ("x = 1\n", "no function def"),
    ]:
        try:
            exec_function_source(bad)
            check(f"reject {label}", False, "no error raised")
        except SandboxError:
            check(f"reject {label}", True)
    good = exec_function_source("import math\ndef f(ctx):\n    return {'v': math.pi}\n")
    check("allowlist import (math) works", good(None)["v"] > 3.14)

    print(f"[7] smart_convert fallback tier")
    from arc.context import Context
    from arc.primitives import bind, smart_convert
    ctx = Context(Embedder(backend="hash"), Thresholds())
    bind(Embedder(backend="hash"), arc.registry, MockLM(routes={"Python literal": "42.5"}))
    v = smart_convert(ctx, "about forty-two and a half", float)
    check("LLM rewrite coerced to float", v == 42.5, f"v={v}")

    print(f"[8] invocation log persisted")
    lines = (tmp / "invocations.jsonl").read_text().strip().splitlines()
    rec = json.loads(lines[-1])
    check("log line valid json with fields", {"ts", "request", "hit", "entry_score"} <= set(rec))

    print(f"[9] exact-match fast path beats weak vector separation")
    from arc.context import Context as C2
    c2 = C2(Embedder(backend="hash"), Thresholds())
    c2.layers[0].add("the molar amount of gas with unit", "10.0 mol", c2.embedder)
    c2.layers[0].add("the volume of the container with unit", "4.860 dm3", c2.embedder)
    v1 = c2.find("the molar amount of gas with unit", scope="load")
    v2 = c2.find("the volume of the container with unit", scope="load")
    check("exact fast path molar", v1 == "10.0 mol", f"v1={v1!r}")
    check("exact fast path volume", v2 == "4.860 dm3", f"v2={v2!r}")
    check("load_log records via=exact",
          all(e["via"] == "exact" for e in c2.load_log[:2]), f"log={c2.load_log}")

    print(f"\nRESULT: {PASS} passed, {len(FAIL)} failed {FAIL if FAIL else ''}")
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
