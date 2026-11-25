"""P1 live test: P0 gas-law cases through the hardened package.

Uses real backends: EmbeddingGemma (:8081) + system llama-server (:8080, Qwen3.8-27B).
Run: python3 tests/test_p1_live.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc import ARC, serve, Thresholds, Embedder, LMBackend, Ledger

CASES = [
    ("original",
     "Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. "
     "Predict the pressure exerted by the ethane from the perfect gas.",
     5132.0),
    ("phrasal-1",
     "A sample of ethane gas, amounting to 10.0 mol, is held in a rigid "
     "container of volume 4.860 dm3 with its temperature at 27 degrees Celsius. "
     "Using the ideal gas law, what pressure does it exert?",
     5132.0),
    ("phrasal-2",
     "Ten moles of ethane are sealed in a 4.860 dm3 vessel at 27 C. "
     "Calculate the pressure assuming perfect gas behavior.",
     5132.0),
    ("parametric-1",
     "Suppose that 5.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. "
     "Predict the pressure exerted by the ethane from the perfect gas.",
     2567.0),
    ("combined-1",
     "A sample of methane gas, amounting to 2.0 mol, is held in a rigid "
     "container of volume 8.0 dm3 with its temperature at 100 degrees Celsius. "
     "Using the ideal gas law, what pressure does it exert?",
     772.6),
    ("unrelated (should miss)",
     "How many positive whole-number divisors does 196 have?",
     None),
]

GAS_ROOT = '''
def compute_ideal_gas_pressure(ctx):
    """Calculate the pressure exerted by a gas using PV = nRT rearranged to P = nRT/V."""
    n = load(ctx, "the molar amount of gas with unit", d_type=float)
    v = load(ctx, "the volume of the container with unit", d_type=float)
    t = load(ctx, "the temperature with unit", d_type=float)
    gas = load(ctx, "the common name of the gas", d_type=str)
    narrate(ctx, f"Calculating pressure for {gas} using the ideal gas law")
    narrate(ctx, f"Given: n = {n} mol, V = {v} dm^3, T = {t} C")
    p = n * 8.314 * (t + 273.15) / v
    narrate(ctx, f"P = nRT/V = {n} * 8.314 * {t + 273.15:.2f} / {v} = {p:.2f} kPa")
    return {
        "the calculated pressure in kPa": round(p, 2),
        "the calculated pressure in MPa": round(p / 1000, 3),
        "summary of the calculation": f"{n} mol of {gas} in {v} dm^3 at {t + 273.15:.1f} K -> {p:.0f} kPa",
    }
'''

PARAMS = ["the molar amount of gas with unit",
          "the volume of the container with unit",
          "the temperature with unit",
          "the common name of the gas"]

ENTRY = CASES[0][1]


def main():
    embedder = Embedder(backend="server", url="http://127.0.0.1:8081")
    lm = LMBackend(url="http://127.0.0.1:8080/v1", model="unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M")
    arc = ARC(embedder, lm,
              Thresholds(entry=0.60, function=0.55, load=0.45),
              db_path=Path("tests/.live_cache"),
              ledger=Ledger(),
              invocation_log=Path("tests/.live_invocations.jsonl"))
    arc.register_entry(ENTRY, GAS_ROOT,
                       "Calculate the pressure exerted by a gas using PV = nRT rearranged to P = nRT/V",
                       PARAMS)

    passed = 0
    for name, text, expected in CASES:
        t0 = time.time()
        try:
            out = arc.use(text)
        except Exception as e:
            out = False
            err = f"EXC: {e}"
        ms = (time.time() - t0) * 1000
        if out is False:
            if expected is None:
                print(f"  ok  {name}: correct MISS")
                passed += 1
            else:
                print(f"  FAIL {name}: unexpected miss")
            continue
        p = out["result"].get("the calculated pressure in kPa")
        if expected is None:
            print(f"  FAIL {name}: unexpected HIT score={out['entry_score']:.3f} p={p}")
            continue
        ok = p is not None and abs(p - expected) / expected < 0.02
        passed += bool(ok)
        print(f"  {'ok ' if ok else 'FAIL'} {name}: P={p} kPa (exp ~{expected}) "
              f"score={out['entry_score']:.3f} {ms:.0f}ms "
              f"tok_in={out['lm_stats']['by_role'].get('extractor', {}).get('in', 0)}")
        if ok:
            for line in out["narration"]:
                print(f"        {line}")

    print(f"\nLIVE RESULT: {passed}/{len(CASES)} passed")
    print("ledger:", arc.ledger.snapshot())
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
