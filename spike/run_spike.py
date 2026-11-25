"""P0 spike: hand-coded ideal-gas-law ARC entry (paper Appendix A) + perturbation tests.

Run: python3 run_spike.py [--embedder hash|server]
"""
from __future__ import annotations

import argparse
import sys

from arc_runtime import ARC, Context, LMBackend, Embedder, CacheInvocationError, load, do, narrate, lm, extract


# --------------------------------------------------------------------------- #
# Semantic functions (hand-coded from paper Appendix A, Figures 5-6)
# --------------------------------------------------------------------------- #
def fn_convert_temperature(ctx: Context) -> dict:
    """Convert a temperature value from Celsius or Kelvin to Kelvin."""
    temp_with_unit = load(ctx, "the temperature value with unit", d_type=str)
    narrate(ctx, f"Received temperature: {temp_with_unit}")

    temp_value = extract(ctx, temp_with_unit, "the numeric temperature value", d_type=float)
    narrate(ctx, f"Numeric value: {temp_value}")

    temp_unit = lm(
        ctx,
        f"What is the temperature unit in '{temp_with_unit}'? Reply with exactly one of: C, K",
    ).strip().upper()
    narrate(ctx, f"Detected unit: {temp_unit}")

    if "C" in temp_unit:
        temp_kelvin = temp_value + 273.15
        narrate(ctx, f"{temp_value} C + 273.15 = {temp_kelvin:.2f} K")
    else:
        temp_kelvin = temp_value
        narrate(ctx, f"Already in Kelvin: {temp_kelvin:.2f} K")

    return {"the temperature in Kelvin": temp_kelvin}


def fn_convert_pressure(ctx: Context) -> dict:
    """Convert a pressure value from kPa to MPa, bar, and atm."""
    p = load(ctx, "the pressure value in kilopascals (kPa)", d_type=float)
    narrate(ctx, f"Converting {p:.2f} kPa to other units")
    return {
        "the pressure in kPa": p,
        "the pressure in MPa": p / 1000.0,
        "the pressure in bar": p / 100.0,
        "the pressure in atm": p / 101.325,
    }


def fn_ideal_gas_pressure(ctx: Context) -> dict:
    """Root: calculate gas pressure using PV = nRT rearranged to P = nRT/V."""
    molar_str = load(ctx, "the molar amount of gas with unit", d_type=str)
    volume_str = load(ctx, "the volume of the container with unit", d_type=str)
    temp_str = load(ctx, "the temperature with unit", d_type=str)
    gas_name = load(ctx, "the common name of the gas", d_type=str)
    narrate(ctx, f"Calculating pressure for {gas_name} using the ideal gas law")
    narrate(ctx, f"Given: n = {molar_str}, V = {volume_str}, T = {temp_str}")

    n = extract(ctx, molar_str, "the numeric molar amount", d_type=float)
    v = extract(ctx, volume_str, "the numeric volume value", d_type=float)
    narrate(ctx, f"Extracted: n = {n} mol, V = {v} dm^3")

    temp_result = do(
        ctx,
        "Convert a temperature value from Celsius or Kelvin to Kelvin given a temperature string with unit",
        payloads={"the temperature value with unit": temp_str},
    )
    T = load(temp_result, "the temperature in Kelvin", d_type=float)

    R = 8.314  # kPa.dm^3.mol^-1.K^-1
    p_kpa = n * R * T / v
    narrate(ctx, f"P = nRT/V = {n} * {R} * {T:.2f} / {v} = {p_kpa:.2f} kPa")

    pres_result = do(
        ctx,
        "Convert a pressure value from kPa to multiple common units including MPa, bar, and atm",
        payloads={"the pressure value in kilopascals (kPa)": p_kpa},
    )
    p_mpa = load(pres_result, "the pressure in MPa", d_type=float)
    p_bar = load(pres_result, "the pressure in bar", d_type=float)
    p_atm = load(pres_result, "the pressure in atm", d_type=float)

    return {
        "the calculated pressure in kPa": round(p_kpa, 2),
        "the calculated pressure in MPa": round(p_mpa, 3),
        "the pressure in bar": round(p_bar, 2),
        "the pressure in atm": round(p_atm, 2),
        "summary of the calculation": (
            f"The pressure exerted by {n} mol of {gas_name} in {v} dm^3 at {T:.1f} K "
            f"is {p_mpa:.2f} MPa (or {p_kpa:.0f} kPa)."
        ),
    }


# --------------------------------------------------------------------------- #
# Test cases: original + phrasal/parametric/combined perturbations
# --------------------------------------------------------------------------- #
CASES = [
    {
        "id": "original",
        "text": ("Suppose that 10.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. "
                 "Predict the pressure exerted by the ethane from the perfect gas."),
        "expected_kpa": 5132.0,  # n=10.0 V=4.860 T=300.15K -> 5133.9 (paper rounds 27C->300K)
    },
    {
        "id": "phrasal-1",
        "text": ("A sample of ethane gas, amounting to 10.0 mol, is held in a rigid "
                 "container of volume 4.860 dm3 with its temperature at 27 degrees Celsius. "
                 "Using the ideal gas law, what pressure does it exert?"),
        "expected_kpa": 5132.0,
    },
    {
        "id": "phrasal-2",
        "text": ("Ten moles of ethane are sealed in a 4.860 dm3 vessel at 27 C. "
                 "Calculate the pressure assuming perfect gas behavior."),
        "expected_kpa": 5132.0,
    },
    {
        "id": "parametric-1",
        "text": ("Suppose that 5.0 mol C2H6(g) is confined to 4.860 dm3 at 27°C. "
                 "Predict the pressure exerted by the ethane from the perfect gas."),
        "expected_kpa": 2567.0,
    },
    {
        "id": "combined-1",
        "text": ("A sample of methane gas, amounting to 2.0 mol, is held in a rigid "
                 "container of volume 8.0 dm3 with its temperature at 100 degrees Celsius. "
                 "Using the ideal gas law, what pressure does it exert?"),
        "expected_kpa": 694.5,  # n=2 V=8 T=373.15 -> 2*8.314*373.15/8 = 775.6... recompute: 6180.9/8=772.6
    },
]

# fix expected for combined-1: P = 2*8.314*373.15/8 = 6181.0/8 = 772.6 kPa
CASES[4]["expected_kpa"] = 772.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hash", choices=["hash", "server"])
    ap.add_argument("--embed-url", default="http://127.0.0.1:8081")
    ap.add_argument("--lm-url", default="http://127.0.0.1:8080/v1")
    args = ap.parse_args()

    embedder = Embedder(backend=args.embedder, url=args.embed_url)
    lm_backend = LMBackend(url=args.lm_url)

    arc = ARC(embedder=lm_backend and embedder, lm_backend=lm_backend, threshold=0.35)
    params = [
        "the molar amount of gas with unit",
        "the volume of the container with unit",
        "the temperature with unit",
        "the common name of the gas",
    ]
    arc.registry.register(
        "Convert a temperature value from Celsius or Kelvin to Kelvin given a temperature string with unit",
        fn_convert_temperature,
    )
    arc.registry.register(
        "Convert a pressure value from kPa to multiple common units including MPa, bar, and atm",
        fn_convert_pressure,
    )
    arc.register_entry(CASES[0]["text"], fn_ideal_gas_pressure,
                       "Calculate the pressure exerted by a gas using PV = nRT rearranged to P = nRT/V",
                       params)

    passed = 0
    for case in CASES:
        print(f"\n=== {case['id']} ===")
        try:
            out = arc.use(case["text"])
            if out is False:
                print("CACHE MISS (entry-point below threshold)")
                continue
            res = out["result"]
            p = res.get("the calculated pressure in kPa")
            ok = p is not None and abs(p - case["expected_kpa"]) / case["expected_kpa"] < 0.02
            passed += bool(ok)
            print(f"  P = {p} kPa (expected ~{case['expected_kpa']}) -> {'PASS' if ok else 'FAIL'}")
            print(f"  summary: {res.get('summary of the calculation')}")
            print(f"  entry_score={out['entry_score']:.3f} lm_calls={out['lm_stats']['calls']} "
                  f"in_tok={out['lm_stats']['in_tokens']} out_tok={out['lm_stats']['out_tokens']}")
            print("  narration:")
            for line in out["narration"]:
                print(f"    {line}")
        except CacheInvocationError as e:
            print(f"  INVOCATION ERROR: {e}")

    print(f"\nRESULT: {passed}/{len(CASES)} passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
