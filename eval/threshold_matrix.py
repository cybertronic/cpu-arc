"""Cross-family threshold calibration + confusion-matrix eval.

Scores a matrix of queries (per-family full/short phrasings + cross-family +
off-topic) against EVERY family's entry store and reports:

  * per-family in-family score landscape (min/p10/med/max, by style)
  * per-family max cross-family score (the false friend, with entry)
  * confusion matrix at a given threshold
  * per-family threshold sweep: in-family hit rate vs cross-family false hits
  * recommended per-family thresholds + margins

Usage:
  python3 eval/threshold_matrix.py \
      --store divisors=tests/.p2_live_cache \
      --store gcd=tests/.gcd_cache \
      --store lcm=tests/.lcm_cache \
      --embed-url http://127.0.0.1:8081 \
      --threshold 0.72 --out tests/threshold_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc import Embedder
from arc.registry import EntryStore

# --------------------------------------------------------------------------- #
# Query matrix
# --------------------------------------------------------------------------- #

DIV_N = [196, 144, 12, 84, 360, 1000, 65536, 999983, 210, 720, 48, 180]
GCD2 = [(24, 36), (48, 180), (105, 175), (84, 126), (72, 96), (15, 25),
        (100, 75), (60, 40)]
GCD3 = [(63, 147, 105), (120, 80, 60), (252, 180, 330),
        (49, 147, 245), (1000, 750, 500), (91, 143, 187)]
LCM2 = [(48, 180), (24, 36), (12, 18), (10, 15), (72, 96), (100, 75),
        (60, 40), (14, 21)]

OFFTOPIC = [
    "Explain the difference between prime and composite numbers in one paragraph.",
    "What is Euler's theorem in number theory?",
    "In what year did the Apollo 11 moon landing take place?",
]


def build_queries() -> list[dict]:
    qs: list[dict] = []
    for n in DIV_N:
        qs.append(dict(family="divisors", style="full",
                       q=f"How many positive whole-number divisors does {n} have?"))
        qs.append(dict(family="divisors", style="short",
                       q=f"How many divisors does {n} have?"))
    for a, b in GCD2:
        qs.append(dict(family="gcd", style="two-int",
                       q=f"What is the greatest common divisor of {a} and {b}?"))
    for a, b, c in GCD3:
        qs.append(dict(family="gcd", style="three-int",
                       q=f"What is the greatest common divisor of {a}, {b}, and {c}?"))
    for a, b in LCM2:
        qs.append(dict(family="lcm", style="full",
                       q=f"What is the least common multiple of {a} and {b}?"))
        qs.append(dict(family="lcm", style="short",
                       q=f"What is the LCM of {a} and {b}?"))
    for q in OFFTOPIC:
        qs.append(dict(family="off-topic", style="n/a", q=q))
    return qs


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="append", default=[],
                    help="name=path to an entries.db parent dir (repeatable)")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8081")
    ap.add_argument("--threshold", type=float, default=0.72,
                    help="threshold for the confusion matrix (current global)")
    ap.add_argument("--out", default="tests/threshold_report.json")
    args = ap.parse_args()
    if not args.store:
        ap.error("at least one --store required")

    t0 = time.perf_counter()
    embedder = Embedder(backend="server", url=args.embed_url)
    stores: dict[str, EntryStore] = {}
    for spec in args.store:
        name, path = spec.split("=", 1)
        stores[name] = EntryStore(embedder, path=Path(path) / "entries.db")
        fams = {e.get("family") for e in stores[name].entries}
        print(f"loaded {name}: {len(stores[name].entries)} entries (families={fams})")

    queries = build_queries()
    t_embed0 = time.perf_counter()

    # --- score every query against every store ----------------------------- #
    rows = []
    for item in queries:
        scores = {}
        for name, store in stores.items():
            entry, score = store.search(item["q"])
            scores[name] = {
                "score": round(score, 4),
                "entry": entry["description"][:70] if entry else None,
                "entry_family": entry.get("family") if entry else None,
            }
        best_name = max(scores, key=lambda n: scores[n]["score"])
        rows.append({
            "family": item["family"], "style": item["style"],
            "q": item["q"], "scores": scores, "top1": best_name,
            "top1_score": scores[best_name]["score"],
        })

    n_embeds = (time.perf_counter() - t_embed0)
    print(f"\n scored {len(rows)} queries x {len(stores)} stores in {n_embeds:.0f}s")

    # --- per-family landscape ---------------------------------------------- #
    names = list(stores)
    landscape = {}
    for fam in names:
        in_full = [r["scores"][fam]["score"] for r in rows
                   if r["family"] == fam and r["style"] == "full"]
        in_short = [r["scores"][fam]["score"] for r in rows
                    if r["family"] == fam and r["style"] == "short"]
        in_all = [r["scores"][fam]["score"] for r in rows if r["family"] == fam]
        cross = [(r["scores"][fam]["score"], r) for r in rows if r["family"] != fam]
        cross_max = max(cross) if cross else (0.0, None)
        landscape[fam] = {
            "in_family_all": _dist(in_all),
            "in_family_full": _dist(in_full),
            "in_family_short": _dist(in_short),
            "max_cross_score": cross_max[0],
            "max_cross_from": (f"{cross_max[1]['family']}/{cross_max[1]['style']}: "
                               f"{cross_max[1]['q'][:50]}") if cross_max[1] else None,
        }

    # --- confusion matrix at the given threshold ---------------------------- #
    T = args.threshold
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["family"] == "off-topic":
            continue
        decision = (r["top1"] if r["top1_score"] >= T else "miss")
        confusion.setdefault(r["family"], {}).setdefault(decision, 0)
        confusion[r["family"]][decision] += 1

    # --- per-family threshold sweep ----------------------------------------- #
    sweep = {}
    for fam in names:
        in_scores = [r["scores"][fam]["score"] for r in rows if r["family"] == fam]
        in_full = [r["scores"][fam]["score"] for r in rows
                   if r["family"] == fam and r["style"] == "full"]
        cross_scores = [r["scores"][fam]["score"] for r in rows
                        if r["family"] != "off-topic" and r["family"] != fam]
        off_scores = [r["scores"][fam]["score"] for r in rows
                      if r["family"] == "off-topic"]
        # Capture target: full-style queries when present, else all in-family
        capture = in_full if in_full else in_scores
        best = None
        out = []
        for T100 in range(300, 951, 5):
            t = T100 / 1000.0
            hits_full = sum(s >= t for s in capture)
            hits_all = sum(s >= t for s in in_scores)
            false_cross = sum(s >= t for s in cross_scores)
            false_off = sum(s >= t for s in off_scores)
            rec = (t, hits_full, len(capture), false_cross, false_off)
            out.append(rec)
            if capture and hits_full == len(capture) and false_cross == 0 \
                    and false_off == 0:
                best = t  # keep raising while perfect
        sweep[fam] = {
            "perfect_max_threshold": best,
            "capture_style": "full" if in_full else "all-in-family",
            "min_capture": min(capture) if capture else None,
            "min_full": min(in_full) if in_full else None,
            "max_cross": max(cross_scores) if cross_scores else None,
            "max_offtopic": max(off_scores) if off_scores else None,
        }

    # --- print --------------------------------------------------------------- #
    print(f"\n=== score landscape (top-1 vs each family) ===")
    for fam, d in landscape.items():
        ia, ifu, ish = d["in_family_all"], d["in_family_full"], d["in_family_short"]
        print(f"\n[{fam}]")
        print(f"  in-family  : min={ia['min']:.3f} p10={ia['p10']:.3f} "
              f"med={ia['med']:.3f} max={ia['max']:.3f}")
        if ifu["n"]:
            print(f"  full style : min={ifu['min']:.3f} med={ifu['med']:.3f} "
                  f"(n={ifu['n']})")
        if ish["n"]:
            print(f"  short style: min={ish['min']:.3f} med={ish['med']:.3f} "
                  f"(n={ish['n']})")
        print(f"  max cross  : {d['max_cross_score']:.3f}  <- {d['max_cross_from']}")

    print(f"\n=== confusion matrix at T={T} (rows=query family) ===")
    cols = names + ["miss"]
    print(f"  {'family':12s} " + "".join(f"{c:>10s}" for c in cols))
    for fam, counts in confusion.items():
        print(f"  {fam:12s} " +
              "".join(f"{counts.get(c, 0):>10d}" for c in cols))

    print(f"\n=== per-family threshold recommendation ===")
    for fam, s in sweep.items():
        t = s["perfect_max_threshold"]
        mc = s["min_capture"] if s["min_capture"] is not None else 0.0
        if t is None:
            print(f"  {fam:12s}: NO threshold separates perfectly; "
                  f"min_capture({s['capture_style']})={mc:.3f} "
                  f"max_cross={s['max_cross']:.3f} max_off={s['max_offtopic']:.3f}")
        else:
            print(f"  {fam:12s}: T* = {t:.3f}  (capture={s['capture_style']}, "
                  f"min={mc:.3f} margin={mc-t:+.3f}, max_cross={s['max_cross']:.3f} "
                  f"gap={t-s['max_cross']:+.3f}, max_off={s['max_offtopic']:.3f} "
                  f"gap={t-s['max_offtopic']:+.3f})")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stores": {n: len(s.entries) for n, s in stores.items()},
        "global_threshold": T,
        "landscape": landscape,
        "confusion_at_T": confusion,
        "sweep": sweep,
        "queries": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out}  ({(time.perf_counter()-t0):.0f}s total)")


def _dist(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": min(xs), "p10": pct(xs, 0.10),
            "med": statistics.median(xs), "max": max(xs)}


if __name__ == "__main__":
    main()
