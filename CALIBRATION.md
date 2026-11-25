# Cross-Family Threshold Calibration — Findings (2026-11-24)

Tooling: `eval/threshold_matrix.py` (score landscape + confusion matrix +
per-family threshold sweep) · Data: `tests/threshold_report.json` (57 queries
× 3 families, incl. live A/B section).

## Families evaluated (real builds, 27B)

| family   | entries | notes                                    |
|----------|---------|------------------------------------------|
| divisors | 2       | concrete 196 + template, P2              |
| gcd      | 4       | 3-int concrete + template (fn_0001/0003), P4 multi |
| lcm      | 2       | concrete 48/180 + template, built 2026-11-24 (299s, 0 rewrites) |

## Score landscape (top-1 cosine vs each family)

- **divisors**: in-family min 0.693 / full-style min 0.729 / med 0.751;
  max cross-family 0.683 (from a gcd two-int query).
  → **T\* = 0.725** captures all full-style queries; 0.690 would also capture
  short-style with zero cross hits (gap to max cross 0.010).
- **gcd**: in-family 0.579–0.733, wide spread. The spread is a *coverage* gap,
  not a threshold gap: 3-int queries matching the 3-int template score
  0.72–0.733, but two-int queries (no two-int entry exists) only reach
  0.579–0.65. No threshold separates gcd from divisors-short (max cross 0.639).
- **lcm**: in-family min 0.651 / full min 0.695; **max cross 0.922** — a GCD
  query sharing the concrete entry's exact numbers.

## The false friend (headline)

`What is the greatest common divisor of 48 and 180?` scores **0.922** against
the LCM concrete entry (`...least common multiple of 48 and 180?`) vs 0.606
against its own GCD entries. When families share one cache, that GCD query is
served by LCM code: **720 returned for gcd(48,180)=12**, confidently, in 100ms.
Same numbers dominate the embedding; the operation word doesn't separate.

Live A/B (merged 8-entry cache, per-family thresholds, runtime dispatch):

| query | guard ON (sep 0.05) | guard OFF (sep 0) |
|-------|---------------------|-------------------|
| divisors 196 (full) | cache hit, 9 ✓ | cache hit, 9 ✓ |
| divisors 48 (short) | fallback — "ambiguous cross-family 0.756 vs 0.733" | fallback — "param extraction failed" |
| gcd(48,180) | **cache hit 720 ✗** | **cache hit 720 ✗** |
| gcd(120,80,60) | fallback — "ambiguous 0.729 vs 0.683" | **cache hit 240 ✗ (LCM!)** |

The cross-family **separation guard** (`--min-separation`, default 0.05)
converts the near-tie case (gcd 120/80/60) from a confident wrong cache answer
into a safe LLM fallback. It does **not** help the same-numbers case (separation
0.316 — the wrong family genuinely wins).

## Recommendations

1. **Deploy per-family today.** Separate caches/gateways per family have no
   in-cache cross-family confusion; the 0.922 danger only exists once families
   share an entries.db.
2. **Per-family thresholds are in** (`--family-threshold divisors=0.725,gcd=0.72,lcm=0.72`);
   entries carry a `family` tag (builder sets it; existing caches backfilled).
   Keep global 0.72 as the default floor.
3. **Before merging families into one cache**, add an operation-level guard:
   entries declare a signature phrase (e.g. "least common multiple"); a query
   containing a *competing* family's signature cannot execute that entry.
   This is the minimal fix for the 0.922 case (design noted, not built).
4. **Build two-int GCD entries** to close the gcd coverage gap (lifts
   0.579–0.65 toward the 0.72 band) — part of the batch-build milestone.

## Reproduce

```bash
python3 tests/build_lcm_family.py          # ~5 min of 27B (skippable if cache exists)
python3 tests/build_merged_cache.py        # merge 3 families -> tests/.merged_cache
python3 eval/threshold_matrix.py \
  --store divisors=tests/.p2_live_cache --store gcd=tests/.gcd_cache \
  --store lcm=tests/.lcm_cache --out tests/threshold_report.json
```
