# CPU-ARC server deployment

Two moving parts, one installer:

| Artifact | What it installs | Purpose |
|---|---|---|
| `arc-gateway.service` | systemd unit | the cache-first gateway (long-running server) |
| `arc-curator.cron` → `nightly_curate.sh` | cron.d job (03:15 daily) | the production feedback loop: mine misses → **gated** auto-promote → report |
| `arc-gateway.env` | `/etc/arc/gateway.env` | all tunables for both (ports, model, thresholds, curator policy) |
| `golden-map.json` | `/etc/arc/golden-map.json` | which families may be **auto**-promoted, and with what validation |

## Install

```bash
sudo ./deploy/install.sh                 # default: user arc, cron 15 3, starts service
sudo ./deploy/install.sh --dry-run       # print every action, change nothing
sudo ./deploy/install.sh --schedule "5 2" --no-start
```

The installer creates the `arc` system user, state dirs
(`/var/lib/arc/{cache,build_queue,curate}`, `/var/log/arc`), and never
overwrites existing `/etc/arc` config. Re-running is safe (idempotent).

```bash
sudo ./deploy/install.sh --uninstall          # keep cache + invocation log
sudo ./deploy/install.sh --uninstall --purge  # remove everything
```

## The nightly loop (what cron actually does)

```
invocations.db (every production miss, P7)
        │  eval/nightly_curate.py
        ▼
cluster near-dup misses (embed cos ≥ 0.92) · rank by count + recency
        │
        ├─ auto-promotable: known family + count ≥ CURATOR_MIN_REPEATS
        │   + golden template in golden-map.json + not ambiguous/off-band
        │        → eval/promote.py: build → recompile → validate N random
        │          cases vs golden on the production compiled path
        │          (any WRONG answer or <60% coverage → full cache rollback)
        │
        └─ everything else → nightly report for human review
```

Reports: `/var/lib/arc/curate/curate-YYYY-MM-DD.json` (machine) and
`/var/log/arc/curate-cron.log` (human). The loop auto-promotes only what it
can *prove* correct — families without a golden template are never
auto-built, they just get ranked candidates in the report.

Run it by hand any time:

```bash
./deploy/nightly_curate.sh                          # honors /etc/arc/gateway.env
CURATOR_AUTO_PROMOTE=0 ./deploy/nightly_curate.sh   # report only
```

## Operational notes

- **Golden map trust**: entries in `golden-map.json` are evaluated as
  Python (`math`, `numdiv`, `p0..pN`, `params`). Only put families you can
  express this way (gcd/lcm/divisors are trivial). A wrong template means
  wrong gates — review before adding.
- **Cost**: each promotion is one local-model build (~2–3 min) + gate
  cases. `CURATOR_MAX_PROMOTIONS` caps nightly spend (default 2).
- **Embedder**: the curator prefers the local embed server and falls back to
  deterministic hash embeddings if it is down (logged as a warning).
- **Sandbox**: the unit runs with `ProtectSystem=full`, `NoNewPrivileges`,
  `MemoryMax=2G`, `CPUQuota=400%`. `CACHE_DIR`/`BUILD_QUEUE_DIR` must live
  under `/var/lib/arc` (the only writable path) for this to hold.
- **Thresholds** (`FAMILY_THRESHOLDS`) must match `CALIBRATION.md` —
  per-family, never a single global number.
