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
        │        → POST /reload on the running gateway: the live serving
        │          path picks up the new entry index + recompiled module
        │          WITHOUT a service restart (loop closure)
        │
        ├─ new-family signal: repeated out-of-band misses grouped into
        │   semantic regions; a region ≥ FAMILY_CANDIDATE_MIN misses is
        │   flagged in the report as a family SEED CANDIDATE (report only —
        │   families are never auto-created)
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

## Pointing the stack at a different model / location

One file: `/etc/arc/gateway.env` (template: `deploy/arc-gateway.env`).
Every component — the gateway service, the promotion gate, the nightly
curator — reads these as its defaults (CLI flags still win):

```
ARC_CHAT_URL    chat/completions base URL    (llama-server :8080)
ARC_EMBED_URL   embedding server base URL    (llama-server :8081)
ARC_MODEL       model name for runtime LLM fallback
ARC_BUILD_MODEL model name for cache builds/promotion
                (empty = use ARC_MODEL; set to point builds at a
                 stronger/slower model than the runtime fallback)

# curator policy
ARC_FAMILY_CANDIDATE_MIN  miss volume that flags a new-family seed
                          candidate region (default 5)
ARC_GATEWAY_URL           base URL the curator POSTs /reload to after a
                          cache change (default http://127.0.0.1:$ARC_PORT)
```

Example — move to a 70B build model on another host, runtime stays on 27B:

```
ARC_CHAT_URL=http://10.0.0.5:8080/v1
ARC_MODEL=Qwen3-27B
ARC_BUILD_MODEL=Qwen3-70B
```
then `systemctl restart arc-gateway` (the cron loop picks up changes on its
next run; it sources the same file).

In code this lives in `arc/endpoints.py` — `ARC_*` environment variables,
which is also how you override any single run without touching the file.

## Operational notes

- **Golden map trust**: entries in `golden-map.json` are evaluated as
  Python (`math`, `numdiv`, `p0..pN`, `params`). Only put families you can
  express this way (gcd/lcm/divisors are trivial). A wrong template means
  wrong gates — review before adding.
- **Cost**: each promotion is one local-model build (~2–3 min) + gate
  cases. `CURATOR_MAX_PROMOTIONS` caps nightly spend (default 2).
- **Loop closure**: after any promotion OR rollback the curator calls
  `POST /reload` on the running gateway (entry index + a fresh worker pool
  importing the recompiled module). If that call fails it is loud in the
  cron log + report (`gateway_reload` field) and the fix is
  `systemctl restart arc-gateway`. Manual reload: `curl -X POST
  http://127.0.0.1:8084/reload`. Note: the reload respawns the worker pool,
  so a query in flight during the swap can error once (safe failure, logged
  as an execution error — the client retries); reload on low traffic.
- **Miss latency**: `avg_miss_ms` in `/stats` and `ms`/`serve_ms` in the
  invocation log are full client-visible latency including the LLM fallback
  (not just the match-decision time) — don't be alarmed by seconds-scale
  miss times; hits stay single-digit ms.
- **Embedder**: the curator prefers the local embed server and falls back to
  deterministic hash embeddings if it is down (logged as a warning).
- **Sandbox**: the unit runs with `ProtectSystem=full`, `NoNewPrivileges`,
  `MemoryMax=2G`, `CPUQuota=400%`. `CACHE_DIR`/`BUILD_QUEUE_DIR` must live
  under `/var/lib/arc` (the only writable path) for this to hold.
- **Thresholds** (`FAMILY_THRESHOLDS`) must match `CALIBRATION.md` —
  per-family, never a single global number.
