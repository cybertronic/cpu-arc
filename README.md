# CPU-ARC

**A semantic cache and execution gateway for verified algorithms.**

CPU-ARC answers natural-language math queries by matching them to *verified,
compiled* implementations instead of asking an LLM every time. Queries like
*"How many positive divisors does 120 have?"* or *"Find the LCM of 48, 75, and
90"* are embedded, matched against a vector registry of proven function
families, and — on a hit — executed in isolated subprocess workers under CPU,
memory, and timeout limits. Misses fall back to the LLM, get logged, and feed a
nightly curation loop that (with gated verification) grows the cache over time.

```
query ──► embed ──► entry match (threshold + cross-family guard)
                        │
              ┌─────────┴──────────┐
            hit                   miss
              │                      │
   extract params (regex / LLM)   LLM fallback answer
   + type coercion (scalar/list)  + logged for curation
              │                      │
   compiled module in             build queue ──► nightly curator
   isolated worker pool               (gated auto-promote)
              │                      └──► POST /reload (hot)
         answer + source:"cache"
```

## Why

LLMs are slow, expensive, and occasionally wrong at exact arithmetic. A
cache-first gateway that only ever executes *code that has been judged,
gated, and statically verified* gives LLM-grade natural-language
understanding with deterministic, sub-100 ms exact answers for the queries
it has seen — and learns new ones without ever letting an unvetted
implementation in.

## What it does

- **Family-based semantic matching** — queries are cosine-matched against
  entry vectors; per-family thresholds (default 0.72) control precision.
- **Cross-family separation guard** — if the top two *different* families
  score within 0.05 of each other, the hit is refused (safe miss) rather
  than risking the wrong family's code.
- **List-aware parameter extraction** — regex fast path for positional
  scalars; number-aware grouping for list-typed params ("126, 168, and 210"
  → `[126, 168, 210]`); LLM extractor as fallback; declared `param_types`
  are persisted per entry and used for coercion.
- **Static compilation** — the compiler v2 turns verified family functions
  into plain static Python modules (multi-entry, per-entry name maps) with a
  `fully_static` verification gate; the compiled path never shells out.
- **Subprocess isolation** — execution runs in a worker pool of subprocesses
  under resource limits (`MALLOC_ARENA_MAX=2`, per-job CPU/memory caps,
  hard timeouts); the gateway is never compromised by a hostile or runaway
  function body.
- **SQLite vector persistence** — entry vectors stored as BLOBs with
  in-place schema migrations (new columns are added automatically, no
  re-embedding).
- **Invocation logging** — every query is logged (WAL-mode SQLite) with
  score, params, reason, latency, and token accounting; 30-day retention.
- **Production feedback loop** — nightly curation mines misses, clusters
  them, and auto-promotes recurring families through the same gated
  verification as manual promotion, then hot-reloads the gateway
  (`POST /reload`, zero downtime).

## Architecture

```
arc/
  gateway.py      HTTP gateway (ThreadingHTTPServer): /query /stats /reload,
                  keep-alive, thread-safe atomic stats
  cache.py        ARC core: match → extract → coerce → execute, thresholds
  registry.py     Registry + EntryStore: SQLite, BLOB vectors, schema
                  migrations, param_types persistence
  builder.py      Family builder: LLM drafts family code against primitives,
                  judge pass, gated promotion
  compiler.py     Compiler v2: verified functions → static compiled modules
  worker.py       Subprocess worker pool (isolation, resource limits)
  execution.py    Sandboxed execution + timeout control
  primitives.py   Fixed numeric primitives (divisors, gcd, lcm, is_prime, …)
                  and number-aware type coercion
  sandbox.py      Source verification (AST allowlist) for function bodies
  backends.py     Embedder (local llama-embedding) + LMBackend (OpenAI-compat)
  invocations.py  Invocation log store + retention + CLI inspector
  endpoints.py    Endpoint configuration
eval/
  promote.py      Gated promotion: 8–10 randomized cases, hit-rate floor,
                  exactness check, token ledger
  nightly_curate.py  Miss mining → clustering → gated auto-promote → report
  curate_misses.py   Miss clustering signals (family candidate detection)
deploy/
  install.sh      Idempotent installer (user, dirs, systemd, cron, env)
  arc-gateway.service   systemd unit for the gateway
  arc-gateway.env       All tunables (see Configuration)
  arc-curator.cron      cron.d job (03:15 daily)
  golden-map.json       Which families may be auto-promoted, with what golden
tests/            Offline + live test suites
```

## Requirements

- **Python 3.10+** with `numpy` (only third-party dependency)
- **llama.cpp** servers:
  - chat model — any OpenAI-compatible `llama-server` (e.g. a Qwen GGUF)
  - embeddings — `llama-embedding` server (e.g. nomic-embed)
- Production: `systemd` + `cron` (the installer handles both)

## Installation

### Production (systemd + nightly curator)

```bash
git clone https://github.com/cybertronic/cybertronic.git && cd cybertronic

# dry run first — prints every action, changes nothing
sudo ./deploy/install.sh --dry-run

# real install: creates the arc system user, state dirs under /var/lib/arc,
# /etc/arc/gateway.env + golden-map.json, systemd unit, cron job; starts it
sudo ./deploy/install.sh
```

The installer is idempotent and never overwrites existing `/etc/arc` config.
Point `ARC_CHAT_URL` / `ARC_EMBED_URL` / `ARC_MODEL` in
`/etc/arc/gateway.env` at your model servers, then:

```bash
sudo systemctl restart arc-gateway
```

Uninstall: `sudo ./deploy/install.sh --uninstall` (keeps cache + logs),
add `--purge` to remove everything.

### From source (dev / eval)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy

# start your model servers first, then:
export ARC_CHAT_URL=http://127.0.0.1:8080/v1   # your chat server
export ARC_EMBED_URL=http://127.0.0.1:8081     # your embedding server
export ARC_MODEL=<your-model-id>

python3 -m arc.gateway --cache-dir /tmp/arc_cache
```

## Quick start

```bash
# cache hit — sub-100 ms, deterministic
curl -s http://127.0.0.1:8084/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How many positive divisors does 120 have?"}'
# → {"answer": 16, "source": "cache", ...}

# miss — falls back to the LLM, query is queued for the curator
curl -s http://127.0.0.1:8084/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Find the LCM of 7065, 1152, and 52007."}'

# stats
curl -s http://127.0.0.1:8084/stats | python3 -m json.tool

# hot reload after promoting a family (no downtime)
curl -s -X POST http://127.0.0.1:8084/reload
```

## Configuration

All tunables live in `/etc/arc/gateway.env` (dev: environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `ARC_CHAT_URL` / `ARC_EMBED_URL` | — | Model server endpoints (OpenAI-compat / llama-embedding) |
| `ARC_MODEL` | — | Chat model id used for fallback + curation |
| `ARC_PORT` / `ARC_HOST` | `8084` / `0.0.0.0` | Gateway bind |
| `ARC_CACHE_DIR` | `/var/lib/arc/cache` | Entry DB (vectors), compiled modules, invocation log |
| `ARC_ENTRY_THRESHOLD` | `0.72` | Default per-family hit threshold |
| `ARC_FAMILY_THRESHOLDS` | — | Per-family overrides, e.g. `gcd=0.72,divisors=0.68` |
| `ARC_MIN_SEPARATION` | `0.05` | Cross-family ambiguity guard (refuse hit when top families are closer) |
| `ARC_BUILD_QUEUE_DIR` | `/var/lib/arc/build_queue` | Misses queued for building |
| `ARC_LOG_RETENTION_DAYS` | `30` | Invocation log retention |
| `ARC_CURATOR_SINCE_HOURS` | `168` | Lookback window for miss mining |
| `ARC_CURATOR_MIN_REPEATS` | `3` | Min repeats before a cluster is promotable |
| `ARC_CURATOR_AUTO_PROMOTE` | `1` | Allow gated auto-promotion (vs. report-only) |
| `ARC_CURATOR_MAX_PROMOTIONS` | `2` | Promotions capped per night |
| `ARC_CURATOR_CASES` / `ARC_CURATOR_TOP` | `10` / `10` | Gate size / report size |
| `ARC_CURATOR_GOLDEN_MAP` | `/etc/arc/golden-map.json` | Which families may auto-promote, with what golden validator |
| `ARC_FAMILY_CANDIDATE_MIN` | `5` | Min cluster size to signal a *new* family candidate |

**Threshold calibration.** List-param families (e.g. LCM of N integers)
embed with wider variance than scalar families — the query's numbers
dominate the vector — so their scores cluster lower. Calibrate each family's
threshold against its observed score distribution before gating promotions
(`eval/threshold_matrix.py`, see `CALIBRATION.md`); a threshold tuned for
scalar families will silently miss list families.

## The nightly loop

At 03:15 the curator:

1. **Mines** the invocation log for misses (default: 7 days, ≥3 repeats)
2. **Clusters** them and scores each cluster (repeats, size, family-candidate
   signal)
3. **Gated auto-promote** (capped per night, golden-map only): each
   candidate must clear the same gate as manual promotion — N randomized
   paraphrases, hit-rate floor, exact answers checked against the golden
   expression
4. **Compiles** accepted families into the static module and **hot-reloads**
   the gateway via `POST /reload`
5. **Reports** everything to `/var/log/arc/`

## Operations

```bash
# invocation log inspector (miss reasons, scores, latency)
PYTHONPATH=/opt/arc python3 -m arc.invocations summary --db /var/lib/arc/cache/invocations.db

# promote a family manually (the full gate runs against your model servers)
python3 eval/promote.py --family lcm --cache-dir /var/lib/arc/cache \
  --seed "Find the LCM of 126, 168, and 210." \
  --golden "math.lcm(*params)" --threshold 0.72
```

## Tests

```bash
python3 -m pytest tests/ -q      # offline suites (extraction, guard, compiler, curation)
```

Live suites (require running model servers): multi-family eval, isolation
tests (13 adversarial payloads), and the end-to-end closure test
(promote → pre-reload fallback → `POST /reload` → cache hit).

## Current status

- Families live in production: **divisors**, **gcd**, **lcm** (list-aware)
- Multi-family live eval: **100% accuracy** on the eval set
- Isolation: **13/13** adversarial-payload tests
- E2E closure (novel seed → gated promote → hot reload → cache hit): **PASS**
- Production: systemd-managed under `/opt/arc`, state in `/var/lib/arc`,
  nightly curator active

## Notes

- `paper-fulltext.txt` — reference paper used during the P0 design phase.
- Research history (P0–P12 phases): see `PLAN.md`, `NOTES.md`, `CALIBRATION.md`.
