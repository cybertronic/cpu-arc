# CPU-ARC Implementation Plan

Status: **P0 + P1 done** (2026-11-21). Paper ingested → `NOTES.md`, full text → `paper-fulltext.txt`.
- P0 spike: `spike/` — 5/5 gas-law cases pass (real EmbeddingGemma + Qwen servers).
- P1 hardened package: `arc/` (~900 LOC) + `tests/` — 21/21 offline + 6/6 live.

## TL;DR
Build ARC as a **standalone Python package (`arc`)** first — the paper's runtime is ~7 primitives + a layered vector context, and nothing in it needs C++. Treat llama.cpp as the **target-model (𝒯) serving backend** behind an `lm()`/`extract()` adapter, not as the host for the cache. A llama.cpp integration can come later as a thin serving adapter; putting this *inside* llama.cpp proper is not recommended (it's a Python-layer feature, and upstream won't take it).

## Recommendation: standalone package + llama.cpp backend
Why not "part of llama.cpp":
- The entire ARC runtime (semantic functions, context stack, registries, builder loop, compiler) is Python. llama.cpp is C++/server code; the paper's compiled artifact is *a standalone Python module*.
- Upstream llama.cpp accepts performance/inference work, not application-layer caching frameworks.
- Standalone keeps it usable with any backend (API SLMs for dev, llama.cpp GGUF for edge/prod) and testable without a GPU.

Deliverable: `arc/` package + optional `arc-llama.cpp` adapter + eval harness.

## Phase 0 — Spike (1–2 days): prove the runtime
Scope: hardcode one cache entry by hand (use the paper's ideal-gas-law example, App A — it exercises every primitive).
- [ ] `Context`: stack of memory layers; each layer = list of (description, value) + embeddings; top-down `load()` with cosine threshold
- [ ] Primitives: `load`, `do`, `narrate`, `lm`, `extract`, `convert`/`smart_convert` (`ast.literal_eval` guard), `invoke_tool` (stub)
- [ ] `LMBackend` interface: `complete(prompt, system=None) -> str`. Two impls: OpenAI-compatible HTTP (dev, any SLM) and llama.cpp `/v1/chat/completions` (same API — llama.cpp server already speaks it; the adapter is mostly config + a GGUF pick)
- [ ] Embeddings: start with an API embedder or `sentence-transformers` locally; swap to EmbeddingGemma GGUF via llama.cpp embedding endpoint later
- [ ] Run the gas-law entry on original + 2 rephrasings + 2 reparameterizations; assert golden answers (≈5130 kPa etc.)
- Exit criteria: all 5 variants correct, narration trace produced, zero vector-store deps beyond numpy/faiss-cpu.

## Phase 1 — Runtime hardening (week 1)  ✅ DONE (shipped as `arc/` package)
- [x] Registry: function descriptions + entry points in a persistent store — **SQLite** (`functions.db`/`entries.db`); functions persist as **source text** (not closures) and re-execute through the sandbox on load. FAISS not needed at this scale (numpy matmul over a flat index; revisit >10k entries).
- [x] Type coercion ladder: native `_coerce` → `smart_convert` (LLM rewrite → `ast.literal_eval`) — both in `primitives.py`, tested.
- [x] Gated serving API: `serve()` = try cache → on miss/any error fall through to source LLM (paper Fig. 1); every invocation (hit+miss+error) logged to `invocations.jsonl` for P3 mining.
- [x] Isolation/safety: `sandbox.py` — restricted builtins, import allowlist (math/string/datetime/json/re/decimal/fractions), AST gate rejecting `exec`/`eval`/dunder-attr/`__import__`, CPU rlimit + wall-clock budget. Documented as containment (not a hostile-code boundary).
- [x] Observability: per-role cost `Ledger` (target/extractor/converter, mirrors paper App C) wired into the LM backend; `load_log` records exact-vs-vector + score for every `load`; narration trace + full invocation records.
- [x] **P1 extras from P0 findings:** per-scope `Thresholds` (entry 0.60 / function 0.55 / load 0.45) + exact/token-substring fast path before vector search (fixes EmbeddingGemma's weak short-description separation); `NOT_FOUND` extraction protocol + one hinted retry.
- [x] Tests: `tests/test_p1_offline.py` (21 checks, MockLM + hash embedder, zero GPU) + `tests/test_p1_live.py` (6/6 gas-law cases vs real servers, incl. an unrelated-question miss).

## Phase 2 — Builder pipeline (weeks 2–3)
Agentic loop implementing the paper's 5 phases; this is where most of the engineering lives.
- [ ] Transcript ingestion: accept chat logs / OpenClaw session exports / API transcripts → (query, source transcript, golden label if known)
- [ ] Phase 1–4 as structured LLM calls with JSON-schema outputs (isolation, decomposition DAG, parameter template + validity ranges, bottom-up plan w/ granularity knob)
- [ ] Phase 5 implementation loop: registry-reuse search → candidate codegen → static checks (syntax, narration density vs control-flow complexity) → test dispatch with original params vs transcript fragment (judge) → register or rewrite with structured feedback; cap rewrites per function
- [ ] Root validation: end-to-end correctness + extraction round-trip ×N (start N=5)
- [ ] Judge policy: single judge for dev, 2-judge unanimity optional (paper default); judge calls are the main build-time cost — prompt-cache aggressively
- [ ] Per-sample cache → merge into master cache (paper's construction pattern)
- [ ] Cost guardrails: builder budget per entry; report Build× against source cost

## Phase 3 — Compiler + target-model production (week 4)
- [ ] `arc compile <cache>`: static resolution pass — do()→direct call, load()→named param/dict key; emit standalone `.py` module with manifest (no runtime vector store needed for resolved paths; lm()/extract() survive)
- [ ] Invocation-log harvester (from Phase 1 ledger): filter target/extractor/converter roles, dedup by deterministic id
- [ ] Synthetic augmentation: 10× expansion via source model (variants keep scaffold verbatim), regex parse + degenerate filtering per paper App D.2
- [ ] SFT recipe for 𝒯: Llama 3.2 1B (or Qwen3 ~1–4B) via trl SFTTrainer, bf16, grad ckpt, AdamW lr 2e-5, early stop patience 3 → GGUF via llama.cpp quantization (Q4_K_M/Q8_0 for the 1B class is fine)
- [ ] Edge deploy: `llama-server` with target GGUF + EmbeddingGemma GGUF (embedding endpoint), ARC compiled module next to it; measure tokens/s and $-equivalent vs source

## Phase 4 — Eval harness (parallel, from week 2)
Mirror the paper's protocol so we can reproduce/compare:
- [ ] Benchmarks: MATH-500 + GSM8K (cheap, large), AIME subset, SciBench subset; golden labels local
- [ ] Metrics: C Acc vs S Acc, fidelity (judge), perturbation modes — phrasal / parametric / combined with generation-judge validation (calibration check: generator must re-derive original golden answer first)
- [ ] Cost report per role; break-even calc per benchmark
- [ ] Regression suite: every merged cache entry re-runs on its construction sample + 3 stored perturbations

## Phase 5 — llama.cpp integration surface (optional, post-MVP)
- [ ] `arc.backends.llamacpp`: config-driven (model paths, endpoints), auto-detect running server; embedding endpoint for the vector store
- [ ] Optionally ship a `llama-server` docker-compose + GGUF set as a reference "edge ARC box"
- [ ] If we ever want in-process: llama.cpp's Python bindings (`llama` pip) can back `LMBackend.complete()` without an HTTP server — still outside the C++ core

## Risks / open questions
1. **Builder quality is the whole game.** Paper uses Opus 4.5 + ext thinking for both 𝒮 and ℬ; our results will track our builder model. Budget for frontier-class API calls at build time (that's by design — build cost amortizes).
2. **Extractor token bloat** (6–18K input tokens/sample in paper) is the real invocation cost; mitigate with compact extraction prompts + few-shot from construction params.
3. **Coverage ≠ correctness**: a wrong-but-confident cache hit is worse than a miss. Mitigations: high entry-point threshold, judge-based async review of popular entries (paper's SLC benefit #2), per-entry confidence from perturbation pass rate, kill-switch to fall through on any primitive error.
4. **No public reference implementation** — expect reimplementation bugs in the semantic-discovery layer; Phase 0 spike de-risks before builder investment.
5. Creative/open-ended workloads won't cache (paper's A1 assumption); scope this to structured, repeat-instruction workloads.

## Suggested first concrete step
Phase 0 spike against the ideal-gas-law example with a local Qwen3-4B via llama.cpp as 𝒯 — validates the full loop (embed → dispatch → extract → narrate) on hardware we already run, in ~2 days.
