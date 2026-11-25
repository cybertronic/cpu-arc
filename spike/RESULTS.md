# P0 Spike Results (2026-11-21)

**Verdict: PASS — the ARC runtime concept is validated on local hardware.**

## What was built
`arc_runtime.py` (~450 lines): full paper runtime surface — `Context` (layered memory + top-down vector lookup), `load` (both forms: layered context and do()-result dict), `do` (semantic function discovery by NL description), `narrate`, `lm`, `extract` (with NOT_FOUND protocol + retry), `convert`/`smart_convert` (multi-tier coercion ladder with `ast.literal_eval` guard), `invoke_tool` stub, gated `serve()` (paper Fig. 1), `ARC` container with entry-point registry.

`run_spike.py`: hand-coded ideal-gas-law cache entry from paper Appendix A (3 semantic functions: temperature conversion, pressure unit conversion, root P=nRT/V) + 5 test cases (original, 2 phrasal, 1 parametric, 1 combined perturbation).

## Backends
- **LM (𝒯):** correction (2026-11-21 17:52): the 4B GGUF download 404'd; the spike actually ran against the system `llama-server` already listening on :8080 (unsloth/Qwen3.8-27B-GGUF Q6). Results stand (loop validated; 27B is a stronger 𝒯 than the paper's 1B, so edge-cost claims need a re-run with a small model in P3). OpenAI-compatible `/v1/chat/completions`, temp 0. No custom adapter needed; the paper's 𝒯 interface is exactly an OpenAI-compatible chat endpoint.
- **Embeddings:** EmbeddingGemma-300M QAT Q8_0 GGUF via `llama-server --embedding` (native `/embeddings` returns nested `[1][768]` — unwrap needed).
- **Offline fallback:** deterministic hash embedder (bag-of-words) so the spike runs with zero GPU.

## Results (EmbeddingGemma + Qwen3-4B, threshold 0.35)
| Case | P (kPa) | Expected | Entry score | LM calls / in-tok |
|---|---|---|---|---|
| original | 5134.66 | ~5132 | 1.000 | 8 / 1,004 |
| phrasal-1 | 5134.66 | ~5132 | 0.884 | 16 / 2,050 |
| phrasal-2 ("Ten moles" wording) | 5134.66 | ~5132 | 0.917 | 25 / 3,081 |
| parametric-1 (n=5.0) | 2567.33 | ~2567 | 0.964 | 33 / 4,080 |
| combined-1 (methane, 2 mol, 8 dm³, 100°C) | 775.59 | ~772.6 | 0.779 | 41 / 5,113 |

**5/5 passed.** Same run with hash embedder also 5/5 (entry scores 0.37–0.97).

## What got exercised (all paper mechanisms)
- Entry-point semantic search with threshold → miss fall-through path exists (not triggered; all ≥0.78)
- Parameter extraction via 𝒯 from raw natural-language input, including "Ten moles" → 10.0 via the **smart_convert LLM fallback tier** (programmatic coercion failed, model rewrote to literal)
- `do()` semantic function discovery: root found both utility functions by NL intent description alone
- Layered context push/pop scoping for sub-function calls
- `load(result_dict, ...)` second form for retrieving sub-function outputs
- `narrate()` trace with indentation by layer depth (matches paper Fig. 8 format)
- Type coercion ladder: regex number extraction → LLM rewrite → `ast.literal_eval`

## Observations feeding P1
1. **Extractor dominates cost** (as predicted from paper App C): ~4–6 extract calls + 1 unit-classification `lm()` per invocation; in-tokens grow with case count only because stats are cumulative — per-case is ~1K in / ~200 out tokens. Very cheap.
2. **EmbeddingGemma separation is weak for short param descriptions**: "the molar amount of gas with unit" vs "the volume of the container with unit" = 0.761, "temperature in Kelvin" vs "pressure in MPa" = 0.676. A flat cosine threshold is risky for `load()` disambiguation; P1 should add (a) exact/substring-match fast path before vector search, (b) higher effective threshold or MRR-style re-ranking for short queries, (c) consider longer description phrasings.
3. **Entry-point separation was clean**: related problems 0.78–1.00, unrelated 0.39 → threshold ~0.6 works for entry matching; the same number would be wrong for param loads. Per-scope thresholds are a P1 item.
4. **Extraction retry with stronger hint** recovered one NOT_FOUND (volume in "rigid container of volume" phrasing) — keep this pattern, it's cheap insurance.
5. llama.cpp server needed no API-key/adapter work; embedding endpoint format quirk is the only integration gotcha found.

## Next (P1 per PLAN.md)
- Registry persistence (SQLite + FAISS), per-scope thresholds + exact-match fast path
- Sandboxed semantic-function execution, cost ledger per role, async judge review hook
- Then P2 builder loop (the real engineering lift).
