# CPU-ARC — Paper Ingest Notes (2026-11-21)

Paper: **CPU-ARC: Agentic Reasoning Caching for on-CPU Reasoning Reuse**
arXiv: 2611.09891v1 [cs.AI], 06 Nov 2026
Authors: Peter Belcak, Greg Heinrich, Saurav Muralidharan, Shizhe Diao, Pavlo Molchanov (NVIDIA; links to "Project Moonshot: Local Frontier Intelligence")
Full text saved: `paper-fulltext.txt`

## Core idea
Decouple **task-solving** (frontier LLM reasons once) from **solution execution** (deterministic CPU code + tiny SLM). Convert a source model's reasoning transcript into an **Agentic Reasoning Cache (ARC)**: a network of small Python "semantic functions" that solve the *class* of problems (equirational variants), not just the singleton instance.

## Key components
- **Roles**: Source 𝒮 (expensive, e.g. Claude Opus 4.5 + ext thinking), Builder ℬ (transcript → code, agentic loop with judge feedback), Target 𝒯 (tiny SLM at invocation: Haiku/Flash-Lite/Nano-class; fine-tuned **Llama 3.2 1B** works for extraction/conversion), Judges 𝒥 (unanimity committees).
- **Semantic functions**: `f: Context -> dict[str, Any]`. Everything referenced by *natural-language descriptions*, not identifiers → "semantic discovery" via vector search; components reusable across caches without renaming.
- **Runtime primitives** (the whole runtime surface):
  - `load(ctx, query[, d_type])` — top-down layered context lookup via embedding similarity
  - `do(ctx, intent, payloads)` — discover+invoke a semantic function by NL description; pushes isolated memory layer
  - `narrate(ctx, msg)` — human-readable reasoning trace (explainability)
  - `lm(ctx, prompt)` — tiny-model call for classification/short generation
  - `extract(ctx, text, prompt[, d_type])` — verbatim value extraction via 𝒯
  - `convert/smart_convert` — programmatic coercion first, else 𝒯 rewrites to Python literal validated by `ast.literal_eval()`
  - `invoke_tool(ctx, name, args)` — external tools (terminal, web)
- **Context/memory**: stack of memory layers, each backed by CPU vector store (EmbeddingGemma ~300M → 768-dim, FAISS flat). do() pushes/pops layers = scoping.
- **Construction (5 phases)**: 1) input-output isolation, 2) decomposition into function-sized logical steps + reusable sub-computations, 3) parameter recognition (generic template q(•,…,•), validity ranges; params must be atomic, not whole narratives), 4) planning (dependency DAG, bottom-up build order, granularity setting fine/medium/coarse), 5) implementation loop (registry reuse search → write → static/narration checks → test vs transcript fragment w/ judge → register or rewrite with structured feedback). Root tested end-to-end: correctness vs golden label + extraction round-trip ×N.
- **Invocation (4 steps)**: entry-point cosine search (threshold → miss falls through to 𝒮) → parameter extraction by 𝒯 → dispatch root on fresh Context → output dict + narration trace. Gated use pattern: try cache, fall back to full LLM on miss/error.
- **Compilation**: once registry stable, statically resolve do()→direct calls, load()→dict lookups → standalone Python module, no embedding engine/vector store at runtime; lm()/extract() survive (need 𝒯). Trade-off: frozen network, recompile on update.

## Results (S=𝒯=Opus 4.5 ext-thinking; T=Haiku 4.5 then Llama-3.2-1B-ft)
| Bench | S Acc | C Acc | Fidelity | Phrasal/Param/Combined | Build× | Savings× |
|---|---|---|---|---|---|---|
| MATH-500 | 98.6 | 98.8 | 99.6 | 96.7/94.5/91.6 | 30.5 | 88.7 |
| AIME '24 | 100 | 100 | 100 | 96.0/99.3/95.1 | 12.9 | 262.4 |
| AIME '25 | 83.3 | **93.3** | 90.0 | 98.4/91.2/93.4 | 12.1 | 291.1 |
| AIME '26 | 90.0 | **96.7** | 90.0 | 97.9/90.9/90.8 | 12.9 | 281.4 |
| GPQA Diamond | 83.8 | 83.8 | 98.5 | 93.8/93.9/94.0 | 25.1 | 38.1 |
| GSM8K | 98.5 | 98.0 | 99.5 | 96.2/94.2/92.5 | 83.7 | 16.3 |
| MMLU-Pro | 88.0 | 89.0 | 100 | 95.7/96.7/91.9 | 40.8 | 22.9 |
| SciBench | 85.2 | **88.2** | 99.4 | 97.5/98.8/96.4 | 29.0 | 83.0 |

- Cache cost per invocation: $0.0004–$0.0019 (mostly extractor calls).
- Break-even ≈ Build× invocations: ~13 for AIME, ~90 for GSM8K.
- Cache > source on hard math: deterministic code execution avoids the model's final-step slips.
- Perturbation robustness ≥90% in all modes.
- **Target model production (App D)**: harvest 20% of real invocation logs → 10× synthetic augmentation by 𝒮 (variants keep instructional scaffold verbatim) → SFT Llama 3.2 1B (trl SFTTrainer, bf16, grad ckpt, AdamW, lr 2e-5, max 20 epochs, early stop patience 3, seq 4096).
- Paper's own vision: "semantic models" — explainable, continually trainable layer above neural "neuralware"; edge deployment of non-creative intelligence.

## Critical notes / caveats for building
1. **No public code** (links are to a private lab page). Reimplementation from paper is feasible; the runtime surface is small and fully specified.
2. Biggest cost driver at invocation = **extractor** calls (6–18K input tokens avg per sample in App C tables) — parameter extraction dominates over target/converter.
3. EmbeddingGemma + FAISS flat on CPU is the retrieval stack; any 768-dim embedder works for a prototype (even API embeddings).
4. Assumption A1 (every solvable problem = code + SLM, generalized) is the weak spot: creative/open-ended tasks won't cache well — paper deliberately tests hardest *structured* cases.
5. Per-sample isolated caches merged into master cache; entry points registered as both generic template and original input for retrieval.
6. Judges are unanimity committees of 2× Opus — expensive at build time, cheap to swap in a prototype (single judge).

## Where we'd use it (Cellery / OpenClaw context)
- Repetitive agentic workloads: same instructions + varying payloads (support triage, storage diagnostics, report generation, benchmark runs).
- Edge/on-prem: compiled ARC = plain Python + 1B SLM → runs on CPU-class hardware; pairs with llama.cpp serving of the target model.
