"""Builder pipeline: transcript -> ARC entry (paper §2.2.1, five phases).

Roles (paper):
  S  source model     — already answered; its transcript is the input
  B  builder model    — analyzes the transcript and constructs the cache entry
  J  LM judges        — verify candidate functions and root outputs
  T  target model     — used at invocation (extraction, lm() calls)

Phases:
  1 isolation     — separate user input from expected output
  2 decomposition — hierarchical list of logical operations
  3 parameters    — generic input template + atomic parameter values + validity
  4 planning      — function DAG, bottom-up build order, root
  5 implementation— agentic loop: reuse-check -> codegen -> static checks ->
                    test vs transcript (judge) -> register or rewrite (bounded)

Root validation: end-to-end dispatch with original parameters + extraction
round-trip xN (paper §2.2.1 Phase 5).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .backends import Ledger
from .context import Context, CacheInvocationError, Thresholds
from .primitives import bind, _primitives
from .sandbox import SandboxError, check_source, exec_function_source
from .cache import ARC

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Param:
    description: str
    value: str          # concrete value in the original instance
    type: str           # python type name
    validity: str       # logical validity range / constraints


@dataclass
class FnSpec:
    id: str
    description: str    # imperative NL description (used for semantic search)
    depends_on: list[str] = field(default_factory=list)
    is_root: bool = False


@dataclass
class BuildResult:
    ok: bool
    query: str
    final_answer: str = ""
    template: str = ""
    params: list[Param] = field(default_factory=list)
    functions: list[FnSpec] = field(default_factory=list)
    registered: dict[str, str] = field(default_factory=dict)  # spec_id -> registry id
    reused: list[str] = field(default_factory=list)
    rewrites: int = 0
    root_output: dict | None = None
    narration: list[str] = field(default_factory=list)
    roundtrips: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    ms: float = 0.0


def _extract_json(text: str) -> dict:
    """Robust JSON extraction from an LLM reply (fences, prose, multiple blocks)."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    return json.loads(t[s:e + 1])


# --------------------------------------------------------------------------- #
# Runtime contract given to the builder when it writes semantic functions
# --------------------------------------------------------------------------- #
RUNTIME_CONTRACT = """You are writing a "semantic function" for the ARC (Agentic Reasoning Cache) runtime.
Strict contract:
- Define exactly one top-level Python function: def __FUNC_NAME__(ctx):
- Do NOT import anything except from: math, string, datetime, json, re, decimal, fractions.
- Available primitives (already in scope, call them with the ctx object):
    load(ctx, "<natural-language description of the value>", d_type=<type>) -> value
        (fetches a value previously placed in the context by description)
    do(ctx, "<imperative NL description of a sub-task>", payloads={{ "<description>": value, ... }}) -> dict
        (discovers and runs another registered semantic function; only use it for
        sub-tasks that are themselves separate registered functions)
    narrate(ctx, "<human-readable reasoning step>")
    lm(ctx, "<prompt for a small language model>") -> str   (only for genuinely linguistic tasks)
    extract(ctx, text, "<what to extract from text>", d_type=<type>) -> value
    convert(value, <type>); smart_convert(ctx, text, <type>)
- The function MUST return a dict mapping natural-language description strings to values.
- Call narrate() for each meaningful reasoning step.
- Reference inputs ONLY by their natural-language description via load(); never assume fixed names.
- Prefer deterministic Python (arithmetic, loops, math) over lm() calls; use lm() only for
  classification/linguistic steps plain code cannot do.
Reply with ONLY a single ```python fenced block containing the complete source. No prose."""


def _extract_source(text: str) -> str:
    """Extract function source from a model reply: prefer a python fence, else JSON
    {"source": ...}, else bare code (first 'def' to end)."""
    t = text.strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.S)
    if m:
        return m.group(1).strip()
    try:
        d = _extract_json(t)
        if isinstance(d.get("source"), str):
            return d["source"].strip()
    except (ValueError, json.JSONDecodeError):
        pass
    i = t.find("def ")
    if i != -1:
        return t[i:].strip()
    raise ValueError(f"no python source found in reply: {text[:200]!r}")


class Builder:
    def __init__(self, arc: ARC, builder_lm, judge_lm=None,
                 granularity: str = "medium", max_rewrites: int = 3,
                 roundtrips: int = 2, budget_tokens: int = 400_000):
        self.arc = arc
        self.b = builder_lm                 # role="build"
        self.j = judge_lm or builder_lm     # role="judge"
        self.ledger: Ledger = arc.ledger
        # record build/judge tokens on the cache's shared ledger (paper App C)
        for lm_obj in (self.b, self.j):
            if hasattr(lm_obj, "ledger"):
                lm_obj.ledger = self.ledger
        self.granularity = granularity
        self.max_rewrites = max_rewrites
        self.roundtrips = max(1, roundtrips)
        self.budget_tokens = budget_tokens

    # -- helpers ---------------------------------------------------------------
    def _gen(self, system: str, user: str, max_tokens: int = 4096,
             tag: str = "") -> str:
        # Root cause of empty replies (reproducible): the 27B is a Qwen3-family
        # thinking model — without reasoning_effort=none it spends the entire
        # completion budget on reasoning_content and returns content="" with
        # finish_reason=length. LMBackend now sends reasoning_effort=none by
        # default; keep one defensive retry in case a server rejects the field.
        t0 = time.time()
        out = self.b.complete(user, system=system, role="build",
                              max_tokens=max_tokens).text
        if not out.strip():
            time.sleep(1.0)
            out = self.b.complete(user, system=system, role="build",
                                  max_tokens=max_tokens).text
        print(f"    [builder] {tag or 'gen'}: {len(out)} chars in "
              f"{time.time()-t0:.0f}s", flush=True)
        return out

    def _gen_json(self, tag: str, system: str, user: str) -> dict:
        last_err = None
        for attempt in range(3):
            suffix = "" if attempt == 0 else (
                f"\n\nYour previous reply was not valid JSON ({last_err}). "
                f"Reply with JSON only, no prose.")
            try:
                return _extract_json(self._gen(system, user + suffix,
                                               max_tokens=8192, tag=tag))
            except (ValueError, json.JSONDecodeError) as e:
                last_err = str(e)
        raise CacheInvocationError(f"phase {tag}: model produced no valid JSON after 3 tries")

    def _judge(self, prompt: str) -> tuple[bool, str]:
        t0 = time.time()
        out = self.j.complete(
            prompt,
            system=("You are a strict correctness judge for a math/science cache-building "
                    'system. Reply with JSON only: {"pass": true|false, '
                    '"critique": "<one short sentence>"}'),
            role="judge", max_tokens=512).text
        print(f"    [builder] judge: {len(out)} chars in "
              f"{time.time()-t0:.0f}s", flush=True)
        try:
            d = _extract_json(out)
            return bool(d.get("pass")), str(d.get("critique", ""))
        except (ValueError, json.JSONDecodeError):
            return ("true" in out.lower()), out[:200]

    # -- Phase 1: input-output isolation ----------------------------------------
    def _phase1(self, query: str, transcript: str) -> dict:
        return self._gen_json(
            "P1-isolation",
            "You separate a problem into its user input and the model final answer. "
            "Reply with JSON only.",
            f"""Given the user query and the model full-reasoning transcript, extract:
- "user_input": the user query, verbatim
- "final_answer": the final answer/result stated in the transcript, as a concise value string
JSON: {{"user_input": "...", "final_answer": "..."}}

Query:
{query}

Transcript:
{transcript}""")

    # -- Phase 2: decomposition ---------------------------------------------------
    def _phase2(self, p1: dict, transcript: str) -> list[dict]:
        d = self._gen_json(
            "P2-decomposition",
            "You decompose a model reasoning transcript into a hierarchical list of "
            "function-sized logical operations. Reply with JSON only.",
            f"""Decompose the reasoning below into logical steps, each small enough to
implement as ONE function. For each step give:
- "id": "s1", "s2", ...
- "description": one imperative sentence naming the operation and the individual
  values it consumes/produces
- "reusable": true if this operation would likely recur in similar problems
  (e.g. a unit conversion, a factorisation)
Granularity: {self.granularity} (fine = many small steps, medium, coarse = few big steps).
JSON: {{"steps": [{{"id":"s1","description":"...","reusable":true}}]}}

Problem:
{p1['user_input']}

Transcript:
{transcript}""")
        steps = d.get("steps") or []
        if not steps:
            raise CacheInvocationError("P2: decomposition returned no steps")
        return steps

    # -- Phase 3: parameter recognition -------------------------------------------
    def _phase3(self, p1: dict, steps: list[dict]) -> dict:
        d = self._gen_json(
            "P3-parameters",
            "You identify the variable parameters of a problem class. Reply with JSON only.",
            f"""The problem below belongs to a class of problems whose reasoning is
identical but whose input varies. Identify the atomic input parameters (each a
single, self-contained piece of information the reasoning directly consumes —
never a multi-sentence narrative; keep variable-size collections as ONE parameter).
For each: "description" (a natural-language phrase the value is described by in the
input), "value" (the concrete value in THIS instance, verbatim where possible),
"type" (python type name), "validity" (constraints/range for valid values).
Also "template": the generic input template with <PLACEHOLDER> markers.
JSON: {{"template":"...","parameters":[{{"description":"...","value":"...","type":"int","validity":"..."}}]}}

Problem:
{p1['user_input']}

Reasoning steps (for context):
{json.dumps(steps, indent=1)}""")
        params = d.get("parameters") or []
        if not params:
            raise CacheInvocationError("P3: no parameters recognized")
        return d

    # -- Phase 4: planning ----------------------------------------------------------
    def _phase4(self, p1: dict, steps: list[dict], p3: dict,
                transcript: str) -> list[FnSpec]:
        d = self._gen_json(
            "P4-planning",
            "You plan the semantic-function network that will re-execute the reasoning. "
            "Reply with JSON only.",
            f"""Plan the functions that will solve this problem class. Rules:
- Each function = one coherent reasoning stage; reuse the step list.
- A function description is an IMPERATIVE self-contained sentence (it is matched
  semantically later, so it must work without context).
- Functions may depend on other functions (their output dict is fetched with
  load(result, "...")).
- Exactly one function is the root (is_root=true): it consumes the problem
  parameters via load(ctx, "<parameter description>") and returns the final
  answer dict.
- The plan must NOT depend on the specific parameter values.
- A named standard algorithm mentioned in the transcript (e.g. the Euclidean
  algorithm) is ONE self-contained function — do not split it into
  per-iteration / per-division sub-functions; the code must compute the
  algorithm generically for any inputs, never the transcript's literal
  intermediate values.
- STRONGLY prefer a SINGLE root function that computes the whole answer in one
  coherent body (e.g. LCM of N numbers = ONE root function that factorizes each
  input and multiplies the highest prime powers; GCD = ONE Euclidean loop). Only
  split into multiple functions when a sub-step is itself a genuinely reusable,
  independently-cached algorithm AND the root genuinely needs it as an input. A
  plan where every non-root function's output is consumed by exactly one other
  function (a linear chain) is almost always over-decomposition — collapse it
  into the root.
JSON: {{"functions":[{{"id":"f1","description":"...","depends_on":["f2"],"is_root":false}}],
       "build_order":["f2","f1"]}}   (dependencies first)

Problem:
{p1['user_input']}

Parameters:
{json.dumps(p3.get('parameters'), indent=1)}

Steps:
{json.dumps(steps, indent=1)}

Transcript:
{transcript}""")
        raw = d.get("functions") or []
        specs = []
        for f in raw:
            specs.append(FnSpec(id=str(f.get("id", f"fn{len(specs) + 1}")),
                                description=str(f.get("description", "")).strip(),
                                depends_on=[str(x) for x in f.get("depends_on", [])],
                                is_root=bool(f.get("is_root"))))
        roots = [s for s in specs if s.is_root]
        if len(roots) != 1:
            raise CacheInvocationError(f"P4: expected exactly one root, got {len(roots)}")
        return specs

    # -- Phase 5: implementation loop -----------------------------------------------
    def _dispatch_candidate(self, fn, params: list[Param]) -> dict:
        ctx = Context(self.arc.embedder, self.arc.thresholds)
        bind(self.arc.embedder, self.arc.registry, self.arc.lm, ctx=ctx, tools=self.arc.tools)
        for p in params:
            ctx.layers[0].add(p.description, p.value, self.arc.embedder)
        return fn(ctx)

    def _loads_compatible(self, fn: dict, params: list[Param]) -> bool:
        """True if every load(ctx, "<desc>") in fn's source resolves against
        this entry's parameters (exact case-insensitive, or embedding cosine
        >= 0.72 — the builder reuses functions across entries whose parameter
        names were paraphrased). A load with no resolvable target means the
        function reads a parameter this entry does not provide."""
        src = fn.get("source") or ""
        if not src:
            return False
        descs = [p.description for p in params]
        load_pat = re.compile(r'load\(\s*ctx\s*,\s*r?"((?:[^"\\]|\\.)*)"')
        queries = [m.group(1).strip().lower() for m in load_pat.finditer(src)]
        if not queries:
            return True
        try:
            import numpy as np
            emb = self.arc.embedder
            d_vecs = []
            for d in descs:
                pv = np.asarray(emb.embed(d), dtype=np.float32)
                d_vecs.append(pv / (np.linalg.norm(pv) + 1e-12))
            for q in queries:
                if any(d.strip().lower() == q for d in descs):
                    continue
                qv = np.asarray(emb.embed(q), dtype=np.float32)
                qn = float(np.linalg.norm(qv))
                if qn < 1e-12:
                    return False
                best = max(float(qv @ dv / qn) for dv in d_vecs)
                if best < 0.72:
                    return False
        except Exception:
            # embedding unavailable: be conservative, require exact matches only
            return all(any(d.strip().lower() == q for d in descs) for q in queries)
        return True

    def _implement_one(self, spec: FnSpec, all_specs: list[FnSpec],
                       p1: dict, params: list[Param], transcript: str,
                       result: BuildResult) -> None:
        # 1) reuse check against existing registry — but only if the reused
        # function's load() calls are satisfiable by THIS entry's parameters.
        # (A sibling entry's function may load params this entry doesn't have,
        # e.g. a 3-int function reused by a 2-int entry -> KeyError at serve.)
        hits = self.arc.registry.search(spec.description, top_k=1)
        if hits and hits[0][1] >= 0.92:
            cand = hits[0][0]
            if self._loads_compatible(cand, params):
                result.reused.append(spec.id)
                result.registered[spec.id] = cand["id"]
                return
            # incompatible -> fall through and generate a fresh implementation

        name = spec.id
        dep_ctx = "\n".join(
            f"- {s.id}: {s.description}" for s in all_specs if s.id in spec.depends_on)
        # CRITICAL: for functions with dependencies, the dispatch context only
        # contains the PROBLEM PARAMETERS — not the dependency outputs. The model
        # must use do(ctx, "<dep description>") to invoke its dependencies at
        # runtime (do() discovers + runs the registered function and returns its
        # output dict). If we tell it to load() the dep output directly, vector
        # search matches a parameter entry instead and coercion produces garbage.
        if spec.depends_on:
            dep_lines = []
            for s in all_specs:
                if s.id not in spec.depends_on:
                    continue
                dep_lines.append(f'  - {s.id}: do(ctx, "{s.description}") -> their output dict')
            dep_instruction = (
                "Depends on (ALREADY REGISTERED — invoke them at runtime with do()):\n"
                + "\n".join(dep_lines)
                + "\n\nCRITICAL RULES:\n"
                "1. Use the EXACT description string shown above as the first argument to do(). Do NOT paraphrase, shorten, or rewrite it.\n"
                "2. Do NOT pass payloads= to do() for dependencies — they call load() themselves for problem parameters. Just call do(ctx, \"<exact description>\").\n"
                "3. The dependency's output dict has natural-language keys. To extract a value, iterate over the dict: `for k, v in result.items(): if isinstance(v, <expected_type>): value = v`. Do NOT use .get() with guessed key names — the actual keys are unknown and will silently return None/default.\n"
                "4. Use load(ctx, \"<parameter description>\") ONLY for the problem parameters listed below."
            )
        else:
            dep_instruction = (
                "Depends on: (none) — use load(ctx, \"<parameter description>\") to fetch inputs."
            )
        user = f"""Implement this semantic function.

Function id: {name}
Its description: {spec.description}

SCOPE RULE (critical): implement EXACTLY what the description above says and
return ONLY that described output. Do NOT solve the whole original problem here
and do NOT return the final answer unless this function IS the root. If your
description says "return a dictionary mapping each prime factor to its
exponent", return exactly that dict — not the LCM, not the GCD, not any other
final value. The root function is the only one that returns the final answer.
OUTPUT KEY: the returned dict's key is a free-form natural-language description
of the value (e.g. "the least common multiple of the given integers"). Pick a
clear descriptive key and keep it stable across rewrites — do not change the
key to fix a judging complaint about its wording; only change the VALUE.
{dep_instruction}

Original problem:
{p1['user_input']}

Parameters of this instance (available via load()):
{json.dumps([{'description': p.description, 'value': p.value, 'type': p.type}
             for p in params], indent=1)}

The source model full transcript (your function outputs must be consistent with it):
{transcript}"""

        feedback = ""
        failures: list[str] = []
        for _attempt in range(self.max_rewrites + 1):
            print(f"    [builder] {name}: codegen attempt {_attempt+1} ...",
                  flush=True)
            reply = self._gen(RUNTIME_CONTRACT.replace("__FUNC_NAME__", name) + feedback,
                              user, tag=f"{name} a{_attempt+1}")
            # llama.cpp quirk: occasionally returns empty content (finish_reason=stop,
            # zero completion tokens) on long structured prompts. Retry once before
            # counting the attempt as a failure.
            if not reply.strip():
                time.sleep(1.0)
                reply = self._gen(RUNTIME_CONTRACT.replace("__FUNC_NAME__", name) + feedback,
                                  user, tag=f"{name} a{_attempt+1}r")
            try:
                src = _extract_source(reply)
                check_source(src)
                # The sandbox names the function after its id (spec.id); if the
                # model emits a differently-named def, dispatch would fail with a
                # confusing AttributeError on ctx. Enforce the name statically.
                import ast as _ast
                _fn_names = [n.name for n in _ast.parse(src).body
                             if isinstance(n, _ast.FunctionDef)]
                if not _fn_names or _fn_names[0] != name:
                    raise ValueError(
                        f"generated function must be named {name!r}, "
                        f"got {_fn_names}")
                fn = exec_function_source(src, _primitives(), timeout_s=30.0)
                out = self._dispatch_candidate(fn, params)
                ok, critique = self._judge(
                    f"""Function under test: "{spec.description}".
Its output: {json.dumps(out, default=str)[:1500]}
Problem: {p1['user_input']}
Source transcript:
{transcript}
OUTPUT CONTRACT: by design every semantic function returns a dict of
{{"natural-language description": value}}. The dict wrapping is CORRECT and
expected — never reject or critique the key wording, the dict shape, or the
fact that the value is wrapped in a dict. Judge ONLY whether the VALUE(s) are
numerically correct and consistent with the transcript reasoning. If the
value is right, pass=true.""" + (
                        "\nNOTE this function is NOT the root; its described output "
                        "is an intermediate value, not the final answer."
                        if not spec.is_root else ""))
                if ok:
                    fid = self.arc.registry.register(spec.description, fn, source=src,
                                                     meta={"spec_id": spec.id})
                    result.registered[spec.id] = fid
                    return
                failures.append(f"judge: {critique}")
                feedback = (
                    f"\n\nYour previous candidate FAILED judging: {critique}\n"
                    f"CRITICAL: keep every load(ctx, \"...\") query string EXACTLY as it "
                    f"was — do NOT reword them (rewording breaks context matching and "
                    f"returns the wrong type). Change ONLY the computation or the "
                    f"returned value. Return ONLY a corrected ```python fenced block.")
            except (ValueError, json.JSONDecodeError, SandboxError) as e:
                failures.append(f"codegen/sandbox/dispatch: {e!r}")
                feedback = (
                    f"\n\nYour previous candidate FAILED: {e!r}\n"
                    f"CRITICAL: keep every load(ctx, \"...\") query string EXACTLY as it "
                    f"was — do NOT reword them. Change ONLY the buggy computation. "
                    f"Return ONLY a corrected ```python fenced block.")
            except AttributeError as e:
                # runtime bug in the generated code (e.g. wrong value type from
                # load()); feed it back so the model can fix its assumptions
                failures.append(f"runtime: {e!r}")
                feedback = (
                    f"\n\nYour previous candidate FAILED at runtime: {e!r}\n"
                    f"Check what each load() actually returns and fix your code. "
                    f"CRITICAL: keep every load(ctx, \"...\") query string EXACTLY as it "
                    f"was — do NOT reword them (rewording breaks context matching). "
                    f"Return ONLY a corrected ```python fenced block.")
            except CacheInvocationError as e:
                failures.append(f"dispatch: {e!r}")
                # dispatch failure is usually a runtime bug in the candidate; feed back.
                feedback = (
                    f"\n\nYour previous candidate FAILED at runtime: {e!r}\n"
                    f"CRITICAL: keep every load(ctx, \"...\") query string EXACTLY as it "
                    f"was — do NOT reword them. Change ONLY the buggy computation. "
                    f"Return ONLY a corrected ```python fenced block.")
        result.errors.append(
            f"function {spec.id} failed after {self.max_rewrites + 1} attempts: "
            + " | ".join(failures))
        raise CacheInvocationError(f"P5: function {spec.id} could not be implemented")

    def _topo_order(self, specs: list[FnSpec]) -> list[FnSpec]:
        by_id = {s.id: s for s in specs}
        seen: set[str] = set()
        out: list[FnSpec] = []

        def visit(s: FnSpec) -> None:
            if s.id in seen:
                return
            seen.add(s.id)
            for dep in s.depends_on:
                if dep in by_id:
                    visit(by_id[dep])
            out.append(s)

        root = next(s for s in specs if s.is_root)
        visit(root)
        for s in specs:  # include any planned functions not reachable from root
            visit(s)
        return out

    # -- top-level ---------------------------------------------------------------
    def build(self, query: str, transcript: str, golden: str | None = None,
              family: str | None = None) -> BuildResult:
        t0 = time.time()
        result = BuildResult(ok=False, query=query)
        try:
            p1 = self._phase1(query, transcript)
            result.final_answer = golden or str(p1.get("final_answer", ""))
            steps = self._phase2(p1, transcript)
            p3 = self._phase3(p1, steps)
            result.template = str(p3.get("template", ""))
            params = [Param(description=str(p["description"]), value=str(p["value"]),
                            type=str(p.get("type", "str")),
                            validity=str(p.get("validity", "")))
                      for p in p3["parameters"]]
            result.params = params
            specs = self._phase4(p1, steps, p3, transcript)
            result.functions = specs

            for spec in self._topo_order(specs):
                self._implement_one(spec, specs, p1, result.params, transcript, result)

            root_spec = next(s for s in specs if s.is_root)
            root_id = result.registered[root_spec.id]
            # register BOTH the generic template and the original input as entry
            # points (paper §2.2.1 Phase 5)
            # family tag drives per-family entry thresholds at serve time
            self.arc.entry_store.register(query, root_id, [p.description for p in params],
                                          family=family,
                                          param_types=[p.type for p in params])
            if result.template and "<" in result.template:
                self.arc.entry_store.register(result.template, root_id,
                                              [p.description for p in params],
                                              family=family,
                                              param_types=[p.type for p in params])

            # Root end-to-end validation (extraction round-trip xN)
            for i in range(self.roundtrips):
                out = self.arc.use(query)
                if out is False:
                    raise CacheInvocationError("root validation: entry-point miss on original")
                result.root_output = out["result"]
                result.narration = out["narration"]
                ok, critique = self._judge(
                    f"""Final answer candidate (from cache): {json.dumps(out['result'], default=str)[:800]}
Correct answer: {result.final_answer}
Problem: {query}
OUTPUT CONTRACT: the candidate is a dict of {{"description": value}} by design —
the dict wrapping and key wording are correct; judge ONLY the numeric VALUE
against the correct answer (which may itself be stated as a bare number).
If the value matches, pass=true.""")
                result.roundtrips.append({"round": i + 1, "result": out["result"],
                                          "pass": ok, "critique": critique})
                if not ok:
                    result.errors.append(f"round {i + 1}: {critique}")

            result.ok = all(r["pass"] for r in result.roundtrips)
        except CacheInvocationError as e:
            result.errors.append(str(e))
        except Exception as e:  # noqa: BLE001 — builder must never crash the caller
            result.errors.append(f"unexpected: {e!r}")
        finally:
            result.cost = self.ledger.snapshot()
            result.ms = (time.time() - t0) * 1000
            print(f"    [builder] build done: ok={result.ok} in "
                  f"{result.ms/1000:.0f}s", flush=True)
        return result
