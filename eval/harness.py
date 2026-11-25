"""P4 Eval harness — benchmark loaders, perturbation modes, metrics, cost report.

Mirrors the paper's protocol:
  - S (source) accuracy: raw source model answers benchmark problems
  - C (cache) accuracy: ARC gateway answers (build on first sight, serve after)
  - Perturbation modes: original / phrasal (rephrased) / parametric (new values)
  - Fidelity: judge compares cached answer vs golden label
  - Cost report: build cost amortized over N served invocations -> break-even

Data: local parquet files in eval/data/ (MATH test split, GSM8K test split).
No network needed at eval time.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class Problem:
    """One benchmark problem with golden answer."""
    id: str
    question: str
    golden: str            # canonical answer string (number or expression)
    source: str            # "gsm8k" | "math"
    subject: str = ""      # MATH subject / level
    solution: str = ""     # full worked solution (for judge context)

    @property
    def golden_number(self) -> float | None:
        """Extract the final numeric answer if present."""
        m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", self.golden.replace(",", ""))
        return float(m[-1]) if m else None


@dataclass
class EvalCase:
    """A problem + perturbation mode to evaluate."""
    problem: Problem
    mode: str              # "original" | "phrasal" | "parametric"
    query: str             # the actual query text sent to the system
    golden: str            # expected answer for THIS variant


@dataclass
class CaseResult:
    case_id: str
    problem_id: str
    mode: str
    source: str            # "cache" | "llm-fallback" | "error"
    correct: bool
    answer: Any = None
    golden: str = ""
    entry_score: float | None = None
    ms: float = 0.0
    error: str | None = None


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_gsm8k(data_dir: Path | None = None, limit: int | None = None) -> list[Problem]:
    """Load GSM8K test split (1319 problems)."""
    import pyarrow.parquet as pq
    data_dir = data_dir or DATA_DIR
    path = data_dir / "gsm8k_test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"GSM8K parquet not found: {path}")
    rows = pq.read_table(path).to_pylist()
    problems = []
    for i, r in enumerate(rows):
        # GSM8K answers end with "#### <number>"
        m = re.search(r"####\s*([\d.,]+)\s*$", r["answer"], re.MULTILINE)
        golden = m.group(1) if m else ""
        problems.append(Problem(
            id=f"gsm8k_{i:04d}",
            question=r["question"],
            golden=golden,
            source="gsm8k",
            subject="grade-school-math",
            solution=r["answer"],
        ))
    return problems[:limit] if limit else problems


def load_math(data_dir: Path | None = None, limit: int | None = None,
              subjects: list[str] | None = None) -> list[Problem]:
    """Load MATH test split (5000 problems across 7 subjects)."""
    import pyarrow.parquet as pq
    data_dir = data_dir or DATA_DIR
    subject_files = {
        "algebra": "math_algebra_test.parquet",
        "counting_and_probability": "math_counting_and_probability_test.parquet",
        "geometry": "math_geometry_test.parquet",
        "intermediate_algebra": "math_intermediate_algebra_test.parquet",
        "number_theory": "math_number_theory_test.parquet",
        "prealgebra": "math_prealgebra_test.parquet",
        "precalculus": "math_precalculus_test.parquet",
    }
    if subjects:
        subject_files = {k: v for k, v in subject_files.items() if k in subjects}

    problems = []
    idx = 0
    for subj, fname in sorted(subject_files.items()):
        path = data_dir / fname
        if not path.exists():
            continue
        rows = pq.read_table(path).to_pylist()
        for r in rows:
            # MATH solutions end with the answer; extract boxed or final number
            sol = r.get("solution", "")
            m = re.search(r"\\boxed\{([^}]+)\}", sol)
            if m:
                golden = m.group(1).strip()
            else:
                # fallback: last number in solution
                nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", sol.replace(",", ""))
                golden = nums[-1] if nums else ""
            problems.append(Problem(
                id=f"math_{idx:04d}",
                question=r["problem"],
                golden=golden,
                source="math",
                subject=subj,
                solution=sol,
            ))
            idx += 1
    return problems[:limit] if limit else problems


def load_benchmark(name: str, data_dir: Path | None = None,
                   limit: int | None = None, **kwargs) -> list[Problem]:
    """Load a benchmark by name: 'gsm8k' | 'math'."""
    if name == "gsm8k":
        return load_gsm8k(data_dir, limit)
    elif name == "math":
        return load_math(data_dir, limit, **kwargs)
    else:
        raise ValueError(f"unknown benchmark: {name!r}")


# --------------------------------------------------------------------------- #
# Perturbation modes (paper protocol)
# --------------------------------------------------------------------------- #

def perturb_original(p: Problem) -> EvalCase:
    """Original problem text, golden answer unchanged."""
    return EvalCase(problem=p, mode="original", query=p.question, golden=p.golden)


def perturb_phrasal(p: Problem, rephrase_fn) -> EvalCase:
    """Rephrased question (same math), same golden answer.

    rephrase_fn(question) -> rephrased_question
    Calibration: the rephrasing must preserve the exact numeric content.
    """
    q = rephrase_fn(p.question)
    return EvalCase(problem=p, mode="phrasal", query=q, golden=p.golden)


def perturb_parametric(p: Problem, mutate_fn) -> tuple[EvalCase | None, str]:
    """Replace numbers with new values; recompute golden via the source model.

    mutate_fn(question) -> (new_question, new_golden_or_None)
    Returns (case, note); case is None if mutation produced an invalid problem.
    """
    new_q, new_golden = mutate_fn(p.question)
    if new_q is None or new_golden is None:
        return None, "mutation rejected"
    return EvalCase(problem=p, mode="parametric", query=new_q, golden=str(new_golden)), "ok"


# --------------------------------------------------------------------------- #
# Answer comparison (fidelity)
# --------------------------------------------------------------------------- #

def extract_answer_number(text: Any) -> float | None:
    """Pull the final numeric answer out of a response (dict, str, or number)."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    if isinstance(text, dict):
        # compiled modules return {"answer": N} or {"the total ...": N}
        for k in ("answer", "result"):
            if k in text:
                return extract_answer_number(text[k])
        # take the last numeric value
        vals = [v for v in text.values() if isinstance(v, (int, float))]
        return float(vals[-1]) if vals else None
    s = str(text)
    # "Answer: 9" / "#### 18" patterns first. Models restate intermediate
    # answers in the reasoning, so prefer the LAST marker, not the first.
    markers = re.findall(r"(?:answer|####)\s*:?\s*(-?[\d,]+(?:\.\d+)?)", s, re.IGNORECASE)
    if markers:
        return float(markers[-1].replace(",", ""))
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s.replace(",", ""))
    return float(nums[-1]) if nums else None


def answers_match(golden: str, answer: Any, tol: float = 1e-6) -> bool:
    """Compare a golden label against a system answer (numeric or exact)."""
    g_num = extract_answer_number(golden)
    a_num = extract_answer_number(answer)
    if g_num is not None and a_num is not None:
        return abs(g_num - a_num) <= tol * max(1.0, abs(g_num))
    # fallback: string containment (for non-numeric answers)
    gs, as_ = str(golden).strip().lower(), str(answer).strip().lower()
    if not gs or not as_:
        return False
    return gs in as_ or as_ in gs


# --------------------------------------------------------------------------- #
# Metrics + cost report
# --------------------------------------------------------------------------- #

@dataclass
class EvalReport:
    benchmark: str
    n_cases: int = 0
    by_mode: dict = field(default_factory=dict)   # mode -> {n, correct, acc}
    s_acc: float | None = None                    # source-only accuracy (baseline)
    c_acc: float | None = None                    # cache accuracy
    cache_hit_rate: float | None = None
    build_cost_tokens: int = 0
    serve_cost_tokens: int = 0
    results: list[CaseResult] = field(default_factory=list)

    def add(self, r: CaseResult) -> None:
        self.results.append(r)
        self.n_cases += 1
        m = self.by_mode.setdefault(r.mode, {"n": 0, "correct": 0})
        m["n"] += 1
        if r.correct:
            m["correct"] += 1

    def finalize(self) -> None:
        for mode, m in self.by_mode.items():
            m["acc"] = round(m["correct"] / max(1, m["n"]), 4)

    def summary(self) -> dict:
        self.finalize()
        hits = [r for r in self.results if r.source == "cache"]
        out = {
            "benchmark": self.benchmark,
            "n_cases": self.n_cases,
            "by_mode": self.by_mode,
            "s_acc": self.s_acc,
            "c_acc": self.c_acc,
            "cache_hit_rate": round(len(hits) / max(1, self.n_cases), 4),
            "build_cost_tokens": self.build_cost_tokens,
            "serve_cost_tokens": self.serve_cost_tokens,
        }
        # Break-even: build cost amortized over served invocations.
        # If a cached entry serves N times, per-invocation cost = build/N + serve.
        if self.build_cost_tokens > 0 and len(hits) > 0:
            out["amortized_build_per_hit"] = round(self.build_cost_tokens / len(hits), 1)
        return out

    def to_json(self, path: Path | None = None) -> str:
        s = json.dumps(self.summary(), indent=2)
        if path:
            path.write_text(s)
        return s


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

class EvalRunner:
    """Runs eval cases against a gateway (or in-process ARC) + source LLM."""

    def __init__(self, gateway=None, source_lm=None, judge_lm=None,
                 rephrase_fn=None, mutate_fn=None):
        self.gateway = gateway          # ArcGateway instance (serve path)
        self.source_lm = source_lm      # LMBackend for S baseline + parametric goldens
        self.judge_lm = judge_lm        # optional judge for non-numeric answers
        self.rephrase_fn = rephrase_fn  # question -> rephrased
        self.mutate_fn = mutate_fn      # question -> (new_q, new_golden)

    def run_cases(self, cases: list[EvalCase]) -> EvalReport:
        report = EvalReport(benchmark="")
        for case in cases:
            r = self._run_one(case)
            report.add(r)
        return report

    def _run_one(self, case: EvalCase) -> CaseResult:
        cid = f"{case.problem.id}:{case.mode}"
        t0 = time.perf_counter()
        try:
            if self.gateway is not None:
                out = self.gateway.handle_query(case.query)
                source = out.get("source", "error")
                answer = out.get("answer")
                score = out.get("entry_score")
                ms = out.get("ms", 0.0)
            elif self.source_lm is not None:
                # S-only baseline (no cache)
                answer = self.source_lm.complete(case.query, role="source").text
                source, score, ms = "llm-fallback", None, 0.0
            else:
                return CaseResult(cid, case.problem.id, case.mode, "error",
                                  False, golden=case.golden, error="no backend")
        except Exception as e:
            return CaseResult(cid, case.problem.id, case.mode, "error",
                              False, golden=case.golden, error=str(e))

        correct = answers_match(case.golden, answer)
        return CaseResult(
            case_id=cid, problem_id=case.problem.id, mode=case.mode,
            source=source, correct=correct, answer=answer,
            golden=case.golden, entry_score=score, ms=ms * 1000 if ms < 1 else ms,
        )

    def run_benchmark(self, name: str, limit: int | None = None,
                      modes: list[str] | None = None,
                      subjects: list[str] | None = None) -> EvalReport:
        """Full eval: load benchmark, generate cases for each mode, run them."""
        modes = modes or ["original", "phrasal", "parametric"]
        problems = load_benchmark(name, limit=limit, subjects=subjects)

        cases: list[EvalCase] = []
        for p in problems:
            if "original" in modes:
                cases.append(perturb_original(p))
            if "phrasal" in modes and self.rephrase_fn:
                try:
                    cases.append(perturb_phrasal(p, self.rephrase_fn))
                except Exception as e:
                    print(f"  [warn] phrasal failed for {p.id}: {e}")
            if "parametric" in modes and self.mutate_fn:
                c, note = perturb_parametric(p, self.mutate_fn)
                if c is not None:
                    cases.append(c)

        report = self.run_cases(cases)
        report.benchmark = name
        return report
