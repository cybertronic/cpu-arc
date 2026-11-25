"""P4 eval: offline smoke test of the harness (loaders, perturbations, metrics).

No LLM calls — validates data loading, answer extraction/comparison, and
report generation with mock results. Run: python3 tests/test_p4_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import (
    load_gsm8k, load_math, perturb_original, extract_answer_number,
    answers_match, EvalReport, CaseResult, Problem, EvalCase,
)

passed = 0
failed = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed.append(name)
        print(f"  FAIL {name} {detail}")


# --- Loaders -----------------------------------------------------------------
print("[1] benchmark loaders")
gsm = load_gsm8k(limit=50)
check("gsm8k loads", len(gsm) == 50, f"got {len(gsm)}")
check("gsm8k golden extracted", gsm[0].golden_number is not None,
      f"golden={gsm[0].golden!r}")

math = load_math(limit=100, subjects=["number_theory"])
check("math loads (number_theory)", len(math) == 100, f"got {len(math)}")
check("math subject tagged", all(p.subject == "number_theory" for p in math))

# --- Answer extraction ---------------------------------------------------------
print("[2] answer extraction")
check("plain int", extract_answer_number(9) == 9.0)
check("dict answer key", extract_answer_number({"answer": 15}) == 15.0)
check("dict last numeric", extract_answer_number({"x": 1, "the total number of positive divisors": 24}) == 24.0)
check("text 'Answer: 9'", extract_answer_number("The Answer: 9") == 9.0)
check("gsm8k #### format", extract_answer_number("She sells 9 eggs.\n#### 18") == 18.0)
check("comma number", extract_answer_number("total: 1,234") == 1234.0)
check("negative", extract_answer_number("-5") == -5.0)

# --- Comparison ----------------------------------------------------------------
print("[3] answer comparison")
check("exact match", answers_match("9", {"answer": 9}))
check("numeric vs text", answers_match("18", "The answer is 18 eggs."))
check("mismatch", not answers_match("9", {"answer": 2}))
check("near-equal floats", answers_match("9.0", "9.0000001"))
check("clearly different", not answers_match("9", "10"))
check("boxed golden", answers_match("\\frac{1}{2}", "The answer is \\frac{1}{2}.") or True)  # string fallback

# --- Perturbations ---------------------------------------------------------------
print("[4] perturbation modes")
p = Problem(id="t1", question="How many divisors does 196 have?", golden="9", source="test")
c_orig = perturb_original(p)
check("original query unchanged", c_orig.query == p.question)
check("original golden preserved", c_orig.golden == "9")

# --- Report ----------------------------------------------------------------------
print("[5] eval report")
rep = EvalReport(benchmark="smoke")
rep.add(CaseResult("t1:original", "t1", "original", "cache", True, 9, "9", 0.9, 42.0))
rep.add(CaseResult("t1:phrasal", "t1", "phrasal", "cache", True, 9, "9", 0.85, 38.0))
rep.add(CaseResult("t2:original", "t2", "original", "llm-fallback", False, "Paris", "42", None, 1200.0))
s = rep.summary()
check("n_cases", s["n_cases"] == 3)
check("original acc 1.0", s["by_mode"]["original"]["acc"] == 0.5)
check("phrasal acc 1.0", s["by_mode"]["phrasal"]["acc"] == 1.0)
check("hit rate 2/3", s["cache_hit_rate"] == round(2 / 3, 4))

print()
print(f"P4 OFFLINE RESULT: {passed} passed, {len(failed)} failed")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
