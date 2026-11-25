"""P2 offline test: scripted full 5-phase builder run (MockLM, zero GPU).

Scenario: a fresh problem class (divisors of N) is "solved" by a source-model
transcript; the builder must produce a cache entry; a FRESH ARC reloaded from
the persisted DB must then serve a reparameterized instance (N=144 -> 15).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc import ARC, Thresholds, Embedder, Ledger, MockLM
from arc.builder import Builder

PASS, FAIL = 0, []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name} {detail}")


QUERY = "How many positive whole-number divisors does 196 have?"
TRANSCRIPT = (
    "I need the prime factorisation of 196 first. 196 = 4 x 49 = 2^2 x 7^2.\n"
    "For a number with prime factorisation p1^a1 x p2^a2 x ..., the number of divisors "
    "is (a1+1)(a2+1)...\n"
    "For 196 = 2^2 x 7^2: (2+1)(2+1) = 3 x 3 = 9.\n"
    "Answer: 9"
)

SRC_F1 = """```python
def f1(ctx):
    n = load(ctx, 'the number n', d_type=int)
    narrate(ctx, f'Factoring {n}')
    d, facs = 2, []
    while d * d <= n:
        c = 0
        while n % d == 0:
            n //= d
            c += 1
        if c:
            facs.append((d, c))
        d += 1
    if n > 1:
        facs.append((n, 1))
    narrate(ctx, f'Prime factorization: {facs}')
    return {'prime factorization of n': facs}
```"""

SRC_F2 = """```python
def f2(ctx):
    n = load(ctx, 'the number n', d_type=int)
    narrate(ctx, f'Counting divisors of {n}')
    fr = do(ctx, 'Compute the prime factorization of a positive integer n', payloads={'the number n': n})
    facs = load(fr, 'prime factorization of n', d_type=list)
    count = 1
    for p, e in facs:
        count *= (e + 1)
    narrate(ctx, f'Number of divisors = {count}')
    return {'the number of positive divisors': count}
```"""

ROUTES = {
    # phase 1 — isolation
    "Given the user query and the model":
        '{"user_input": "How many positive whole-number divisors does 196 have?", '
        '"final_answer": "9"}',
    # phase 2 — decomposition
    "Decompose the reasoning below":
        '{"steps": [{"id": "s1", "description": "Compute the prime factorization of the '
        'number n", "reusable": true}, {"id": "s2", "description": "Count the positive '
        'divisors of n from its prime factorization", "reusable": false}]}',
    # phase 3 — parameters
    "belongs to a class of problems":
        '{"template": "How many positive whole-number divisors does <N> have?", '
        '"parameters": [{"description": "the number n", "value": "196", "type": "int", '
        '"validity": "n is a positive integer >= 2"}]}',
    # phase 4 — planning
    "Plan the functions that will solve":
        '{"functions": [{"id": "f1", "description": "Compute the prime factorization of '
        'a positive integer n", "depends_on": [], "is_root": false}, '
        '{"id": "f2", "description": "Count the positive divisors of a number from its '
        'prime factorization", "depends_on": ["f1"], "is_root": true}], '
        '"build_order": ["f1", "f2"]}',
    # phase 5 — implementation (per-function source)
    "Function id: f1": SRC_F1,
    "Function id: f2": SRC_F2,
    # judge
    "Is the function output correct": '{"pass": true, "critique": "consistent with transcript"}',
    "Final answer candidate": '{"pass": true, "critique": "matches golden answer"}',
    # extraction (IS2)
    "the number n": "196",
}


def main():
    tmp = Path(tempfile.mkdtemp(prefix="arc_p2_"))
    embedder = Embedder(backend="hash")
    ledger = Ledger()
    arc = ARC(embedder, MockLM(routes=ROUTES, default="NOT_FOUND"),
              Thresholds(entry=0.3, function=0.3, load=0.3),
              db_path=tmp / "cache", ledger=ledger,
              invocation_log=tmp / "invocations.jsonl")
    b = Builder(arc, builder_lm=MockLM(routes=ROUTES, default="NOT_FOUND"),
                judge_lm=MockLM(routes=ROUTES, default="NOT_FOUND"),
                max_rewrites=2, roundtrips=2)

    print("[1] build from transcript (5 phases)")
    res = b.build(QUERY, TRANSCRIPT, golden="9")
    print(f"  ok={res.ok} rewrites={res.rewrites} ms={res.ms:.0f} errors={res.errors}")
    for r in res.roundtrips:
        print(f"    round {r['round']}: {r['result']} pass={r['pass']}")
    check("build succeeded", res.ok, f"errors={res.errors}")
    check("two functions registered", len(res.registered) == 2,
          f"registered={res.registered}")
    check("no rewrites needed (mock is correct)", res.rewrites == 0, f"r={res.rewrites}")
    check("params recognized", len(res.params) == 1 and res.params[0].value == "196")
    check("template produced", "<N>" in res.template)
    check("build cost recorded", res.cost["by_role"].get("build", {}).get("calls", 0) >= 6,
          f"cost={res.cost}")
    check("root output correct",
          res.root_output and res.root_output.get("the number of positive divisors") == 9,
          f"out={res.root_output}")

    print("[2] fresh ARC from persisted DB serves a REPARAMETERIZED instance")
    fresh = ARC(Embedder(backend="hash"),
                MockLM(routes={"the number n": "144", "look carefully": "144"},
                       default="NOT_FOUND"),
                Thresholds(entry=0.3, function=0.3, load=0.3), db_path=tmp / "cache",
                ledger=Ledger())
    check("functions reloaded", len(fresh.registry.functions) == 2,
          f"n={len(fresh.registry.functions)}")
    check("entries reloaded", len(fresh.entry_store.entries) == 2,
          f"n={len(fresh.entry_store.entries)}")
    out = fresh.use("How many positive whole-number divisors does 144 have?")
    ok = out is not False and out["result"].get("the number of positive divisors") == 15
    check("144 -> 15 (144 = 2^4 x 3^2)", bool(ok), f"out={out if out is False else out.get('result')}")
    if ok:
        for line in out["narration"]:
            print(f"        {line}")
        print(f"        entry_score={out['entry_score']:.3f} "
              f"lm_in={out['lm_stats']['by_role'].get('extractor', {}).get('in', 0)} tok")

    print("[3] template entry point also resolves (generic form)")
    out2 = fresh.use("How many positive whole-number divisors does <N> have?")
    # generic template hit: extraction from a template is meaningless -> expect
    # either a miss or a graceful error, never a crash
    check("template entry handled gracefully", out2 is False or isinstance(out2, dict))

    print(f"\nP2 OFFLINE RESULT: {PASS} passed, {len(FAIL)} failed {FAIL if FAIL else ''}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
