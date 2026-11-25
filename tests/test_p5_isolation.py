"""P5 isolation test: builder-generated code executes in worker subprocesses.

Verifies, against a real WorkerPool (no LLM, no embedder):
  1. normal calls return correct results
  2. a runaway infinite loop is bounded (per-call CPU alarm) and the worker
     keeps serving
  3. a memory hog fails and its worker is killed + respawned
  4. a worker-suicidal call (os._exit) is contained; pool respawns
  5. a broken module (syntax error) fails pool startup cleanly
  6. concurrent calls across workers never cross wires
  7. warm-call latency stays in single-digit milliseconds
"""
import sys
import tempfile
import time
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc.execution import WorkerPool, ExecutionError, ExecutionTimeout

PASS, FAIL = 0, []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name}  {detail}")


TMP = Path(tempfile.mkdtemp(prefix="arc_p5_"))

DIVISORS_SRC = '''
PARAMS = ["n"]

def _count_divisors(n):
    d, i = 0, 1
    while i * i <= n:
        if n % i == 0:
            d += 1
            if i * i != n:
                d += 1
        i += 1
    return d

def solve(params: dict):
    n = int(str(params["n"]).replace(",", ""))
    return {"answer": _count_divisors(n)}
'''

RUNAWAY_SRC = '''
PARAMS = ["n"]

def solve(params: dict):
    x = 0
    while True:
        x += 1
'''

MEMHOG_SRC = '''
PARAMS = ["n"]

def solve(params: dict):
    return {"answer": len(bytearray(10**12))}
'''

KILLER_SRC = '''
import os
PARAMS = ["n"]

def solve(params: dict):
    os._exit(3)
'''

BAD_SRC = "this is not python (["  # syntax error at module load


def make_module(name, src):
    p = TMP / f"{name}.py"
    p.write_text(src)
    return p


def main():
    tmp_div = make_module("divisors", DIVISORS_SRC)
    tmp_run = make_module("runaway", RUNAWAY_SRC)
    tmp_mem = make_module("memhog", MEMHOG_SRC)
    tmp_kill = make_module("killer", KILLER_SRC)
    tmp_bad = make_module("bad", BAD_SRC)

    # 1. normal calls
    pool = WorkerPool(tmp_div, n_workers=2, cpu_seconds=2.0, call_timeout_s=5.0)
    r = pool.submit({"n": 196})
    check("divisors(196) == 9", r == {"answer": 9}, str(r))  # 196 = 2^2 * 7^2
    r = pool.submit({"n": 144})
    check("divisors(144) == 15", r == {"answer": 15}, str(r))
    r = pool.submit({"n": "1,000"})
    check("string param coerced: divisors('1,000') == 16", r == {"answer": 16}, str(r))

    # 2. runaway loop: per-call CPU alarm fires (~1s), error reported, and
    # the ORIGINAL divisors pool is untouched the whole time
    pool_run = WorkerPool(tmp_run, n_workers=1, cpu_seconds=1.0, call_timeout_s=6.0)
    t0 = time.perf_counter()
    try:
        pool_run.submit({"n": 1})
        check("runaway loop bounded", False, "no error raised")
    except (ExecutionError, ExecutionTimeout) as e:
        dt = time.perf_counter() - t0
        check("runaway loop bounded", dt < 6.0, f"{dt:.1f}s: {e}")
    # same worker keeps serving after a CPU-budget trip: a second call must
    # fail with the CPU error again (worker alive), not a death error
    try:
        pool_run.submit({"n": 1})
        alive_check = False
    except ExecutionError as e2:
        alive_check = "CPU budget" in str(e2) or "infinite" in str(e2)
        if not alive_check:
            alive_check = "died" not in str(e2) and "closed stdout" not in str(e2)
    check("worker still serving after CPU-budget trip", alive_check)
    pool_run.shutdown()
    check("divisors pool unaffected by other module's runaway",
          pool.submit({"n": 360}) == {"answer": 24})

    # 3. memory hog: worker killed + respawned, pool still serves
    pool_mem = WorkerPool(tmp_mem, n_workers=1, cpu_seconds=5.0, call_timeout_s=10.0)
    restarts_before = pool_mem.restarts
    try:
        pool_mem.submit({"n": 1})
        check("memory hog contained", False, "no error raised")
    except ExecutionError as e:
        check("memory hog contained", "MemoryError" in str(e) or "died" in str(e), str(e))
    time.sleep(0.2)
    # respawn proof: next submit reaches a FRESH worker (fails the same way,
    # but the restart counter must move)
    try:
        pool_mem.submit({"n": 1})
        check("worker respawned after memhog", False, "no error raised")
    except ExecutionError:
        check("worker respawned after memhog", pool_mem.restarts > restarts_before,
              f"restarts={pool_mem.restarts}")
    pool_mem.shutdown()

    # 4. worker suicide: os._exit(3) contained
    pool_kill = WorkerPool(tmp_kill, n_workers=1, cpu_seconds=5.0, call_timeout_s=10.0)
    try:
        pool_kill.submit({"n": 1})
        check("worker suicide contained", False, "no error raised")
    except ExecutionError as e:
        check("worker suicide contained", True, str(e))
    # now point the SAME pool at a healthy module: restarted workers must serve
    pool_kill.restart(tmp_div)
    check("pool restart serves healthy module", pool_kill.submit({"n": 196}) == {"answer": 9})
    pool_kill.shutdown()

    # 5. broken module: clean startup failure
    try:
        WorkerPool(tmp_bad, n_workers=1)
        check("broken module rejected at startup", False, "pool started")
    except ExecutionError as e:
        check("broken module rejected at startup", "load" in str(e).lower() or "died" in str(e), str(e))

    # 6. concurrency: 8 threads, 2 workers, distinct params
    pool_cc = WorkerPool(tmp_div, n_workers=2, cpu_seconds=5.0, call_timeout_s=15.0)
    ns = [12, 84, 65536, 999983, 1000, 360, 196, 144]

    def one(n):
        return n, pool_cc.submit({"n": n})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, ns))
    expected = {12: 6, 84: 12, 65536: 17, 999983: 2, 1000: 16, 360: 24, 196: 9, 144: 15}
    ok = all(res == {"answer": expected[n]} for n, res in results)
    check("8 concurrent calls, no cross-talk", ok, str(results))
    pool_cc.shutdown()

    # 7. warm latency
    pool_lat = WorkerPool(tmp_div, n_workers=1, cpu_seconds=5.0, call_timeout_s=5.0)
    pool_lat.submit({"n": 1})  # warm
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        pool_lat.submit({"n": 196})
        ts.append(time.perf_counter() - t0)
    med = sorted(ts)[len(ts) // 2]
    check("warm call latency < 15ms (median)", med < 0.015, f"{med*1000:.1f}ms")
    pool_lat.shutdown()

    print(f"\nP5 ISOLATION RESULT: {PASS} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
