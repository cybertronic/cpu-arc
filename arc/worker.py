"""ARC code-execution worker subprocess.

Isolation boundary for builder-generated code: the gateway NEVER execs
generated code in its own process. Each worker is a fresh Python
interpreter that loads the compiled module once and then services calls
over JSON lines on stdin/stdout.

Per-worker defenses (threat model: builder bugs + runaway loops, not a
hostile model on this host):
- RLIMIT_AS  — virtual-memory ceiling; runaway allocations die with
  MemoryError instead of OOM-killing the host.
- RLIMIT_CPU — cumulative CPU budget over the worker's lifetime; the hard
  wall SIGKILLs the worker and the pool lazily respawns it.
- setitimer  — per-call CPU budget; the SIGALRM raises inside the call,
  which is reported as an error and the worker keeps serving.

Protocol (JSON lines):
  -> {"id": 1, "kind": "ping"}
  <- {"id": 1, "ok": true, "result": "pong"}
  -> {"id": 2, "kind": "call", "params": {...}}
  <- {"id": 2, "ok": true, "result": {"answer": 15}}
  <- {"id": 2, "ok": false, "error": "CPU budget exceeded (5s)"}
Startup:
  <- {"ok": true, "result": "ready"}   (first line, after module load)

A worker that dies (hard CPU wall, os._exit, segfault) simply stops
responding; the pool detects EOF/exit and respawns on next use.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import signal
import sys


class _CPUBudgetExceeded(Exception):
    """Raised by the SIGALRM handler when a call exceeds its CPU budget."""


def _install_cpu_alarm() -> None:
    def handler(signum, frame):
        raise _CPUBudgetExceeded()
    signal.signal(signal.SIGALRM, handler)


def _apply_rlimits(cpu_lifetime_s: float, mem_mb: int) -> None:
    """Apply process-level budgets BEFORE anything big is loaded."""
    try:
        mem_bytes = int(mem_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError) as e:
        sys.stderr.write(f"warn: RLIMIT_AS not applied: {e}\n")
    try:
        soft = int(max(1.0, cpu_lifetime_s))
        resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 5))
    except (ValueError, OSError) as e:
        sys.stderr.write(f"warn: RLIMIT_CPU not applied: {e}\n")


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("arc_worker_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="ARC code-execution worker")
    ap.add_argument("--module", required=True, help="path to compiled arc_compiled.py")
    ap.add_argument("--cpu-seconds", type=float, default=5.0,
                    help="per-call CPU budget (SIGALRM; seconds)")
    ap.add_argument("--cpu-lifetime", type=float, default=300.0,
                    help="cumulative CPU budget for the worker's lifetime (rlimit; seconds)")
    ap.add_argument("--mem-mb", type=int, default=512,
                    help="virtual-memory ceiling (rlimit; MiB)")
    args = ap.parse_args()

    _apply_rlimits(args.cpu_lifetime, args.mem_mb)
    _install_cpu_alarm()

    try:
        mod = _load_module(args.module)
        solve = getattr(mod, "solve")
        if not callable(solve):
            raise TypeError("compiled module has no callable solve(params)")
    except Exception as e:  # noqa: BLE001 — report + exit, pool falls back
        sys.stderr.write(f"worker: failed to load module: {type(e).__name__}: {e}\n")
        sys.stdout.write(json.dumps({"ok": False, "error": f"module load failed: {e}"}) + "\n")
        sys.stdout.flush()
        sys.exit(2)

    sys.stdout.write(json.dumps({"ok": True, "result": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id", 0)
        kind = req.get("kind")
        if kind == "ping":
            out = {"id": rid, "ok": True, "result": "pong"}
        elif kind == "call":
            signal.setitimer(signal.ITIMER_REAL, args.cpu_seconds)
            try:
                result = solve(req.get("params", {}))
                if not isinstance(result, dict):
                    raise TypeError(
                        f"semantic function must return dict, got {type(result).__name__}")
                out = {"id": rid, "ok": True, "result": result}
            except BaseException as e:  # noqa: BLE001 — CPU alarm included
                err = f"CPU budget exceeded ({args.cpu_seconds:g}s)" \
                    if isinstance(e, _CPUBudgetExceeded) else f"{type(e).__name__}: {e}"
                out = {"id": rid, "ok": False, "error": err}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        else:
            out = {"id": rid, "ok": False, "error": f"unknown kind {kind!r}"}
        try:
            sys.stdout.write(json.dumps(out, default=str) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, ValueError):
            return  # gateway went away


if __name__ == "__main__":
    main()
