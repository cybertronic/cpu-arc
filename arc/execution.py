"""Worker-pool executor: builder-generated code runs in child processes.

The gateway treats the pool as an opaque executor:

    pool.submit(params) -> result dict

Workers are fresh Python interpreters (see arc/worker.py) with rlimit
budgets and a per-call CPU alarm. A dead worker (hard CPU wall, os._exit,
segfault) is detected via EOF/exit code and lazily respawned on the next
submit. One call in flight per worker; concurrent gateway threads are
spread across workers by round-robin.
"""
from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("arc.execution")


class ExecutionError(Exception):
    """Worker reported an execution failure (or protocol broke)."""


class ExecutionTimeout(ExecutionError):
    """Call exceeded the wall-clock budget (worker was killed + respawned)."""


class _Worker:
    """One child process + its stdio pipes. NOT thread-safe by itself;
    the pool serializes access (one call in flight per worker)."""

    def __init__(self, proc: subprocess.Popen, index: int):
        self.proc = proc
        self.index = index
        self.busy = False
        self.dead = False
        self.buf = b""
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._err_thread.start()

    def _drain_stderr(self) -> None:
        """Keep the stderr pipe from filling (would block the worker)."""
        try:
            for line in iter(self.proc.stderr.readline, b""):
                text = line.decode(errors="replace").rstrip()
                if text:
                    log.warning("worker[%d] stderr: %s", self.index, text)
        except (ValueError, OSError):
            pass

    def read_line(self) -> str | None:
        """Blocking read of one newline-terminated line; None on EOF."""
        while b"\n" not in self.buf:
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                if self.buf:
                    line, self.buf = self.buf, b""
                    return line.decode(errors="replace")
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode(errors="replace")

    def alive(self) -> bool:
        if self.dead:
            return False
        rc = self.proc.poll()
        if rc is not None:
            self.dead = True
            log.warning("worker[%d] exited rc=%s", self.index, rc)
            return False
        return True

    def kill(self) -> None:
        try:
            self.proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        self.dead = True


class WorkerPool:
    """Fixed pool of worker subprocesses executing one compiled module."""

    def __init__(self, module_path: str | Path, n_workers: int = 2,
                 cpu_seconds: float = 5.0, mem_mb: int = 1024,
                 call_timeout_s: float = 10.0, cpu_lifetime_s: float = 300.0,
                 startup_timeout_s: float = 30.0):
        self.module_path = Path(module_path).resolve()  # workers chdir to its parent
        self.n_workers = n_workers
        self.cpu_seconds = cpu_seconds
        self.mem_mb = mem_mb
        self.call_timeout_s = call_timeout_s
        self.cpu_lifetime_s = cpu_lifetime_s
        self._startup_timeout_s = startup_timeout_s
        self._lock = threading.Lock()
        self._rr = 0
        self._next_id = 1
        self.restarts = 0
        self._workers: list[_Worker] = []
        self._shutdown = False
        self._spawn_all()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _env(self) -> dict:
        env = dict(os.environ)
        arc_root = str(Path(__file__).resolve().parent.parent)
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = arc_root + (os.pathsep + pp if pp else "")
        # Cap glibc malloc arenas: without this, multi-core hosts reserve
        # ~64MB of VA per arena (up to 8*cores), which can blow the
        # RLIMIT_AS ceiling during normal numpy/python imports.
        env["MALLOC_ARENA_MAX"] = "2"
        return env

    def _spawn(self, slot: int) -> _Worker:
        proc = subprocess.Popen(
            [sys.executable, "-m", "arc.worker",
             "--module", str(self.module_path),
             "--cpu-seconds", str(self.cpu_seconds),
             "--cpu-lifetime", str(self.cpu_lifetime_s),
             "--mem-mb", str(self.mem_mb)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(self.module_path.parent), env=self._env())
        w = _Worker(proc, slot)
        line = w.read_line()
        if line is None:
            w.kill()
            raise ExecutionError("worker died during startup (see logs)")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            w.kill()
            raise ExecutionError(f"worker sent non-JSON startup line: {line[:120]!r}") from e
        if not msg.get("ok"):
            w.kill()
            raise ExecutionError(f"worker failed to start: {msg.get('error')}")
        return w

    def _spawn_all(self) -> None:
        self._workers = [self._spawn(i) for i in range(self.n_workers)]

    def restart(self, new_module_path: str | Path | None = None) -> None:
        """Replace all workers (used after a cache/module reload)."""
        if new_module_path is not None:
            self.module_path = Path(new_module_path).resolve()
        with self._lock:
            for w in self._workers:
                w.kill()
            self._spawn_all()
        log.info("worker pool restarted: %d workers, module=%s",
                 self.n_workers, self.module_path)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            for w in self._workers:
                w.kill()

    # ------------------------------------------------------------------ #
    # Serving
    # ------------------------------------------------------------------ #

    def _acquire(self, deadline: float) -> _Worker:
        """Round-robin pick of a live, idle worker; wait for one to free."""
        while True:
            with self._lock:
                if self._shutdown:
                    raise ExecutionError("worker pool is shutting down")
                for i in range(self.n_workers):
                    slot = (self._rr + i) % self.n_workers
                    w = self._workers[slot]
                    if w.dead:
                        w = self._spawn(slot)  # may raise ExecutionError
                        self._workers[slot] = w
                        self.restarts += 1
                        log.info("worker respawned at slot %d (restart #%d)",
                                 slot, self.restarts)
                    if not w.busy:
                        w.busy = True
                        self._rr = (slot + 1) % self.n_workers
                        return w
            if time.monotonic() >= deadline:
                raise ExecutionTimeout(
                    f"no idle worker within {self.call_timeout_s:g}s")
            time.sleep(0.005)

    def _release(self, w: _Worker, broken: bool = False) -> None:
        with self._lock:
            w.busy = False
            if broken and not w.dead:
                w.kill()

    def _read_response(self, w: _Worker, deadline: float) -> dict:
        """Wait for this worker's next JSON line; kill the worker on
        timeout or EOF."""
        while True:
            if not w.alive():
                raise ExecutionError(f"worker {w.index} died mid-call")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ExecutionTimeout(
                    f"call exceeded {self.call_timeout_s:g}s wall budget")
            ready, _, _ = select.select([w.proc.stdout], [], [], min(remaining, 0.25))
            if not ready:
                continue
            line = w.read_line()
            if line is None:
                raise ExecutionError(f"worker {w.index} closed stdout mid-call")
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                log.warning("worker[%d] sent non-JSON line: %r", w.index, line[:120])

    def submit(self, params: dict) -> dict:
        """Execute solve(params) in a worker; return its result dict."""
        rid = self._next_id
        self._next_id += 1
        deadline = time.monotonic() + self.call_timeout_s
        w = self._acquire(deadline)
        broken = False
        try:
            req = json.dumps({"id": rid, "kind": "call", "params": params},
                             default=str) + "\n"
            w.proc.stdin.write(req.encode())
            w.proc.stdin.flush()
            msg = self._read_response(w, deadline)
            if msg.get("id") != rid:
                raise ExecutionError(
                    f"protocol desync: expected id {rid}, got {msg.get('id')!r}")
            if not msg.get("ok"):
                raise ExecutionError(msg.get("error", "unknown worker error"))
            return msg["result"]
        except (ExecutionTimeout, ExecutionError):
            broken = True
            raise
        except Exception as e:  # noqa: BLE001 — socket/pipe errors
            broken = True
            raise ExecutionError(f"executor failure: {type(e).__name__}: {e}") from e
        finally:
            if broken:
                self._release(w, broken=True)
            else:
                self._release(w)
