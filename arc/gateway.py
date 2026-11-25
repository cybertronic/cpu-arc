"""ARC Gateway — sidecar proxy that serves cached answers in front of an LLM.

Production architecture (Option A):
    Client → [arc-gateway :8082] → hit?  → run compiled fn → respond (0 LLM tokens)
                                     miss? → forward to llama-server → respond
                                           + queue for async cache build

The gateway:
  1. Embeds the incoming query (EmbeddingGemma)
  2. Vector-searches the entry store
  3. On hit: extracts params (regex or LLM), runs the compiled module, returns answer
  4. On miss: proxies to the source LLM, optionally queues for lazy build

Run:
    python -m arc.gateway --port 8082 \
        --cache-dir ./cache \
        --chat-url http://127.0.0.1:8080/v1 \
        --embed-url http://127.0.0.1:8081 \
        --model unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("arc.gateway")


# --------------------------------------------------------------------------- #
# Parameter extraction
# --------------------------------------------------------------------------- #

# Patterns for common numeric parameter extraction
_NUM_PATTERNS = [
    r"\b(\d{1,15})\b",                          # plain integer
    r"\b(\d+\.\d+)\b",                           # float
]


def extract_params_regex(query: str, param_descriptions: list[str]) -> dict | None:
    """Try to extract parameters via regex (zero LLM calls).

    Returns {description: value} or None if extraction fails.
    Positional: the Nth number in the query maps to the Nth parameter
    description. Requires an exact count match; anything else defers to
    the LLM extractor (which can disambiguate extra numbers).
    """
    nums = re.findall(r"\b(\d{1,15})\b", query)
    if not nums:
        return None
    # Filter out small numbers that are likely operators/indices
    candidates = [n for n in nums if int(n) > 1]
    if not candidates:
        candidates = nums
    if len(candidates) != len(param_descriptions):
        return None
    return {desc: cand for desc, cand in zip(param_descriptions, candidates)}


def extract_params_llm(query: str, param_descriptions: list[str], lm_backend) -> dict | None:
    """Extract parameters via a small LLM call (fallback when regex fails)."""
    prompt = (
        f"From the following problem, extract these values. Reply as JSON.\n\n"
        f"Problem: {query}\n\n"
        f"Values to extract:\n"
        + "\n".join(f'- "{d}"' for d in param_descriptions)
        + "\n\nReply with ONLY a JSON object like {\"<description>\": <value>, ...}."
    )
    try:
        out = lm_backend.complete(prompt, role="extractor", max_tokens=200).text
        out = out.strip()
        if out.startswith("```"):
            out = re.sub(r"^```\w*\n?", "", out)
            out = re.sub(r"\n?```$", "", out)
        parsed = json.loads(out)
        return {d: str(parsed.get(d, "")) for d in param_descriptions}
    except (json.JSONDecodeError, Exception) as e:
        log.warning("LLM param extraction failed: %s", e)
        return None


# --------------------------------------------------------------------------- #
# Gateway state
# --------------------------------------------------------------------------- #

@dataclass
class GatewayConfig:
    port: int = 8082
    cache_dir: str = "./cache"
    chat_url: str = "http://127.0.0.1:8080/v1"
    embed_url: str = "http://127.0.0.1:8081"
    model: str = ""
    entry_threshold: float = 0.72
    function_threshold: float = 0.55
    load_threshold: float = 0.45
    enable_lazy_build: bool = False
    build_queue_dir: str = "./build_queue"
    # Code-execution isolation (worker subprocesses, see arc/execution.py)
    workers: int = 2
    worker_cpu_seconds: float = 5.0
    worker_mem_mb: int = 1024
    call_timeout_s: float = 10.0
    # Per-family entry thresholds (calibrated, see eval/threshold_matrix.py).
    # Falls back to entry_threshold for families not listed here.
    family_thresholds: dict[str, float] = field(default_factory=dict)
    # Cross-family separation guard: if the best entry from ANOTHER family
    # scores within this many of the top-1 entry, the match is ambiguous
    # (e.g. a GCD query vs a same-numbers LCM entry) and we fall back to the
    # LLM instead of executing the wrong family's code. 0 disables. Single-
    # family caches are unaffected (no other family to compare against).
    min_separation: float = 0.05
    # Persistent invocation log (arc/invocations.py): one row per /query —
    # audit trail + dataset for ongoing threshold monitoring.
    invocation_log: bool = True
    invocation_log_path: str | None = None  # default <cache_dir>/invocations.db
    log_retention_days: float | None = None


@dataclass
class GatewayStats:
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_fallbacks: int = 0
    errors: int = 0
    build_jobs_queued: int = 0
    total_hit_ms: float = 0.0
    total_miss_ms: float = 0.0

    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def avg_hit_ms(self) -> float:
        return self.total_hit_ms / max(1, self.cache_hits)

    def avg_miss_ms(self) -> float:
        return self.total_miss_ms / max(1, self.cache_misses)


class ArcGateway:
    """The sidecar proxy that sits in front of the LLM server."""

    def __init__(self, config: GatewayConfig):
        self.config = config
        self.stats = GatewayStats()
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()  # ThreadingHTTPServer: stats are mutated per-request from worker threads

        # Lazy imports (avoid requiring full ARC at import time)
        from arc import Embedder, LMBackend, Thresholds
        from arc.registry import Registry, EntryStore
        from arc.compiler import Compiler

        self.embedder = Embedder(backend="server", url=config.embed_url)
        self.lm = LMBackend(url=config.chat_url, model=config.model) if config.model else None

        cache_dir = Path(config.cache_dir)
        thresholds = Thresholds(entry=config.entry_threshold,
                                function=config.function_threshold,
                                load=config.load_threshold)

        # Load registry + entry store from the cache DB
        self.registry = Registry(self.embedder, path=cache_dir / "functions.db",
                                 function_threshold=config.function_threshold)
        self.entry_store = EntryStore(self.embedder, path=cache_dir / "entries.db",
                                      entry_threshold=config.entry_threshold)

        # Compiled module: never exec'd in this process — the worker pool
        # (child interpreters with rlimit budgets) owns generated code.
        self._compiled_path: Path | None = None
        self._worker_pool = None
        self._load_compiled(cache_dir)

        # Persistent invocation log (audit + threshold-monitoring dataset)
        self.invocation_log = None
        if config.invocation_log:
            from arc.invocations import InvocationLog
            inv_path = (config.invocation_log_path or
                        str(cache_dir / "invocations.db"))
            self.invocation_log = InvocationLog(
                inv_path, retention_days=config.log_retention_days)
            log.info("Invocation log: %s (retention=%s)",
                     inv_path,
                     f"{config.log_retention_days}d"
                     if config.log_retention_days else "unbounded")

        log.info("Gateway initialized: %d entries, %d functions, compiled=%s",
                 len(self.entry_store.entries), len(self.registry.functions),
                 "yes" if self._compiled_path else "no")

    def _stat_bump(self, *inc: str, add: dict[str, float] | None = None) -> None:
        """Atomically increment named int fields and/or add to float fields."""
        with self._stats_lock:
            for f in inc:
                setattr(self.stats, f, getattr(self.stats, f) + 1)
            for f, v in (add or {}).items():
                setattr(self.stats, f, getattr(self.stats, f) + v)

    def _load_compiled(self, cache_dir: Path) -> None:
        """Point the worker pool at the compiled module (if present).

        The gateway only checks for the file; loading and executing the
        generated code happens exclusively in worker subprocesses. A bad
        module (syntax error, missing solve) disables the compiled path
        instead of crashing the gateway.
        """
        mod_path = cache_dir / "compiled" / "arc_compiled.py"
        if not mod_path.exists():
            log.info("No compiled module found at %s — will use runtime dispatch", mod_path)
            self._compiled_path = None
            if self._worker_pool is not None:
                self._worker_pool.shutdown()
                self._worker_pool = None
            return
        from arc.execution import WorkerPool
        try:
            pool = WorkerPool(mod_path, n_workers=self.config.workers,
                              cpu_seconds=self.config.worker_cpu_seconds,
                              mem_mb=self.config.worker_mem_mb,
                              call_timeout_s=self.config.call_timeout_s)
        except Exception as e:  # noqa: BLE001 — bad generated module must not kill the gateway
            log.error("Failed to start worker pool for %s: %s — falling back to runtime dispatch",
                      mod_path, e)
            self._compiled_path = None
            return
        old = self._worker_pool
        self._worker_pool = pool
        self._compiled_path = mod_path
        if old is not None:
            old.shutdown()
        log.info("Worker pool ready: %d workers, cpu=%.0fs/call, mem=%dMB, module=%s",
                 pool.n_workers, self.config.worker_cpu_seconds,
                 self.config.worker_mem_mb, mod_path)

    def reload_cache(self) -> None:
        """Hot-reload the entry index + compiled module (called by build worker)."""
        with self._lock:
            cache_dir = Path(self.config.cache_dir)
            # Reload entry store
            self.entry_store.entries.clear()
            if (cache_dir / "entries.db").exists():
                self.entry_store._load()
            # Reload compiled module
            self._load_compiled(cache_dir)
            log.info("Cache reloaded: %d entries, compiled=%s",
                     len(self.entry_store.entries),
                     "yes" if self._compiled_path else "no")

    # ------------------------------------------------------------------ #
    # Core serve logic
    # ------------------------------------------------------------------ #

    def handle_query(self, query: str) -> dict:
        """Process a single query; persists one row to the invocation log."""
        self._stat_bump("total_requests")
        t0 = time.perf_counter()
        inv: dict = {"query": query, "ts": time.time()}
        tok_mark = self.lm.ledger.mark() if self.lm else None
        try:
            resp = self._handle_query(query, t0, inv)
        except Exception as e:
            log.exception("Unhandled gateway error")
            self._stat_bump("errors")
            inv.update(status="error", phase="unhandled",
                       reason=f"gateway error: {e}")
            resp = {"source": "gateway-error", "error": str(e),
                    "ms": round((time.perf_counter() - t0) * 1000, 2)}
        inv.setdefault("source", resp.get("source"))
        inv["serve_ms"] = resp.get("ms")
        if tok_mark is not None:
            delta = self.lm.ledger.since(tok_mark)
            inv["tokens_in"] = sum(v["in"] for v in delta["by_role"].values())
            inv["tokens_out"] = sum(v["out"] for v in delta["by_role"].values())
        self._record_invocation(inv)
        return resp

    def _record_invocation(self, inv: dict) -> None:
        """Append to the invocation log; a log failure must never break serving."""
        if self.invocation_log is None:
            return
        try:
            self.invocation_log.record(
                ts=inv.get("ts"), query=inv.get("query", ""),
                status=inv.get("status") or "unknown", phase=inv.get("phase"),
                source=inv.get("source"), family=inv.get("family"),
                entry=inv.get("entry"), score=inv.get("score"),
                params=inv.get("params"), reason=inv.get("reason"),
                serve_ms=inv.get("serve_ms"),
                tokens_in=inv.get("tokens_in"), tokens_out=inv.get("tokens_out"))
        except Exception as e:
            log.warning("Invocation log write failed: %s", e)

    def _handle_query(self, query: str, t0: float, inv: dict) -> dict:
        """Core serve logic. Every decision point marks `inv` (for the log)."""

        def mark(status: str, phase: str, reason: str | None = None,
                 family: str | None = None, entry: dict | None = None,
                 score: float | None = None, params: dict | None = None) -> None:
            inv.update(status=status, phase=phase, reason=reason,
                       family=family,
                       entry=entry["description"] if entry else None,
                       score=None if score is None else round(score, 4),
                       params=params)

        # 1. Entry search
        if not self.entry_store.entries:
            mark("miss", "search", reason="empty cache")
            return self._miss(query, t0, reason="empty cache")

        entries, scores = self.entry_store.scores_all(query)
        if not entries:
            mark("miss", "search", reason="empty cache")
            return self._miss(query, t0, reason="empty cache")
        i = int(np.argmax(scores))
        entry, score = entries[i], scores[i]
        # Per-family threshold (calibrated); global fallback otherwise.
        thr = self.config.entry_threshold
        fam = entry.get("family")
        if fam:
            thr = self.config.family_thresholds.get(fam, thr)
        if score < thr:
            mark("miss", "threshold", reason=f"score {score:.3f} < {thr:.3f}" +
                                    (f" (family={fam})" if fam else ""),
                 family=fam, entry=entry, score=score)
            return self._miss(query, t0, reason=f"score {score:.3f} < {thr:.3f}" +
                                 (f" (family={fam})" if fam else ""))
        # Cross-family separation guard: a near-tie with another family's
        # entry is too ambiguous to execute (false-friend, see
        # eval/threshold_matrix.py + tests/threshold_report.json).
        if self.config.min_separation > 0 and fam:
            best_other = max((s for e, s in zip(entries, scores)
                              if e.get("family") and e.get("family") != fam),
                             default=None)
            if best_other is not None and score - best_other < self.config.min_separation:
                mark("miss", "separation",
                     reason=(f"ambiguous cross-family: top1 {score:.3f} vs "
                             f"other family {best_other:.3f} "
                             f"(separation < {self.config.min_separation})"),
                     family=fam, entry=entry, score=score)
                return self._miss(query, t0, reason=(
                    f"ambiguous cross-family: top1 {score:.3f} vs other family "
                    f"{best_other:.3f} (separation < {self.config.min_separation})"))

        # 2. Parameter extraction
        params_desc = entry["params"]
        params = extract_params_regex(query, params_desc)
        if params is None and self.lm:
            # Only ask the LLM to extract when the query plausibly contains a
            # number for every parameter; otherwise it will hallucinate the
            # missing ones (e.g. a 2-number query vs a 3-param GCD entry).
            n_nums = len(re.findall(r"\b\d{1,15}\b", query))
            if n_nums >= len(params_desc):
                params = extract_params_llm(query, params_desc, self.lm)
        if params is None:
            mark("miss", "extract", reason="param extraction failed",
                 family=fam, entry=entry, score=score)
            return self._miss(query, t0, reason="param extraction failed")
        # Guard: compiled entries coerce numeric params. Refuse non-numeric
        # values ("" / "None" / text from a partial LLM extraction) instead of
        # executing with garbage.
        for desc, val in params.items():
            if not re.fullmatch(r"\s*-?[\d,]+(?:\.\d+)?\s*", str(val)):
                mark("miss", "extract",
                     reason=f"param {desc!r} not numeric: {val!r}",
                     family=fam, entry=entry, score=score)
                return self._miss(query, t0, reason=f"param {desc!r} not numeric: {val!r}")

        # 3. Execute
        try:
            result = self._execute(entry, params)
        except Exception as e:
            log.warning("Execution error: %s", e)
            self._stat_bump("errors")
            mark("error", "execute", reason=f"execution error: {e}",
                 family=fam, entry=entry, score=score, params=params)
            return self._miss(query, t0, reason=f"execution error: {e}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._stat_bump("cache_hits", add={"total_hit_ms": elapsed_ms})
        mark("hit", "execute", family=fam, entry=entry, score=score, params=params)

        return {
            "source": "cache",
            "answer": result,
            "entry_score": round(score, 4),
            "params": params,
            "ms": round(elapsed_ms, 2),
            "hit_rate": round(self.stats.hit_rate(), 3),
        }

    def _execute(self, entry: dict, params: dict) -> dict:
        """Run the compiled module (in a worker subprocess) or fall back
        to runtime dispatch (builder-sandboxed, in-process)."""
        if self._worker_pool is not None:
            return self._worker_pool.submit(params)

        # Runtime path (no compiled module): use the full ARC context
        from arc.context import Context, Thresholds
        from arc.primitives import bind, _primitives
        ctx = Context(self.embedder, Thresholds(
            entry=self.config.entry_threshold,
            function=self.config.function_threshold,
            load=self.config.load_threshold,
        ))
        bind(self.embedder, self.registry, self.lm or _NullLM(), ctx=ctx, tools={})
        for desc in params:
            ctx.layers[0].add(desc, params[desc], self.embedder)

        # Find and run the root function
        fn = None
        for f in self.registry.functions:
            if f["id"] == entry["root_function_id"]:
                fn = f["fn"]
                break
        if fn is None:
            raise RuntimeError(f"root function {entry['root_function_id']} not found")
        return fn(ctx)

    def _miss(self, query: str, t0: float, reason: str) -> dict:
        """Handle a cache miss: proxy to LLM + optionally queue for build."""
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._stat_bump("cache_misses", add={"total_miss_ms": elapsed_ms})

        response = {
            "source": "llm-fallback",
            "entry_score": None,
            "reason": reason,
            "ms": round(elapsed_ms, 2),
            "hit_rate": round(self.stats.hit_rate(), 3),
        }

        # Proxy to the source LLM
        if self.lm:
            try:
                answer = self.lm.complete(query, role="source", max_tokens=2000).text
                response["answer"] = answer
                self._stat_bump("llm_fallbacks")
            except Exception as e:
                log.error("LLM fallback failed: %s", e)
                response["error"] = str(e)

        # Queue for lazy build (optional)
        if self.config.enable_lazy_build and self.lm and "answer" in response:
            self._queue_build(query, response["answer"])

        return response

    def _queue_build(self, query: str, transcript: str) -> None:
        """Queue a (query, transcript) pair for the async build worker."""
        queue_dir = Path(self.config.build_queue_dir)
        queue_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"build_{int(time.time() * 1000)}"
        job_path = queue_dir / f"{job_id}.json"
        job_path.write_text(json.dumps({
            "id": job_id,
            "query": query,
            "transcript": transcript,
            "queued_at": time.time(),
            "status": "pending",
        }))
        self._stat_bump("build_jobs_queued")
        log.info("Queued build job %s for: %s", job_id, query[:60])


class _NullLM:
    """Placeholder LM for runtime dispatch when no chat model is configured."""
    def complete(self, prompt, role="target", max_tokens=512):
        raise RuntimeError("no LLM backend configured for runtime dispatch")


# --------------------------------------------------------------------------- #
# HTTP server (stdlib http.server — zero dependencies)
# --------------------------------------------------------------------------- #

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler


class _GatewayHandler(BaseHTTPRequestHandler):
    gateway: ArcGateway = None  # set by factory

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", **self._stats()})
        elif self.path == "/stats":
            self._json(200, self._stats())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/query":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid JSON"})
                return
            query = req.get("query", "")
            if not query:
                self._json(400, {"error": "missing 'query' field"})
                return
            result = self.gateway.handle_query(query)
            self._json(200, result)
        elif self.path == "/reload":
            self.gateway.reload_cache()
            self._json(200, {"status": "reloaded"})
        else:
            self._json(404, {"error": "not found"})

    def _stats(self) -> dict:
        s = self.gateway.stats
        return {
            "total_requests": s.total_requests,
            "cache_hits": s.cache_hits,
            "cache_misses": s.cache_misses,
            "llm_fallbacks": s.llm_fallbacks,
            "errors": s.errors,
            "build_jobs_queued": s.build_jobs_queued,
            "hit_rate": round(s.hit_rate(), 4),
            "avg_hit_ms": round(s.avg_hit_ms(), 2),
            "avg_miss_ms": round(s.avg_miss_ms(), 2),
            "compiled": self.gateway._compiled_path is not None,
            "workers": self.gateway._worker_pool.n_workers if self.gateway._worker_pool else 0,
            "worker_restarts": self.gateway._worker_pool.restarts if self.gateway._worker_pool else 0,
            "invocations": (self.gateway.invocation_log.count()
                            if self.gateway.invocation_log else None),
        }

    def _json(self, code: int, data: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, format, *args):
        log.debug(format, *args)


def make_handler(gateway: ArcGateway):
    handler = type("BoundHandler", (_GatewayHandler,), {"gateway": gateway})
    return handler


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="ARC Gateway — cache-first LLM proxy")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", default="0.0.0.0",
                        help="bind address (use 127.0.0.1 unless the sidecar is exposed)")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--chat-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--embed-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="")
    parser.add_argument("--entry-threshold", type=float, default=0.72)
    parser.add_argument("--function-threshold", type=float, default=0.55)
    parser.add_argument("--load-threshold", type=float, default=0.45)
    parser.add_argument("--lazy-build", action="store_true", help="Enable lazy cache build on miss")
    parser.add_argument("--build-queue-dir", default="./build_queue")
    parser.add_argument("--workers", type=int, default=2,
                        help="worker subprocesses for executing generated code")
    parser.add_argument("--worker-cpu-seconds", type=float, default=5.0,
                        help="per-call CPU budget in each worker")
    parser.add_argument("--worker-mem-mb", type=int, default=1024,
                        help="virtual-memory ceiling per worker (MiB)")
    parser.add_argument("--call-timeout", type=float, default=10.0,
                        help="wall-clock budget per cached-code call")
    parser.add_argument("--family-threshold", default="",
                        help="comma list of family=threshold, e.g. 'divisors=0.68,gcd=0.70'")
    parser.add_argument("--min-separation", type=float, default=0.05,
                        help="min score gap between top-1 entry and best entry "
                             "of another family; smaller => ambiguous, fall back (0 disables)")
    parser.add_argument("--invocation-log", default=None,
                        help="path to invocations.db (default <cache-dir>/invocations.db)")
    parser.add_argument("--no-invocation-log", action="store_true",
                        help="disable the persistent invocation log")
    parser.add_argument("--log-retention-days", type=float, default=None,
                        help="prune invocation rows older than N days at startup")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = GatewayConfig(
        port=args.port,
        cache_dir=args.cache_dir,
        chat_url=args.chat_url,
        embed_url=args.embed_url,
        model=args.model,
        entry_threshold=args.entry_threshold,
        function_threshold=args.function_threshold,
        load_threshold=args.load_threshold,
        enable_lazy_build=args.lazy_build,
        build_queue_dir=args.build_queue_dir,
        workers=args.workers,
        worker_cpu_seconds=args.worker_cpu_seconds,
        worker_mem_mb=args.worker_mem_mb,
        call_timeout_s=args.call_timeout,
        family_thresholds=_parse_family_thresholds(args.family_threshold),
        min_separation=args.min_separation,
        invocation_log=not args.no_invocation_log,
        invocation_log_path=args.invocation_log,
        log_retention_days=args.log_retention_days,
    )

    gateway = ArcGateway(config)
    handler = make_handler(gateway)
    # ThreadingHTTPServer: a slow LLM-fallback request must not serialize
    # cache-hit traffic behind it. daemon_threads keeps shutdown snappy.
    server = ThreadingHTTPServer((args.host, config.port), handler)
    server.daemon_threads = True
    log.info("ARC Gateway listening on %s:%d (cache=%s)", args.host, config.port, config.cache_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down. Stats: %s", json.dumps(gateway.stats.__dict__))
    finally:
        if gateway._worker_pool is not None:
            gateway._worker_pool.shutdown()
        if gateway.invocation_log is not None:
            gateway.invocation_log.close()


def _parse_family_thresholds(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in s.split(","))):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"bad --family-threshold entry: {part!r}")
    return out


if __name__ == "__main__":
    main()
