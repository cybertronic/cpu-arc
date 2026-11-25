"""ARC cache container + gated serving (paper §2.2.2, Figure 1)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .backends import Ledger
from .context import Context, CacheInvocationError, Thresholds
from .primitives import bind, extract, _primitives
from .registry import Registry, EntryStore


@dataclass
class InvocationRecord:
    """One cache invocation, persisted for mining (P3 target-model data)."""
    ts: float
    request: str
    hit: bool
    entry_score: float
    result: dict | None
    error: str | None
    narration: list[str]
    load_log: list[dict]
    ms: float
    lm_stats: dict

    def to_json(self) -> str:
        return json.dumps(asdict_safe(self))


def asdict_safe(r: InvocationRecord) -> dict:
    return {
        "ts": r.ts, "request": r.request, "hit": r.hit,
        "entry_score": r.entry_score, "result": r.result, "error": r.error,
        "narration": r.narration, "load_log": r.load_log,
        "ms": r.ms, "lm_stats": r.lm_stats,
    }


class ARC:
    """An agentic reasoning cache: registry + entry points + serving config."""

    def __init__(self, embedder, lm, thresholds: Thresholds | None = None,
                 db_path: str | Path | None = None, ledger: Ledger | None = None,
                 invocation_log: str | Path | None = None,
                 tools: dict | None = None):
        self.embedder = embedder
        self.lm = lm
        self.thresholds = thresholds or Thresholds()
        self.ledger = ledger or Ledger()
        self.tools = tools or {}
        db = Path(db_path) if db_path else None
        if db:
            db.mkdir(parents=True, exist_ok=True)
        # wire the shared ledger into the backend so per-role tokens are captured
        if hasattr(lm, "ledger"):
            lm.ledger = self.ledger
        self.registry = Registry(embedder, path=db / "functions.db" if db else None,
                                 function_threshold=self.thresholds.function)
        self.entry_store = EntryStore(embedder, path=db / "entries.db" if db else None,
                                      entry_threshold=self.thresholds.entry)
        self.invocation_log = Path(invocation_log) if invocation_log else None
        if self.invocation_log:
            self.invocation_log.parent.mkdir(parents=True, exist_ok=True)

    # -- construction API ------------------------------------------------------
    def register_entry(self, input_text: str, root_source: str,
                       root_description: str, param_descriptions: list[str],
                       utility_sources: list[tuple[str, str]] | None = None) -> str:
        """Register a root function (from source) plus its utilities and an entry point.

        root_source must define a function taking (ctx). Utility sources likewise.
        Returns the root function id.
        """
        from .sandbox import exec_function_source
        prims = _primitives()
        for desc, src in (utility_sources or []):
            self.registry.register(desc, exec_function_source(src, prims), source=src)
        root_id = self.registry.register(
            root_description, exec_function_source(root_source, prims),
            source=root_source, meta={"params": param_descriptions})
        self.entry_store.register(input_text, root_id, param_descriptions)
        return root_id

    # -- invocation (IS1-IS4) ----------------------------------------------------
    def use(self, request: str) -> dict | bool:
        entry, best = self.entry_store.search(request)
        if entry is None or best < self.thresholds.entry:
            return False

        ctx = Context(self.embedder, self.thresholds)
        bind(self.embedder, self.registry, self.lm, ctx=ctx, tools=self.tools)
        _mark = self.ledger.mark()

        # IS2: parameter extraction via T, with one hinted retry for misses
        for attempt in range(2):
            missing = []
            for desc in entry["params"]:
                try:
                    val = extract(ctx, request, f"{desc} (as stated in the problem)")
                    ctx.layers[0].add(desc, val, self.embedder)
                except CacheInvocationError:
                    missing.append(desc)
            if not missing:
                break
            for desc in missing:
                val = extract(
                    ctx, request,
                    f"{desc} — look carefully: it is stated somewhere in the problem, "
                    f"possibly embedded in a longer phrase or sentence",
                )
                ctx.layers[0].add(desc, val, self.embedder)

        # IS3: dispatch root
        fn = self._resolve_root(entry)
        if fn is None:
            raise CacheInvocationError("root function not found in registry")
        result = fn(ctx)

        # IS4: output + narration + record
        _stats = self.ledger.since(_mark)
        rec = InvocationRecord(
            ts=time.time(), request=request, hit=True, entry_score=best,
            result=result, error=None, narration=ctx.narration,
            load_log=ctx.load_log, ms=0.0, lm_stats=_stats)
        self._log(rec)
        return {
            "result": result,
            "narration": ctx.narration,
            "entry_score": best,
            "entry_description": entry["description"],
            "load_log": ctx.load_log,
            "lm_stats": _stats,
        }

    def _is_id(self, s: str) -> bool:
        return s.startswith("fn_")

    def _resolve_root(self, entry: dict) -> Callable | None:
        rid = entry["root_function_id"]
        for f in self.registry.functions:
            if f["id"] == rid:
                return f["fn"]
        # fall back to description search
        hits = self.registry.search(rid, top_k=1)
        if hits and hits[0][1] >= self.thresholds.function:
            return hits[0][0]["fn"]
        return None

    def _log(self, rec: InvocationRecord) -> None:
        if self.invocation_log is None:
            return
        with open(self.invocation_log, "a") as fh:
            fh.write(rec.to_json() + "\n")


def serve(cache: ARC, request: str,
          source_llm: Callable[[str], str] | None = None) -> dict:
    """Gated use (paper Figure 1): cache first; on miss/error, fall through."""
    try:
        result = cache.use(request)
        if result is not False:
            return {"source": "cache", **result}
    except (CacheInvocationError, Exception) as e:  # kill-switch: any error falls through
        if source_llm is None:
            raise
        return {"source": "llm-fallback", "answer": source_llm(request),
                "cache_error": str(e)}
    if source_llm is None:
        raise CacheInvocationError("cache miss and no source LLM configured")
    return {"source": "llm-fallback", "answer": source_llm(request)}
