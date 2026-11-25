"""ARC P0 spike — minimal agentic reasoning cache runtime (CPU-ARC paper reimplementation).

Implements the paper's runtime surface:
  Context (layered memory + vector lookup), load, do, narrate, lm, extract,
  convert/smart_convert, and a gated serve().
"""
from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class Embedder:
    """CPU embedding provider. Backends: 'hash' (offline fallback) or llama.cpp server."""

    def __init__(self, backend: str = "hash", url: str = "http://127.0.0.1:8081", dim: int = 768):
        self.backend = backend
        self.url = url.rstrip("/")
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        if self.backend == "hash":
            return self._hash_embed(text)
        import urllib.request
        req = urllib.request.Request(
            f"{self.url}/embeddings",
            data=json.dumps({"input": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        if isinstance(data, list):
            emb = data[0]["embedding"]  # llama.cpp native format
        else:
            emb = data["data"][0]["embedding"]
        while isinstance(emb, list) and len(emb) == 1:
            emb = emb[0]
        return np.asarray(emb, dtype=np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic bag-of-words hash embedder — offline fallback only."""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            h = abs(hash(tok)) % self.dim
            vec[h] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec


# --------------------------------------------------------------------------- #
# LM backend (target model T)
# --------------------------------------------------------------------------- #
class LMBackend:
    """OpenAI-compatible chat completion client (llama.cpp server or any API)."""

    def __init__(self, url: str = "http://127.0.0.1:8080/v1", api_key: str = "none"):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.stats = {"calls": 0, "in_tokens": 0, "out_tokens": 0}

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        import urllib.request
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps(
            {"model": "default", "messages": messages, "temperature": temperature, "max_tokens": 512}
        ).encode()
        req = urllib.request.Request(
            f"{self.url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
        out = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        self.stats["calls"] += 1
        self.stats["in_tokens"] += usage.get("prompt_tokens", 0)
        self.stats["out_tokens"] += usage.get("completion_tokens", 0)
        return out


# --------------------------------------------------------------------------- #
# Context: stack of memory layers, vector lookup top-down
# --------------------------------------------------------------------------- #
@dataclass
class _Item:
    description: str
    value: Any
    vec: np.ndarray


@dataclass
class Layer:
    items: list[_Item] = field(default_factory=list)

    def add(self, description: str, value: Any, embedder: Embedder):
        self.items.append(_Item(description, value, embedder.embed(description)))

    def search(self, query_vec: np.ndarray, top_k: int = 1):
        if not self.items:
            return []
        vecs = np.stack([it.vec for it in self.items])
        scores = vecs @ query_vec
        order = np.argsort(-scores)[:top_k]
        return [(self.items[i], float(scores[i])) for i in order]


class Context:
    def __init__(self, embedder: Embedder, threshold: float = 0.35):
        self.embedder = embedder
        self.threshold = threshold
        self.layers: list[Layer] = [Layer()]  # base layer
        self.narration: list[str] = []

    def push(self) -> None:
        self.layers.append(Layer())

    def pop(self) -> Layer:
        return self.layers.pop()


# --------------------------------------------------------------------------- #
# Registry of semantic functions
# --------------------------------------------------------------------------- #
class Registry:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.functions: list[dict] = []  # {description, vec, fn}

    def register(self, description: str, fn: Callable[[Context], dict]):
        self.functions.append(
            {"description": description, "vec": self.embedder.embed(description), "fn": fn}
        )

    def search(self, query: str, top_k: int = 1):
        if not self.functions:
            return []
        qv = self.embedder.embed(query)
        vecs = np.stack([f["vec"] for f in self.functions])
        scores = vecs @ qv
        order = np.argsort(-scores)[:top_k]
        return [(self.functions[i], float(scores[i])) for i in order]


# --------------------------------------------------------------------------- #
# Runtime primitives (module-level; bound to active ctx/registry via globals)
# --------------------------------------------------------------------------- #
_STATE: dict[str, Any] = {}


def _require(key: str) -> Any:
    if key not in _STATE:
        raise RuntimeError(f"primitive used outside runtime ({key})")
    return _STATE[key]


def load(ctx: Context | None = None, query: str = "", d_type: type | None = None,
         result_dict: "dict | None" = None) -> Any:
    """Retrieve a value from layered context memory via vector similarity.

    Two forms (paper's usage):
      load(ctx, description, d_type=...)          — layered context search
      load(result_dict, description, d_type=...)  — search within a do() result dict
    """
    if isinstance(ctx, dict) or ctx is None:
        # second form: first positional arg is the result dict
        if ctx is None:
            raise CacheInvocationError("load: missing result dict")
        result_dict = ctx
        items = [
            _Item(k, v, _STATE["embedder"].embed(k)) for k, v in result_dict.items()
        ]
        qv = _STATE["embedder"].embed(query)
        vecs = np.stack([it.vec for it in items])
        scores = vecs @ qv
        i = int(np.argmax(scores))
        if float(scores[i]) < 0.2:
            raise CacheInvocationError(f"load: no match for '{query}' in result (best={scores[i]:.3f})")
        return coerce_or_convert(_STATE.get("ctx"), items[i].value, d_type)
    # normal layered search, top-down
    qv = ctx.embedder.embed(query)
    for layer in reversed(ctx.layers):
        hits = layer.search(qv)
        if hits and hits[0][1] >= ctx.threshold:
            return coerce_or_convert(ctx, hits[0][0].value, d_type)
    raise CacheInvocationError(f"load: no match for '{query}' (threshold={ctx.threshold})")


def do(ctx: Context, intent: str, payloads: dict[str, Any] | None = None) -> dict:
    """Discover and invoke a semantic function by natural-language description."""
    registry = _require("registry")
    hits = registry.search(intent)
    if not hits or hits[0][1] < ctx.threshold:
        raise CacheInvocationError(f"do: no function for intent '{intent}'")
    fn, score = hits[0]
    ctx.push()
    try:
        for desc, val in (payloads or {}).items():
            ctx.layers[-1].add(desc, val, ctx.embedder)
        result = fn["fn"](ctx)
        if not isinstance(result, dict):
            raise CacheInvocationError(f"do: function '{fn['description']}' returned non-dict")
        return result
    finally:
        ctx.pop()


def narrate(ctx: Context, message: str) -> None:
    depth = len(ctx.layers) - 1
    ctx.narration.append(("  " * depth) + message)


def lm(ctx: Context, prompt: str) -> str:
    return _require("lm").complete(prompt)


def extract(ctx: Context, text: str, what: str, d_type: type | None = None) -> Any:
    """Use target model T for verbatim value extraction."""
    out = lm(
        ctx,
        f"From the text below, extract exactly: {what}.\n"
        f"Reply with ONLY the extracted value (include its unit if one is present).\n"
        f"If the text does not contain it, reply with exactly: NOT_FOUND.\n\n"
        f"Text: {text}",
    )
    out = out.strip()
    if out.upper() == "NOT_FOUND":
        raise CacheInvocationError(f"extract: '{what}' not found in text")
    return coerce_or_convert(ctx, out, d_type)


def convert(value: Any, d_type: type) -> Any:
    return _coerce(value, d_type)


def smart_convert(ctx: Context, text: str, d_type: type) -> Any:
    """Multi-tier coercion: programmatic first, else ask T to rewrite as Python literal."""
    try:
        return _coerce(text, d_type)
    except Exception:
        out = lm(
            ctx,
            f"Rewrite the following value as a valid Python literal of type {d_type.__name__}. "
            f"Reply with ONLY the literal.\n\nValue: {text}",
        )
        parsed = ast.literal_eval(out.strip())
        return _coerce(parsed, d_type)


def coerce_or_convert(ctx: Context | None, value: Any, d_type: type | None) -> Any:
    """Coerce; on failure fall back to smart_convert (paper's multi-tier ladder)."""
    try:
        return _coerce(value, d_type)
    except CacheInvocationError:
        if ctx is None or d_type is None:
            raise
        return smart_convert(ctx, str(value), d_type)


def invoke_tool(ctx: Context, name: str, args: dict) -> Any:
    tools = _STATE.get("tools") or {}
    if name not in tools:
        raise CacheInvocationError(f"invoke_tool: unknown tool '{name}'")
    return tools[name](**args)


class CacheInvocationError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Type coercion
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _coerce(value: Any, d_type: type | None) -> Any:
    if d_type is None or value is None:
        return value
    if isinstance(value, d_type) and not (d_type is float and isinstance(value, int)):
        return value
    if d_type in (int, float):
        if isinstance(value, str):
            m = _NUM_RE.search(value.replace(",", ""))
            if m:
                v = m.group(0)
                return float(v) if d_type is float else int(float(v))
        if isinstance(value, (int, float)):
            return d_type(value)
    if d_type is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)
    if d_type is list:
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            try:
                v = ast.literal_eval(value)
                return list(v) if isinstance(v, (list, tuple)) else [value]
            except Exception:
                return [x.strip() for x in value.split(",") if x.strip()]
    if d_type is set:
        return set(_coerce(value, list))
    if d_type is dict:
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except Exception:
                pass
        return value
    raise CacheInvocationError(f"coerce: cannot coerce {value!r} to {d_type}")


# --------------------------------------------------------------------------- #
# Gated serving (paper Figure 1)
# --------------------------------------------------------------------------- #
def serve(cache: "ARC", request: str, source_llm: Callable[[str], str] | None = None):
    try:
        result = cache.use(request)
        if result is not False:
            return result
    except CacheInvocationError:
        pass
    if source_llm is None:
        raise CacheInvocationError("cache miss and no source LLM configured")
    return {"answer": source_llm(request), "source": "llm-fallback"}


# --------------------------------------------------------------------------- #
# ARC cache container
# --------------------------------------------------------------------------- #
class ARC:
    def __init__(self, embedder: Embedder, lm_backend: LMBackend, threshold: float = 0.35):
        self.embedder = embedder
        self.lm = lm_backend
        self.threshold = threshold
        self.registry = Registry(embedder)
        self.entry_points: list[dict] = []  # {description, vec, root_desc, params}

    def register_entry(self, input_text: str, root_fn: Callable[[Context], dict],
                       root_description: str, param_descriptions: list[str]):
        self.registry.register(root_description, root_fn)
        self.entry_points.append(
            {
                "description": input_text,
                "vec": self.embedder.embed(input_text),
                "root_desc": root_description,
                "params": param_descriptions,
            }
        )

    def use(self, request: str) -> dict | bool:
        # IS1: entry-point search
        if not self.entry_points:
            return False
        qv = self.embedder.embed(request)
        vecs = np.stack([e["vec"] for e in self.entry_points])
        scores = vecs @ qv
        i = int(np.argmax(scores))
        best = float(scores[i])
        if best < self.threshold:
            return False
        entry = self.entry_points[i]

        # IS2: parameter extraction via T
        ctx = Context(self.embedder, threshold=self.threshold)
        _STATE.update({"embedder": self.embedder, "registry": self.registry, "lm": self.lm})
        _STATE["ctx"] = ctx
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
            # retry only the missing ones with a stronger hint
            for desc in missing:
                val = extract(
                    ctx,
                    request,
                    f"{desc} — look carefully: it is stated somewhere in the problem, "
                    f"possibly with words like 'rigid container', 'vessel', or 'amounting to'",
                )
                ctx.layers[0].add(desc, val, self.embedder)
            if all(any(d == m for d in entry["params"]) for m in missing):
                break

        # IS3: dispatch root
        hits = self.registry.search(entry["root_desc"])
        if not hits or hits[0][1] < self.threshold:
            raise CacheInvocationError("root function not found")
        root_fn, _score = hits[0]
        result = root_fn["fn"](ctx)

        # IS4: output + narration
        return {
            "result": result,
            "narration": ctx.narration,
            "entry_score": best,
            "lm_stats": dict(self.lm.stats),
        }
