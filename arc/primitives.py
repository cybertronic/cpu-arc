"""Runtime primitives (paper §2.1) + type-coercion ladder.

Two forms of load (paper usage):
  load(ctx, description, d_type=...)         — layered context search
  load(result_dict, description, d_type=...) — search within a do() result dict
"""
from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np

from .context import Context, CacheInvocationError

# --------------------------------------------------------------------------- #
# Active-runtime state (bound once per ARC via bind())
# --------------------------------------------------------------------------- #
_STATE: dict[str, Any] = {}


def bind(embedder, registry, lm, ctx: Context | None = None,
         tools: dict | None = None) -> None:
    _STATE.clear()
    _STATE["embedder"] = embedder
    _STATE["registry"] = registry
    _STATE["lm"] = lm
    _STATE["ctx"] = ctx
    _STATE["tools"] = tools or {}


def get_state() -> dict:
    return _STATE


# --------------------------------------------------------------------------- #
# Type coercion (multi-tier: programmatic -> LLM rewrite -> ast.literal_eval)
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
                v = ast.literal_eval(value)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
        if isinstance(value, dict):
            return value
        raise CacheInvocationError(f"coerce: cannot coerce {value!r} to dict")
    raise CacheInvocationError(f"coerce: cannot coerce {value!r} to {d_type}")


def smart_convert(ctx: Context, text: str, d_type: type) -> Any:
    """Tier 2: ask target T to rewrite the value as a Python literal, validate with ast."""
    lm = _STATE["lm"]
    out = lm.complete(
        f"Rewrite the following value as a valid Python literal of type {d_type.__name__}. "
        f"Reply with ONLY the literal.\n\nValue: {text}",
        role="converter",
    ).text
    out = out.strip()
    try:
        parsed = ast.literal_eval(out)
    except (ValueError, SyntaxError) as e:
        raise CacheInvocationError(f"smart_convert: invalid literal {out!r}: {e}") from e
    return _coerce(parsed, d_type)


def coerce_or_convert(ctx: Context | None, value: Any, d_type: type | None) -> Any:
    """Full ladder: programmatic coercion, else smart_convert via T."""
    try:
        return _coerce(value, d_type)
    except CacheInvocationError:
        if ctx is None or d_type is None:
            raise
        converted = smart_convert(ctx, str(value), d_type)
        # Guard: a value that cannot become the requested type must fail loudly,
        # not silently pass through (e.g. load() matched an int where a dict was
        # expected). A silent passthrough corrupts downstream generated code.
        try:
            _coerce(converted, d_type)
        except CacheInvocationError:
            raise CacheInvocationError(
                f"coerce: value {value!r} could not be converted to {d_type.__name__} "
                f"(smart_convert produced {converted!r})")
        return converted


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def load(ctx, query: str = "", d_type: type | None = None):
    """P1: first positional arg is either a Context or a result dict."""
    embedder = _STATE["embedder"]
    if isinstance(ctx, dict):
        # do()-result form: search the dict's keys
        items = [(k, v, embedder.embed(k)) for k, v in ctx.items()]
        q = query.strip().lower()
        for k, v, _ in items:
            if k.strip().lower() == q:
                return coerce_or_convert(_STATE.get("ctx"), v, d_type)
        qv = embedder.embed(query)
        vecs = np.stack([e for _, _, e in items])
        scores = vecs @ qv
        i = int(np.argmax(scores))
        if float(scores[i]) < 0.3:
            raise CacheInvocationError(
                f"load: no match for '{query}' in result dict (best={scores[i]:.3f})")
        return coerce_or_convert(_STATE.get("ctx"), items[i][1], d_type)
    if not isinstance(ctx, Context):
        raise CacheInvocationError(f"load: bad first arg type {type(ctx).__name__}")
    return ctx.find(query, d_type=d_type, scope="load")


def do(ctx: Context, intent: str, payloads: dict[str, Any] | None = None) -> dict:
    """Discover + invoke a semantic function by natural-language description."""
    registry = _STATE["registry"]
    th = ctx.thresholds.function
    hits = registry.search(intent)
    if not hits:
        raise CacheInvocationError(f"do: registry empty, cannot dispatch '{intent}'")
    entry, score = hits[0]
    if score < th:
        raise CacheInvocationError(
            f"do: no function for intent '{intent}' (best={score:.3f} < {th})")
    ctx.push()
    try:
        for desc, val in (payloads or {}).items():
            ctx.layers[-1].add(desc, val, ctx.embedder)
        result = entry["fn"](ctx)
        if not isinstance(result, dict):
            raise CacheInvocationError(
                f"do: function '{entry['description']}' returned non-dict")
        return result
    finally:
        ctx.pop()


def narrate(ctx_or_msg, message: str | None = None) -> None:
    """Record a narration step.

    Accepts both the contract form `narrate(ctx, msg)` and the common model
    mistake `narrate(msg)` — where the first arg is a string. Without this,
    generated code that omits ctx fails at dispatch with a confusing TypeError.
    """
    if message is None:
        # narrate("message") form: first arg IS the message
        msg = str(ctx_or_msg)
        ctx = _STATE.get("ctx")
        if ctx is not None:
            ctx.narrate(msg)
        return
    ctx_or_msg.narrate(str(message))


def lm(ctx: Context, prompt: str, role: str = "target") -> str:
    """Query target T for an elementary NL task (classification, short generation)."""
    return _STATE["lm"].complete(prompt, role=role).text


def extract(ctx: Context, text: str, what: str, d_type: type | None = None) -> Any:
    """Use T for verbatim value extraction (role=extractor in the ledger)."""
    out = lm(
        ctx,
        f"From the text below, extract exactly: {what}.\n"
        f"Reply with ONLY the extracted value (include its unit if one is present).\n"
        f"If the text does not contain it, reply with exactly: NOT_FOUND.\n\n"
        f"Text: {text}",
        role="extractor",
    )
    out = out.strip()
    if out.upper() == "NOT_FOUND":
        raise CacheInvocationError(f"extract: '{what}' not found in text")
    return coerce_or_convert(ctx, out, d_type)


def convert(value: Any, d_type: type) -> Any:
    return _coerce(value, d_type)


def invoke_tool(ctx: Context, name: str, args: dict) -> Any:
    """Call a registered external tool — the only network/IO path for cached code."""
    tools = _STATE.get("tools") or {}
    if name not in tools:
        raise CacheInvocationError(
            f"invoke_tool: unknown tool '{name}' (registered={sorted(tools)})")
    return tools[name](**args)


def _primitives() -> dict:
    return {"load": load, "do": do, "narrate": narrate, "lm": lm,
            "extract": extract, "convert": convert,
            "smart_convert": lambda ctx, text, t: smart_convert(ctx, text, t),
            "invoke_tool": invoke_tool}
