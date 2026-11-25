"""Backend abstractions: embeddings, LM (target model T), and a cost ledger.

Design goal: ARC's 𝒯 interface is exactly an OpenAI-compatible chat endpoint, so
llama.cpp `llama-server` works with zero adapter code (P0 finding). Embeddings
support llama.cpp's native `/embeddings` (nested [1][dim]) and OpenAI format,
plus a deterministic offline hash embedder for tests.
"""
from __future__ import annotations

import http.client
import json
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any
from urllib.parse import urlparse

import numpy as np


# --------------------------------------------------------------------------- #
# Keep-alive HTTP client (thread-safe, one connection per thread)
# --------------------------------------------------------------------------- #
class _KeepAliveClient:
    """Minimal keep-alive POST/JSON client over http.client.

    Replaces per-call urllib.urlopen (fresh TCP handshake per request) with a
    persistent connection per thread (threading.local), auto-reconnecting on
    stale/broken connections. Both Embedder and LMBackend use it.
    """

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self._local = threading.local()

    def post_json(self, url: str, payload: bytes, headers: dict) -> bytes:
        p = urlparse(url)
        host = p.hostname
        port = p.port or (443 if p.scheme == "https" else 80)
        path = p.path or "/"
        if p.query:
            path = f"{path}?{p.query}"
        conn_cls = (http.client.HTTPSConnection if p.scheme == "https"
                    else http.client.HTTPConnection)
        for attempt in (1, 2):
            conn = getattr(self._local, "conn", None)
            if conn is not None and (conn.host, conn.port) != (host, port):
                conn.close()
                conn = None
            if conn is None:
                conn = conn_cls(host, port, timeout=self.timeout)
                self._local.conn = conn
            try:
                conn.request("POST", path, body=payload, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {data[:200]!r}")
                return data
            except (http.client.HTTPException, ConnectionError, OSError) as e:
                # Stale/recycled connection: drop and retry once.
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.conn = None
                if attempt == 2:
                    raise RuntimeError(f"POST {url} failed: {e}") from e


# --------------------------------------------------------------------------- #
# Cost ledger (per-role token accounting, mirrors paper Appendix C)
# --------------------------------------------------------------------------- #
class Ledger:
    """Accumulates token/cost by role. Roles: target, extractor, converter, build."""

    def __init__(self, prices: dict[str, dict[str, float]] | None = None):
        # prices[role] = {"in": $/tok, "out": $/tok}
        self.prices = prices or {}
        self.by_role: dict[str, dict[str, int]] = {}

    def add(self, role: str, in_tokens: int = 0, out_tokens: int = 0) -> None:
        r = self.by_role.setdefault(role, {"in": 0, "out": 0, "calls": 0})
        r["in"] += in_tokens
        r["out"] += out_tokens
        r["calls"] += 1

    def total_tokens(self) -> int:
        return sum(v["in"] + v["out"] for v in self.by_role.values())

    def cost(self) -> float:
        c = 0.0
        for role, v in self.by_role.items():
            p = self.prices.get(role, {})
            c += v["in"] * p.get("in", 0.0) + v["out"] * p.get("out", 0.0)
        return c

    def snapshot(self) -> dict:
        return {"by_role": {k: dict(v) for k, v in self.by_role.items()},
                "total_tokens": self.total_tokens(), "cost": self.cost()}

    def mark(self) -> dict:
        """Snapshot current totals for later delta computation (per-invocation stats)."""
        return {k: dict(v) for k, v in self.by_role.items()}

    def since(self, mark: dict) -> dict:
        """Snapshot of only what was added after mark (mirrors paper App C per-role)."""
        out: dict[str, dict[str, int]] = {}
        for k, v in self.by_role.items():
            m = mark.get(k, {"in": 0, "out": 0, "calls": 0})
            d = {"in": v["in"] - m["in"], "out": v["out"] - m["out"],
                 "calls": v["calls"] - m["calls"]}
            if any(d.values()):
                out[k] = d
        total = sum(v["in"] + v["out"] for v in out.values())
        c = 0.0
        for role, v in out.items():
            p = self.prices.get(role, {})
            c += v["in"] * p.get("in", 0.0) + v["out"] * p.get("out", 0.0)
        return {"by_role": out, "total_tokens": total, "cost": c}


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class Embedder:
    """CPU embedding provider. Backends: 'hash' (offline) or 'server' (llama.cpp / OpenAI)."""

    def __init__(self, backend: str = "hash", url: str = "http://127.0.0.1:8081",
                 dim: int = 768, api_key: str = "none"):
        self.backend = backend
        self.url = url.rstrip("/")
        self.dim = dim
        self.api_key = api_key
        self._cache: dict[str, np.ndarray] = {}
        self._http = _KeepAliveClient(timeout=60.0)

    def embed(self, text: str) -> np.ndarray:
        key = text if len(text) < 256 else text[:256]
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self.backend == "hash":
            vec = self._hash_embed(text)
        else:
            vec = self._server_embed(text)
        self._cache[key] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]

    def _server_embed(self, text: str) -> np.ndarray:
        data = json.loads(self._http.post_json(
            f"{self.url}/embeddings",
            json.dumps({"input": text}).encode(),
            {"Content-Type": "application/json",
             "Authorization": f"Bearer {self.api_key}"},
        ))
        if isinstance(data, list):
            emb = data[0]["embedding"]          # llama.cpp native format
        else:
            emb = data["data"][0]["embedding"]  # OpenAI format
        while isinstance(emb, list) and len(emb) == 1:
            emb = emb[0]                        # unwrap [1][dim]
        return np.asarray(emb, dtype=np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic bag-of-words hash embedder — offline/test fallback only.

        Not semantically meaningful, but stable, so unit tests are reproducible.
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            vec[abs(hash(tok)) % self.dim] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec


# --------------------------------------------------------------------------- #
# LM backend (target model T / extractor / converter)
# --------------------------------------------------------------------------- #
@dataclass
class Completion:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    ms: float = 0.0


class LMBackend:
    """OpenAI-compatible chat completion client (llama.cpp server or any API)."""

    def __init__(self, url: str = "http://127.0.0.1:8080/v1", api_key: str = "none",
                 model: str = "default", ledger: Ledger | None = None,
                 reasoning_effort: str | None = "none"):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.ledger = ledger or Ledger()
        # Qwen3-family thinking models (e.g. Qwen3.8-27B on the system llama.cpp
        # server) spend ALL completion tokens on reasoning_content, leaving
        # message.content empty with finish_reason=length — reproducible on long
        # structured prompts. reason=none / reasoning_effort:none disables it.
        self.reasoning_effort = reasoning_effort
        self._http = _KeepAliveClient(timeout=600.0)

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, role: str = "target",
                 max_tokens: int = 512) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        payload = json.dumps(body).encode()
        t0 = time.perf_counter()
        data = json.loads(self._http.post_json(
            f"{self.url}/chat/completions", payload,
            {"Content-Type": "application/json",
             "Authorization": f"Bearer {self.api_key}"},
        ))
        dt = (time.perf_counter() - t0) * 1000.0
        out = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        it = usage.get("prompt_tokens", 0)
        ot = usage.get("completion_tokens", 0)
        self.ledger.add(role, it, ot)
        return Completion(out, it, ot, dt)


class MockLM:
    """Deterministic LM for unit tests. Route on prompt substrings -> canned text.

    Lets us test the full coercion/extraction/dispatch logic with zero GPU.
    """

    def __init__(self, routes: dict[str, str] | None = None,
                 default: str = "NOT_FOUND", ledger: Ledger | None = None):
        # routes: ordered dict; first key that is a substring of the prompt wins.
        self.routes = routes or {}
        self.default = default
        self.ledger = ledger or Ledger()
        self.calls: list[str] = []

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, role: str = "target",
                 max_tokens: int = 512) -> Completion:
        self.calls.append(prompt)
        text = self.default
        for key, val in self.routes.items():
            if key in prompt:
                text = val
                break
        self.ledger.add(role, len(prompt) // 4, len(text) // 4)
        return Completion(text, len(prompt) // 4, len(text) // 4, 0.0)
