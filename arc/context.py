"""Context: stack of memory layers with vector + exact-match lookup.

P1 hardening over the P0 spike:
- exact/substring fast path BEFORE vector search (EmbeddingGemma conflates
  short param descriptions, e.g. molar-amount vs volume scored 0.76)
- per-scope thresholds (entry matching vs param loads need different cutoffs)
- structured miss diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class CacheInvocationError(Exception):
    """Raised on any cache-side failure; serve() catches this to fall through to the LLM."""


class Thresholds:
    """Per-scope similarity thresholds (P1: one global threshold was unsafe).

    entry   — entry-point matching (problems); paper shows clean separation here
    function— do() semantic-function discovery
    load    — layered param/value lookup (weak embedding separation => higher bar
              + exact-match fast path)
    """

    def __init__(self, entry: float = 0.55, function: float = 0.55, load: float = 0.45):
        self.entry = entry
        self.function = function
        self.load = load


@dataclass
class _Item:
    description: str
    value: Any
    vec: np.ndarray


@dataclass
class Layer:
    items: list[_Item] = field(default_factory=list)

    def add(self, description: str, value: Any, embedder) -> None:
        self.items.append(_Item(description, value, embedder.embed(description)))

    def exact_match(self, query: str) -> _Item | None:
        """P1 fast path: case-insensitive exact or token-substring match."""
        q = query.strip().lower()
        for it in self.items:
            d = it.description.strip().lower()
            if d == q:
                return it
        # token overlap: require >=70% of query tokens present in the description
        qtok = set(q.split())
        if len(qtok) >= 2:
            for it in self.items:
                dtok = set(it.description.strip().lower().split())
                if qtok and len(qtok & dtok) / len(qtok) >= 0.7:
                    return it
        return None

    def search(self, query_vec: np.ndarray, top_k: int = 1) -> list[tuple[_Item, float]]:
        if not self.items:
            return []
        vecs = np.stack([it.vec for it in self.items])
        scores = vecs @ query_vec
        order = np.argsort(-scores)[:top_k]
        return [(self.items[i], float(scores[i])) for i in order]


class Context:
    """Layered memory: base layer + one pushed layer per do() call."""

    def __init__(self, embedder, thresholds: Thresholds | None = None):
        self.embedder = embedder
        self.thresholds = thresholds or Thresholds()
        self.layers: list[Layer] = [Layer()]
        self.narration: list[str] = []
        self.load_log: list[dict] = []  # P1 observability: how each load resolved

    # -- layer management ----------------------------------------------------
    def push(self) -> None:
        self.layers.append(Layer())

    def pop(self) -> Layer:
        return self.layers.pop()

    # -- lookup ---------------------------------------------------------------
    def find(self, query: str, d_type=None, scope: str = "load") -> Any:
        """Top-down lookup: exact fast path first, then vector similarity."""
        from .primitives import coerce_or_convert  # local import avoids cycle
        th = getattr(self.thresholds, scope)
        for layer in reversed(self.layers):
            hit = layer.exact_match(query)
            if hit is not None:
                self.load_log.append({"query": query, "scope": scope, "via": "exact",
                                      "score": 1.0, "layer": len(self.layers)})
                return coerce_or_convert(self, hit.value, d_type)
            hits = layer.search(self.embedder.embed(query))
            if hits and hits[0][1] >= th:
                item, score = hits[0]
                self.load_log.append({"query": query, "scope": scope, "via": "vector",
                                      "score": score, "layer": len(self.layers)})
                return coerce_or_convert(self, item.value, d_type)
        best = self._best_score(query)
        raise CacheInvocationError(
            f"load: no match for '{query}' (scope={scope}, threshold={th}, best={best:.3f})"
        )

    def _best_score(self, query: str) -> float:
        qv = self.embedder.embed(query)
        best = 0.0
        for layer in self.layers:
            hits = layer.search(qv)
            if hits:
                best = max(best, hits[0][1])
        return best

    def narrate(self, message: str) -> None:
        depth = len(self.layers) - 1
        self.narration.append(("  " * depth) + message)
