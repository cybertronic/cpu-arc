"""ARC (Agentic Reasoning Cache) runtime — P1 hardened package.

Reimplementation of the CPU-ARC paper (arXiv:2611.09891) runtime surface:
layered Context, semantic functions discovered by natural-language description,
runtime primitives (load/do/narrate/lm/extract/convert/smart_convert/invoke_tool),
gated serving with LLM fall-through, SQLite persistence, sandboxed execution,
and a per-role cost ledger.
"""
from .backends import Embedder, LMBackend, MockLM, Ledger
from .context import Context, Layer, CacheInvocationError
from .primitives import (
    load, do, narrate, lm, extract, convert, smart_convert, invoke_tool,
    bind, get_state,
)
from .registry import Registry
from .sandbox import SandboxError, exec_function_source
from .cache import ARC, serve, Thresholds

__version__ = "0.1.0"
__all__ = [
    "ARC", "serve", "Thresholds", "Embedder", "LMBackend", "MockLM", "Ledger",
    "Context", "Layer", "CacheInvocationError", "Registry",
    "load", "do", "narrate", "lm", "extract", "convert", "smart_convert",
    "invoke_tool", "bind", "get_state", "SandboxError", "exec_function_source",
    "__version__",
]
