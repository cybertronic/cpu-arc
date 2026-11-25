"""Sandboxed execution of semantic-function source code.

The paper's compiled artifact is plain Python; the builder generates function
source as text. P1 requirement: execute that source with a restricted
namespace (no arbitrary imports, only the primitive surface), a CPU wall-clock
budget, and a no-network guarantee (network is only reachable via invoke_tool
with an explicitly registered tool).

This is a *containment* sandbox, not a security boundary against a hostile
model — the threat model is builder bugs and runaway loops, not adversarial
code. Real isolation (seccomp/worker process) is a P1-later hardening note.
"""
from __future__ import annotations

import ast
import builtins
import resource
import time
from typing import Any, Callable

# The only names a semantic function may reference from the environment.
# Everything else must be defined locally or imported through the allowlist.
ALLOWED_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "max": max,
    "min": min, "range": range, "repr": repr, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None, "print": None,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "AttributeError": AttributeError, "Exception": Exception,
}

# import allowlist (math/string/datetime are the realistic needs of cached logic).
ALLOWED_IMPORTS = {"math", "string", "datetime", "json", "re", "decimal", "fractions"}

# Primitives are injected by the caller (bound to the active ctx/registry/lm).


class SandboxError(Exception):
    pass


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise SandboxError(f"relative import not allowed: {name}")
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise SandboxError(f"import not allowed: {name!r} (allowlist={sorted(ALLOWED_IMPORTS)})")
    import importlib
    return importlib.import_module(name)


# The import hook must live in the builtins dict itself, not just globals, or the
# C-level IMPORT_NAME opcode reports "__import__ not found".
ALLOWED_BUILTINS["__import__"] = _restricted_import


def build_namespace(primitives: dict[str, Any]) -> dict:
    ns: dict[str, Any] = {"__builtins__": ALLOWED_BUILTINS}
    ns.update(primitives)
    return ns


def check_source(source: str) -> None:
    """Static gate before exec: must be a def, must not dangle dangerous tokens."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e}") from e
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        raise SandboxError("semantic function source must contain a function definition")
    # reject explicit exec/eval and ALL dunder attribute access (ctx.__class__ etc.)
    banned = {"__subclasses__", "__globals__", "__loader__", "__spec__", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in banned or (node.attr.startswith("__") and node.attr.endswith("__")):
                raise SandboxError(f"forbidden attribute access: {node.attr}")
        if isinstance(node, ast.Name) and node.id in banned:
            raise SandboxError(f"forbidden name: {node.id}")
    # reject calls to exec/eval
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
            raise SandboxError(f"forbidden call: {node.func.id}()")


def exec_function_source(source: str, primitives: dict[str, Any] | None = None,
                         timeout_s: float = 30.0) -> Callable[[Any], dict]:
    """Compile and execute function source; return the bound callable.

    Enforces a CPU+wall clock budget around the returned callable via a wrapper.
    """
    tree = ast.parse(source)
    func_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_names:
        raise SandboxError("no function found in source")
    check_source(source)
    ns = build_namespace(primitives or {})
    code = compile(source, "<arc-function>", "exec")
    exec(code, ns)  # noqa: S102 — contained namespace, static-gated above
    func = ns.get(func_names[0])
    if not callable(func):
        raise SandboxError(f"function {func_names[0]!r} not defined after exec")

    def bounded(ctx) -> dict:
        soft, hard = int(timeout_s), int(timeout_s) + 1
        # per-process CPU limit (best effort; reset after so we don't poison the host)
        try:
            old = resource.getrlimit(resource.RLIMIT_CPU)
            resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
        except (ValueError, OSError):
            old = None
        t0 = time.perf_counter()
        try:
            result = func(ctx)
        finally:
            if old is not None:
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, old)
                except (ValueError, OSError):
                    pass
        if time.perf_counter() - t0 > timeout_s:
            raise SandboxError(f"function exceeded wall-clock budget {timeout_s}s")
        if not isinstance(result, dict):
            raise SandboxError(f"semantic function must return dict, got {type(result).__name__}")
        return result

    return bounded
