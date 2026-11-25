"""P3 Compiler: static resolution of semantic functions into a standalone module.

The paper's compiled artifact is a plain Python module that needs NO runtime
vector store for resolved paths:
  - do(ctx, "<intent>")   -> direct function call _fn_<id>(ctx)
  - load(ctx, "<desc>")   -> ctx.params["<desc>"]  (entry parameters only)

What survives in the compiled module (still needs the ARC runtime):
  - extract(ctx, ...) / lm(ctx, ...) — LLM-backed value extraction (rare on hot path)
  - narrate(ctx, ...) — optional tracing

The compiler walks each function's AST, resolves do() intents against the
registry by exact description match (with vector fallback), and rewrites the
source. load() calls are resolved against the entry's parameter descriptions;
unresolvable loads are left as-is (they'll hit the runtime context).

Output: a single .py file with all functions + a manifest JSON, importable and
executable without the registry/vector store for the resolved call graph.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import Registry, EntryStore


class CompileError(Exception):
    pass


@dataclass
class ResolvedCall:
    """A do() call resolved to a direct function reference."""
    func_id: str          # registry fn id (fn_0001)
    intent: str           # original NL intent string
    match_method: str     # "exact" | "vector"
    score: float = 1.0


@dataclass
class ResolvedLoad:
    """A load() call resolved to a parameter or left for runtime."""
    query: str
    param_index: int | None = None   # index into entry params, or None
    match_method: str = "runtime"    # "param-exact" | "runtime"


@dataclass
class FunctionCompileResult:
    func_id: str
    original_source: str
    compiled_source: str
    do_calls: list[ResolvedCall] = field(default_factory=list)
    load_calls: list[ResolvedLoad] = field(default_factory=list)
    unresolved_do: list[str] = field(default_factory=list)  # intents that couldn't resolve


@dataclass
class CompileResult:
    entry_description: str
    root_func_id: str
    param_descriptions: list[str]
    functions: dict[str, FunctionCompileResult]  # keyed by func_id
    manifest: dict = field(default_factory=dict)

    @property
    def all_resolved(self) -> bool:
        """True if every do() call was statically resolved (no runtime vector lookup)."""
        return all(not fr.unresolved_do for fr in self.functions.values())


# --------------------------------------------------------------------------- #
# AST analysis helpers
# --------------------------------------------------------------------------- #

def _find_function_def(tree: ast.Module) -> ast.FunctionDef:
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        raise CompileError("no function definition found in source")
    return funcs[0]


def _iter_calls(node: ast.AST, name: str):
    """Yield (call_node, arg_nodes) for every call to `name` in the tree."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name:
            yield n


def _string_arg(call: ast.Call, idx: int = 0) -> str | None:
    """Extract a constant string argument at position idx, or None."""
    if idx < len(call.args):
        a = call.args[idx]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value
    return None


# --------------------------------------------------------------------------- #
# Source rewriting (token-level, preserves comments/formatting where possible)
# --------------------------------------------------------------------------- #

def _rewrite_do_calls(source: str, replacements: dict[str, str]) -> str:
    """Replace do(ctx, "<intent>", ...) with direct calls.

    replacements maps the EXACT intent string to the replacement expression.
    We use a regex that matches do() calls with a string literal first arg.
    """
    def _sub(match: re.Match) -> str:
        # Capture: do( ctx , "intent" [rest] )
        intent = match.group("intent")
        if intent in replacements:
            return replacements[intent]
        return match.group(0)  # leave unresolved calls as-is

    # Match do(ctx, "literal", ...) — the string must be a plain literal.
    pattern = re.compile(
        r'\bdo\(\s*ctx\s*,\s*r?"(?P<intent>(?:[^"\\]|\\.)*)"\s*(?P<rest>,[^\)]*)?\)',
        re.DOTALL,
    )
    return pattern.sub(_sub, source)


def _rewrite_load_calls(source: str, param_map: dict[str, int]) -> str:
    """Replace load(ctx, "<param desc>", ...) with ctx.params["<desc>"] lookups.

    Only rewrites loads whose query exactly matches a known entry parameter
    description (case-insensitive). Everything else stays as runtime load().
    If a d_type kwarg is present, wraps the lookup in a coercion call.
    """
    def _sub(match: re.Match) -> str:
        query = match.group("query")
        q = query.strip().lower()
        for desc, idx in param_map.items():
            if desc.strip().lower() == q:
                rest = match.group("rest") or ""
                d_type_m = re.search(r'd_type\s*=\s*(\w+)', rest)
                if d_type_m:
                    dt = d_type_m.group(1)
                    return f'_coerce_param(ctx.params["{desc}"], {dt})'
                return f'ctx.params["{desc}"]'
        return match.group(0)

    pattern = re.compile(
        r'\bload\(\s*ctx\s*,\s*r?"(?P<query>(?:[^"\\]|\\.)*)"\s*(?P<rest>,[^\)]*)?\)',
        re.DOTALL,
    )
    return pattern.sub(_sub, source)


# --------------------------------------------------------------------------- #
# Compiler
# --------------------------------------------------------------------------- #

class Compiler:
    """Resolves a cache's function graph into a standalone module."""

    def __init__(self, registry: Registry, entry_store: EntryStore):
        self.registry = registry
        self.entry_store = entry_store

    def compile_entry(self, entry_description: str | None = None) -> CompileResult:
        """Compile the entry (default: first/only entry) into a standalone module.

        Walks the root function's dependency graph via do() intents, resolves
        each against the registry, and rewrites all sources.
        """
        if not self.entry_store.entries:
            raise CompileError("entry store is empty — nothing to compile")

        # Pick the entry
        if entry_description is None:
            entry = self.entry_store.entries[0]
        else:
            matches = [e for e in self.entry_store.entries
                       if e["description"].lower() == entry_description.lower()]
            if not matches:
                # vector fallback
                best, score = self.entry_store.search(entry_description)
                if best is None or score < 0.8:
                    raise CompileError(f"entry not found: {entry_description!r}")
                entry = best

        root_id = entry["root_function_id"]
        params = entry["params"]

        # Build func_id -> registry entry map
        funcs_by_id = {f["id"]: f for f in self.registry.functions}
        if root_id not in funcs_by_id:
            raise CompileError(f"root function {root_id!r} not in registry")

        # Walk the dependency graph from the root via do() intents
        compiled: dict[str, FunctionCompileResult] = {}
        visited: set[str] = set()
        self._compile_func(root_id, funcs_by_id, params, compiled, visited)

        # Order functions: dependencies first (topological), root last
        ordered = self._topo_order(compiled, funcs_by_id, root_id)

        result = CompileResult(
            entry_description=entry["description"],
            root_func_id=root_id,
            param_descriptions=params,
            functions={fid: compiled[fid] for fid in ordered},
        )
        result.manifest = self._build_manifest(result)
        return result

    def _compile_func(self, func_id: str, funcs_by_id: dict, params: list[str],
                      compiled: dict, visited: set) -> None:
        if func_id in visited:
            return
        visited.add(func_id)

        entry = funcs_by_id[func_id]
        source = entry["source"] or ""
        if not source:
            raise CompileError(f"function {func_id} has no source (closure-only)")

        fr = FunctionCompileResult(
            func_id=func_id,
            original_source=source,
            compiled_source=source,  # start with original; rewrite below
        )

        # Parse and find do() calls
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise CompileError(f"{func_id}: syntax error in source: {e}") from e

        # Build param_map for load() rewriting (exact description match)
        param_map = {desc: i for i, desc in enumerate(params)}

        do_replacements: dict[str, str] = {}  # intent -> replacement expr

        for call in _iter_calls(tree, "do"):
            intent = _string_arg(call, 1)  # do(ctx, "<intent>", ...)
            if intent is None:
                # Non-literal intent (variable) — cannot resolve statically
                fr.unresolved_do.append("<dynamic>")
                continue

            # Resolve intent against registry
            resolved = self._resolve_intent(intent)
            if resolved is None:
                fr.unresolved_do.append(intent)
                continue

            dep_id, method, score = resolved
            fr.do_calls.append(ResolvedCall(func_id=dep_id, intent=intent,
                                            match_method=method, score=score))

            # Recursively compile the dependency (ensures it's in `compiled`)
            if dep_id in funcs_by_id:
                self._compile_func(dep_id, funcs_by_id, params, compiled, visited)

            # Build the replacement: _fn_<id>(ctx) with optional payloads
            dep_fn_name = self._fn_name(dep_id)
            rest = ""
            if len(call.args) > 2:
                # There's a third arg (payloads dict literal) — keep it as kwarg
                # We'll handle this in the regex rewrite by matching the full call.
                pass
            do_replacements[intent] = f"{dep_fn_name}(ctx)"

        # Rewrite source
        new_source = _rewrite_do_calls(source, do_replacements)
        new_source = _rewrite_load_calls(new_source, param_map)
        fr.compiled_source = new_source

        # Track load() calls for the manifest
        for call in _iter_calls(tree, "load"):
            query = _string_arg(call, 1)  # load(ctx, "<desc>", ...)
            if query is None:
                continue
            q = query.strip().lower()
            param_idx = next((i for d, i in param_map.items()
                              if d.strip().lower() == q), None)
            fr.load_calls.append(ResolvedLoad(
                query=query,
                param_index=param_idx,
                match_method="param-exact" if param_idx is not None else "runtime",
            ))

        compiled[func_id] = fr

    def _resolve_intent(self, intent: str) -> tuple[str, str, float] | None:
        """Resolve a do() intent to (func_id, match_method, score)."""
        # Exact description match first (case-insensitive)
        q = intent.strip().lower()
        for f in self.registry.functions:
            if f["description"].strip().lower() == q:
                return f["id"], "exact", 1.0

        # Vector fallback with the function threshold
        hits = self.registry.search(intent, top_k=1)
        if hits and hits[0][1] >= self.registry.function_threshold:
            return hits[0][0]["id"], "vector", hits[0][1]
        return None

    @staticmethod
    def _fn_name(func_id: str) -> str:
        """fn_0003 -> _fn_0003 (module-level name in compiled output)."""
        return f"_{func_id}"

    def _topo_order(self, compiled: dict, funcs_by_id: dict, root_id: str) -> list[str]:
        """Order functions so dependencies come before dependents."""
        order: list[str] = []
        visiting: set[str] = set()

        def visit(fid: str):
            if fid in order or fid in visiting:
                return
            visiting.add(fid)
            fr = compiled.get(fid)
            if fr:
                for dc in fr.do_calls:
                    visit(dc.func_id)
            visiting.discard(fid)
            order.append(fid)

        visit(root_id)
        # Include any compiled funcs not reachable from root (shouldn't happen, but safe)
        for fid in compiled:
            if fid not in order:
                order.append(fid)
        return order

    def _build_manifest(self, result: CompileResult) -> dict:
        total_do = sum(len(fr.do_calls) for fr in result.functions.values())
        unresolved = sum(len(fr.unresolved_do) for fr in result.functions.values())
        total_load = sum(len(fr.load_calls) for fr in result.functions.values())
        param_loads = sum(1 for fr in result.functions.values()
                          for lc in fr.load_calls if lc.param_index is not None)
        return {
            "entry": result.entry_description,
            "root": result.root_func_id,
            "params": result.param_descriptions,
            "functions": [
                {
                    "id": fid,
                    "name": self._fn_name(fid),
                    "do_calls": len(fr.do_calls),
                    "unresolved_do": fr.unresolved_do,
                    "load_calls": len(fr.load_calls),
                }
                for fid, fr in result.functions.items()
            ],
            "stats": {
                "total_functions": len(result.functions),
                "total_do_resolved": total_do,
                "total_do_unresolved": unresolved,
                "total_load": total_load,
                "load_param_exact": param_loads,
                "fully_static": result.all_resolved,
            },
        }

    # ------------------------------------------------------------------ #
    # Emission
    # ------------------------------------------------------------------ #

    def emit_module(self, result: CompileResult) -> str:
        """Emit the standalone Python module source."""
        lines = [
            '"""ARC compiled module — generated by arc.compiler.',
            f'Entry: {result.entry_description!r}',
            "No runtime vector store needed for resolved do()/load() paths.",
            '"""',
            "from __future__ import annotations",
            "import re as _re",
            "",
            "# --- runtime stubs (no-op / minimal) -----------------------------",
            "def narrate(ctx, msg):", 
            '    """No-op narration in compiled mode (enable via ctx.verbose)."""',
            "    if getattr(ctx, 'verbose', False):",
            "        ctx.narration.append(str(msg))",
            "",
            "def _coerce_param(value, d_type):",
            '    """Lightweight type coercion for pre-extracted parameters."""',
            "    if d_type is None or value is None:",
            "        return value",
            "    if isinstance(value, d_type):",
            "        return value",
            "    if d_type in (int, float):",
            "        try:",
            '            s = str(value).replace(",", "")',
            "            m = _re.search(r'-?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?', s)",
            "            if m:",
            "                v = m.group(0)",
            "                return float(v) if d_type is float else int(float(v))",
            "        except (ValueError, TypeError):",
            "            pass",
            "    raise ValueError(f'cannot coerce {value!r} to {d_type.__name__}')",
            "",
            "# --- entry parameters (resolved at serve time) -------------------",
            f"PARAMS = {json.dumps(result.param_descriptions)}",
            f"ROOT = '{result.root_func_id}'",
            "",
        ]

        # Emit functions in topological order
        for fid, fr in result.functions.items():
            lines.append(f"# --- {fid} ({self._fn_name(fid)}) " + "-" * 40)
            # Rewrite the def name to _fn_<id>
            src = fr.compiled_source
            # Replace the original function name with the module-level name
            tree = ast.parse(src)
            func_def = _find_function_def(tree)
            old_name = func_def.name
            new_name = self._fn_name(fid)
            if old_name != new_name:
                src = re.sub(rf'^\s*def\s+{re.escape(old_name)}\s*\(',
                             f'def {new_name}(', src, count=1, flags=re.MULTILINE)
            lines.append(src.rstrip())
            lines.append("")

        # Entry point helper
        lines += [
            "# --- entry point ------------------------------------------------",
            "def solve(params: dict):",
            '    """Run the root function with extracted parameters.',
            "",
            "    params: {description: value} for each entry parameter.",
            "    Returns the root function's output dict.",
            '    """',
            "    from arc.context import Context, Thresholds",
            "    from arc.primitives import bind, _primitives",
            "",
            "    # Minimal context: params are pre-extracted (no vector search)",
            "    class _CompiledCtx:",
            "        def __init__(self, params):",
            "            self.params = params",
            "            self.narration = []",
            "            self.load_log = []",
            "        def narrate(self, msg):",
            "            self.narration.append(str(msg))",
            "",
            "    ctx = _CompiledCtx(params)",
            f"    return {self._fn_name(result.root_func_id)}(ctx)",
            "",
        ]
        return "\n".join(lines)

    def write(self, result: CompileResult, out_dir: str | Path) -> Path:
        """Write the compiled module + manifest to out_dir. Returns module path."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mod_path = out_dir / "arc_compiled.py"
        mod_path.write_text(self.emit_module(result))

        man_path = out_dir / "manifest.json"
        man_path.write_text(json.dumps(result.manifest, indent=2))

        return mod_path
