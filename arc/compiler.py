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

import numpy as np

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
    functions: dict[str, FunctionCompileResult]  # keyed by module-level name
    manifest: dict = field(default_factory=dict)
    # v2 multi-entry: one plan per unique (root, params) entry
    entries: list[dict] = field(default_factory=list)
    # per-entry ordered module names (dependencies first)
    entry_order: list[list[str]] = field(default_factory=list)

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


def _rewrite_load_calls(source: str, param_map: dict[str, int],
                        fuzzy_map: dict[str, str] | None = None) -> str:
    """Replace load(ctx, "<param desc>", ...) with ctx.params["<desc>"] lookups.

    Rewrites loads whose query exactly matches a known entry parameter
    description (case-insensitive), or — when provided — a fuzzy_map entry
    (load query -> parameter description, resolved by embedding similarity:
    the builder sometimes reuses a function whose source was written against
    a sibling entry's parameter names). Everything else stays as runtime
    load(). If a d_type kwarg is present, wraps the lookup in a coercion call.
    """
    fuzzy_map = fuzzy_map or {}

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
        if q in fuzzy_map:
            desc = fuzzy_map[q]
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

        return self._compile_entries([entry])

    def compile_all(self) -> CompileResult:
        """Compile EVERY entry in the store into one multi-entry module.

        Entries sharing (root_function_id, params) share one compiled tree,
        but each entry description gets its own dispatch row (the gateway
        matches by exact description). Trees get suffixed function names when
        more than one distinct tree exists, because load() rewrites depend on
        the entry's parameter descriptions — a shared function can
        legitimately have different compiled sources per entry context.
        """
        return self._compile_entries(list(self.entry_store.entries))

    def _compile_entries(self, entries: list[dict]) -> CompileResult:
        if not entries:
            raise CompileError("entry store is empty — nothing to compile")

        funcs_by_id = {f["id"]: f for f in self.registry.functions}
        all_funcs: dict[str, FunctionCompileResult] = {}
        plans: list[dict] = []
        tree_cache: dict[tuple, tuple] = {}  # (root, params) -> (name_map, ordered_names, root_name)
        n_trees = 0

        # One compiled tree per unique (root, params); EVERY entry gets its own
        # dispatch row (the gateway matches by exact entry description).
        for entry in entries:
            root_id = entry["root_function_id"]
            params = entry["params"]
            key = (root_id, tuple(params))
            if key not in tree_cache:
                if root_id not in funcs_by_id:
                    raise CompileError(f"root function {root_id!r} not in registry")
                suffix = "" if n_trees == 0 else f"_{n_trees}"
                fids = self._collect_fids(root_id, funcs_by_id)
                name_map = {fid: f"{self._fn_name(fid)}{suffix}" for fid in fids}

                compiled: dict[str, FunctionCompileResult] = {}
                visited: set[str] = set()
                self._compile_func(root_id, funcs_by_id, params, compiled,
                                   visited, name_map)
                ordered = self._topo_order(compiled, funcs_by_id, root_id)
                for fid in ordered:
                    all_funcs[name_map[fid]] = compiled[fid]
                tree_cache[key] = (name_map, [name_map[fid] for fid in ordered],
                                   name_map[root_id])
                n_trees += 1
            name_map, ordered_names, root_name = tree_cache[key]
            plans.append({"entry": entry, "root_name": root_name,
                          "ordered": ordered_names})

        first = entries[0]
        result = CompileResult(
            entry_description=first["description"],
            root_func_id=first["root_function_id"],
            param_descriptions=list(first["params"]),
            functions=all_funcs,
        )
        result.entries = [{"description": p["entry"]["description"],
                           "params": list(p["entry"]["params"]),
                           "root_func_id": p["entry"]["root_function_id"],
                           "root_name": p["root_name"]} for p in plans]
        result.entry_order = [p["ordered"] for p in plans]
        result.manifest = self._build_manifest(result)
        return result

    def _collect_fids(self, root_id: str, funcs_by_id: dict) -> set[str]:
        """DFS the do() intent graph from root; return every reachable func_id."""
        out: set[str] = set()

        def visit(fid: str) -> None:
            if fid in out or fid not in funcs_by_id:
                return
            out.add(fid)
            src = funcs_by_id[fid].get("source") or ""
            try:
                tree = ast.parse(src)
            except SyntaxError:
                return
            for call in _iter_calls(tree, "do"):
                intent = _string_arg(call, 1)
                if intent is None:
                    continue
                resolved = self._resolve_intent(intent)
                if resolved is not None:
                    visit(resolved[0])

        visit(root_id)
        return out

    def _fuzzy_param_match(self, query: str, params: list[str]) -> str | None:
        """Best entry parameter by embedding cosine (>= 0.72).

        Handles the case where the builder reused a function whose source was
        written against a sibling entry's parameter names (e.g. 'the first
        integer in the list' vs 'the first integer in the GCD query').
        """
        if not params:
            return None
        try:
            qv = np.asarray(self.registry.embedder.embed(query),
                            dtype=np.float32)
            qn = float(np.linalg.norm(qv))
        except Exception:
            return None
        if qn < 1e-12:
            return None
        best, best_s = None, 0.0
        for p in params:
            try:
                pv = np.asarray(self.registry.embedder.embed(p),
                                dtype=np.float32)
            except Exception:
                continue
            pn = float(np.linalg.norm(pv))
            if pn < 1e-12:
                continue
            s = float(qv @ pv / (qn * pn))
            if s > best_s:
                best, best_s = p, s
        return best if best_s >= 0.72 else None

    def _compile_func(self, func_id: str, funcs_by_id: dict, params: list[str],
                      compiled: dict, visited: set,
                      name_map: dict | None = None) -> None:
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

            # Build the replacement: <module name>(ctx)
            dep_fn_name = (name_map or {}).get(dep_id, self._fn_name(dep_id))
            rest = ""
            if len(call.args) > 2:
                # There's a third arg (payloads dict literal) — keep it as kwarg
                # We'll handle this in the regex rewrite by matching the full call.
                pass
            do_replacements[intent] = f"{dep_fn_name}(ctx)"

        # Fuzzy resolution for load() queries that match no entry parameter
        # exactly (reused functions may reference a sibling entry's names).
        load_fuzzy: dict[str, str] = {}
        for call in _iter_calls(tree, "load"):
            query = _string_arg(call, 1)
            if query is None:
                continue
            q = query.strip().lower()
            if any(d.strip().lower() == q for d in params):
                continue
            fp = self._fuzzy_param_match(query, params)
            if fp is not None:
                load_fuzzy[q] = fp

        # Rewrite source
        new_source = _rewrite_do_calls(source, do_replacements)
        new_source = _rewrite_load_calls(new_source, param_map, load_fuzzy)
        fr.compiled_source = new_source

        # Track load() calls for the manifest
        for call in _iter_calls(tree, "load"):
            query = _string_arg(call, 1)  # load(ctx, "<desc>", ...)
            if query is None:
                continue
            q = query.strip().lower()
            param_idx = next((i for d, i in param_map.items()
                              if d.strip().lower() == q), None)
            if param_idx is None and q in load_fuzzy:
                param_idx = next(i for i, d in enumerate(params)
                                 if d == load_fuzzy[q])
                method = "param-fuzzy"
            else:
                method = "param-exact" if param_idx is not None else "runtime"
            fr.load_calls.append(ResolvedLoad(
                query=query,
                param_index=param_idx,
                match_method=method,
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
            "format": 2 if len(result.entries) > 1 else 1,
            "entry": result.entry_description,
            "root": result.root_func_id,
            "params": result.param_descriptions,
            "entries": result.entries,
            "functions": [
                {
                    "id": fid,
                    "name": fid if str(fid).startswith("_") else self._fn_name(fid),
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
        """Emit the standalone Python module source (multi-entry format).

        One function tree per unique (root, params) entry, with entry-suffixed
        names where a shared function compiles differently per entry context.
        Dispatch: execute(entry_description, params) (v2) or solve(params)
        (legacy — runs the first entry).
        """
        lines = [
            '"""ARC compiled module — generated by arc.compiler (v2).',
            f'Entries: {len(result.entries)}',
            "No runtime vector store needed for resolved do()/load() paths.",
            '"""',
            "from __future__ import annotations",
            "import re as _re",
            "",
            "# --- runtime stubs (no-op / minimal) -----------------------------",
            "def narrate(*args, **kwargs):",
            '    """No-op narration in compiled mode. Accepts the runtime primitive\'s',
            '    narrate(msg) calling convention (and the legacy narrate(ctx, msg))."""',
            "    return None",
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
            "def load(ctx, query, **kwargs):",
            '    """Runtime parameter lookup for load() calls left unresolved at compile',
            "    time (e.g. the generated code paraphrased the parameter description).",
            "    Exact case-insensitive match first, then a unique substring match,",
            "    then a unique best token-overlap (Jaccard >= 0.5) match.",
            '    Fails loudly if nothing binds."""',
            "    q = str(query).strip().lower()",
            "    params = ctx.params",
            "    def _num(v):",
            "        if isinstance(v, str) and _re.fullmatch(r'-?\\d+', v.strip()):",
            "            return int(v)",
            "        return v",
            "    for k, v in params.items():",
            "        if str(k).strip().lower() == q:",
            "            return _num(v)",
            "    hits = [v for k, v in params.items()",
            "            if q in str(k).strip().lower() or str(k).strip().lower() in q]",
            "    if len(hits) == 1:",
            "        return _num(hits[0])",
            "    best, best_s = None, 0.0",
            "    for k, v in params.items():",
            "        ta, tb = set(str(k).lower().split()), set(q.split())",
            "        s = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0",
            "        if s > best_s:",
            "            best, best_s = v, s",
            "    if best is not None and best_s >= 0.5:",
            "        return _num(best)",
            "    raise KeyError(query)",
            "",
            "def do(ctx, intent, *args, **kwargs):",
            '    """Fail loudly: an unresolved do() means the module is not fully static."""',
            "    raise RuntimeError(f'unresolved do() in compiled module: {intent!r}')",
            "",
            "def _mk_ctx(params):",
            "    from arc.context import Context, Thresholds",
            "    from arc.primitives import bind, _primitives",
            "    # Minimal context: params are pre-extracted (no vector search)",
            "    class _CompiledCtx:",
            "        def __init__(self, params):",
            "            self.params = params",
            "            self.narration = []",
            "            self.load_log = []",
            "        def narrate(self, msg):",
            "            self.narration.append(str(msg))",
            "    return _CompiledCtx(params)",
            "",
        ]

        # Emit functions in per-entry topological order (dedup by name)
        seen: set[str] = set()
        for names in result.entry_order:
            for nm in names:
                if nm in seen:
                    continue
                seen.add(nm)
                fr = result.functions[nm]
                lines.append(f"# --- {nm} " + "-" * 48)
                src = fr.compiled_source
                tree = ast.parse(src)
                old_name = _find_function_def(tree).name
                if old_name != nm:
                    src = re.sub(rf'^\s*def\s+{re.escape(old_name)}\s*\(',
                                 f'def {nm}(', src, count=1, flags=re.MULTILINE)
                lines.append(src.rstrip())
                lines.append("")

        # Entry wrappers + dispatch table
        lines += ["# --- entry points (v2 dispatch) --------------------------------"]
        for i, ep in enumerate(result.entries):
            lines += [
                f"def _entry_{i}(params: dict):",
                f'    """Entry: {ep["description"]!r}"""',
                "    ctx = _mk_ctx(params)",
                f"    return {ep['root_name']}(ctx)",
                "",
            ]
        table_rows = ",\n".join(
            f"    ({json.dumps(ep['description'])}, _entry_{i})"
            for i, ep in enumerate(result.entries))
        lines += [
            "ENTRY_TABLE = [",
            table_rows,
            "]",
            "",
            "def execute(entry_description: str, params: dict):",
            '    """Run the compiled code for the matched entry."""',
            "    for d, fn in ENTRY_TABLE:",
            "        if d == entry_description:",
            "            return fn(params)",
            "    raise KeyError(entry_description)",
            "",
            "def solve(params: dict):",
            '    """Legacy single-entry entry point (runs the first entry)."""',
            "    return _entry_0(params)",
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


def main() -> None:
    """CLI: recompile a cache's multi-entry module.

    Usage: python3 -m arc.compiler <cache_dir>
    """
    import argparse
    from arc import Embedder

    ap = argparse.ArgumentParser(description="Recompile ARC cache module (v2)")
    ap.add_argument("cache_dir")
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)

    embedder = Embedder(backend="hash")
    registry = Registry(embedder, path=cache_dir / "functions.db")
    entry_store = EntryStore(embedder, path=cache_dir / "entries.db")
    compiler = Compiler(registry, entry_store)
    crec = compiler.compile_all()
    out = compiler.write(crec, cache_dir / "compiled")
    n_trees = len({(e["root_func_id"], tuple(e["params"])) for e in crec.entries})
    print(f"compiled {len(entry_store.entries)} entries "
          f"({n_trees} distinct trees) -> {out}")
    print(f"  fully_static={crec.manifest['stats']['fully_static']} "
          f"all_resolved={crec.all_resolved}")


if __name__ == "__main__":
    main()
