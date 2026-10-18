"""LDL010: a streamed OpenAI-style tool-call reducer that can't handle deltas without ``index``.

Gemini's OpenAI-compatible endpoint sends each parallel tool call whole in its own chunk, with no
``tool_calls[].index`` (the openai SDK then gives ``index=None``). Three shapes break on that:

- the index is the only key: every call lands on ``None`` and they collapse (or the lookup fails);
- deltas without an index are skipped (``if tc.index is not None:``): every call is dropped;
- the fallback is the delta's position in the chunk (``enumerate(delta.tool_calls)``), which is 0
  in every Gemini chunk: the calls still collapse.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register

_MESSAGES = {
    "none": (
        "Streamed tool calls are merged by `{var}.index` alone; Gemini's OpenAI-compatible endpoint "
        "sends each parallel call in its own delta without `index`, so the calls collapse into one "
        "(or the lookup fails)"
    ),
    "drop": (
        "Streamed tool-call deltas without `index` are skipped; Gemini's OpenAI-compatible endpoint "
        "never sends `index`, so every tool call is dropped"
    ),
    "position": (
        "Streamed tool-call deltas without `index` fall back to their position in the chunk, which is "
        "0 in every chunk from Gemini's OpenAI-compatible endpoint (one call per chunk), so the calls "
        "collapse into one"
    ),
}


def _chain(node: ast.expr) -> list[str]:
    """Names along an attribute / subscript / call chain: ``chunk.choices[0].delta.tool_calls``
    -> ["chunk", "choices", "delta", "tool_calls"]."""
    out = []
    while True:
        if isinstance(node, ast.Attribute):
            out.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                out.append(node.slice.value)
            node = node.value
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
                key = node.args[0]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    out.append(key.value)
            node = node.func.value if isinstance(node.func, ast.Attribute) else node.func
        elif isinstance(node, ast.Name):
            out.append(node.id)
            break
        else:
            break
    return list(reversed(out))


def _unwrap(node: ast.expr) -> tuple[ast.expr, bool]:
    """Strip ``x or []``, ``cast(T, x)``, ``list(x)`` and ``enumerate(x)``; says whether enumerate was seen."""
    enumerated = False
    while True:
        if isinstance(node, ast.BoolOp):
            node = node.values[0]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            name = node.func.id
            if name == "enumerate":
                enumerated, node = True, node.args[0]
            elif name == "cast" and len(node.args) == 2:
                node = node.args[1]
            elif name in ("list", "tuple", "iter"):
                node = node.args[0]
            else:
                return node, enumerated
        else:
            return node, enumerated


def _is_delta_tool_calls(node: ast.expr) -> bool:
    names = _chain(node)
    return bool(names) and names[-1] == "tool_calls" and any("delta" in n for n in names[:-1])


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _str_is(node: ast.expr, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


class _Scope:
    """One body that handles the delta ``var``; ``counters`` are names holding its position in the chunk."""

    def __init__(self, body: list[ast.stmt], var: str, counters: set[str]):
        self.body, self.var, self.counters = body, var, counters
        self.reads: list[ast.expr] = []
        self.defaults: list[ast.expr] = []  # explicit defaults: .get("index", d) / getattr(..., d)
        self.names: set[str] = set()  # names bound to the index
        self._collect()

    def _walk(self) -> Iterator[ast.AST]:
        for stmt in self.body:
            yield from ast.walk(stmt)

    def _is_read(self, node: ast.AST) -> bool:
        var = self.var
        if isinstance(node, ast.Attribute) and node.attr == "index":
            return isinstance(node.value, ast.Name) and node.value.id == var
        if isinstance(node, ast.Subscript) and _str_is(node.slice, "index"):
            return isinstance(node.value, ast.Name) and node.value.id == var
        if isinstance(node, ast.Call):
            func, args = node.func, node.args
            # tc.get("index"[, d])
            if isinstance(func, ast.Attribute) and func.attr == "get":
                return (
                    isinstance(func.value, ast.Name)
                    and func.value.id == var
                    and bool(args)
                    and _str_is(args[0], "index")
                )
            # getattr(tc, "index"[, d]) and helpers such as _get(tc, "index")
            if isinstance(func, ast.Name) and len(args) >= 2:
                return isinstance(args[0], ast.Name) and args[0].id == var and _str_is(args[1], "index")
        return False

    def _collect(self) -> None:
        for node in self._walk():
            if self._is_read(node):
                self.reads.append(node)  # type: ignore[arg-type]
                if isinstance(node, ast.Call):
                    extra = node.args[2:] if isinstance(node.func, ast.Name) else node.args[1:]
                    self.defaults += [a for a in extra if not _is_none(a)]
        for node in self._walk():
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if self._mentions_index(node.value):
                    self.names |= {t.id for t in targets if isinstance(t, ast.Name)}

    def _mentions_index(self, expr: ast.AST) -> bool:
        return any(self._is_index_expr(n) for n in ast.walk(expr))

    def _is_index_expr(self, expr: ast.AST) -> bool:
        return self._is_read(expr) or (isinstance(expr, ast.Name) and expr.id in self.names)

    def keyed(self) -> bool:
        """True when the index picks the accumulated call: a subscript, ``.get/.setdefault/.pop``
        argument or ``in`` test."""
        for node in self._walk():
            if (
                isinstance(node, ast.Subscript)
                and not self._is_read(node)
                and self._mentions_index(node.slice)
            ):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "setdefault", "pop")
                and node.args
                and self._is_index_expr(node.args[0])
            ):
                return True
            if (
                isinstance(node, ast.Compare)
                and self._is_index_expr(node.left)
                and isinstance(node.ops[0], (ast.In, ast.NotIn))
            ):
                return True
        return False

    def fallbacks(self) -> list[ast.expr]:
        """Values used in place of a missing index."""
        out = list(self.defaults)
        for node in self._walk():
            # x if <index> is not None else y  /  y if <index> is None else x
            if isinstance(node, ast.IfExp) and (cmp := self._none_compare(node.test)) is not None:
                out.append(node.orelse if cmp == "is_not" else node.body)
            # <index> or y
            if (
                isinstance(node, ast.BoolOp)
                and isinstance(node.op, ast.Or)
                and self._is_index_expr(node.values[0])
            ):
                out += node.values[1:]
            # if <index> is None: idx = y
            if isinstance(node, ast.If) and self._none_compare(node.test) == "is":
                out += [stmt.value for stmt in node.body if isinstance(stmt, ast.Assign)]
        return out

    def drops(self) -> bool:
        """``if <index> is not None:`` around the accumulation with no else, or
        ``if <index> is None: continue``."""
        for node in self._walk():
            if not isinstance(node, ast.If):
                continue
            cmp = self._none_compare(node.test)
            if cmp == "is_not" and not node.orelse:
                return True
            if cmp == "is" and node.body and isinstance(node.body[0], (ast.Continue, ast.Return)):
                return True
        return False

    def _none_compare(self, test: ast.expr) -> str | None:
        """Returns "is" for ``<index> is None`` (or ``== None``), "is_not" for the negation and for
        ``hasattr(tc, "index")`` / ``"index" in tc``, else None."""
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            return "is_not" if any(self._none_compare(v) == "is_not" for v in test.values) else None
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op, right = test.ops[0], test.comparators[0]
            if self._is_index_expr(test.left) and _is_none(right):
                if isinstance(op, (ast.Is, ast.Eq)):
                    return "is"
                if isinstance(op, (ast.IsNot, ast.NotEq)):
                    return "is_not"
            if _str_is(test.left, "index") and isinstance(op, ast.In):
                return "is_not"
        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and test.func.id == "hasattr"
            and len(test.args) == 2
            and _str_is(test.args[1], "index")
        ):
            return "is_not"
        return None

    def verdict(self) -> str | None:
        """Returns "none" / "drop" / "position" for a broken reducer, None when it copes (or isn't one)."""
        if not self.reads:
            return None
        if self.drops():
            return "drop"
        if not self.keyed():
            return None
        values = self.fallbacks()
        if not values:
            return "none"
        if all(isinstance(v, ast.Name) and v.id in self.counters for v in values):
            return "position"
        return None


def _local_functions(ctx: FileContext) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {n.name: n for n in ast.walk(ctx.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _scopes(loop: ast.For | ast.AsyncFor, functions) -> Iterator[_Scope]:
    """The loop body, plus local helpers it hands the delta to (``_accum_tc(tc, idx)``)."""
    target = loop.target
    _, enumerated = _unwrap(loop.iter)
    counters: set[str] = set()
    if enumerated and isinstance(target, ast.Tuple) and len(target.elts) == 2:
        counter, target = target.elts
        if isinstance(counter, ast.Name):
            counters.add(counter.id)
    if not isinstance(target, ast.Name):
        return
    var = target.id
    yield _Scope(loop.body, var, counters)
    for stmt in loop.body:
        for call in (n for n in ast.walk(stmt) if isinstance(n, ast.Call)):
            name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
            fn = functions.get(name) if name else None
            if fn is None:
                continue
            params = [a.arg for a in fn.args.posonlyargs + fn.args.args]
            if params and params[0] in ("self", "cls") and isinstance(call.func, ast.Attribute):
                params = params[1:]
            mapped_var, mapped_counters = None, set()
            for param, arg in zip(params, call.args):
                if isinstance(arg, ast.Name) and arg.id == var:
                    mapped_var = param
                elif isinstance(arg, ast.Name) and arg.id in counters:
                    mapped_counters.add(param)
            if mapped_var is not None:
                yield _Scope(fn.body, mapped_var, mapped_counters)


@register
class StreamToolCallIndex(Rule):
    id = "LDL010"
    name = "stream-tool-call-index"
    summary = "streamed tool calls merged by tool_calls[].index with no usable fallback (Gemini omits it)"
    # The LLPhant, neurolink, PraisonAI and nanobot fixes aren't public yet; add their links when they are.
    references = ("https://ai.google.dev/gemini-api/docs/openai",)

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        functions = _local_functions(ctx)
        for loop in ast.walk(ctx.tree):
            if not isinstance(loop, (ast.For, ast.AsyncFor)):
                continue
            iterable, _ = _unwrap(loop.iter)
            if not _is_delta_tool_calls(iterable):
                continue
            for scope in _scopes(loop, functions):
                verdict = scope.verdict()
                if verdict is not None:
                    yield self.finding(
                        loop,
                        ctx,
                        _MESSAGES[verdict].format(var=scope.var),
                        "When `index` is missing, start a new call if the delta has an `id` or "
                        "`function.name`, otherwise append to the latest call.",
                    )
                    break
                if scope.reads:
                    break  # this scope reads the index and copes with a missing one
