"""LDL010: a streamed OpenAI-style tool-call reducer that keys calls only by ``tool_calls[].index``."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register


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


def _is_delta_tool_calls(node: ast.expr) -> bool:
    if isinstance(node, ast.BoolOp):  # delta.tool_calls or []
        node = node.values[0]
    names = _chain(node)
    return bool(names) and names[-1] == "tool_calls" and any("delta" in n for n in names[:-1])


def _index_reads(body: list[ast.stmt], var: str) -> list[ast.expr]:
    """``var.index`` / ``var["index"]`` / ``var.get("index")`` reads in the loop body."""
    found = []
    for stmt in body:
        for sub in ast.walk(stmt):
            if (
                (
                    isinstance(sub, ast.Attribute)
                    and sub.attr == "index"
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == var
                )
                or (
                    isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == var
                    and isinstance(sub.slice, ast.Constant)
                    and sub.slice.value == "index"
                )
                or (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "get"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == var
                    and len(sub.args) == 1
                    and isinstance(sub.args[0], ast.Constant)
                    and sub.args[0].value == "index"
                )
            ):
                found.append(sub)
    return found


def _handles_missing_index(ctx: FileContext, fn: ast.AST, var: str, reads: list[ast.expr]) -> bool:
    """True when the code visibly handles a delta without an index."""
    for sub in ast.walk(fn):
        # getattr(tc, "index", None) / tc.get("index", default)
        if isinstance(sub, ast.Call) and len(sub.args) >= 2:
            if isinstance(sub.func, ast.Name) and sub.func.id == "getattr":
                key = sub.args[1]
                if isinstance(key, ast.Constant) and key.value == "index":
                    return True
            if isinstance(sub.func, ast.Attribute) and sub.func.attr == "get":
                key = sub.args[0]
                if isinstance(key, ast.Constant) and key.value == "index":
                    return True
        # index is None / is not None / == None / "index" in tc / hasattr(tc, "index")
        if isinstance(sub, ast.Compare):
            for op, right in zip(sub.ops, sub.comparators):
                if isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)) and (
                    isinstance(right, ast.Constant) and right.value is None
                ):
                    return True
                if isinstance(op, (ast.In, ast.NotIn)) and (
                    isinstance(sub.left, ast.Constant) and sub.left.value == "index"
                ):
                    return True
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "hasattr"
            and len(sub.args) == 2
            and isinstance(sub.args[1], ast.Constant)
            and sub.args[1].value == "index"
        ):
            return True
    for read in reads:
        parent = ctx.parent(read)
        # tc.index or len(calls) / tc.index if ... else ...
        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or):
            return True
    return False


def _used_as_key(ctx: FileContext, read: ast.expr) -> bool:
    """True when the index picks the accumulated call: ``calls[tc.index]``, ``calls.get(tc.index)``,
    ``calls.setdefault(tc.index, ...)``, ``idx = tc.index`` later used as a key, ``tc.index in calls``."""
    parent = ctx.parent(read)
    if isinstance(parent, ast.Subscript) and parent.slice is read:
        return True
    if isinstance(parent, ast.Call) and read in parent.args and isinstance(parent.func, ast.Attribute):
        return parent.func.attr in ("get", "setdefault", "pop")
    if (
        isinstance(parent, ast.Compare)
        and parent.left is read
        and isinstance(parent.ops[0], (ast.In, ast.NotIn))
    ):
        return True
    # Bound to a name (``index = tc.index``): the reducer keys by it in practice (checked by the FP scan).
    return isinstance(parent, (ast.Assign, ast.AnnAssign))


@register
class StreamToolCallIndex(Rule):
    id = "LDL010"
    name = "stream-tool-call-index"
    summary = "streamed tool calls merged only by tool_calls[].index, which some servers (Gemini) omit"
    # The LLPhant and neurolink fixes aren't public yet; add their PR links when they are.
    references = ("https://ai.google.dev/gemini-api/docs/openai",)

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        for loop in ast.walk(ctx.tree):
            if not isinstance(loop, (ast.For, ast.AsyncFor)) or not isinstance(loop.target, ast.Name):
                continue
            if not _is_delta_tool_calls(loop.iter):
                continue
            var = loop.target.id
            reads = [r for r in _index_reads(loop.body, var) if _used_as_key(ctx, r)]
            if not reads:
                continue
            scope = ctx.enclosing_function(loop) or loop
            if _handles_missing_index(ctx, scope, var, reads):
                continue
            yield self.finding(
                loop,
                ctx,
                f"Streamed tool calls are merged by `{var}.index` alone; Gemini's OpenAI-compatible "
                "endpoint sends each parallel call in its own delta without `index`, so the calls "
                "collapse into one (or the lookup fails)",
                "When `index` is missing, start a new call if the delta has an `id` or `function.name`, "
                "otherwise append to the latest call.",
            )
