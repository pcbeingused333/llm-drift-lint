"""LDL011: an OpenAI→Anthropic converter that maps finish_reason "stop" to "end_turn" whatever the content.

Gemini's OpenAI-compatible endpoint ends a turn with tool calls with ``finish_reason: "stop"``, and
native Gemini says ``STOP``. A gateway that maps "stop" to Anthropic's ``end_turn`` without looking at
the content tells Anthropic clients the turn is over: the official SDK tool runners then run no tools.
Converters that set ``tool_use`` whenever the message has tool calls are fine.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register

# finish_reason values that already mean "tool calls"; a test comparing against one is a finish-reason
# lookup, not a check of the content.
_FINISH_VALUES = frozenset({"stop", "tool_calls", "function_call"})


def _is_stop(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.lower() == "stop"


def _is_end_turn(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "end_turn"


def _yields_end_turn(nodes: list[ast.stmt] | ast.expr) -> bool:
    roots = nodes if isinstance(nodes, list) else [nodes]
    return any(_is_end_turn(n) for root in roots for n in ast.walk(root))


def _tests_stop(test: ast.expr) -> bool:
    """``fr == "stop"``, ``fr in ("stop", ...)``, ``fr.lower() == "stop"``."""
    for node in ast.walk(test):
        if isinstance(node, ast.Compare):
            for side in [node.left, *node.comparators]:
                if _is_stop(side):
                    return True
                if isinstance(side, (ast.Tuple, ast.List, ast.Set)) and any(_is_stop(e) for e in side.elts):
                    return True
    return False


def _mentions_tool(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        text = None
        if isinstance(sub, ast.Name):
            text = sub.id
        elif isinstance(sub, ast.Attribute):
            text = sub.attr
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            text = sub.value
        elif isinstance(sub, ast.arg):
            text = sub.arg
        if text and "tool" in text.lower():
            return True
    return False


def _compares_finish_value(test: ast.expr) -> bool:
    for node in ast.walk(test):
        if isinstance(node, ast.Compare):
            for side in [node.left, *node.comparators]:
                elts = side.elts if isinstance(side, (ast.Tuple, ast.List, ast.Set)) else [side]
                if any(isinstance(e, ast.Constant) and e.value in _FINISH_VALUES for e in elts):
                    return True
    return False


def _has_tool_override(scope: ast.AST) -> bool:
    """True when ``scope`` sets ``tool_use`` from the content: an ``if`` / conditional expression whose
    test looks at tool calls (not at the finish reason) and whose branch yields "tool_use"."""
    for node in ast.walk(scope):
        if isinstance(node, ast.If):
            branch: list[ast.AST] = list(node.body)
        elif isinstance(node, ast.IfExp):
            branch = [node.body]
        else:
            continue
        test = node.test
        parts = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or) else [test]
        if not any(_mentions_tool(p) and not _compares_finish_value(p) for p in parts):
            continue
        if any(_sets_tool_use(b) for b in branch):
            return True
    return False


def _target_name(target: ast.expr) -> str:
    if isinstance(target, ast.Name):
        return target.id.lower()
    if isinstance(target, ast.Attribute):
        return target.attr.lower()
    return ""


def _sets_tool_use(root: ast.AST) -> bool:
    """A "tool_use" that is a stop reason: not a block type (``{"type": "tool_use"}``,
    ``ToolUseBlock(type="tool_use")``, ``state.block_type = "tool_use"``) and not a comparison."""
    skip: set[int] = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Dict):
            skip |= {
                id(v)
                for k, v in zip(node.keys, node.values)
                if isinstance(k, ast.Constant) and k.value == "type"
            }
        elif isinstance(node, ast.keyword) and node.arg == "type":
            skip.add(id(node.value))
        elif isinstance(node, ast.Assign) and any("type" in _target_name(t) for t in node.targets):
            skip.add(id(node.value))  # block_type = "tool_use"
        elif isinstance(node, ast.Compare):
            skip |= {id(n) for n in ast.walk(node)}
    return any(
        isinstance(n, ast.Constant) and n.value == "tool_use" and id(n) not in skip for n in ast.walk(root)
    )


def _uses_of(ctx: FileContext, name: str) -> list[ast.AST]:
    """Functions that read module- or class-level ``name`` (as a name or an attribute)."""
    out = []
    for fn in ast.walk(ctx.tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)) or (
                isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.ctx, ast.Load)
            ):
                out.append(fn)
                break
    return out


def _assigned_name(ctx: FileContext, node: ast.AST) -> str | None:
    parent = ctx.parent(node)
    if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
        target = parent.targets[0]
    elif isinstance(parent, ast.AnnAssign):
        target = parent.target
    else:
        return None
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


@register
class StopToEndTurn(Rule):
    id = "LDL011"
    name = "stop-to-end-turn"
    summary = 'finish_reason "stop" mapped to Anthropic "end_turn" with no tool-call check (Gemini ends tool turns with "stop")'
    references = (
        "https://ai.google.dev/gemini-api/docs/openai",
        "https://platform.claude.com/docs/en/api/handling-stop-reasons",
    )

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        for node, scopes in self._mappings(ctx):
            if not scopes or any(_has_tool_override(s) for s in scopes):
                continue
            yield self.finding(
                node,
                ctx,
                'finish reason "stop" is mapped to `stop_reason` "end_turn" without checking for tool calls; '
                'Gemini (OpenAI-compatible "stop", native "STOP") ends turns that call tools this way, so '
                "Anthropic clients and the SDK tool runners see end_turn and run no tools",
                'Set `stop_reason` to "tool_use" whenever the converted message has tool_use blocks, '
                "whatever the finish reason.",
            )

    def _mappings(self, ctx: FileContext) -> Iterator[tuple[ast.AST, list[ast.AST]]]:
        """(mapping node, scopes where its result is used)."""
        for node, scopes in self._raw_mappings(ctx):
            # A helper such as ``_finish_to_stop_reason(fr)``: its callers may fix the result up.
            callers = [
                c
                for fn in scopes
                if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
                for c in _uses_of(ctx, fn.name)
            ]
            yield node, scopes + [c for c in callers if c not in scopes]

    def _raw_mappings(self, ctx: FileContext) -> Iterator[tuple[ast.AST, list[ast.AST]]]:
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Dict):
                if not any(_is_stop(k) and _is_end_turn(v) for k, v in zip(node.keys, node.values)):
                    continue
                fn = ctx.enclosing_function(node)
                if fn is not None:
                    yield node, [fn]
                elif (name := _assigned_name(ctx, node)) is not None:
                    yield node, _uses_of(ctx, name)
            elif isinstance(node, (ast.If, ast.IfExp)):
                if _tests_stop(node.test) and _yields_end_turn(node.body):
                    fn = ctx.enclosing_function(node)
                    yield node, [fn] if fn is not None else []
            elif isinstance(node, ast.match_case):
                pattern = node.pattern
                values = pattern.patterns if isinstance(pattern, ast.MatchOr) else [pattern]
                if any(
                    isinstance(p, ast.MatchValue) and _is_stop(p.value) for p in values
                ) and _yields_end_turn(node.body):
                    fn = ctx.enclosing_function(node.body[0])
                    yield node.pattern, [fn] if fn is not None else []
