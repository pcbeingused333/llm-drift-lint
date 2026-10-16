"""LDL003: reading the text from the first content block of a response that may start with thinking."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..core import CallSite, FileContext, Finding, Rule, register
from ._common import dict_field, dict_str, has_splat, site_traits

# Calls that return a complete Message. Streams are left out: their final message comes from a helper.
_RESPONSE_CALLS = ("messages.create", "messages.parse")
# Methods that turn a Message into a dict, for the ``["content"][0]["text"]`` form.
_TO_DICT = frozenset({"model_dump", "to_dict"})
_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


@register
class ContentFirstBlockText(Rule):
    id = "LDL003"
    name = "content-first-block-text"
    summary = "response.content[0].text on a Claude response whose first block can be a thinking block"
    references = (
        "https://platform.claude.com/docs/en/build-with-claude/extended-thinking",
        "https://github.com/allenai/reward-bench/pull/290",
        "https://github.com/AllYourBot/hostedgpt/pull/823",
    )

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        sites = {id(site.node): site for site in ctx.calls(*_RESPONSE_CALLS)}
        if not sites:
            return
        findings = []
        for scope in ast.walk(ctx.tree):
            if isinstance(scope, _SCOPES):
                findings.extend(self._check_scope(ctx, scope, sites))
        yield from sorted(findings, key=lambda f: (f.line, f.col))

    def _check_scope(self, ctx: FileContext, scope: ast.AST, sites: dict[int, CallSite]) -> Iterator[Finding]:
        nodes = list(_scope_nodes(scope))
        # Assignment targets ``X = <request>`` / ``X = <request>.model_dump()`` -> the producer.
        produced: dict[int, tuple[CallSite, bool]] = {}
        for node in nodes:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                producer = _producer(node.value, sites)
                if producer is not None:
                    produced[id(node.targets[0])] = producer
        # name -> every binding of it in this scope: (line, col, producer or None).
        bindings: dict[str, list[tuple[int, int, tuple[CallSite, bool] | None]]] = {}
        for node in nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bindings.setdefault(node.id, []).append(
                    (node.lineno, node.col_offset, produced.get(id(node)))
                )

        guarded = _guarded_names(nodes)
        for node in nodes:
            match = _first_block_text(node)
            if match is None:
                continue
            base, as_dict = match
            if isinstance(base, ast.Name):
                producer = None if base.id in guarded else _latest_binding(bindings.get(base.id, []), base)
            else:
                producer = _producer(base, sites)
            # A Message has no ["content"], and a dict from model_dump() has no .content.
            if producer is None or producer[1] != as_dict:
                continue
            reason = _thinking_reason(producer[0])
            if reason is None:
                continue
            yield self.finding(
                node,
                ctx,
                f"`{ast.unparse(node)}` assumes the first content block is text, but {reason}, so block 0 "
                "can be a thinking block, which has no text",
                'Read the first block whose type is "text", e.g. '
                'next(b.text for b in response.content if b.type == "text").',
            )


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Nodes that belong to ``scope`` itself, not to functions, lambdas or classes nested in it."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (*_SCOPES, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def _unwrap(node: ast.expr) -> ast.expr:
    return node.value if isinstance(node, ast.Await) else node


def _producer(node: ast.expr, sites: dict[int, CallSite]) -> tuple[CallSite, bool] | None:
    """The request call ``node`` is the result of, and whether it went through model_dump()/to_dict()."""
    node = _unwrap(node)
    via_dict = False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _TO_DICT:
        node, via_dict = _unwrap(node.func.value), True
    site = sites.get(id(node)) if isinstance(node, ast.Call) else None
    return None if site is None else (site, via_dict)


def _latest_binding(bindings: list, use: ast.Name) -> tuple[CallSite, bool] | None:
    before = [b for b in bindings if (b[0], b[1]) < (use.lineno, use.col_offset)]
    return max(before, key=lambda b: (b[0], b[1]))[2] if before else None


def _first_block_text(node: ast.AST) -> tuple[ast.expr, bool] | None:
    """For ``X.content[0].text`` -> (X, False); for ``X["content"][0]["text"]`` -> (X, True)."""
    if isinstance(node, ast.Attribute) and node.attr == "text":
        block = node.value
        if (
            isinstance(block, ast.Subscript)
            and _is_const(block.slice, 0)
            and isinstance(block.value, ast.Attribute)
            and block.value.attr == "content"
        ):
            return block.value.value, False
    if isinstance(node, ast.Subscript) and _is_const(node.slice, "text"):
        block = node.value
        if (
            isinstance(block, ast.Subscript)
            and _is_const(block.slice, 0)
            and isinstance(block.value, ast.Subscript)
            and _is_const(block.value.slice, "content")
        ):
            return block.value.value, True
    return None


def _is_const(node: ast.expr, value: object) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is type(value) and node.value == value


# Builtins that take ``X.content`` without looking for its text block.
_NON_INSPECTING = frozenset(
    {"str", "repr", "print", "len", "bool", "type", "hasattr", "isinstance", "getattr", "cast"}
)


def _guarded_names(nodes: list[ast.AST]) -> set[str]:
    """Names whose content blocks are looked at beyond block 0: iterated, indexed with something other
    than 0, or passed on to a function. Checking block 0 alone (``.type``, hasattr, isinstance) doesn't
    count: reading block 0 afterwards still misses the text when a thinking block comes first."""
    parents = {id(child): node for node in nodes for child in ast.iter_child_nodes(node)}
    guarded = set()
    for node in nodes:
        is_content = (isinstance(node, ast.Attribute) and node.attr == "content") or (
            isinstance(node, ast.Subscript) and _is_const(node.slice, "content")
        )
        if not (is_content and isinstance(node.value, ast.Name)):
            continue
        parent = parents.get(id(node))
        if (
            (isinstance(parent, ast.Subscript) and parent.value is node and not _is_const(parent.slice, 0))
            or (isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is node)
            or (
                isinstance(parent, ast.Call)
                and parent.func is not node
                and not (isinstance(parent.func, ast.Name) and parent.func.id in _NON_INSPECTING)
            )
            or (isinstance(parent, ast.keyword) and parent.value is node)
        ):
            guarded.add(node.value.id)
    return guarded


def _thinking_reason(site: CallSite) -> str | None:
    """Why the request's response can start with a thinking block, or None if it can't be told."""
    thinking = site.kwargs.get("thinking")
    if thinking is None:
        thinking = dict_field(site.kwargs.get("extra_body"), "thinking")
    if thinking is not None:
        kind = dict_str(thinking, "type")
        return (
            f'the request enables thinking ({{"type": "{kind}"}})'
            if kind in ("enabled", "adaptive")
            else None
        )
    if has_splat(site) or "extra_body" in site.kwargs and not isinstance(site.kwargs["extra_body"], ast.Dict):
        return None  # thinking may be set where we can't see it
    traits = site_traits(site)
    if traits is not None and traits.thinks_by_default:
        return f"{site.model()} thinks by default when the request has no `thinking` field"
    return None
