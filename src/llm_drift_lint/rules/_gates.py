"""Model gates: hand-written checks such as ``if model not in NO_TEMPERATURE_MODELS`` or
``model.startswith(("claude-opus-4-7", "claude-opus-5"))`` that decide what a request sends.

Rules evaluate a gate's test against the Claude ids in ``models.CLAUDE_IDS`` to see which models
reach each branch. Anything the evaluator doesn't understand is "unknown" for every model, and a
rule never fires on an unknown outcome.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .. import models
from ..core import FileContext

# A value for one model: True / False, or None when the test can't be decided statically.
Outcome = bool | None
Predicate = Callable[[str], Outcome]

_REGEX_CHARS = re.compile(r"[\\|()\[\]*+?^$]")


def _is_model_operand(node: ast.expr) -> bool:
    """True for an expression that reads like the model id: ``model``, ``self.model_name.lower()``,
    ``model_id.split("/")[-1]``. Other operands (``provider``, ``temperature``) make the test unknown."""
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Attribute):
            if "model" in node.attr.lower():
                return True
            node = node.value
        elif isinstance(node, ast.Name):
            return "model" in node.id.lower()
        else:
            return False


def _str(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower()
    return None


@dataclass
class GateFile:
    """Per-file state: the string collections bound to names, for ``model in NAME``."""

    ctx: FileContext

    def __post_init__(self) -> None:
        self.collections: dict[str, list[str]] = {}
        for node in ast.walk(self.ctx.tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            else:
                continue
            name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
            items = self.strings(value)
            if name and items is not None:
                self.collections[name] = items

    def strings(self, node: ast.expr | None) -> list[str] | None:
        """The strings of a list/tuple/set display (or ``frozenset(...)``/``set(...)`` of one), else None."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in (
                "frozenset",
                "set",
                "tuple",
                "list",
            )
        ):
            node = node.args[0] if len(node.args) == 1 else None
        if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return None
        out = [_str(e) for e in node.elts]
        if not out or any(s is None for s in out):
            return None
        return out  # type: ignore[return-value]

    def collection(self, node: ast.expr) -> list[str] | None:
        if isinstance(node, ast.Name):
            return self.collections.get(node.id)
        if isinstance(node, ast.Attribute):
            return self.collections.get(node.attr)
        return self.strings(node)

    def claude_strings(self, test: ast.expr) -> list[str]:
        """Every string in the test, and in the collections it names, that mentions a Claude model."""
        found = []
        for sub in ast.walk(test):
            items: list[str] = []
            if (s := _str(sub)) is not None:  # type: ignore[arg-type]
                items = [s]
            elif isinstance(sub, (ast.Name, ast.Attribute)):
                items = self.collection(sub) or []
            found += [
                s for s in items if "claude" in s or re.search(r"\b(opus|sonnet|haiku|fable|mythos)\b", s)
            ]
        return found

    # -- evaluation ---------------------------------------------------------------------------

    def predicate(self, test: ast.expr) -> Predicate:
        """``predicate(model_id)`` -> whether ``test`` is true when the model variable holds that id."""
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self.predicate(test.operand)
            return lambda m: None if (r := inner(m)) is None else not r
        if isinstance(test, ast.BoolOp):
            parts = [self.predicate(v) for v in test.values]
            if isinstance(test.op, ast.And):
                return lambda m: _all([p(m) for p in parts])
            return lambda m: _any([p(m) for p in parts])
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            return self._compare(test.left, test.ops[0], test.comparators[0])
        if isinstance(test, ast.Call):
            return self._call(test)
        return _unknown

    def _compare(self, left: ast.expr, op: ast.cmpop, right: ast.expr) -> Predicate:
        negate = isinstance(op, (ast.NotIn, ast.NotEq))
        base: Predicate = _unknown
        if isinstance(op, (ast.In, ast.NotIn)):
            needle = _str(left)
            if needle is not None and _is_model_operand(right):
                base = _matcher([needle], "substring")
            elif _is_model_operand(left) and (items := self.collection(right)) is not None:
                base = _matcher(items, "exact")
        elif isinstance(op, (ast.Eq, ast.NotEq)):
            for a, b in ((left, right), (right, left)):
                if _is_model_operand(a) and (s := _str(b)) is not None:
                    base = _matcher([s], "exact")
        if not negate:
            return base
        return lambda m: None if (r := base(m)) is None else not r

    def _call(self, call: ast.Call) -> Predicate:
        func = call.func
        # model.startswith("claude-opus-4-7") / model.startswith(PREFIXES)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "startswith"
            and _is_model_operand(func.value)
            and len(call.args) == 1
        ):
            arg = call.args[0]
            items = [s] if (s := _str(arg)) is not None else self.collection(arg)
            return _matcher(items, "prefix") if items is not None else _unknown
        # any(model.startswith(p) for p in PREFIXES) / any(p in model for p in NAMES)
        if (
            isinstance(func, ast.Name)
            and func.id == "any"
            and len(call.args) == 1
            and isinstance(gen := call.args[0], ast.GeneratorExp)
            and len(gen.generators) == 1
            and not gen.generators[0].ifs
            and isinstance(var := gen.generators[0].target, ast.Name)
        ):
            items = self.collection(gen.generators[0].iter)
            elt = gen.elt
            if items is None:
                return _unknown
            if (
                isinstance(elt, ast.Call)
                and isinstance(elt.func, ast.Attribute)
                and elt.func.attr == "startswith"
                and _is_model_operand(elt.func.value)
                and _names(elt.args, var.id)
            ):
                return _matcher(items, "prefix")
            if (
                isinstance(elt, ast.Compare)
                and len(elt.ops) == 1
                and isinstance(elt.ops[0], ast.In)
                and _names([elt.left], var.id)
                and _is_model_operand(elt.comparators[0])
            ):
                return _matcher(items, "substring")
        return _unknown


def _names(args: list[ast.expr], name: str) -> bool:
    return len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == name


def _unknown(model: str) -> Outcome:
    return None


def _all(values: list[Outcome]) -> Outcome:
    if any(v is False for v in values):
        return False
    return None if any(v is None for v in values) else True


def _any(values: list[Outcome]) -> Outcome:
    if any(v is True for v in values):
        return True
    return None if any(v is None for v in values) else False


def _matcher(items: list[str], mode: str) -> Predicate:
    if any(_REGEX_CHARS.search(s) for s in items):
        return _unknown
    if mode == "exact":
        return lambda m: m in items
    if mode == "prefix":
        return lambda m: any(m.startswith(s) for s in items)
    return lambda m: any(s in m for s in items)


@dataclass
class Branch:
    body: list[ast.stmt]
    reaches: Callable[[str], Outcome]  # whether a model id reaches this branch


def branches(gf: GateFile, node: ast.If | ast.IfExp) -> Iterator[Branch]:
    """The branches of an if / elif / else chain, each with the models that reach it.

    A chain without ``else`` gets an implicit else: empty, or the statements after the chain when
    every branch ends in ``return`` / ``raise`` (``if fixed(model): return {}`` then ``return {...}``)."""
    earlier: list[Predicate] = []
    exits = True
    cur: ast.AST | None = node
    while isinstance(cur, (ast.If, ast.IfExp)):
        pred = gf.predicate(cur.test)
        prior = list(earlier)
        body = cur.body if isinstance(cur, ast.If) else [ast.Expr(cur.body)]

        def reach(m: str, pred: Predicate = pred, prior: list[Predicate] = prior) -> Outcome:
            return _all([*(_not(p(m)) for p in prior), pred(m)])

        yield Branch(body, reach)
        exits = exits and isinstance(cur, ast.If) and isinstance(body[-1], (ast.Return, ast.Raise))
        earlier.append(pred)
        orelse = cur.orelse if isinstance(cur, ast.If) else [ast.Expr(cur.orelse)]
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            cur = orelse[0]
        elif len(orelse) == 1 and isinstance(orelse[0], ast.Expr) and isinstance(orelse[0].value, ast.IfExp):
            cur = orelse[0].value
        else:
            final = list(earlier)
            if not orelse and exits:
                orelse = _following(gf.ctx, node)
            yield Branch(orelse, lambda m, final=final: _all([_not(p(m)) for p in final]))
            cur = None


def _following(ctx: FileContext, node: ast.AST) -> list[ast.stmt]:
    """The statements after ``node`` in the block that contains it."""
    parent = ctx.parent(node)
    for field in ("body", "orelse", "finalbody"):
        block = getattr(parent, field, None)
        if isinstance(block, list) and node in block:
            return block[block.index(node) + 1 :]
    return []


def _not(v: Outcome) -> Outcome:
    return None if v is None else not v


def chain_heads(ctx: FileContext) -> Iterator[ast.If | ast.IfExp]:
    """Every if / conditional expression that starts a chain (an ``elif`` is part of its parent's chain)."""
    for node in ast.walk(ctx.tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            parent = ctx.parent(node)
            if isinstance(parent, ast.If) and parent.orelse == [node]:
                continue
            if isinstance(parent, ast.Expr) and isinstance(ctx.parent(parent), ast.If):
                grand = ctx.parent(parent)
                if isinstance(grand, ast.If) and grand.orelse == [parent]:
                    continue
            if isinstance(parent, ast.IfExp) and parent.orelse is node:
                continue
            yield node


def chain_tests(node: ast.If | ast.IfExp) -> list[ast.expr]:
    tests = []
    cur: ast.AST | None = node
    while isinstance(cur, (ast.If, ast.IfExp)):
        tests.append(cur.test)
        orelse = cur.orelse
        if isinstance(orelse, list):
            cur = orelse[0] if len(orelse) == 1 and isinstance(orelse[0], ast.If) else None
        else:
            cur = orelse if isinstance(orelse, ast.IfExp) else None
    return tests


# -- what a branch sends ------------------------------------------------------------------------


def sets_field(stmts: list[ast.stmt], names: tuple[str, ...]) -> ast.AST | None:
    """The first node in ``stmts`` that puts one of ``names`` into a request with a value other than
    None: a keyword argument, a dict key, ``params["name"] = v`` or ``obj.name = v``."""
    for stmt in stmts:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.keyword) and sub.arg in names and not _none(sub.value):
                return sub.value
            if isinstance(sub, ast.Dict):
                for k, v in zip(sub.keys, sub.values, strict=True):
                    if k is not None and _str(k) in names and not _none(v):
                        return k
            if isinstance(sub, ast.Assign) and not _none(sub.value):
                for t in sub.targets:
                    if isinstance(t, ast.Subscript) and _str(t.slice) in names:
                        return t
    return None


def mentions(stmts: list[ast.stmt], words: tuple[str, ...]) -> bool:
    """True when any string, name, attribute or keyword in ``stmts`` is one of ``words``."""
    for stmt in stmts:
        for sub in ast.walk(stmt):
            if (
                (_str(sub) in words if isinstance(sub, ast.Constant) else False)
                or (isinstance(sub, ast.Name) and sub.id in words)
                or (isinstance(sub, ast.Attribute) and sub.attr in words)
                or (isinstance(sub, ast.keyword) and sub.arg in words)
            ):
                return True
    return False


def _none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def claude_models(trait: Callable[[models.ModelTraits], bool]) -> list[str]:
    """The ids in ``models.CLAUDE_IDS`` whose traits satisfy ``trait``."""
    out = []
    for m in models.CLAUDE_IDS:
        t = models.traits(m)
        if t is not None and trait(t):
            out.append(m)
    return out
