"""Core types shared by every rule. This module is the contract between the engine and the rules:
change it only through the lead session (see COORDINATION.md)."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .project import ProjectInfo, project_for


@dataclass(frozen=True)
class Finding:
    rule_id: str  # e.g. "LDL001"
    path: Path
    line: int
    col: int
    message: str  # what is wrong, one sentence
    hint: str = ""  # how to fix it, one sentence
    references: tuple[str, ...] = ()  # docs / real PRs that motivated the rule


@dataclass
class CallSite:
    """A call such as ``client.messages.create(model=..., temperature=0)``."""

    node: ast.Call
    # Dotted attribute chain of the callee without its root variable:
    # ``client.messages.create(...)`` -> "messages.create",
    # ``self.client.beta.messages.stream(...)`` -> "client.beta.messages.stream",
    # ``anthropic.Anthropic().messages.create(...)`` -> "messages.create".
    # Plain function calls keep their name: ``completion(...)`` -> "completion".
    # Match with ``ctx.calls(*suffixes)``, which compares suffixes, rather than on the full target.
    target: str
    kwargs: dict[str, ast.expr]
    ctx: FileContext

    @property
    def line(self) -> int:
        return self.node.lineno

    @property
    def col(self) -> int:
        return self.node.col_offset

    @property
    def function(self) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """The function or method the call is in (None at module level)."""
        return self.ctx.enclosing_function(self.node)

    def kwarg_literal(self, name: str) -> object | None:
        """The keyword argument's value if it can be resolved statically, else None.

        Resolves literals, then names: a local assignment before the call in the same function, the
        enclosing function's parameter default, then a module-level constant."""
        node = self.kwargs.get(name)
        return None if node is None else self.ctx.literal(node, scope=self.node)

    def model(self) -> str | None:
        """The ``model=`` value when it can be resolved statically."""
        value = self.kwarg_literal("model")
        return value if isinstance(value, str) else None


# Callee suffixes that identify each SDK's request methods. Rules match on these.
ANTHROPIC_MESSAGE_CALLS = ("messages.create", "messages.stream", "messages.parse", "messages.count_tokens")
OPENAI_CHAT_CALLS = ("chat.completions.create", "chat.completions.parse", "chat.completions.stream")
OPENAI_RESPONSES_CALLS = ("responses.create", "responses.parse", "responses.stream")


@dataclass
class FileContext:
    path: Path
    source: str
    tree: ast.Module
    _constants: dict[str, object] = field(default_factory=dict)
    _parents: dict[ast.AST, ast.AST] = field(default_factory=dict)

    @classmethod
    def parse(cls, source: str, path: Path = Path("<string>")) -> FileContext | None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        ctx = cls(path, source, tree)
        ctx._collect_constants()
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                ctx._parents[child] = parent
        return ctx

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self._parents.get(node)

    def enclosing_function(self, node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        cur = self._parents.get(node)
        while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cur = self._parents.get(cur)
        return cur

    def _collect_constants(self) -> None:
        # NAME = "literal" anywhere in the file. Last assignment wins; good enough for model ids.
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    self._constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError, TypeError):
                    pass
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                try:
                    self._constants[node.target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError, TypeError):
                    pass

    @property
    def project(self) -> ProjectInfo:
        """Dependencies declared or locked by the project this file belongs to."""
        if self.path == Path("<string>"):
            return ProjectInfo(None)
        return project_for(self.path)

    def min_version(self, package: str) -> tuple[int, ...] | None:
        """Lowest version of ``package`` the project runs with (locked version, else declared lower bound).
        None when the project doesn't say."""
        return self.project.min_version(package)

    def literal(self, node: ast.expr, scope: ast.AST | None = None) -> object | None:
        """Statically resolve ``node``. With ``scope`` (a node inside a function), names are looked up
        in that function first: the last simple assignment before ``scope``, then a parameter default."""
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            pass
        if not isinstance(node, ast.Name):
            return None
        fn = self.enclosing_function(scope) if scope is not None else None
        if fn is not None:
            found, value = _local_value(fn, node.id, scope)
            if found:
                return value
        return self._constants.get(node.id)

    def _expanded_kwargs(self, call: ast.Call) -> dict[str, ast.expr]:
        """Keys of ``**mapping`` expansions that can be resolved: a dict display, or a name bound to a
        dict display (plus ``name["key"] = value`` assignments) earlier in the same function."""
        out: dict[str, ast.expr] = {}
        for kw in call.keywords:
            if kw.arg is not None:
                continue
            value = kw.value
            if isinstance(value, ast.Name):
                fn = self.enclosing_function(call)
                value = _local_dict(fn, value.id, call) if fn is not None else None
            if isinstance(value, ast.Dict):
                for k, v in zip(value.keys, value.values, strict=True):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        out[k.value] = v
        return out

    def calls(self, *suffixes: str) -> Iterator[CallSite]:
        """Every call whose target ends with one of ``suffixes`` (all calls if none given), in source order."""
        nodes = sorted(
            (n for n in ast.walk(self.tree) if isinstance(n, ast.Call)),
            key=lambda n: (n.lineno, n.col_offset),
        )
        for node in nodes:
            target = _dotted_target(node.func)
            if target is None:
                continue
            if suffixes and not any(target == s or target.endswith("." + s) for s in suffixes):
                continue
            # Explicit keywords win over keys found in ** expansions, as at runtime for later keys.
            kwargs = self._expanded_kwargs(node)
            kwargs.update({k.arg: k.value for k in node.keywords if k.arg is not None})
            yield CallSite(node, target, kwargs, self)


def _before(node: ast.AST, limit: ast.AST) -> bool:
    return (node.lineno, node.col_offset) < (limit.lineno, limit.col_offset)  # type: ignore[attr-defined]


def _local_value(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, name: str, limit: ast.AST
) -> tuple[bool, object]:
    """(found, value) for ``name`` inside ``fn`` before ``limit``. A non-literal assignment counts as
    found-but-unknown, so a module constant with the same name doesn't leak in."""
    found, value = False, None
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in [
        *zip(positional, defaults, strict=True),
        *zip(args.kwonlyargs, args.kw_defaults, strict=True),
    ]:
        if arg.arg == name:
            found = True
            try:
                value = ast.literal_eval(default) if default is not None else None
            except (ValueError, SyntaxError, TypeError):
                value = None
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not _before(node, limit):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
            found = True
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                value = None
    return found, value


def _local_dict(fn: ast.FunctionDef | ast.AsyncFunctionDef, name: str, limit: ast.AST) -> ast.Dict | None:
    """The dict display bound to ``name`` before ``limit``, merged with later ``name["k"] = v``
    assignments (also before ``limit``). None when the binding isn't a plain dict display."""
    base: ast.Dict | None = None
    extra: list[tuple[ast.expr, ast.expr]] = []
    for node in sorted(
        (n for n in ast.walk(fn) if isinstance(n, ast.Assign) and _before(n, limit)),
        key=lambda n: (n.lineno, n.col_offset),
    ):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(node.value, ast.Dict):
                    base, extra = node.value, []
                elif (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "dict"
                    and not node.value.args
                ):
                    base = ast.Dict(
                        keys=[ast.Constant(k.arg) for k in node.value.keywords if k.arg],
                        values=[k.value for k in node.value.keywords if k.arg],
                    )
                    extra = []
                else:
                    base, extra = None, []
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
                and isinstance(target.slice, ast.Constant)
            ):
                extra.append((target.slice, node.value))
    if base is None:
        return None
    return ast.Dict(keys=[*base.keys, *(k for k, _ in extra)], values=[*base.values, *(v for _, v in extra)])


def _dotted_target(func: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        if not parts:
            return func.id  # plain function call
        # Drop the receiver variable (client, self, anthropic...) but keep the method chain.
        return ".".join(reversed(parts))
    if isinstance(func, ast.Call) and parts:
        return ".".join(reversed(parts))  # e.g. Anthropic().messages.create
    return None


class Rule:
    """Base class. Subclasses set the class attributes and implement ``check``."""

    id: str = ""  # "LDL001"...
    name: str = ""  # kebab-case, e.g. "anthropic-sampling-params"
    summary: str = ""  # one line for --list-rules
    references: tuple[str, ...] = ()

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        raise NotImplementedError

    def finding(self, site: CallSite | ast.AST, ctx: FileContext, message: str, hint: str = "") -> Finding:
        line = site.line if isinstance(site, CallSite) else getattr(site, "lineno", 1)
        col = site.col if isinstance(site, CallSite) else getattr(site, "col_offset", 0)
        return Finding(self.id, ctx.path, line, col, message, hint, self.references)


_REGISTRY: dict[str, type[Rule]] = {}


def register(rule_cls: type[Rule]) -> type[Rule]:
    if not rule_cls.id or rule_cls.id in _REGISTRY:
        raise ValueError(f"rule id missing or duplicated: {rule_cls.id!r}")
    _REGISTRY[rule_cls.id] = rule_cls
    return rule_cls


def all_rules() -> list[Rule]:
    from . import rules  # noqa: F401  (imports every rule module, which registers it)

    return [cls() for _, cls in sorted(_REGISTRY.items())]
