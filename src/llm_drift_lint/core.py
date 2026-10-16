"""Core types shared by every rule. This module is the contract between the engine and the rules:
change it only through the lead session (see COORDINATION.md)."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


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
    # Dotted attribute chain of the callee, without the receiver variable:
    # ``self.client.messages.create(...)`` -> "messages.create",
    # ``anthropic.Anthropic().beta.messages.stream(...)`` -> "beta.messages.stream".
    # Plain function calls keep their name: ``completion(...)`` -> "completion".
    target: str
    kwargs: dict[str, ast.expr]
    ctx: FileContext

    @property
    def line(self) -> int:
        return self.node.lineno

    @property
    def col(self) -> int:
        return self.node.col_offset

    def kwarg_literal(self, name: str) -> object | None:
        """The keyword argument's value if it is a literal (or a module/function constant), else None."""
        node = self.kwargs.get(name)
        return None if node is None else self.ctx.literal(node)

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

    @classmethod
    def parse(cls, source: str, path: Path = Path("<string>")) -> FileContext | None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        ctx = cls(path, source, tree)
        ctx._collect_constants()
        return ctx

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

    def literal(self, node: ast.expr) -> object | None:
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            pass
        if isinstance(node, ast.Name):
            return self._constants.get(node.id)
        return None

    def calls(self, *suffixes: str) -> Iterator[CallSite]:
        """Every call whose target ends with one of ``suffixes`` (all calls if none given)."""
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            target = _dotted_target(node.func)
            if target is None:
                continue
            if suffixes and not any(target == s or target.endswith("." + s) for s in suffixes):
                continue
            kwargs = {k.arg: k.value for k in node.keywords if k.arg is not None}
            yield CallSite(node, target, kwargs, self)


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
