"""LDL007: an OpenAI model that the API is shutting down, used as a default model."""

from __future__ import annotations

import ast
import datetime
import re
from collections.abc import Iterator

from .. import models
from ..core import FileContext, Finding, Rule, register

# Only defaults count: a parameter or field default, a ``*MODEL*`` constant, an env-var or CLI
# fallback. A model passed at one call site (examples, tests, token counters) isn't a default.
_TEST_FILE = re.compile(r"^(test_.*|.*_test|conftest)\.py$")


def today() -> datetime.date:
    """The date shutdowns are compared with. Tests replace it."""
    return datetime.date.today()


def _is_model_name(name: str | None) -> bool:
    return bool(name) and "model" in name.lower() and "embed" not in name.lower()


@register
class OpenAIShutdownDefault(Rule):
    id = "LDL007"
    name = "openai-shutdown-default"
    summary = "default model is an OpenAI model that the API has shut down or will shut down (models.OPENAI_SHUTDOWNS)"
    references = (models.OPENAI_DEPRECATIONS,)

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        if _TEST_FILE.search(ctx.path.name):
            return
        seen: set[int] = set()
        for node, how in _defaults(ctx):
            model = node.value if isinstance(node, ast.Constant) else None
            shutdown = models.openai_shutdown(model) if isinstance(model, str) else None
            if shutdown is None or id(node) in seen or _azure(ctx, node):
                continue
            seen.add(id(node))
            if datetime.date.fromisoformat(shutdown.date) <= today():
                when = f"OpenAI shut it down on {shutdown.date}, so requests with it fail"
            else:
                when = f"OpenAI shuts it down on {shutdown.date}"
            yield self.finding(
                node,
                ctx,
                f"`{model}` is the {how}, and {when}",
                f"Use {shutdown.replacement} (OpenAI's replacement); check which parameters it accepts "
                "before swapping (sampling, max_completion_tokens).",
            )


def _defaults(ctx: FileContext) -> Iterator[tuple[ast.expr, str]]:
    for node in ast.walk(ctx.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            pairs = list(zip(positional[len(positional) - len(args.defaults) :], args.defaults))
            pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
            for arg, default in pairs:
                if _is_model_name(arg.arg):
                    yield default, f"default of `{arg.arg}`"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id if isinstance(t, ast.Name) else getattr(t, "attr", None) for t in targets]
            name = next((n for n in names if _is_model_name(n)), None)
            if name is None or node.value is None:
                continue
            for value in _default_values(node.value):
                yield value, f"value of `{name}`"
        elif isinstance(node, ast.Call):
            target = _callee(node)
            if target.endswith(("getenv", "environ.get")) and len(node.args) == 2:
                env = node.args[0]
                if (
                    isinstance(env, ast.Constant)
                    and isinstance(env.value, str)
                    and "MODEL" in env.value.upper()
                ):
                    yield node.args[1], f"fallback for `${env.value}`"
            if target.endswith("add_argument") and node.args:
                flag = node.args[0]
                if (
                    isinstance(flag, ast.Constant)
                    and isinstance(flag.value, str)
                    and _is_model_name(flag.value)
                ):
                    for kw in node.keywords:
                        if kw.arg == "default":
                            yield kw.value, f"default of `{flag.value}`"


def _default_values(value: ast.expr) -> Iterator[ast.expr]:
    """A literal, or the fallback inside ``os.getenv(..., "x")`` / ``Field(default="x")`` / ``a or "x"``."""
    if isinstance(value, ast.Constant):
        yield value
    elif isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
        yield value.values[-1]
    elif isinstance(value, ast.Call):
        target = _callee(value)
        if target.endswith(("getenv", "environ.get", ".get")) and len(value.args) == 2:
            yield value.args[1]
        for kw in value.keywords:
            if kw.arg == "default":
                yield kw.value
        if target.endswith("Field") and value.args:
            yield value.args[0]


def _callee(call: ast.Call) -> str:
    parts = []
    func = call.func
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _azure(ctx: FileContext, node: ast.AST) -> bool:
    """Azure OpenAI deployments have their own retirement dates; skip code that targets Azure."""
    cur = ctx.parent(node)
    while cur is not None:
        if isinstance(cur, (ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(ctx.source, cur) or ""
            if "azure" in segment.lower():
                return True
            if isinstance(cur, ast.ClassDef):
                break
        cur = ctx.parent(cur)
    return False
