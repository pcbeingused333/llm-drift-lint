"""LDL008: a hand-written Claude model gate that stops sending sampling parameters to some of the
models that reject them, but not to all of them."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register
from . import _gates
from ._common import SAMPLING_PARAMS

# Names of collections that list the models a request must not send sampling parameters to:
# NO_SUPPORT_TEMPERATURE_MODELS, samplingParamsDisallowedPrefixes, FIXED_TEMPERATURE_MODELS...
_REJECT_LIST_NAME = re.compile(
    r"(^|_)(no|not|non|un|without|reject|rejects|fixed|skip|omit|drop|deny|disallow(ed)?|unsupported)_"
    r"(\w*_)?(temp|temperature|sampling)(_|$)"
    r"|(^|_)(temp|temperature|sampling)_(\w*_)?"
    r"(disallow(ed)?|unsupported|not_supported|rejected|fixed|omit(ted)?|skip(ped)?|denied|blocked)(_|$)"
)


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _rejects(t) -> bool:
    return t.rejects_sampling is not None


@register
class SamplingModelGate(Rule):
    id = "LDL008"
    name = "incomplete-sampling-gate"
    summary = "a Claude model gate drops temperature/top_p/top_k for some models that reject them, not all"
    references = (
        "https://platform.claude.com/docs/en/about-claude/model-deprecations#api-parameter-deprecations",
        "https://github.com/assafelovic/gpt-researcher/pull/2217",
        "https://github.com/danielmiessler/fabric/pull/2270",
        "https://github.com/CherryHQ/cherry-studio/pull/21939",
        "https://github.com/fishjar/kiss-translator/pull/1185",
    )

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        gf = _gates.GateFile(ctx)
        rejecting = _gates.claude_models(_rejects)
        used_in_chain: set[str] = set()
        for head in _gates.chain_heads(ctx):
            tests = _gates.chain_tests(head)
            if not any(gf.claude_strings(t) for t in tests):
                continue
            for t in tests:
                used_in_chain |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
                used_in_chain |= {n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute)}
            exempt, sent = set(), set()
            for branch in _gates.branches(gf, head):
                sends = _gates.sets_field(branch.body, SAMPLING_PARAMS) is not None
                for m in rejecting:
                    if branch.reaches(m) is True:
                        (sent if sends else exempt).add(m)
            if exempt and sent:
                yield self._finding(head, ctx, sorted(exempt), [m for m in rejecting if m in sent])
        # A reject list defined here and used elsewhere (gpt-researcher reads it in another module).
        for name, items in gf.collections.items():
            if name in used_in_chain or not _REJECT_LIST_NAME.search(snake(name)):
                continue
            covered = [m for m in rejecting if any(s in m for s in items)]
            missing = [m for m in rejecting if m not in covered]
            if covered and missing:
                node = _definition(ctx, name)
                if node is not None:
                    yield self._finding(node, ctx, covered, missing, name)

    def _finding(self, node, ctx, exempt, missing, name=None) -> Finding:
        where = f"`{name}`" if name else "This model gate"
        return self.finding(
            node,
            ctx,
            f"{where} stops sampling parameters for {', '.join(exempt)} but still lets them through "
            f"for {', '.join(missing)}, which also reject non-default temperature/top_p/top_k (400)",
            "Cover every Claude model from Opus 4.7 on (Opus/Sonnet/Haiku 5.x, Fable, Mythos), "
            "e.g. by family prefix instead of exact ids.",
        )


def _definition(ctx: FileContext, name: str) -> ast.AST | None:
    for node in ast.walk(ctx.tree):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        for t in targets:
            if (isinstance(t, ast.Name) and t.id == name) or (
                isinstance(t, ast.Attribute) and t.attr == name
            ):
                return node
    return None
