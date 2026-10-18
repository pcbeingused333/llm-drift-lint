"""LDL009: a hand-written Claude model gate that sends a manual thinking budget to models that only
take adaptive thinking."""

from __future__ import annotations

from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register
from . import _gates


def _rejects(t) -> bool:
    return t.rejects_budget_tokens


@register
class ThinkingModelGate(Rule):
    id = "LDL009"
    name = "incomplete-thinking-gate"
    summary = "a Claude model gate sends budget_tokens to models that only take adaptive thinking (400)"
    references = (
        "https://platform.claude.com/docs/en/build-with-claude/thinking#configuring-thinking",
        "https://github.com/danielmiessler/fabric/pull/2270",
    )

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        gf = _gates.GateFile(ctx)
        rejecting = _gates.claude_models(_rejects)
        for head in _gates.chain_heads(ctx):
            tests = _gates.chain_tests(head)
            if not any(gf.claude_strings(t) for t in tests):
                continue
            adaptive, budget = set(), set()
            for branch in _gates.branches(gf, head):
                sends_budget = _gates.sets_field(branch.body, ("budget_tokens",)) is not None
                sends_adaptive = _gates.mentions(branch.body, ("adaptive",))
                if sends_budget == sends_adaptive:
                    continue  # neither, or both (can't tell which one is sent)
                for m in rejecting:
                    if branch.reaches(m) is True:
                        (budget if sends_budget else adaptive).add(m)
            # Fire when the gate already gives some of these models adaptive thinking, so the
            # budget branch is the legacy path rather than an intentional choice.
            if adaptive and budget:
                missing = [m for m in rejecting if m in budget]
                yield self.finding(
                    head,
                    ctx,
                    f"This model gate gives adaptive thinking to {', '.join(sorted(adaptive))} but sends "
                    f'{{"type": "enabled", "budget_tokens": N}} to {", ".join(missing)}, '
                    "which reject manual thinking budgets (400)",
                    'Send thinking={"type": "adaptive"} (with output_config.effort) to every model from '
                    "Opus 4.7 on; only Opus/Sonnet 4.6 and older and Mythos Preview take budget_tokens.",
                )
