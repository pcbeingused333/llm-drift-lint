"""LDL005: a manual thinking budget sent to a Claude model that only takes adaptive thinking."""

from __future__ import annotations

from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register
from ._common import ANTHROPIC_REQUEST_CALLS, dict_str, request_fields, site_traits


@register
class ThinkingBudgetTokens(Rule):
    id = "LDL005"
    name = "thinking-budget-tokens"
    summary = 'thinking {"type": "enabled", "budget_tokens": N} sent to a Claude model that rejects it (400)'
    references = (
        "https://platform.claude.com/docs/en/docs/build-with-claude/adaptive-thinking",
        "https://github.com/posit-dev/chatlas/pull/448",
    )

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        for site in ctx.calls(*ANTHROPIC_REQUEST_CALLS):
            traits = site_traits(site)
            if traits is None or not traits.rejects_budget_tokens:
                continue
            for node in request_fields(site, "thinking"):
                if dict_str(node, "type") == "enabled":
                    yield self.finding(
                        site,
                        ctx,
                        f'thinking {{"type": "enabled", "budget_tokens": ...}} is sent to {site.model()}, '
                        "which rejects manual thinking budgets (400)",
                        'Use thinking={"type": "adaptive"} and set the effort in '
                        'output_config={"effort": ...} instead.',
                    )
