"""LDL004: a forced tool_choice sent to a Claude model that rejects it."""

from __future__ import annotations

from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register
from ._common import ANTHROPIC_REQUEST_CALLS, dict_str, request_fields, site_traits

_FORCED = ("tool", "any")


@register
class ForcedToolChoice(Rule):
    id = "LDL004"
    name = "forced-tool-choice"
    summary = 'tool_choice {"type": "tool"|"any"} sent to a Claude model that rejects forcing a tool (400)'
    references = (
        "https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools",
        "https://github.com/Arize-ai/phoenix/pull/17027",
        "https://github.com/valentinfrlch/ha-llmvision/pull/747",
        "https://github.com/truera/trulens/pull/2948",
    )

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        for site in ctx.calls(*ANTHROPIC_REQUEST_CALLS):
            traits = site_traits(site)
            if traits is None or not traits.rejects_forced_tool_choice:
                continue
            for node in request_fields(site, "tool_choice"):
                kind = dict_str(node, "type")
                if kind in _FORCED:
                    yield self.finding(
                        site,
                        ctx,
                        f'tool_choice {{"type": "{kind}"}} is sent to {site.model()}, which rejects a forced '
                        "tool choice (400)",
                        'Use tool_choice={"type": "auto"} for this model and handle a reply without a '
                        "tool call.",
                    )
