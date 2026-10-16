"""LDL006: a small max_tokens on a Claude model that thinks by default, so thinking can use it all."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register
from ._common import ANTHROPIC_REQUEST_CALLS, dict_field, has_splat, site_traits

# At or below this, a model that thinks by default often spends the whole budget thinking and returns no
# text: langchainrb#1073 hit it at 256, and livekit/agents#7726 measured it at exactly 1024 (no text in
# 5/5 runs on Sonnet 5, 4/5 on Haiku 5.5).
MAX_TOKENS_THRESHOLD = 1024


@register
class LowMaxTokensThinking(Rule):
    id = "LDL006"
    name = "low-max-tokens-thinking"
    summary = f"max_tokens <= {MAX_TOKENS_THRESHOLD} on a Claude model that thinks by default (no text in the reply)"
    references = (
        "https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost",
        "https://github.com/patterns-ai-core/langchainrb/pull/1073",
        "https://github.com/livekit/agents/issues/7726",
    )

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        for site in ctx.calls(*ANTHROPIC_REQUEST_CALLS):
            max_tokens = site.kwarg_literal("max_tokens")
            if (
                isinstance(max_tokens, bool)
                or not isinstance(max_tokens, int)
                or max_tokens > MAX_TOKENS_THRESHOLD
            ):
                continue
            if _may_set_thinking(site):
                continue
            traits = site_traits(site)
            if traits is None or not traits.thinks_by_default:
                continue
            yield self.finding(
                site,
                ctx,
                f"max_tokens={max_tokens} is sent to {site.model()}, which thinks by default and counts "
                "thinking against max_tokens, so the reply can stop before any text",
                f"Raise max_tokens (above {MAX_TOKENS_THRESHOLD}; Anthropic recommends 16000 for non-streaming "
                "requests) or set `thinking` explicitly.",
            )


def _may_set_thinking(site) -> bool:
    """True when the request sets `thinking`, or may set it where we can't see."""
    if "thinking" in site.kwargs or has_splat(site):
        return True
    extra_body = site.kwargs.get("extra_body")
    if extra_body is None:
        return False
    return not isinstance(extra_body, ast.Dict) or dict_field(extra_body, "thinking") is not None
