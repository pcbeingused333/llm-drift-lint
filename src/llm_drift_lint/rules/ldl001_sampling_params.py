"""LDL001: temperature / top_p / top_k sent to a Claude model that rejects them."""

from __future__ import annotations

from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register
from ._common import ANTHROPIC_REQUEST_CALLS, SAMPLING_PARAMS, may_be_unset, request_fields, site_traits


@register
class AnthropicSamplingParams(Rule):
    id = "LDL001"
    name = "anthropic-sampling-params"
    summary = "temperature/top_p/top_k sent to a Claude model that rejects them (400)"
    references = (
        "https://platform.claude.com/docs/en/docs/about-claude/model-deprecations",
        "https://github.com/patterns-ai-core/langchainrb/pull/1055",
        "https://github.com/Tencent/WeKnora/pull/4155",
        "https://github.com/truera/trulens/pull/2948",
    )

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        for site in ctx.calls(*ANTHROPIC_REQUEST_CALLS):
            traits = site_traits(site)
            if traits is None or traits.rejects_sampling is None:
                continue
            model = site.model()
            for param in SAMPLING_PARAMS:
                for node in request_fields(site, param):
                    if traits.rejects_sampling == "all":
                        if may_be_unset(node):
                            continue
                        message = f"`{param}` is sent to {model}, which rejects it with any value (400)"
                    elif traits.rejects_sampling == "non_default":
                        # Only a literal other than the API default (1.0) is certain to fail. top_k has
                        # no numeric default to compare against, so it is left out.
                        value = ctx.literal(node)
                        if (
                            param == "top_k"
                            or isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or value == 1.0
                        ):
                            continue
                        message = (
                            f"`{param}={value}` is sent to {model}, which rejects any value "
                            "other than the default 1.0 (400)"
                        )
                    else:
                        continue
                    yield self.finding(
                        site,
                        ctx,
                        message,
                        f"Don't send `{param}` to this model; leave it out of the request.",
                    )
