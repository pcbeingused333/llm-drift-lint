"""LDL001: temperature / top_p / top_k sent to a Claude model that rejects them."""

from __future__ import annotations

from collections.abc import Iterator

from .. import models
from ..core import FileContext, Finding, Rule, register
from ._common import ANTHROPIC_REQUEST_CALLS, SAMPLING_PARAMS, may_be_unset, request_fields, site_traits


@register
class AnthropicSamplingParams(Rule):
    id = "LDL001"
    name = "anthropic-sampling-params"
    summary = "temperature/top_p/top_k sent to a Claude model that rejects them (400)"
    references = (
        "https://platform.claude.com/docs/en/build-with-claude/thinking",
        "https://platform.claude.com/docs/en/about-claude/model-deprecations",
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
            flagged = set()
            for param in SAMPLING_PARAMS:
                for node in request_fields(site, param):
                    if traits.rejects_sampling == "all":
                        if may_be_unset(node):
                            continue
                        message = f"`{param}` is sent to {model}, which rejects it with any value (400)"
                    elif traits.rejects_sampling == "non_default":
                        # Only a literal other than the model's default is certain to fail. top_k has no
                        # documented default, so any literal top_k counts as non-default
                        # (docs/rules-evidence.md, "SAMPLING_DEFAULTS").
                        default = models.SAMPLING_DEFAULTS.get(param)
                        value = ctx.literal(node)
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            continue
                        if default is None:
                            message = (
                                f"`{param}={value}` is sent to {model}, which rejects non-default "
                                f"`{param}` and documents no default for it (400)"
                            )
                        elif value == default:
                            continue
                        else:
                            message = (
                                f"`{param}={value}` is sent to {model}, which rejects any value "
                                f"other than the default {default} (400)"
                            )
                    else:
                        continue
                    flagged.add(param)
                    yield self.finding(
                        site,
                        ctx,
                        message,
                        f"Don't send `{param}` to this model; leave it out of the request.",
                    )
            if traits.rejects_sampling == "non_default" and not flagged & {"temperature", "top_p"}:
                # The docs: "a request that includes both temperature and top_p, even at their
                # defaults" returns a 400 on these models.
                sent = [
                    p
                    for p in ("temperature", "top_p")
                    if any(not may_be_unset(n) for n in request_fields(site, p))
                ]
                if len(sent) == 2:
                    yield self.finding(
                        site,
                        ctx,
                        f"`temperature` and `top_p` are both sent to {model}, which rejects them together "
                        "even at their defaults (400)",
                        "Send at most one of them, or neither.",
                    )
