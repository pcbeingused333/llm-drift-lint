"""LDL002: sampling parameters passed as keyword arguments to the anthropic 1.x SDK, which removed them."""

from __future__ import annotations

from collections.abc import Iterator

from ..core import FileContext, Finding, Rule, register
from ._common import ANTHROPIC_REQUEST_CALLS, SAMPLING_PARAMS, imports


@register
class AnthropicSamplingKwargsSdk1(Rule):
    id = "LDL002"
    name = "anthropic-sampling-kwargs-sdk1"
    summary = (
        "temperature/top_p/top_k as keyword arguments to messages.create() with anthropic>=1 (TypeError)"
    )
    references = (
        "https://github.com/Arize-ai/phoenix/pull/17026",
        "https://github.com/allenai/reward-bench/pull/290",
    )

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        # Model-independent, but SDK-dependent: only when the project is known to run anthropic 1.x
        # or later, and only in files that use that SDK.
        version = ctx.min_version("anthropic")
        if version is None or version[0] < 1 or not imports(ctx, "anthropic"):
            return
        for site in ctx.calls(*ANTHROPIC_REQUEST_CALLS):
            for param in SAMPLING_PARAMS:
                if param in site.kwargs:
                    yield self.finding(
                        site,
                        ctx,
                        f"`{param}` is passed as a keyword argument, which anthropic>=1 removed from "
                        f"`{site.target.rsplit('.', 1)[-1]}()` (TypeError before the request is sent)",
                        f'Send it in the request body instead: extra_body={{"{param}": ...}}, '
                        "and only to models that accept it.",
                    )
