"""LDL012: non-default temperature / top_p sent to OpenAI reasoning models or Gemini 3 models.

OpenAI's o-series and gpt-5 / gpt-5.5 / gpt-5.6 answer a non-default value with a 400 ("Only the
default (1) value is supported"). Gemini 3 accepts a lower temperature but Google warns it can loop or
degrade, so that case is worded as a warning.

Evidence: for gpt-5.5 and gpt-5.6 the 400 comes from issue reports (khoj#1377, openless#888), not
from OpenAI's documentation, which says nothing about sampling on those models. The Gemini 3 case comes
from Google's Gemini 3 guide.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from .. import models
from ..core import OPENAI_CHAT_CALLS, OPENAI_RESPONSES_CALLS, CallSite, FileContext, Finding, Rule, register
from ._common import dict_field, request_fields

GENAI_CALLS = ("models.generate_content", "models.generate_content_stream")


def _number(ctx: FileContext, node: ast.expr, scope: ast.AST) -> float | None:
    value = ctx.literal(node, scope=scope)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _effort(site: CallSite) -> tuple[bool, object]:
    """(sent, value) for the reasoning effort: ``reasoning_effort=`` (Chat Completions) or
    ``reasoning={"effort": ...}`` (Responses). value is None when it can't be resolved."""
    if "reasoning_effort" in site.kwargs:
        return True, site.kwarg_literal("reasoning_effort")
    reasoning = site.kwargs.get("reasoning")
    if reasoning is None:
        return False, None
    if not isinstance(reasoning, ast.Dict):
        return True, None  # built elsewhere: can't see the effort
    effort = dict_field(reasoning, "effort")
    if effort is None:
        return False, None
    return True, site.ctx.literal(effort, scope=site.node)


def _local_binding(ctx: FileContext, name: str, limit: ast.AST) -> ast.expr | None:
    """The last ``name = <expr>`` before ``limit`` in the same function."""
    fn = ctx.enclosing_function(limit)
    if fn is None:
        return None
    bindings = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        and (node.lineno, node.col_offset) < (limit.lineno, limit.col_offset)  # type: ignore[attr-defined]
    ]
    if not bindings:
        return None
    return max(bindings, key=lambda n: (n.lineno, n.col_offset)).value


def _config_field(site: CallSite, name: str) -> ast.expr | None:
    """``config=GenerateContentConfig(temperature=...)`` or ``config={"temperature": ...}``, also
    through a local variable."""
    config = site.kwargs.get("config")
    if isinstance(config, ast.Name):
        config = _local_binding(site.ctx, config.id, site.node)
    if isinstance(config, ast.Dict):
        return dict_field(config, name)
    if isinstance(config, ast.Call):
        return next((k.value for k in config.keywords if k.arg == name), None)
    return None


@register
class OpenAIGeminiSampling(Rule):
    id = "LDL012"
    name = "openai-gemini-sampling"
    summary = "non-default temperature/top_p sent to OpenAI o-series/gpt-5 models (400) or a low temperature to Gemini 3"
    references = (models.KHOJ_1377, models.OPENLESS_888, models.GEMINI_3_GUIDE)

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        for site in ctx.calls(*OPENAI_CHAT_CALLS, *OPENAI_RESPONSES_CALLS, *GENAI_CALLS):
            if site.target.endswith(GENAI_CALLS):
                yield from self._gemini(site, _config_field(site, "temperature"))
            else:
                yield from self._openai(site)
                yield from self._gemini(site, site.kwargs.get("temperature"))

    def _openai(self, site: CallSite) -> Iterator[Finding]:
        model = site.model()
        fixed = models.openai_fixed_sampling(model)
        if fixed is None:
            return
        if fixed.allowed_without_reasoning:
            sent, effort = _effort(site)
            if sent and effort in (None, "none"):
                return  # sampling is allowed with effort "none", and we can't tell otherwise
        for param, default in models.OPENAI_SAMPLING_DEFAULTS.items():
            for node in request_fields(site, param):
                value = _number(site.ctx, node, site.node)
                if value is None or value == default:
                    continue
                hint = f"Leave `{param}` out of requests to this model."
                if fixed.allowed_without_reasoning:
                    hint += ' It is accepted only with reasoning effort "none".'
                yield self.finding(
                    site,
                    site.ctx,
                    f"`{param}={value:g}` is sent to {model}, which only accepts the default "
                    f"({default:g}) and answers anything else with a 400",
                    hint,
                )

    def _gemini(self, site: CallSite, node: ast.expr | None) -> Iterator[Finding]:
        model = site.model()
        if node is None or not models.is_gemini_3(model):
            return
        value = _number(site.ctx, node, site.node)
        if value is None or value >= models.GEMINI_DEFAULT_TEMPERATURE:
            return
        yield self.finding(
            site,
            site.ctx,
            f"`temperature={value:g}` is sent to {model}; Google recommends the default 1.0 for Gemini 3 "
            "models and warns that lower values can cause looping or degraded output",
            "Leave `temperature` unset for Gemini 3 models.",
        )
