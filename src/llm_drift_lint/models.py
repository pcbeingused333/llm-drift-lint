"""What each model family accepts. Owned by the data session (see COORDINATION.md).

Every entry must cite a source in ``SOURCES`` (provider docs or a real upstream PR). Rules read this
table through the helper functions below and never hard-code model names themselves.

Where sources disagree, the table follows Anthropic's documentation and the disagreement is written
up in ``docs/rules-evidence.md``. A trait nobody documents stays at its default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTraits:
    # "all": temperature/top_p/top_k rejected with any value; "non_default": only the default
    # (see SAMPLING_DEFAULTS) is accepted; None: accepted.
    rejects_sampling: str | None = None
    # tool_choice {"type": "tool"} / {"type": "any"} returns a 400.
    rejects_forced_tool_choice: bool = False
    # Thinks when the request has no `thinking` field; thinking counts against max_tokens and the
    # first content block may be a thinking block.
    thinks_by_default: bool = False
    # {"type": "enabled", "budget_tokens": N} is rejected in favor of effort levels.
    rejects_budget_tokens: bool = False


# Values that "non_default" models accept. top_p's default is 0.99 on those models, so top_p=1 is a
# 400, and so is sending temperature and top_p together even at these values. top_k has no
# documented default, so any top_k counts as non-default.
SAMPLING_DEFAULTS: dict[str, float] = {"temperature": 1.0, "top_p": 0.99}


def _model(name: str) -> str:
    """Regex for one model id in any of its platform spellings.

    Matches ``claude-opus-5-5``, ``anthropic.claude-opus-5-5`` (Bedrock), ``us.anthropic.``/
    ``global.anthropic.`` inference profiles, ``anthropic/claude-opus-5-5`` (router prefixes),
    dated snapshots (``claude-sonnet-4-5-20250929``, ``claude-opus-4-5@20251101`` on Vertex) and
    Bedrock version suffixes (``-v1``, ``-v1:0``). It doesn't match a longer version:
    ``claude-opus-5`` doesn't match ``claude-opus-5-5``, and ``claude-opus-4`` doesn't match
    ``claude-opus-4-8`` (or the ``claude-opus-4-0`` alias, which has its own entry).
    """
    return rf"(?<![a-z0-9]){re.escape(name)}(?!\d|[-.]\d{{1,2}}(?!\d))"


_REJECTS_FORCED = ModelTraits("non_default", True, True, True)  # Opus/Sonnet 5.5, Fable/Mythos 5.1
_THINKS_BY_DEFAULT = ModelTraits("non_default", False, True, True)  # Fable/Mythos 5, Opus/Sonnet 5, Haiku 5.5
_NO_SAMPLING = ModelTraits("non_default", False, False, True)  # Opus 4.7, Opus 4.8
_ACCEPTS_ALL = ModelTraits()  # Opus/Sonnet 4.6 and older: none of the restrictions apply

_CLAUDE_3 = r"(?<![a-z0-9])claude-3-(?:5-|7-)?(?:opus|sonnet|haiku)"

# (regex over the model id, traits). First match wins. The patterns don't overlap (see _model), so
# the order only groups models by traits.
MODEL_TABLE: list[tuple[str, ModelTraits]] = [
    (_model("claude-fable-5-1"), _REJECTS_FORCED),
    (_model("claude-mythos-5-1"), _REJECTS_FORCED),
    (_model("claude-opus-5-5"), _REJECTS_FORCED),
    (_model("claude-sonnet-5-5"), _REJECTS_FORCED),
    (_model("claude-fable-5"), _THINKS_BY_DEFAULT),
    (_model("claude-mythos-5"), _THINKS_BY_DEFAULT),
    (_model("claude-opus-5"), _THINKS_BY_DEFAULT),
    (_model("claude-sonnet-5"), _THINKS_BY_DEFAULT),
    (_model("claude-haiku-5-5"), _THINKS_BY_DEFAULT),
    # Mythos Preview still takes manual budgets ("Extended thinking" in the thinking table) and its
    # forced tool_choice behavior isn't documented, so those two traits stay at their defaults.
    (_model("claude-mythos-preview"), ModelTraits("non_default", False, True, False)),
    (_model("claude-opus-4-8"), _NO_SAMPLING),
    (_model("claude-opus-4-7"), _NO_SAMPLING),
    (_model("claude-opus-4-6"), _ACCEPTS_ALL),
    (_model("claude-sonnet-4-6"), _ACCEPTS_ALL),
    (_model("claude-opus-4-5"), _ACCEPTS_ALL),
    (_model("claude-sonnet-4-5"), _ACCEPTS_ALL),
    (_model("claude-haiku-4-5"), _ACCEPTS_ALL),
    (_model("claude-opus-4-1"), _ACCEPTS_ALL),
    (_model("claude-opus-4-0"), _ACCEPTS_ALL),
    (_model("claude-opus-4"), _ACCEPTS_ALL),  # claude-opus-4-20250514, claude-opus-4@20250514
    (_model("claude-sonnet-4-0"), _ACCEPTS_ALL),
    (_model("claude-sonnet-4"), _ACCEPTS_ALL),
    (_CLAUDE_3, _ACCEPTS_ALL),
]

_DOCS = "https://platform.claude.com/docs/en"
THINKING = f"{_DOCS}/build-with-claude/thinking"  # per-model thinking table, "Sampling parameters"
DEPRECATIONS = f"{_DOCS}/about-claude/model-deprecations"  # "API parameter deprecations"
FORCED_TOOL_USE = f"{_DOCS}/agents-and-tools/tool-use/define-tools#forcing-tool-use"
MODELS_OVERVIEW = f"{_DOCS}/about-claude/models/overview"  # Claude API / Bedrock / Google Cloud ids
BEDROCK_IDS = f"{_DOCS}/build-with-claude/claude-in-amazon-bedrock"  # anthropic.claude-... ids
BEDROCK_LEGACY_IDS = f"{_DOCS}/build-with-claude/claude-on-amazon-bedrock-legacy"  # -v1:0, us./global.
VERTEX_IDS = f"{_DOCS}/build-with-claude/claude-on-vertex-ai"  # claude-...@YYYYMMDD
PHOENIX_17027 = "https://github.com/Arize-ai/phoenix/pull/17027"
TRULENS_2948 = "https://github.com/truera/trulens/pull/2948"
LANGCHAINRB_1055 = "https://github.com/patterns-ai-core/langchainrb/pull/1055"
LANGCHAINRB_1073 = "https://github.com/patterns-ai-core/langchainrb/pull/1073"
WEKNORA_4155 = "https://github.com/Tencent/WeKnora/pull/4155"
CHATLAS_448 = "https://github.com/posit-dev/chatlas/pull/448"
HOSTEDGPT_823 = "https://github.com/AllYourBot/hostedgpt/pull/823"
LLMVISION_747 = "https://github.com/valentinfrlch/ha-llmvision/pull/747"

_IDS = [MODELS_OVERVIEW, BEDROCK_IDS, VERTEX_IDS]
_IDS_PRE_5 = [BEDROCK_IDS, VERTEX_IDS]  # models after 4.6 that the overview lists as legacy
_IDS_LEGACY = [BEDROCK_LEGACY_IDS, VERTEX_IDS]  # Opus/Sonnet 4.6 and older

# model regex -> URLs backing its traits and its platform ids. The PRs are listed where they state
# the same fact; docs/rules-evidence.md says which claim each one makes.
SOURCES: dict[str, list[str]] = {
    _model("claude-fable-5-1"): [THINKING, FORCED_TOOL_USE, *_IDS, PHOENIX_17027, LLMVISION_747],
    _model("claude-mythos-5-1"): [THINKING, FORCED_TOOL_USE, *_IDS_PRE_5, PHOENIX_17027],
    _model("claude-opus-5-5"): [THINKING, FORCED_TOOL_USE, *_IDS, PHOENIX_17027, TRULENS_2948, HOSTEDGPT_823],
    _model("claude-sonnet-5-5"): [THINKING, FORCED_TOOL_USE, *_IDS, PHOENIX_17027, LANGCHAINRB_1073],
    _model("claude-fable-5"): [THINKING, FORCED_TOOL_USE, *_IDS_PRE_5, WEKNORA_4155],
    _model("claude-mythos-5"): [THINKING, FORCED_TOOL_USE, *_IDS_PRE_5],
    _model("claude-opus-5"): [THINKING, FORCED_TOOL_USE, *_IDS_PRE_5, PHOENIX_17027, WEKNORA_4155],
    _model("claude-sonnet-5"): [THINKING, FORCED_TOOL_USE, *_IDS_PRE_5, CHATLAS_448, WEKNORA_4155],
    _model("claude-haiku-5-5"): [THINKING, FORCED_TOOL_USE, *_IDS, PHOENIX_17027, LANGCHAINRB_1073],
    _model("claude-mythos-preview"): [THINKING, DEPRECATIONS, BEDROCK_IDS],
    _model("claude-opus-4-8"): [THINKING, DEPRECATIONS, *_IDS_PRE_5, WEKNORA_4155],
    _model("claude-opus-4-7"): [THINKING, DEPRECATIONS, *_IDS_PRE_5, WEKNORA_4155],
    _model("claude-opus-4-6"): [THINKING, *_IDS_LEGACY],
    _model("claude-sonnet-4-6"): [THINKING, *_IDS_LEGACY, LANGCHAINRB_1055, WEKNORA_4155, TRULENS_2948],
    _model("claude-opus-4-5"): [THINKING, *_IDS_LEGACY],
    _model("claude-sonnet-4-5"): [THINKING, *_IDS_LEGACY],
    _model("claude-haiku-4-5"): [THINKING, BEDROCK_IDS, *_IDS_LEGACY, TRULENS_2948],
    _model("claude-opus-4-1"): [DEPRECATIONS, *_IDS_LEGACY],
    _model("claude-opus-4-0"): [DEPRECATIONS],
    _model("claude-opus-4"): [DEPRECATIONS, VERTEX_IDS],
    _model("claude-sonnet-4-0"): [DEPRECATIONS],
    _model("claude-sonnet-4"): [DEPRECATIONS, *_IDS_LEGACY],
    _CLAUDE_3: [DEPRECATIONS, THINKING],
}


def traits(model: str | None) -> ModelTraits | None:
    """Traits for a model id (also handles Bedrock/Vertex ids such as ``us.anthropic.claude-...``).

    ``None`` means the model is unknown; rules treat that like an unresolved model and stay silent.
    """
    if not model:
        return None
    model = model.lower()
    for pattern, t in MODEL_TABLE:
        if re.search(pattern, model):
            return t
    return None
