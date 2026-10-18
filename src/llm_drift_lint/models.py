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


# Current Claude ids in their plain Claude API spelling, one per row of MODEL_TABLE that has
# restrictions. Rules that check a model gate (a hand-written list of ids or prefixes) test the gate
# against these ids to find the models it misses.
CLAUDE_IDS: tuple[str, ...] = (
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-5-5",
    "claude-mythos-preview",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
)

OPENAI_DEPRECATIONS = "https://developers.openai.com/api/docs/deprecations"


@dataclass(frozen=True)
class Shutdown:
    date: str  # ISO date the API stops serving the model
    replacement: str  # the replacement OpenAI recommends


def _shutdowns(date: str, replacements: dict[str, str]) -> dict[str, Shutdown]:
    return {model: Shutdown(date, repl) for model, repl in replacements.items()}


_TERRA, _SOL, _LUNA = "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna"

# OpenAI model ids and the date the API stops serving them (OPENAI_DEPRECATIONS, checked 2026-10-18).
# Keys are exact ids: an alias is listed only when the notice names it, so the 2027-01-14 notice
# contributes snapshots only (it doesn't say what the gpt-4o / gpt-4.1 aliases point to afterwards).
OPENAI_SHUTDOWNS: dict[str, Shutdown] = {
    # Announced 2026-04-22.
    **_shutdowns(
        "2026-10-23",
        {
            "gpt-3.5-turbo": _TERRA,
            "gpt-3.5-turbo-0125": _TERRA,
            "gpt-4": _SOL,
            "gpt-4-0613": _SOL,
            "gpt-4-1106-preview": _SOL,
            "gpt-4-turbo": _SOL,
            "gpt-4-turbo-2024-04-09": _SOL,
            "gpt-4o-2024-05-13": _SOL,
            "gpt-4.1-nano": _LUNA,
            "gpt-4.1-nano-2025-04-14": _LUNA,
            "o1": _SOL,
            "o1-2024-12-17": _SOL,
            "o1-pro": 'gpt-5.6-sol with reasoning {"mode": "pro"}',
            "o3-mini": _SOL,
            "o3-mini-2025-01-31": _SOL,
            "o4-mini": _TERRA,
            "o4-mini-2025-04-16": _TERRA,
            "gpt-image-1": "gpt-image-2.5-sunburst or gpt-image-2.5-flare",
        },
    ),
    # Announced 2026-10-14.
    **_shutdowns(
        "2027-01-14",
        {
            "gpt-4o-2024-08-06": "gpt-6-luna",
            "gpt-4o-2024-11-20": "gpt-6-luna",
            "gpt-4o-mini-2024-07-18": "gpt-6-luna",
            "gpt-4.1-2025-04-14": "gpt-6-luna",
            "gpt-4.1-mini-2025-04-14": "gpt-6-luna",
        },
    ),
    # Already shut down.
    **_shutdowns("2026-09-28", {"gpt-3.5-turbo-instruct": _TERRA, "gpt-3.5-turbo-1106": _TERRA}),
    **_shutdowns("2026-03-26", {"gpt-4-0314": _SOL, "gpt-4-0125-preview": _SOL}),
}


def openai_shutdown(model: str | None) -> Shutdown | None:
    """The shutdown for an OpenAI model id, also behind an ``openai/`` router prefix. A fine-tune
    (``ft:gpt-4-0613:org::id``) follows its base model, and ``ft:babbage-002`` / ``ft:davinci-002``
    shut down on 2026-10-23. None when the model isn't listed."""
    if not model:
        return None
    model = model.lower().removeprefix("openai/")
    if model.startswith("ft:"):
        base = model.split(":")[1]
        if base in ("babbage-002", "davinci-002"):
            return Shutdown("2026-10-23", _TERRA)
        model = base
    return OPENAI_SHUTDOWNS.get(model)


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
