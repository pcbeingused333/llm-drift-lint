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
GEMINI_DEPRECATIONS = "https://ai.google.dev/gemini-api/docs/deprecations"
VERTEX_MODEL_VERSIONS = "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions"


@dataclass(frozen=True)
class Shutdown:
    date: str  # ISO date the API stops serving the model
    replacement: str  # the replacement the provider recommends
    provider: str = "OpenAI"  # who stops serving it, for the message
    source: str = OPENAI_DEPRECATIONS


def _shutdowns(
    date: str, replacements: dict[str, str], provider: str = "OpenAI", source: str = OPENAI_DEPRECATIONS
) -> dict[str, Shutdown]:
    return {model: Shutdown(date, repl, provider, source) for model, repl in replacements.items()}


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

_SONNET_55, _OPUS_48, _HAIKU_45 = "claude-sonnet-5-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"

# Claude API retirements (DEPRECATIONS, "Deprecation history", read 2026-10-19). The page names
# snapshots; the aliases that pointed at them are listed too, since an alias of a retired snapshot
# has nothing left to serve. Bedrock and Vertex set their own dates, so only Claude API spellings
# (plus an ``anthropic/`` router prefix) are keys here.
CLAUDE_RETIREMENTS: dict[str, Shutdown] = {
    # Announced 2026-09-30.
    **_shutdowns(
        "2026-11-30",
        {"claude-sonnet-4-5-20250929": _SONNET_55, "claude-sonnet-4-5": _SONNET_55},
        "Anthropic",
        DEPRECATIONS,
    ),
    **_shutdowns(
        "2026-08-05",
        {"claude-opus-4-1-20250805": _OPUS_48, "claude-opus-4-1": _OPUS_48},
        "Anthropic",
        DEPRECATIONS,
    ),
    **_shutdowns(
        "2026-06-15",
        {
            "claude-sonnet-4-20250514": _SONNET_55,
            "claude-sonnet-4-0": _SONNET_55,
            "claude-opus-4-20250514": _OPUS_48,
            "claude-opus-4-0": _OPUS_48,
        },
        "Anthropic",
        DEPRECATIONS,
    ),
    **_shutdowns("2026-04-20", {"claude-3-haiku-20240307": _HAIKU_45}, "Anthropic", DEPRECATIONS),
    **_shutdowns(
        "2026-02-19",
        {
            "claude-3-5-haiku-20241022": _HAIKU_45,
            "claude-3-5-haiku-latest": _HAIKU_45,
            "claude-3-7-sonnet-20250219": _SONNET_55,
            "claude-3-7-sonnet-latest": _SONNET_55,
        },
        "Anthropic",
        DEPRECATIONS,
    ),
    **_shutdowns(
        "2026-01-05",
        {"claude-3-opus-20240229": _OPUS_48, "claude-3-opus-latest": _OPUS_48},
        "Anthropic",
        DEPRECATIONS,
    ),
    **_shutdowns(
        "2025-10-28",
        {
            "claude-3-5-sonnet-20240620": _SONNET_55,
            "claude-3-5-sonnet-20241022": _SONNET_55,
            "claude-3-5-sonnet-latest": _SONNET_55,
        },
        "Anthropic",
        DEPRECATIONS,
    ),
    **_shutdowns(
        "2025-07-21",
        {"claude-2.0": _OPUS_48, "claude-2.1": _OPUS_48, "claude-3-sonnet-20240229": _SONNET_55},
        "Anthropic",
        DEPRECATIONS,
    ),
}

# Gemini API (GEMINI_DEPRECATIONS, checked 2026-10-18/19). gemini-2.5-* answers 404 "no longer
# available to new users" there but has no shutdown date, so it isn't listed for the Gemini API.
GEMINI_SHUTDOWNS: dict[str, Shutdown] = {
    **_shutdowns(
        "2026-06-01",
        {
            "gemini-2.0-flash": "gemini-3.6-flash",
            "gemini-2.0-flash-001": "gemini-3.6-flash",
            "gemini-2.0-flash-lite": "gemini-3.1-flash-lite",
            "gemini-2.0-flash-lite-001": "gemini-3.1-flash-lite",
        },
        "Google",
        GEMINI_DEPRECATIONS,
    ),
    **_shutdowns(
        "2026-08-17",
        {
            "imagen-4.0-generate-001": "gemini-nano-banana-2.1",
            "imagen-4.0-ultra-generate-001": "gemini-nano-banana-2.1",
            "imagen-4.0-fast-generate-001": "gemini-nano-banana-2.1",
        },
        "Google",
        GEMINI_DEPRECATIONS,
    ),
    # "Earliest shutdown" date on the page.
    **_shutdowns(
        "2027-03-15",
        {"gemini-2.5-flash-image": "gemini-3.1-flash-lite-image"},
        "Google",
        GEMINI_DEPRECATIONS,
    ),
}

# Vertex AI retires Gemini models on its own schedule (VERTEX_MODEL_VERSIONS, checked 2026-10-19).
# These apply only to code that targets Vertex: gemini-3.6-flash stays stable on the Gemini API.
VERTEX_GEMINI_RETIREMENTS: dict[str, Shutdown] = {
    **_shutdowns(
        "2026-10-20",
        {
            "gemini-2.5-pro": "gemini-3.8-flash",
            "gemini-2.5-flash": "gemini-3.8-flash",
            "gemini-2.5-flash-lite": "gemini-3.8-flash (or gemini-3.1-flash-lite)",
        },
        "Vertex AI",
        VERTEX_MODEL_VERSIONS,
    ),
    **_shutdowns("2026-11-19", {"gemini-3.6-flash": "gemini-3.8-flash"}, "Vertex AI", VERTEX_MODEL_VERSIONS),
}

_ROUTER_PREFIXES = ("openai/", "anthropic/", "gemini/", "google/", "models/")
_VERTEX_PREFIXES = ("vertex_ai/", "vertexai/", "google-vertex/")


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


def retirement(model: str | None, vertex: bool = False) -> Shutdown | None:
    """The shutdown for a model id from any provider: OpenAI, Claude API or Gemini API. With
    ``vertex`` (the code targets Vertex AI), or a ``vertex_ai/`` prefix on the id, Vertex's own Gemini
    dates apply instead of the Gemini API's. None when the model isn't listed."""
    if not model:
        return None
    lowered = model.lower()
    if lowered.startswith(_VERTEX_PREFIXES):
        vertex, lowered = True, lowered.split("/", 1)[1]
    if (found := openai_shutdown(lowered)) is not None:
        return found
    for prefix in _ROUTER_PREFIXES:
        lowered = lowered.removeprefix(prefix)
    if vertex:
        # Vertex also sets its own dates for the Claude and image models it serves.
        return VERTEX_GEMINI_RETIREMENTS.get(lowered)
    return CLAUDE_RETIREMENTS.get(lowered) or GEMINI_SHUTDOWNS.get(lowered)


GEMINI_3_GUIDE = "https://ai.google.dev/gemini-api/docs/gemini-3"
KHOJ_1377 = "https://github.com/khoj-ai/khoj/issues/1377"
OPENLESS_888 = "https://github.com/Open-Less/openless/pull/888"


@dataclass(frozen=True)
class FixedSampling:
    """An OpenAI model that answers non-default ``temperature`` / ``top_p`` with a 400 ("Only the
    default (1) value is supported")."""

    pattern: str  # regex over the lowercased id, after an ``openai/`` prefix is removed
    # True when the request may send sampling values with reasoning effort "none" (newer gpt-5.x);
    # the rule then stays silent unless the effort is known to be something else.
    allowed_without_reasoning: bool
    sources: tuple[str, ...]


# Defaults OpenAI accepts on these models. Anything else is a 400.
OPENAI_SAMPLING_DEFAULTS: dict[str, float] = {"temperature": 1.0, "top_p": 1.0}

# gpt-5.1 to gpt-5.4 are left out: their default reasoning effort is "none" and we have no report of
# them rejecting sampling. ``-chat`` variants are not reasoning models.
OPENAI_FIXED_SAMPLING: list[FixedSampling] = [
    FixedSampling(r"o[134](?:-[a-z0-9-]+)?", False, (KHOJ_1377,)),
    FixedSampling(r"gpt-5(?:-(?!chat)[a-z0-9-]+)?", False, (OPENLESS_888,)),
    # gpt-5.5: OPENLESS_888; gpt-5.6-terra/sol/luna: KHOJ_1377 (400 at temperature 0.8 on gpt-5.6-luna).
    FixedSampling(r"gpt-5\.[56](?:-(?!chat)[a-z0-9-]+)?", True, (OPENLESS_888, KHOJ_1377)),
]

# Gemini 3 models: Google "strongly recommend[s] keeping the temperature parameter at its default value
# of 1.0"; lower values "may lead to unexpected behavior, such as looping or degraded performance"
# (GEMINI_3_GUIDE, read 2026-10-19). Not an error, so the rule words it as a warning.
GEMINI_3 = r"gemini-3(?:\.\d+)?-[a-z0-9.-]+"
GEMINI_DEFAULT_TEMPERATURE = 1.0


def openai_fixed_sampling(model: str | None) -> FixedSampling | None:
    if not model:
        return None
    model = model.lower().removeprefix("openai/")
    return next((m for m in OPENAI_FIXED_SAMPLING if re.fullmatch(m.pattern, model)), None)


def is_gemini_3(model: str | None) -> bool:
    if not model:
        return False
    model = model.lower()
    for prefix in (*_VERTEX_PREFIXES, *_ROUTER_PREFIXES):
        model = model.removeprefix(prefix)
    # The guide is about text generation; image, speech and embedding models aren't covered.
    if any(word in model for word in ("image", "tts", "embedding")):
        return False
    return re.fullmatch(GEMINI_3, model) is not None


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
