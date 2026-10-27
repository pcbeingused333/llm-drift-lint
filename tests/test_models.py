import re

import pytest

from llm_drift_lint import models
from llm_drift_lint.models import MODEL_TABLE, SOURCES, ModelTraits, traits

REJECTS_FORCED = ModelTraits("non_default", True, True, True)
THINKS_BY_DEFAULT = ModelTraits("non_default", False, True, True)
NO_SAMPLING = ModelTraits("non_default", False, False, True)


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-opus-5-5", REJECTS_FORCED),
        ("claude-sonnet-5-5", REJECTS_FORCED),
        ("claude-fable-5-1", REJECTS_FORCED),
        ("claude-mythos-5-1", REJECTS_FORCED),
        ("claude-opus-5", THINKS_BY_DEFAULT),
        ("claude-sonnet-5", THINKS_BY_DEFAULT),
        ("claude-haiku-5-5", THINKS_BY_DEFAULT),
        ("claude-fable-5", THINKS_BY_DEFAULT),
        ("claude-mythos-5", THINKS_BY_DEFAULT),
        ("claude-opus-4-8", NO_SAMPLING),
        ("claude-opus-4-7", NO_SAMPLING),
        ("claude-sonnet-4-6", ModelTraits()),
        ("claude-opus-4-6", ModelTraits()),
        ("claude-haiku-4-5", ModelTraits()),
    ],
)
def test_traits_by_model(model, expected):
    assert traits(model) == expected


@pytest.mark.parametrize(
    "model, same_as",
    [
        ("anthropic.claude-opus-5-5", "claude-opus-5-5"),  # Bedrock
        ("us.anthropic.claude-sonnet-5", "claude-sonnet-5"),  # chatlas#448's default
        ("global.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),  # legacy Bedrock profile
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
        ("claude-opus-4-5@20251101", "claude-opus-4-5"),  # Vertex snapshot
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("anthropic/claude-opus-5-5", "claude-opus-5-5"),
        ("Claude-Opus-5-5", "claude-opus-5-5"),
    ],
)
def test_platform_spellings(model, same_as):
    assert traits(model) == traits(same_as)


def test_shorter_id_does_not_match_a_longer_version():
    assert traits("claude-opus-5") != traits("claude-opus-5-5")
    assert traits("claude-sonnet-5") != traits("claude-sonnet-5-5")
    assert traits("claude-fable-5") != traits("claude-fable-5-1")


@pytest.mark.parametrize("model", [None, "", "gpt-4o", "claude-opus-5-6", "claude-haiku-5", "my-model"])
def test_unknown_models(model):
    assert traits(model) is None


def test_every_entry_is_sourced():
    assert [p for p, _ in MODEL_TABLE] == list(SOURCES)
    for pattern, urls in SOURCES.items():
        assert urls, pattern
        assert all(re.match(r"https://", u) for u in urls), pattern


def test_openai_shutdowns():
    assert models.openai_shutdown("gpt-4").replacement == "gpt-5.6-sol"
    assert models.openai_shutdown("openai/o4-mini").date == "2026-10-23"
    assert models.openai_shutdown("ft:gpt-3.5-turbo-0125:acme::abc123").replacement == "gpt-5.6-terra"
    assert models.openai_shutdown("ft:davinci-002:acme::x").date == "2026-10-23"
    assert models.openai_shutdown("gpt-4o-2024-08-06").date == "2027-01-14"
    assert models.openai_shutdown("gpt-5-mini").date == "2026-12-11"
    assert models.openai_shutdown("o3").replacement == "gpt-5.6-sol"
    for alive in ("gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-5.6-sol", "davinci-002", None):
        assert models.openai_shutdown(alive) is None


def test_claude_ids_resolve():
    assert all(models.traits(m) is not None for m in models.CLAUDE_IDS)
