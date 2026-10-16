"""What each model family accepts. Owned by the data session (see COORDINATION.md).

Every entry must cite a source in ``SOURCES`` (provider docs or a real upstream PR). Rules read this
table through the helper functions below and never hard-code model names themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTraits:
    # "all": temperature/top_p/top_k rejected with any value; "non_default": only the default (1.0)
    # is accepted; None: accepted.
    rejects_sampling: str | None = None
    # tool_choice {"type": "tool"} / {"type": "any"} returns a 400.
    rejects_forced_tool_choice: bool = False
    # Thinks when the request has no `thinking` field; thinking counts against max_tokens and the
    # first content block may be a thinking block.
    thinks_by_default: bool = False
    # {"type": "enabled", "budget_tokens": N} is rejected in favor of effort levels.
    rejects_budget_tokens: bool = False


# (regex over the model id, traits). First match wins, so list specific patterns first.
# PLACEHOLDER ENTRIES: the data session replaces these with sourced, complete entries.
MODEL_TABLE: list[tuple[str, ModelTraits]] = [
    (r"claude-opus-5-5", ModelTraits("all", True, True, True)),
    (r"claude-sonnet-5-5", ModelTraits("non_default", True, True, True)),
]

SOURCES: dict[str, list[str]] = {
    # model regex -> URLs backing its traits
}


def traits(model: str | None) -> ModelTraits | None:
    """Traits for a model id (also handles Bedrock/Vertex ids such as ``us.anthropic.claude-...``)."""
    if not model:
        return None
    for pattern, t in MODEL_TABLE:
        if re.search(pattern, model):
            return t
    return None
