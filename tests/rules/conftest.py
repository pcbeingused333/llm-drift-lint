"""A fixed model table for the rule tests, so they don't move when models.py is updated.

The traits follow what the motivating PRs state; models.py remains the source of truth for the linter.
"""

from textwrap import dedent

import pytest

from llm_drift_lint import models
from llm_drift_lint.core import FileContext
from llm_drift_lint.models import ModelTraits

TEST_TABLE = [
    # rejects_sampling, rejects_forced_tool_choice, thinks_by_default, rejects_budget_tokens
    (r"claude-opus-5-5", ModelTraits("all", True, True, True)),
    (r"claude-sonnet-5-5", ModelTraits("non_default", True, True, True)),
    (r"claude-haiku-5-5", ModelTraits("non_default", False, True, False)),
    (r"claude-sonnet-4-6", ModelTraits()),
]


@pytest.fixture(autouse=True)
def fixed_model_table(monkeypatch):
    monkeypatch.setattr(models, "MODEL_TABLE", TEST_TABLE)


@pytest.fixture
def lint():
    """``lint(rule_cls, source, path=None)`` -> the rule's findings over the dedented source."""

    def _lint(rule_cls, src, path=None):
        kwargs = {} if path is None else {"path": path}
        ctx = FileContext.parse(dedent(src), **kwargs)
        assert ctx is not None
        return list(rule_cls().check(ctx))

    return _lint
