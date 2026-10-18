import datetime
from pathlib import Path

import pytest

from llm_drift_lint.rules import ldl007_openai_shutdown_default
from llm_drift_lint.rules.ldl007_openai_shutdown_default import OpenAIShutdownDefault as Rule


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(ldl007_openai_shutdown_default, "today", lambda: datetime.date(2026, 10, 18))


def test_parameter_and_constant_defaults(lint):
    findings = lint(
        Rule,
        """
        import os

        DEFAULT_MODEL = "gpt-3.5-turbo"
        JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4")

        def summarize(text, model="gpt-4-turbo"):
            ...
        """,
    )
    assert [(f.line, f.message.split("`")[1]) for f in findings] == [
        (4, "gpt-3.5-turbo"),
        (5, "gpt-4"),
        (7, "gpt-4-turbo"),
    ]
    assert "2026-10-23" in findings[0].message
    assert "gpt-5.6-terra" in findings[0].hint and "gpt-5.6-sol" in findings[1].hint


def test_field_router_prefix_and_argparse(lint):
    findings = lint(
        Rule,
        """
        from pydantic import BaseModel, Field

        class Settings(BaseModel):
            model_name: str = Field(default="o4-mini")
            judge_model = "openai/gpt-3.5-turbo"

        parser.add_argument("--model", default="o1-pro")
        """,
    )
    assert [f.line for f in findings] == [5, 6, 8]
    assert "reasoning" in findings[2].hint


def test_call_sites_and_test_files_are_silent(lint):
    src = """
        client.chat.completions.create(model="o1", messages=messages)
        payload = {"model": "gpt-3.5-turbo", "messages": []}
        token_counter(model="gpt-3.5-turbo", text=text)
        """
    assert not lint(Rule, src)
    assert not lint(Rule, 'MODEL = "gpt-4"\n', path=Path("pkg/test_judge.py"))


def test_current_models_tokenizers_azure_and_catalogs_are_silent(lint):
    assert not lint(
        Rule,
        """
        import tiktoken

        DEFAULT_MODEL = "gpt-5.6-sol"
        enc = tiktoken.encoding_for_model("gpt-4")
        enc = tiktoken.encoding_for_model(model="gpt-4")
        PRICES = {"gpt-4": 30.0, "gpt-3.5-turbo": 0.5}
        SUPPORTED = ["gpt-4", "o1"]
        if model == "gpt-4":
            pass
        llm = AzureChatOpenAI(azure_deployment="prod", model="gpt-4")
        def embed(text, embedding_model="gpt-4"):
            ...
        MODEL_4O = "gpt-4o"
        """,
    )


def test_past_and_upcoming_shutdowns_are_worded_apart(lint):
    findings = lint(
        Rule,
        """
        A_MODEL = "gpt-3.5-turbo-instruct"
        B_MODEL = "gpt-4o-2024-08-06"
        C_MODEL = "gpt-4o"
        """,
    )
    assert [f.line for f in findings] == [2, 3]
    assert "shut it down on 2026-09-28, so requests with it fail" in findings[0].message
    assert "shuts it down on 2027-01-14" in findings[1].message and "gpt-6-luna" in findings[1].hint
    assert [f.severity for f in findings] == ["error", "warning"]


def test_shutdown_day_counts_as_past(lint, monkeypatch):
    monkeypatch.setattr(ldl007_openai_shutdown_default, "today", lambda: datetime.date(2026, 10, 23))
    [finding] = lint(Rule, 'DEFAULT_MODEL = "gpt-4"\n')
    assert "shut it down on 2026-10-23" in finding.message and finding.severity == "error"
