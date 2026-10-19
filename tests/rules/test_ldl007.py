import datetime
from pathlib import Path

import pytest

from llm_drift_lint.rules import ldl007_retired_model_default
from llm_drift_lint.rules.ldl007_retired_model_default import RetiredModelDefault as Rule


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(ldl007_retired_model_default, "today", lambda: datetime.date(2026, 10, 18))


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
    monkeypatch.setattr(ldl007_retired_model_default, "today", lambda: datetime.date(2026, 10, 23))
    [finding] = lint(Rule, 'DEFAULT_MODEL = "gpt-4"\n')
    assert "shut it down on 2026-10-23" in finding.message and finding.severity == "error"


def test_claude_retirements(lint):
    findings = lint(
        Rule,
        """
        DEFAULT_MODEL = "claude-3-5-sonnet-latest"
        JUDGE_MODEL = "anthropic/claude-sonnet-4-20250514"

        def ask(prompt, model="claude-sonnet-4-5-20250929"):
            ...

        CURRENT_MODEL = "claude-sonnet-4-6"
        BEDROCK_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        """,
    )
    assert [f.line for f in findings] == [2, 3, 5]
    assert "Anthropic shut it down on 2025-10-28" in findings[0].message
    assert "claude-sonnet-5-5" in findings[1].hint
    assert "Anthropic shuts it down on 2026-11-30" in findings[2].message
    assert [f.severity for f in findings] == ["error", "error", "warning"]
    assert findings[2].references == ("https://platform.claude.com/docs/en/about-claude/model-deprecations",)


def test_gemini_api_retirements(lint):
    findings = lint(
        Rule,
        """
        import os

        GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        IMAGE_MODEL = "models/imagen-4.0-generate-001"
        FLASH_MODEL = "gemini-2.5-flash"
        NEW_MODEL = "gemini-3.6-flash"
        """,
    )
    assert [f.line for f in findings] == [4, 5]
    assert "Google shut it down on 2026-06-01" in findings[0].message
    assert "gemini-3.6-flash" in findings[0].hint
    assert "gemini-nano-banana-2.1" in findings[1].hint


def test_vertex_dates_apply_only_to_vertex_code(lint):
    findings = lint(
        Rule,
        """
        from google import genai

        def vertex_client(model="gemini-2.5-flash"):
            return genai.Client(vertexai=True, project=PROJECT)

        def api_client(model="gemini-2.5-flash"):
            return genai.Client(api_key=KEY)

        LITELLM_MODEL = "vertex_ai/gemini-3.6-flash"
        API_MODEL = "gemini/gemini-3.6-flash"
        """,
    )
    assert [f.line for f in findings] == [4, 10]
    assert "Vertex AI shuts it down on 2026-10-20" in findings[0].message
    assert "Vertex AI shuts it down on 2026-11-19" in findings[1].message
    assert "gemini-3.8-flash" in findings[1].hint
    assert "vertex-ai" in findings[1].references[0]


def test_vertex_sdk_import_marks_the_file(lint):
    [finding] = lint(
        Rule,
        """
        import vertexai
        from vertexai.generative_models import GenerativeModel

        MODEL_NAME = "gemini-2.5-pro"
        GEMINI_IMAGE_MODEL = "gemini-2.0-flash"
        """,
    )
    assert finding.line == 5 and "Vertex AI" in finding.message
