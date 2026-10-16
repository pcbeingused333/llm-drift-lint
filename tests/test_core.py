from textwrap import dedent

from llm_drift_lint.core import ANTHROPIC_MESSAGE_CALLS, FileContext


def ctx(src: str) -> FileContext:
    parsed = FileContext.parse(dedent(src))
    assert parsed is not None
    return parsed


def test_calls_match_sdk_method_chains_regardless_of_receiver():
    c = ctx(
        """
        import anthropic
        client = anthropic.Anthropic()
        client.messages.create(model="m", max_tokens=1, messages=[])
        self.client.beta.messages.stream(model="m")
        anthropic.Anthropic().messages.create(model="m")
        other.create(model="m")
        """
    )
    targets = [site.target for site in c.calls(*ANTHROPIC_MESSAGE_CALLS)]
    assert targets == ["messages.create", "client.beta.messages.stream", "messages.create"]


def test_model_resolves_literals_and_module_constants():
    c = ctx(
        """
        MODEL = "claude-opus-5-5"
        client.messages.create(model=MODEL)
        client.messages.create(model="claude-sonnet-5-5")
        client.messages.create(model=get_model())
        """
    )
    assert [site.model() for site in c.calls("messages.create")] == [
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        None,
    ]


def test_kwarg_literal():
    c = ctx('client.messages.create(temperature=0.0, tool_choice={"type": "tool", "name": "x"})')
    (site,) = c.calls("messages.create")
    assert site.kwarg_literal("temperature") == 0.0
    assert site.kwarg_literal("tool_choice") == {"type": "tool", "name": "x"}
    assert site.kwarg_literal("missing") is None


def test_unparseable_source_returns_none():
    assert FileContext.parse("def broken(:") is None
