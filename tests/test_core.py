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


def test_kwargs_from_a_dict_built_in_the_same_function():
    c = ctx(
        """
        def judge(client, prompt, model="claude-opus-5-5"):
            params = {"model": model, "max_tokens": 256}
            params["temperature"] = 0.0
            return client.messages.create(**params, messages=[{"role": "user", "content": prompt}])
        """
    )
    (site,) = c.calls("messages.create")
    assert site.kwarg_literal("temperature") == 0.0
    assert site.kwarg_literal("max_tokens") == 256
    assert site.model() == "claude-opus-5-5"  # parameter default, through the dict
    assert "messages" in site.kwargs


def test_dict_call_and_inline_dict_expansion():
    c = ctx(
        """
        def f(client):
            opts = dict(model="claude-sonnet-5-5", top_p=0.9)
            client.messages.create(**opts)
            client.messages.create(**{"temperature": 0.2}, model="m")
        """
    )
    first, second = c.calls("messages.create")
    assert first.model() == "claude-sonnet-5-5" and first.kwarg_literal("top_p") == 0.9
    assert second.kwarg_literal("temperature") == 0.2 and second.model() == "m"


def test_assignments_after_the_call_and_unknown_dicts_are_ignored():
    c = ctx(
        """
        def f(client, extra):
            params = build_params()
            client.messages.create(**params, model="m")
            params = {"temperature": 0}
        """
    )
    (site,) = c.calls("messages.create")
    assert site.kwarg_literal("temperature") is None
    assert site.model() == "m"


def test_local_binding_shadows_module_constant():
    c = ctx(
        """
        MODEL = "claude-opus-5-5"

        def f(client, MODEL):
            client.messages.create(model=MODEL)

        def g(client):
            client.messages.create(model=MODEL)
        """
    )
    in_f, in_g = c.calls("messages.create")
    assert in_f.model() is None  # parameter without default: unknown, not the module constant
    assert in_g.model() == "claude-opus-5-5"
