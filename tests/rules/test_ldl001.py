from llm_drift_lint.rules.ldl001_sampling_params import AnthropicSamplingParams as Rule


def test_default_temperature_sent_to_model_that_rejects_sampling(lint):
    # langchainrb#1055 / trulens#2948: a hard-coded temperature=0.0 on every request.
    findings = lint(
        Rule,
        """
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": "hi"}],
        )
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL001", 4)]
    assert "temperature" in findings[0].message and "claude-opus-5-5" in findings[0].message


def test_any_value_and_every_sampling_param_for_reject_all_models(lint):
    findings = lint(
        Rule,
        """
        MODEL = "claude-opus-5-5"
        def judge(client, temperature):
            return client.messages.create(model=MODEL, max_tokens=8, temperature=temperature, top_p=1.0, top_k=5)
        """,
    )
    assert [f.message.split("`")[1] for f in findings] == ["temperature", "top_p", "top_k"]


def test_non_default_literal_for_non_default_models(lint):
    # WeKnora#4155: pipelines that set 0.3 / 0.1 on Sonnet 5.x.
    findings = lint(
        Rule,
        """
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, temperature=0.3)
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, top_p=0.9)
        """,
    )
    assert [f.line for f in findings] == [2, 3]


def test_extra_body_is_sent_too(lint):
    # The reward-bench#290 fix moved temperature into extra_body; the API still receives it.
    findings = lint(
        Rule,
        """
        client.messages.create(model="claude-opus-5-5", max_tokens=8, extra_body={"temperature": t})
        """,
    )
    assert [f.line for f in findings] == [2]


def test_stream_and_parse_are_requests_too(lint):
    findings = lint(
        Rule,
        """
        self.client.beta.messages.stream(model="claude-opus-5-5", max_tokens=8, temperature=0)
        client.messages.parse(model="claude-opus-5-5", max_tokens=8, temperature=0)
        """,
    )
    assert [f.line for f in findings] == [2, 3]


def test_unresolved_model_is_silent(lint):
    assert not lint(
        Rule,
        """
        def chat(client, model):
            return client.messages.create(model=model, max_tokens=8, temperature=0.0)
        client.messages.create(model=settings.model, max_tokens=8, temperature=0.0)
        client.messages.create(max_tokens=8, temperature=0.0, **kwargs)
        """,
    )


def test_models_that_accept_sampling_are_silent(lint):
    assert not lint(
        Rule,
        """
        client.messages.create(model="claude-sonnet-4-6", max_tokens=8, temperature=0.0, top_k=5)
        client.messages.create(model="claude-some-future-model", max_tokens=8, temperature=0.0)
        """,
    )


def test_non_default_models_accept_default_and_unknown_values(lint):
    assert not lint(
        Rule,
        """
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, temperature=1.0)
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, temperature=1)
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, temperature=user_temperature)
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, top_k=40)
        """,
    )


def test_unset_values_are_silent(lint):
    assert not lint(
        Rule,
        """
        from anthropic import NOT_GIVEN
        client.messages.create(model="claude-opus-5-5", max_tokens=8, temperature=None)
        client.messages.create(model="claude-opus-5-5", max_tokens=8, temperature=NOT_GIVEN)
        client.messages.create(model="claude-opus-5-5", max_tokens=8, temperature=anthropic.omit)
        client.messages.create(
            model="claude-opus-5-5", max_tokens=8, temperature=t if t is not None else NOT_GIVEN
        )
        """,
    )


def test_other_apis_are_silent(lint):
    assert not lint(
        Rule,
        """
        client.chat.completions.create(model="claude-opus-5-5", temperature=0.0)
        client.messages.count_tokens(model="claude-opus-5-5", messages=[])
        llm.invoke("hi", model="claude-opus-5-5", temperature=0.0)
        """,
    )
