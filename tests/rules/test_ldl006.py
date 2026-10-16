from llm_drift_lint.rules.ldl006_low_max_tokens_thinking import LowMaxTokensThinking as Rule


def test_langchainrb_default_of_256(lint):
    # langchainrb#1073: a 256-token default with chat_model "claude-sonnet-5-5".
    findings = lint(
        Rule,
        """
        DEFAULTS = {"max_tokens": 256}
        MAX_TOKENS = 256

        def chat(client, messages):
            return client.messages.create(model="claude-sonnet-5-5", max_tokens=MAX_TOKENS, messages=messages)
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL006", 6)]
    assert "max_tokens=256" in findings[0].message and "claude-sonnet-5-5" in findings[0].message


def test_just_below_the_threshold(lint):
    findings = lint(
        Rule,
        """
        client.messages.create(model="claude-haiku-5-5", max_tokens=1023, messages=m)
        client.messages.stream(model="claude-opus-5-5", max_tokens=1, messages=m, extra_body={"metadata": {}})
        """,
    )
    assert [f.line for f in findings] == [2, 3]


def test_enough_tokens_or_non_literal_are_silent(lint):
    assert not lint(
        Rule,
        """
        client.messages.create(model="claude-sonnet-5-5", max_tokens=1024, messages=m)
        client.messages.create(model="claude-sonnet-5-5", max_tokens=16000, messages=m)
        client.messages.create(model="claude-sonnet-5-5", max_tokens=self.max_tokens, messages=m)
        client.messages.create(model="claude-sonnet-5-5", messages=m)
        """,
    )


def test_explicit_or_hidden_thinking_is_silent(lint):
    assert not lint(
        Rule,
        """
        client.messages.create(model="claude-sonnet-5-5", max_tokens=256, thinking={"type": "disabled"}, messages=m)
        client.messages.create(model="claude-sonnet-5-5", max_tokens=256, thinking=cfg, messages=m)
        client.messages.create(model="claude-sonnet-5-5", max_tokens=256, extra_body={"thinking": cfg})
        client.messages.create(model="claude-sonnet-5-5", max_tokens=256, extra_body=body)
        client.messages.create(model="claude-sonnet-5-5", max_tokens=256, **kwargs)
        """,
    )


def test_models_that_dont_think_by_default_or_unresolved_are_silent(lint):
    assert not lint(
        Rule,
        """
        client.messages.create(model="claude-sonnet-4-6", max_tokens=256, messages=m)
        client.messages.create(model=model, max_tokens=256, messages=m)
        client.chat.completions.create(model="claude-sonnet-5-5", max_tokens=256, messages=m)
        """,
    )
