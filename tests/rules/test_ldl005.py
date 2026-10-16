from llm_drift_lint.rules.ldl005_thinking_budget_tokens import ThinkingBudgetTokens as Rule


def test_chatlas_bedrock_reasoning_budget(lint):
    # chatlas#448: ChatBedrockAnthropic(reasoning=4000) on its default model, a Bedrock id.
    findings = lint(
        Rule,
        """
        from anthropic import AnthropicBedrock

        client = AnthropicBedrock()
        response = client.messages.create(
            model="us.anthropic.claude-sonnet-5-5",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": reasoning},
            messages=messages,
        )
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL005", 5)]
    assert "us.anthropic.claude-sonnet-5-5" in findings[0].message


def test_extra_body_budget(lint):
    findings = lint(
        Rule,
        """
        client.messages.stream(
            model="claude-opus-5-5", max_tokens=8, extra_body={"thinking": {"type": "enabled", "budget_tokens": 2048}}
        )
        """,
    )
    assert [f.line for f in findings] == [2]


def test_adaptive_disabled_and_unknown_thinking_are_silent(lint):
    assert not lint(
        Rule,
        """
        client.messages.create(model="claude-opus-5-5", max_tokens=8, thinking={"type": "adaptive"})
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, thinking={"type": "disabled"})
        client.messages.create(model="claude-opus-5-5", max_tokens=8, thinking=thinking)
        client.messages.create(model="claude-opus-5-5", max_tokens=8, thinking=dict(type="enabled", budget_tokens=1))
        """,
    )


def test_models_that_take_budgets_or_unresolved_are_silent(lint):
    assert not lint(
        Rule,
        """
        client.messages.create(model="claude-haiku-5-5", max_tokens=8, thinking={"type": "enabled", "budget_tokens": 1024})
        client.messages.create(model="claude-sonnet-4-6", max_tokens=8, thinking={"type": "enabled", "budget_tokens": 1024})
        client.messages.create(model=model, max_tokens=8, thinking={"type": "enabled", "budget_tokens": 1024})
        client.messages.count_tokens(model="claude-opus-5-5", thinking={"type": "enabled", "budget_tokens": 1024})
        """,
    )
