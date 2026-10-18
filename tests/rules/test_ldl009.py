from llm_drift_lint.rules.ldl009_thinking_gate import ThinkingModelGate as Rule

# Test table: opus-5-5 and sonnet-5-5 reject budget_tokens; haiku-5-5 and sonnet-4-6 accept them.


def test_adaptive_prefix_list_missing_a_model(lint):
    # fabric#2270: adaptiveThinkingPrefixes lacked several models, which got budget_tokens.
    findings = lint(
        Rule,
        """
        ADAPTIVE_THINKING_PREFIXES = ("claude-opus-5",)

        def thinking_for(model, budget):
            if model.startswith(ADAPTIVE_THINKING_PREFIXES):
                return {"type": "adaptive"}
            return {"type": "enabled", "budget_tokens": budget}
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL009", 5)]
    assert "to claude-opus-5-5" in findings[0].message
    assert 'budget_tokens": N} to claude-sonnet-5-5,' in findings[0].message


def test_elif_chain(lint):
    findings = lint(
        Rule,
        """
        def params(model_name, budget):
            if "opus-5" in model_name:
                thinking = {"type": "adaptive"}
            elif "claude" in model_name:
                thinking = {"type": "enabled", "budget_tokens": budget}
            else:
                thinking = None
            return thinking
        """,
    )
    assert [f.line for f in findings] == [3]
    assert "claude-sonnet-5-5" in findings[0].message


def test_complete_or_budget_only_gates_are_silent(lint):
    assert not lint(
        Rule,
        """
        def a(model, budget):
            if model.startswith(("claude-opus-5", "claude-sonnet-5")):
                return {"type": "adaptive"}
            return {"type": "enabled", "budget_tokens": budget}

        def b(model, budget):
            # Not aware of adaptive thinking: LDL005's territory when the model is known.
            if "claude" in model:
                return {"type": "enabled", "budget_tokens": budget}
            return None

        def c(model, budget, effort):
            if effort and model.startswith("claude-opus-5"):
                return {"type": "adaptive"}
            return {"type": "enabled", "budget_tokens": budget}
        """,
    )
