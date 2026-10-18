from llm_drift_lint.rules.ldl008_sampling_gate import SamplingModelGate as Rule

# The test model table (conftest.py): opus-5-5 and sonnet-5-5 and haiku-5-5 reject sampling,
# sonnet-4-6 accepts it.


def test_exact_list_that_stops_early(lint):
    # gpt-researcher#2217: `if model not in NO_SUPPORT_TEMPERATURE_MODELS: kwargs["temperature"] = t`.
    findings = lint(
        Rule,
        """
        NO_SUPPORT_TEMPERATURE_MODELS = ["o1", "claude-sonnet-4-6", "claude-opus-5-5"]

        def kwargs_for(model, temperature):
            kwargs = {}
            if model not in NO_SUPPORT_TEMPERATURE_MODELS:
                kwargs["temperature"] = temperature
            return kwargs
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL008", 6)]
    message = findings[0].message
    assert "for claude-opus-5-5" in message
    assert "claude-sonnet-5-5" in message and "claude-haiku-5-5" in message
    assert "claude-sonnet-4-6" not in message


def test_prefix_gate_with_else_branch(lint):
    # fabric#2270 / cherry-studio#21939: the prefix list is missing newer families.
    findings = lint(
        Rule,
        """
        PREFIXES = ("claude-opus-5",)

        def build(model, temperature):
            if any(model.startswith(p) for p in PREFIXES):
                return {"model": model}
            else:
                return {"model": model, "temperature": temperature}
        """,
    )
    assert [f.line for f in findings] == [5]
    assert "for claude-opus-5-5" in findings[0].message
    assert "claude-sonnet-5-5, claude-haiku-5-5" in findings[0].message


def test_reject_list_used_in_another_module(lint):
    findings = lint(Rule, 'NO_TEMPERATURE_MODELS = ["gpt-5", "claude-opus-5-5"]\n')
    assert [f.line for f in findings] == [1]
    assert "`NO_TEMPERATURE_MODELS`" in findings[0].message


def test_complete_gates_and_unrelated_conditions_are_silent(lint):
    assert not lint(
        Rule,
        """
        def a(model, t):
            if model.startswith(("claude-opus-5", "claude-sonnet-5", "claude-haiku-5")):
                return {}
            return {"temperature": t}

        def b(model, t):
            # No model is exempt: the gate isn't about sampling.
            if "claude" in model:
                return {"temperature": t}
            return {}

        def c(provider, t):
            if provider == "claude-opus-5-5":
                return {}
            return {"temperature": t}

        def d(model, t):
            if model in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]:
                return {"temperature": None}
            return {"temperature": t}

        def e(model, t):
            if re.match(r"claude-(opus|sonnet)-5", model):
                return {}
            return {"temperature": t}

        SUPPORTS_TEMPERATURE = ["claude-sonnet-4-6"]
        NO_TEMPERATURE_MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-5"]
        """,
    )
