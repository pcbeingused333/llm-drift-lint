from llm_drift_lint.rules.ldl004_forced_tool_choice import ForcedToolChoice as Rule


def test_phoenix_structured_output_forces_the_tool(lint):
    # phoenix#17027: AnthropicAdapter._generate_with_tool_calling before the fix.
    findings = lint(
        Rule,
        """
        MODEL = "claude-opus-5-5"

        class AnthropicAdapter:
            def _generate_with_tool_calling(self, messages, tool_definition, **kwargs):
                response = self.client.messages.create(
                    model=MODEL,
                    messages=messages,
                    tools=[tool_definition],
                    tool_choice={"type": "tool", "name": "extract_structured_data"},
                )
                return response
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL004", 6)]
    assert '"tool"' in findings[0].message and "claude-opus-5-5" in findings[0].message


def test_any_and_non_literal_tool_name(lint):
    findings = lint(
        Rule,
        """
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, tools=t, tool_choice={"type": "any"})
        client.messages.parse(
            model="claude-sonnet-5-5", max_tokens=8, tools=t, tool_choice={"type": "tool", "name": tool.name}
        )
        client.messages.create(model="claude-sonnet-5-5", max_tokens=8, extra_body={"tool_choice": {"type": "any"}})
        """,
    )
    assert [f.line for f in findings] == [2, 3, 6]


def test_auto_none_and_unknown_choices_are_silent(lint):
    assert not lint(
        Rule,
        """
        client.messages.create(model="claude-opus-5-5", max_tokens=8, tools=t, tool_choice={"type": "auto"})
        client.messages.create(model="claude-opus-5-5", max_tokens=8, tools=t, tool_choice={"type": "none"})
        client.messages.create(model="claude-opus-5-5", max_tokens=8, tools=t, tool_choice=self._tool_choice())
        client.messages.create(model="claude-opus-5-5", max_tokens=8, tools=t, tool_choice=choice)
        """,
    )


def test_models_that_accept_forcing_or_unresolved_are_silent(lint):
    assert not lint(
        Rule,
        """
        client.messages.create(model="claude-haiku-5-5", max_tokens=8, tools=t, tool_choice={"type": "tool", "name": "x"})
        client.messages.create(model="claude-sonnet-4-6", max_tokens=8, tools=t, tool_choice={"type": "any"})
        client.messages.create(model=self.model, max_tokens=8, tools=t, tool_choice={"type": "any"})
        client.messages.count_tokens(model="claude-opus-5-5", tools=t, tool_choice={"type": "any"}, messages=m)
        client.chat.completions.create(model="claude-opus-5-5", tool_choice={"type": "tool", "name": "x"})
        """,
    )
