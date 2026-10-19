from llm_drift_lint.rules.ldl011_stop_to_end_turn import StopToEndTurn as Rule


def test_module_level_map_without_tool_check(lint):
    # The gateway pattern behind axonhub's fix: the map alone decides the stop_reason.
    findings = lint(
        Rule,
        """
        FINISH_TO_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}

        def to_anthropic(resp):
            choice = resp.choices[0]
            return {
                "content": convert(choice.message),
                "stop_reason": FINISH_TO_STOP.get(choice.finish_reason, "end_turn"),
            }
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL011", 2)]
    assert "tool_use" in findings[0].hint


def test_if_chain_and_gemini_native_stop(lint):
    findings = lint(
        Rule,
        """
        def stop_reason(finish_reason):
            if finish_reason == "stop":
                return "end_turn"
            elif finish_reason == "tool_calls":
                return "tool_use"
            return "max_tokens"

        def from_gemini(candidate):
            reason = "end_turn" if candidate.finish_reason.name == "STOP" else "max_tokens"
            return reason
        """,
    )
    assert [f.line for f in findings] == [3, 10]


def test_match_statement(lint):
    findings = lint(
        Rule,
        """
        def map_reason(reason):
            match reason:
                case "stop" | "eos":
                    return "end_turn"
                case "tool_calls":
                    return "tool_use"
        """,
    )
    assert [f.line for f in findings] == [4]


def test_converters_that_check_the_content_are_silent(lint):
    assert not lint(
        Rule,
        """
        FINISH_MAP = {"stop": "end_turn", "length": "max_tokens"}

        def a(choice, blocks):
            stop_reason = FINISH_MAP.get(choice.finish_reason, "end_turn")
            if any(b["type"] == "tool_use" for b in blocks):
                stop_reason = "tool_use"
            return stop_reason

        def b(choice):
            if choice.finish_reason == "stop":
                reason = "end_turn"
            if choice.message.tool_calls:
                reason = "tool_use"
            return reason

        def c(choice, tool_calls):
            return "tool_use" if tool_calls else {"stop": "end_turn"}.get(choice.finish_reason)

        def d(fr, has_tool_calls):
            if fr == "tool_calls" or has_tool_calls:
                return "tool_use"
            if fr == "stop":
                return "end_turn"
        """,
    )


def test_reverse_direction_and_unused_maps_are_silent(lint):
    assert not lint(
        Rule,
        """
        STOP_TO_FINISH = {"end_turn": "stop", "tool_use": "tool_calls"}
        EXPORTED = {"stop": "end_turn"}  # used by another module: can't see how

        def to_openai(msg):
            return STOP_TO_FINISH[msg.stop_reason]
        """,
    )


def test_block_types_are_not_an_override(lint):
    # open-webui's stream converter builds {"type": "tool_use"} blocks but maps "stop" to end_turn.
    findings = lint(
        Rule,
        """
        def convert_stream(chunks):
            for chunk in chunks:
                for tool in chunk.tool_calls or []:
                    state.current_block_type = "tool_use"
                    yield {"type": "content_block_start", "content_block": {"type": "tool_use", "id": tool.id}}
                if chunk.finish_reason is not None:
                    stop_reason = {"stop": "end_turn", "tool_calls": "tool_use"}.get(chunk.finish_reason, "end_turn")
        """,
    )
    assert [f.line for f in findings] == [8]


def test_helper_callers(lint):
    findings = lint(
        Rule,
        """
        def _to_stop_reason(fr):
            return {"stop": "end_turn", "tool_calls": "tool_use"}.get(fr, "end_turn")

        def convert(choice):
            return {"stop_reason": _to_stop_reason(choice.finish_reason)}

        class Adapter:
            def _map(self, fr):
                if fr == "stop":
                    return "end_turn"

            def convert(self, choice):
                reason = self._map(choice.finish_reason)
                if choice.message.tool_calls:
                    reason = "tool_use"
                return reason
        """,
    )
    assert [f.line for f in findings] == [3]
