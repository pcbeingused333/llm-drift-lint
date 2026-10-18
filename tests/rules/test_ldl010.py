from llm_drift_lint.rules.ldl010_stream_tool_call_index import StreamToolCallIndex as Rule


def test_reducer_keyed_by_index(lint):
    # LLPhant / neurolink: Gemini sends each parallel call whole, without `index`.
    findings = lint(
        Rule,
        """
        async def collect(stream):
            calls = {}
            async for chunk in stream:
                for tc in chunk.choices[0].delta.tool_calls or []:
                    entry = calls.setdefault(tc.index, {"id": tc.id, "name": "", "arguments": ""})
                    entry["arguments"] += tc.function.arguments or ""
            return calls
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL010", 5)]
    assert "`tc.index`" in findings[0].message


def test_dict_chunks_and_list_index(lint):
    findings = lint(
        Rule,
        """
        def collect(chunks):
            calls = []
            for chunk in chunks:
                delta = chunk["choices"][0]["delta"]
                for call in delta.get("tool_calls", []):
                    if call["index"] >= len(calls):
                        calls.append(call)
                    else:
                        calls[call["index"]]["function"]["arguments"] += call["function"]["arguments"]
            return calls
        """,
    )
    assert [f.line for f in findings] == [6]


def test_reducers_that_handle_a_missing_index_are_silent(lint):
    assert not lint(
        Rule,
        """
        def a(chunks):
            calls = {}
            for chunk in chunks:
                for tc in chunk.choices[0].delta.tool_calls or []:
                    key = tc.index if tc.index is not None else len(calls)
                    calls.setdefault(key, tc)

        def b(chunks):
            for chunk in chunks:
                for tc in chunk.choices[0].delta.tool_calls:
                    i = getattr(tc, "index", None)
                    calls[i] = tc

        def c(chunks):
            for chunk in chunks:
                for tc in chunk.choices[0].delta.tool_calls:
                    calls[tc.index or len(calls)] = tc

        def d(message):
            # Not a stream delta: a complete message has whole calls.
            for tc in message.tool_calls:
                calls[tc.index] = tc

        def e(chunks):
            for chunk in chunks:
                for tc in chunk.choices[0].delta.tool_calls:
                    print(tc.index)
        """,
    )
