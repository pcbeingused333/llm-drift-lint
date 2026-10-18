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
                    if i is None:
                        i = len(calls)
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


def test_none_default_is_not_a_fallback(lint):
    findings = lint(
        Rule,
        """
        def collect(chunks):
            for chunk in chunks:
                for tc in chunk.choices[0].delta.tool_calls:
                    i = getattr(tc, "index", None)
                    calls[i] = tc
        """,
    )
    assert [f.line for f in findings] == [4]


def test_deltas_without_index_are_skipped(lint):
    # PraisonAI (TS): `if (tc.index !== undefined) { ... }` dropped every Gemini tool call.
    findings = lint(
        Rule,
        """
        def collect(chunks):
            calls = []
            for chunk in chunks:
                for tc in chunk.choices[0].delta.tool_calls or []:
                    if tc.index is not None:
                        if tc.index >= len(calls):
                            calls.append({"id": tc.id, "arguments": ""})
                        calls[tc.index]["arguments"] += tc.function.arguments or ""
            return calls
        """,
    )
    assert [f.line for f in findings] == [5]
    assert "dropped" in findings[0].message


def test_position_in_chunk_fallback_through_a_helper(lint):
    # nanobot: _accum_tc(tc, idx) with idx from enumerate(delta.tool_calls) is 0 in every Gemini chunk.
    findings = lint(
        Rule,
        """
        def parse(chunks):
            bufs = {}

            def _accum_tc(tc, idx_hint):
                tc_index = _get(tc, "index") if _get(tc, "index") is not None else idx_hint
                buf = bufs.setdefault(tc_index, {"id": "", "arguments": ""})
                buf["arguments"] += _get(_get(tc, "function"), "arguments") or ""

            for chunk in chunks:
                delta = chunk["choices"][0]["delta"]
                for idx, tc in enumerate(cast(Iterable[object], delta.get("tool_calls") or [])):
                    _accum_tc(tc, idx)
            return bufs
        """,
    )
    assert [f.line for f in findings] == [12]
    assert "position in the chunk" in findings[0].message


def test_helper_with_a_real_fallback_is_silent(lint):
    assert not lint(
        Rule,
        """
        def parse(chunks):
            bufs = {}

            def _accum_tc(tc):
                tc_index = _get(tc, "index")
                if tc_index is None:
                    tc_index = _unindexed_key(tc)
                bufs.setdefault(tc_index, {})

            for chunk in chunks:
                for tc in chunk["choices"][0]["delta"].get("tool_calls") or []:
                    _accum_tc(tc)
        """,
    )
