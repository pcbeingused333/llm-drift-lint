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


def test_result_of_a_call_taking_the_index_is_not_the_index(lint):
    # pydantic-ai: the parts manager starts a new part when vendor_part_id is None and a tool name is
    # given, so Gemini works; `maybe_event is not None` is not a guard on the index.
    findings = lint(
        Rule,
        """
        def map_deltas(self, choice):
            for dtc in choice.delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function.name,
                    args=dtc.function.arguments,
                )
                if maybe_event is not None:
                    yield maybe_event
        """,
    )
    assert findings == []


def test_isinstance_checks_count_as_none_checks(lint):
    # ag2 mistral_client: a missing index gets a fresh slot.
    assert (
        lint(
            Rule,
            """
            def collect(chunks, accs):
                for chunk in chunks:
                    for position, tc in enumerate(chunk.choices[0].delta.tool_calls or []):
                        index = tc.index if isinstance(tc.index, int) else len(accs) + position
                        acc = accs.setdefault(index, {"arguments": ""})
                        acc["arguments"] += tc.function.arguments or ""
            """,
        )
        == []
    )
    findings = lint(
        Rule,
        """
        def collect(chunks, accs):
            for chunk in chunks:
                for tc in chunk.choices[0].delta.tool_calls or []:
                    if not isinstance(tc.index, int):
                        continue
                    accs[tc.index] = tc
        """,
    )
    assert [f.line for f in findings] == [4] and "skipped" in findings[0].message


def test_early_return_of_a_sent_index_is_not_a_drop(lint):
    # opik mistral aggregator: a missing index falls through to a fresh key.
    findings = lint(
        Rule,
        """
        def key(tool_call, calls):
            if isinstance(tool_call.index, int):
                return tool_call.index
            return min((k for k in calls if k < 0), default=0) - 1

        def collect(chunks, calls):
            for chunk in chunks:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    calls.setdefault(key(tool_call, calls), {}).update(tool_call)
        """,
    )
    assert findings == []
