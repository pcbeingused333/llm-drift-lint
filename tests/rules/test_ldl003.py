from llm_drift_lint.rules.ldl003_content_first_block_text import ContentFirstBlockText as Rule


def test_reward_bench_judge_with_a_model_that_thinks_by_default(lint):
    # reward-bench#290, inside the retry loop of chat_completion_anthropic.
    findings = lint(
        Rule,
        """
        import anthropic

        def chat_completion_anthropic(conv, max_tokens):
            c = anthropic.Anthropic()
            for _ in range(API_MAX_RETRY):
                try:
                    response = c.messages.create(
                        model="claude-sonnet-5-5",
                        messages=conv.messages,
                        max_tokens=max_tokens,
                    )
                    output = response.content[0].text
                    break
                except anthropic.APIError as e:
                    print(type(e), e)
            return output
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL003", 13)]
    assert "response.content[0].text" in findings[0].message
    assert "claude-sonnet-5-5" in findings[0].message


def test_thinking_enabled_in_the_request_needs_no_model(lint):
    findings = lint(
        Rule,
        """
        async def ask(client, model, prompt):
            resp = await client.messages.create(
                model=model,
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 4000},
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        def ask_adaptive(client, model):
            r = client.beta.messages.create(model=model, max_tokens=8, extra_body={"thinking": {"type": "adaptive"}})
            return r.content[0].text
        """,
    )
    assert [f.line for f in findings] == [9, 13]


def test_chained_access_and_dict_form(lint):
    # hostedgpt#823 read dig("content", 0, "text") from the response hash; this is its Python shape.
    findings = lint(
        Rule,
        """
        text = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m).content[0].text
        data = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m).model_dump()
        title = data["content"][0]["text"]
        """,
    )
    assert [f.line for f in findings] == [2, 4]


def test_unresolved_or_non_thinking_models_are_silent(lint):
    assert not lint(
        Rule,
        """
        def judge(c, model):
            response = c.messages.create(model=model, messages=m, max_tokens=8)
            return response.content[0].text

        response = client.messages.create(model="claude-sonnet-4-6", max_tokens=8, messages=m)
        print(response.content[0].text)
        """,
    )


def test_thinking_disabled_unknown_or_hidden_is_silent(lint):
    assert not lint(
        Rule,
        """
        r1 = client.messages.create(model="claude-sonnet-5-5", max_tokens=8, thinking={"type": "disabled"})
        r1.content[0].text
        r2 = client.messages.create(model="claude-sonnet-5-5", max_tokens=8, thinking=thinking_config)
        r2.content[0].text
        r3 = client.messages.create(model="claude-sonnet-5-5", max_tokens=8, **request_kwargs)
        r3.content[0].text
        r4 = client.messages.create(model="claude-sonnet-5-5", max_tokens=8, extra_body=body)
        r4.content[0].text
        """,
    )


def test_checking_only_block_0_is_not_a_guard(lint):
    # phoenix#17026/#17027 (generate_text): block 0 is checked, then read, so a leading thinking block
    # still loses the answer ("unexpected content format").
    findings = lint(
        Rule,
        """
        def generate_text(self, messages):
            response = self.client.messages.create(model="claude-opus-5-5", messages=messages, max_tokens=4096)
            if hasattr(response.content[0], "text"):
                return cast(str, response.content[0].text)
            raise ValueError(f"Anthropic returned unexpected content format: {response.content}")

        def by_type(client):
            response = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m)
            if response.content[0].type == "text":
                return response.content[0].text

        def by_isinstance(client):
            response = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m)
            if isinstance(response.content[0], TextBlock):
                return response.content[0].text
        """,
    )
    assert [f.line for f in findings] == [5, 11, 16]


def test_looking_beyond_block_0_is_a_guard(lint):
    assert not lint(
        Rule,
        """
        def fixed(client):
            response = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m)
            return next((b.text for b in response.content if b.type == "text"), "")

        def loop(client):
            response = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m)
            for block in response.content:
                log(block)
            return response.content[0].text

        def other_index(client):
            response = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m)
            if len(response.content) > 1:
                return response.content[-1].text
            return response.content[0].text

        def helper(client):
            response = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m)
            check_blocks(blocks=response.content)
            return response.content[0].text
        """,
    )


def test_name_bound_to_something_else_is_silent(lint):
    assert not lint(
        Rule,
        """
        def rebound(client):
            response = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m)
            response = convert(response)
            return response.content[0].text

        def other_function():
            return response.content[0].text

        def other_api(client):
            response = client.chat.completions.create(model="claude-opus-5-5", messages=m)
            return response.content[0].text

        def stream(client):
            with client.messages.stream(model="claude-opus-5-5", max_tokens=8, messages=m) as s:
                msg = s.get_final_message()
            return msg.content[0].text

        def wrong_shape(client):
            response = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m)
            return response["content"][0]["text"], response.content[1].text, response.content[0].text_len
        """,
    )


def test_binding_after_the_read_does_not_count(lint):
    assert not lint(
        Rule,
        """
        def f(client):
            print(response.content[0].text)
            response = client.messages.create(model="claude-opus-5-5", max_tokens=8, messages=m)
        """,
    )
