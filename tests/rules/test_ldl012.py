from llm_drift_lint.rules.ldl012_openai_gemini_sampling import OpenAIGeminiSampling as Rule


def test_openai_reasoning_models_reject_non_default_sampling(lint):
    # khoj#1377: temperature 0.8 on gpt-5.6-luna -> 400 "Only the default (1) value is supported".
    findings = lint(
        Rule,
        """
        TEMPERATURE = 0.8

        def chat(client, messages):
            client.chat.completions.create(model="gpt-5.6-luna", messages=messages, temperature=TEMPERATURE)
            client.chat.completions.create(model="o4-mini", messages=messages, top_p=0.9)
            client.responses.create(model="openai/gpt-5-mini", input=messages, temperature=0)
        """,
    )
    assert [(f.rule_id, f.line) for f in findings] == [("LDL012", 5), ("LDL012", 6), ("LDL012", 7)]
    assert "`temperature=0.8` is sent to gpt-5.6-luna" in findings[0].message
    assert 'reasoning effort "none"' in findings[0].hint
    assert "`top_p=0.9`" in findings[1].message
    assert "none" not in findings[2].hint
    assert {f.severity for f in findings} == {"error"}


def test_defaults_older_models_and_unknown_values_are_silent(lint):
    assert not lint(
        Rule,
        """
        def chat(client, messages, temperature):
            client.chat.completions.create(model="gpt-5.6-sol", messages=messages, temperature=1)
            client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0)
            client.chat.completions.create(model="gpt-5.1", messages=messages, temperature=0)
            client.chat.completions.create(model="gpt-5-chat-latest", messages=messages, temperature=0)
            client.chat.completions.create(model="o3", messages=messages, temperature=temperature)
            client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
        """,
    )


def test_reasoning_effort_none_allows_sampling_on_newer_gpt5(lint):
    findings = lint(
        Rule,
        """
        def chat(client, messages, effort):
            client.chat.completions.create(model="gpt-5.5", messages=messages, reasoning_effort="none", temperature=0)
            client.responses.create(model="gpt-5.6-terra", input=messages, reasoning={"effort": "none"}, temperature=0)
            client.chat.completions.create(model="gpt-5.6-sol", messages=messages, reasoning_effort=effort, temperature=0)
            client.chat.completions.create(model="gpt-5.6-sol", messages=messages, reasoning_effort="high", temperature=0)
            client.chat.completions.create(model="gpt-5", messages=messages, reasoning_effort="minimal", temperature=0)
        """,
    )
    assert [f.line for f in findings] == [6, 7]


def test_gemini_3_low_temperature(lint):
    findings = lint(
        Rule,
        """
        from google import genai
        from google.genai import types

        def ask(client, prompt):
            client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            config = {"temperature": 0}
            client.aio.models.generate_content_stream(model="gemini-3.6-flash", contents=prompt, config=config)
            # Gemini's OpenAI-compatible endpoint
            openai_client.chat.completions.create(model="gemini-3.8-flash", messages=m, temperature=0.1)
        """,
    )
    assert [f.line for f in findings] == [6, 12, 14]
    assert "looping" in findings[0].message and "400" not in findings[0].message
    assert {f.severity for f in findings} == {"warning"}


def test_gemini_default_and_older_models_are_silent(lint):
    assert not lint(
        Rule,
        """
        def ask(client, prompt):
            client.models.generate_content(model="gemini-3.8-flash", contents=prompt, config={"temperature": 1.0})
            client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config={"temperature": 0})
            client.models.generate_content(model="gemini-3.1-flash-image", contents=prompt, config={"temperature": 0})
            client.models.generate_content(model="gemini-3.8-flash", contents=prompt, config=build_config())
        """,
    )
