# llm-drift-lint

Static checks for LLM API calls that break when providers change model behavior.

LLM providers change what their models accept faster than client code changes. Every rule here comes from a bug that shipped in a real open-source project and was fixed upstream, often after it had already broken every request for users on current models:

- `temperature: 0.0` sent by default to models that now reject it;
- `response.content[0].text` read when the first block is now a thinking block;
- forced `tool_choice` sent to models that reject it while thinking;
- manual `budget_tokens` sent to models that only take effort levels;
- a default model that its provider has already shut down, or will next week;
- streamed tool calls merged by an `index` that Gemini never sends.

The linter reads your Python source, finds the SDK calls, resolves the model id when it's a literal or a constant, and checks the call against what that model accepts.

> Status: v0.1 targets the Anthropic Python SDK (direct, Bedrock and Vertex model ids). The unreleased rules LDL007–LDL012 extend it to OpenAI and Gemini: retired default models, OpenAI-compatible stream reducers, OpenAI→Anthropic gateways and OpenAI/Gemini sampling parameters.

## Rules

| Id | Name | Flags | Real-world fix that motivated it |
|---|---|---|---|
| LDL001 | `anthropic-sampling-params` | a non-default `temperature` / `top_p` / `top_k` (or `temperature` and `top_p` together) sent to a model that rejects them | [langchainrb#1055](https://github.com/patterns-ai-core/langchainrb/pull/1055), [WeKnora#4155](https://github.com/Tencent/WeKnora/pull/4155), [trulens#2948](https://github.com/truera/trulens/pull/2948) |
| LDL002 | `anthropic-sampling-kwargs-sdk1` | sampling parameters passed as keyword arguments to `messages.create()` in a project on `anthropic>=1`, where they are no longer keyword arguments | [phoenix#17026](https://github.com/Arize-ai/phoenix/pull/17026) |
| LDL003 | `content-first-block-text` | `response.content[0].text`, where block 0 can be a thinking block | [reward-bench#290](https://github.com/allenai/reward-bench/pull/290), [hostedgpt#823](https://github.com/AllYourBot/hostedgpt/pull/823) |
| LDL004 | `forced-tool-choice` | `tool_choice={"type": "tool" \| "any"}` sent to a model that rejects a forced tool choice | [phoenix#17027](https://github.com/Arize-ai/phoenix/pull/17027), [ha-llmvision#747](https://github.com/valentinfrlch/ha-llmvision/pull/747) |
| LDL005 | `thinking-budget-tokens` | `thinking={"type": "enabled", "budget_tokens": N}` sent to a model that takes effort levels instead | [chatlas#448](https://github.com/posit-dev/chatlas/pull/448) |
| LDL006 | `low-max-tokens-thinking` | `max_tokens` ≤ 1024 sent to a model that thinks by default, so thinking can use the whole budget and the answer comes back empty | [langchainrb#1073](https://github.com/patterns-ai-core/langchainrb/pull/1073), [livekit/agents#7726](https://github.com/livekit/agents/issues/7726) |
| LDL007 | `retired-model-default` | a default model (parameter default, `*MODEL*` constant, env-var or CLI fallback) that OpenAI, the Claude API, the Gemini API or Vertex AI has shut down or has scheduled for shutdown; Vertex's own Gemini dates apply only to code that targets Vertex | [OpenAI](https://developers.openai.com/api/docs/deprecations), [Claude](https://platform.claude.com/docs/en/about-claude/model-deprecations), [Gemini API](https://ai.google.dev/gemini-api/docs/deprecations) and [Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) deprecation pages |
| LDL008 | `incomplete-sampling-gate` | a hand-written Claude model check that drops `temperature` / `top_p` / `top_k` for some of the models that reject them, but not all | [gpt-researcher#2217](https://github.com/assafelovic/gpt-researcher/pull/2217), [fabric#2270](https://github.com/danielmiessler/fabric/pull/2270), [cherry-studio#21939](https://github.com/CherryHQ/cherry-studio/pull/21939), [kiss-translator#1185](https://github.com/fishjar/kiss-translator/pull/1185) |
| LDL009 | `incomplete-thinking-gate` | a hand-written Claude model check that still sends `budget_tokens` to models that only take adaptive thinking | [fabric#2270](https://github.com/danielmiessler/fabric/pull/2270) |
| LDL010 | `stream-tool-call-index` | a streamed tool-call reducer keyed by `tool_calls[].index` with no usable fallback; Gemini's OpenAI-compatible endpoint omits `index`, so parallel calls collapse into one or are dropped | [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai); upstream fixes in review |
| LDL011 | `stop-to-end-turn` | an OpenAI→Anthropic converter that maps finish reason `"stop"` to `stop_reason: "end_turn"` without checking for tool calls; Gemini ends tool-call turns with `"stop"`, so Anthropic SDK tool runners run no tools | [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai); found in OpenAI→Anthropic gateways |
| LDL012 | `openai-gemini-sampling` | a non-default `temperature` / `top_p` sent to OpenAI o-series, gpt-5, gpt-5.5 or gpt-5.6 models (400 unless gpt-5.5+ runs with reasoning effort `"none"`), or a `temperature` below 1.0 sent to a Gemini 3 model (Google warns of looping) | [khoj#1377](https://github.com/khoj-ai/khoj/issues/1377), [openless#888](https://github.com/Open-Less/openless/pull/888), [Gemini 3 guide](https://ai.google.dev/gemini-api/docs/gemini-3) |

`llm-drift-lint --list-rules` prints the rules in the installed version.

**False positives are treated as bugs.** When a rule needs to know the model and can't tell statically which one a call uses, it stays silent. LDL003 is the exception: reading `content[0].text` is fragile on every current Claude model, so it only stays silent when the request visibly disables thinking or the model is known not to think by default.

## Does it work?

`tests/corpus/` holds the real files from the upstream fixes above, as they were just before and just after each fix, plus the findings a correct linter must report. The test suite checks all of them. On 16 large open-source code bases (langchain, llama_index, crewAI, dspy, agno, langgraph, semantic-kernel, autogen and others) v0.1 reports exactly two findings. Both are real bugs: `AnthropicLLM` in `langchain-anthropic` returns `response.content[0].text` and breaks on models that think by default (reported as [langchain-ai/langchain#41263](https://github.com/langchain-ai/langchain/issues/41263)).

## Install

```console
pip install git+https://github.com/pcbeingused333/llm-drift-lint
```

## Use

```console
llm-drift-lint src/                       # text report, exit code 1 if anything is found
llm-drift-lint src/ --format json         # machine-readable
llm-drift-lint src/ --select LDL001,LDL004
llm-drift-lint src/ --ignore LDL006
llm-drift-lint src/ --include-tests       # test directories are skipped by default
llm-drift-lint src/ --fail-on error       # exit 0 when there are only warnings
```

Findings are errors, except LDL007 on a model whose shutdown date is still ahead, which is a warning until that day. Warnings fail the run too unless you pass `--fail-on error`.

Silence one line:

```python
client.messages.create(model=MODEL, temperature=0)  # ldl: ignore[LDL001]
```

or put `# ldl: ignore[LDL001]` alone on the line above. A bare `# ldl: ignore` silences every rule on that line.

### In GitHub Actions

`--format github` emits workflow annotations, so findings show up on the PR diff:

```yaml
- run: pip install git+https://github.com/pcbeingused333/llm-drift-lint
- run: llm-drift-lint src/ --format github
```

### As a pre-commit hook

```yaml
- repo: https://github.com/pcbeingused333/llm-drift-lint
  rev: v0.1.0
  hooks:
    - id: llm-drift-lint
```

## How model knowledge is kept

What each model accepts lives in one table, `src/llm_drift_lint/models.py`, and every entry cites its source: provider documentation or the upstream PR that hit the problem. Rules never hard-code model names. When a provider changes behavior again, updating the linter means adding a row and a test.

Retirement dates live there too (`OPENAI_SHUTDOWNS`, `CLAUDE_RETIREMENTS`, `GEMINI_SHUTDOWNS`, `VERTEX_GEMINI_RETIREMENTS`), each with the date the provider page was read. LDL007 compares them with today's date, so the same default is reported as "shuts down on …" before the date and "shut down on …, so requests fail" after it.

## License

MIT
