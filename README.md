# llm-drift-lint

Static checks for LLM API calls that break when providers change model behavior.

LLM providers change what their models accept faster than client code changes. Every rule here comes from a bug that shipped in a real open-source project and was fixed upstream, often after it had already broken every request for users on current models:

- `temperature: 0.0` sent by default to models that now reject it;
- `response.content[0].text` read when the first block is now a thinking block;
- forced `tool_choice` sent to models that reject it while thinking;
- manual `budget_tokens` sent to models that only take effort levels.

The linter reads your Python source, finds the SDK calls, resolves the model id when it's a literal or a constant, and checks the call against what that model accepts.

> Status: v0.1. Rules target the Anthropic Python SDK (direct, Bedrock and Vertex model ids).

## Rules

| Id | Name | Flags | Real-world fix that motivated it |
|---|---|---|---|
| LDL001 | `anthropic-sampling-params` | a non-default `temperature` / `top_p` / `top_k` (or `temperature` and `top_p` together) sent to a model that rejects them | [langchainrb#1055](https://github.com/patterns-ai-core/langchainrb/pull/1055), [WeKnora#4155](https://github.com/Tencent/WeKnora/pull/4155), [trulens#2948](https://github.com/truera/trulens/pull/2948) |
| LDL002 | `anthropic-sampling-kwargs-sdk1` | sampling parameters passed as keyword arguments to `messages.create()` in a project on `anthropic>=1`, where they are no longer keyword arguments | [phoenix#17026](https://github.com/Arize-ai/phoenix/pull/17026) |
| LDL003 | `content-first-block-text` | `response.content[0].text`, where block 0 can be a thinking block | [reward-bench#290](https://github.com/allenai/reward-bench/pull/290), [hostedgpt#823](https://github.com/AllYourBot/hostedgpt/pull/823) |
| LDL004 | `forced-tool-choice` | `tool_choice={"type": "tool" \| "any"}` sent to a model that rejects a forced tool choice | [phoenix#17027](https://github.com/Arize-ai/phoenix/pull/17027), [ha-llmvision#747](https://github.com/valentinfrlch/ha-llmvision/pull/747) |
| LDL005 | `thinking-budget-tokens` | `thinking={"type": "enabled", "budget_tokens": N}` sent to a model that takes effort levels instead | [chatlas#448](https://github.com/posit-dev/chatlas/pull/448) |
| LDL006 | `low-max-tokens-thinking` | `max_tokens` ≤ 1024 sent to a model that thinks by default, so thinking can use the whole budget and the answer comes back empty | [langchainrb#1073](https://github.com/patterns-ai-core/langchainrb/pull/1073), [livekit/agents#7726](https://github.com/livekit/agents/issues/7726) |

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
```

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

## License

MIT
