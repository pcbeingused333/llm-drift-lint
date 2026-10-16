# Rules evidence

What backs `models.py` and the regression corpus, where the sources disagree, and
what that means for each rule. Checked on 2026-10-16. Owned by the data session.

## Sources

Anthropic's documentation, in order of weight:

- [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking): the per-model
  table of what each `thinking` value does (including "No `thinking` field"), and the
  "Sampling parameters" section.
- [Forcing tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use):
  the table of models that reject `tool_choice` `any`/`tool`.
- [Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations),
  "API parameter deprecations": sampling parameters on 4.7+ models, and the Python SDK 1.x
  `TypeError`.
- Model ids: [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview),
  [Claude in Amazon Bedrock](https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock),
  [Bedrock (Opus 4.6 and earlier)](https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy),
  [Google Cloud](https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai).

The PR descriptions (read with `gh pr view`) are secondary: they state what the author
believed or tested, and several were written without a live API call (Phoenix #17027,
TruLens #2948, ha-llmvision #747, chatlas #448 and langchainrb #1073 say so).

## Traits by model

| Model | `rejects_sampling` | `rejects_forced_tool_choice` | `thinks_by_default` | `rejects_budget_tokens` |
|---|---|---|---|---|
| Fable 5.1, Mythos 5.1, Opus 5.5, Sonnet 5.5 | non_default | yes | yes | yes |
| Fable 5, Mythos 5, Opus 5, Sonnet 5, Haiku 5.5 | non_default | no | yes | yes |
| Mythos Preview | non_default | no (unknown) | yes | no |
| Opus 4.8, Opus 4.7 | non_default | no | no | yes |
| Opus/Sonnet 4.6 and older | — | no | no | no |
| anything else (`claude-opus-5-6`, `claude-haiku-5`, non-Claude) | unknown: `traits()` returns `None` | | | |

`SAMPLING_DEFAULTS` = `{"temperature": 1.0, "top_p": 0.99}`.

## Disagreements and decisions

1. **"all" vs "non_default" sampling.** Two PRs say Opus 4.7 and later reject the sampling
   parameters with *any* value:
   - langchainrb#1055: "Claude Opus 4.7 and later (Opus 4.8, Opus 5, Opus 5.5) don't accept
     `temperature`, `top_p` or `top_k` at all. Any value returns a 400." and "Claude Sonnet
     5, Sonnet 5.5 and Haiku 5.5 return a 400 for any non-default value."
   - TruLens#2948: "Claude Opus 4.7 and later (Opus 4.8, Opus 5, Opus 5.5, Fable 5 / 5.1)
     reject `temperature` with a 400. Claude Sonnet 5 / 5.5 and Haiku 5.5 reject any
     non-default value, and 0.0 is not the default."

   WeKnora#4155 is often grouped with them, but it says *non-default*: "Claude Opus 4.7
   and later (Opus 4.8, Opus 5, Fable 5 and Fable 5.1) and Claude Sonnet 5 reject
   non-default values for `temperature`, `top_p` and `top_k` with a 400", citing the
   deprecations page. Anthropic's documentation says non-default on every model:
   - [Thinking → Sampling parameters](https://platform.claude.com/docs/en/build-with-claude/thinking):
     "On Claude Fable 5.1, Claude Mythos 5.1, Claude Fable 5, Claude Mythos 5, Claude
     Mythos Preview, Claude Opus 5.5, Claude Opus 5, Claude Opus 4.8, Claude Opus 4.7,
     Claude Sonnet 5.5, Claude Sonnet 5, and Claude Haiku 5.5, non-default `temperature`,
     `top_p`, or `top_k` values return a 400 error on every request, regardless of whether
     thinking is used."
   - [Model deprecations → API parameter deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations),
     row `temperature`, `top_p`, `top_k`: "Deprecated (Claude 4.7 and later models)" /
     "Returns a 400 error when set to a non-default value on Claude 4.7 and later models
     and Claude Mythos Preview."

   **Decision:** `non_default` for all of them; `"all"` is unused. Every source agrees
   that non-default values fail, so this choice can't add a false positive. It can miss
   one case: if "any value" is right, `temperature=1.0` sent to an Opus is still a 400.
   No source says it tested `temperature=1.0` on an Opus against the live API.
2. **What "default" means.** The contract says `non_default` means "any value other than
   1.0". The thinking docs say `top_p`'s default is **0.99** on these models, so `top_p=1`
   is a 400, and so is a request that sets **both** `temperature` and `top_p`, even at
   their defaults. No default is documented for `top_k`. **Decision:** `SAMPLING_DEFAULTS`
   in `models.py`; any `top_k` counts as non-default. → RULES: LDL001 should read
   `SAMPLING_DEFAULTS` instead of hard-coding 1.0, and could flag temperature + top_p
   together.
3. **`budget_tokens` on 4.6.** chatlas#448 says fixed budgets "only work on models before
   Claude Opus 4.6 / Sonnet 4.6". The thinking table says Opus/Sonnet 4.6 still accept them
   ("Extended thinking (deprecated)"). **Decision:** `rejects_budget_tokens=False` for 4.6.
4. **Mythos Preview.** It isn't in the forced-tool-use table, and nothing says whether it
   rejects forced `tool_choice`. **Decision:** left at the default (`False`). It still accepts
   `budget_tokens` per the thinking table.
5. **Model families vs exact ids.** TruLens#2948 matches whole families
   (`claude-opus-5`, `claude-sonnet-5`, `claude-fable-5` prefixes) to cover future 5.x
   releases, and mentions a `claude-haiku-5` that no Anthropic page lists. **Decision:**
   exact ids only, so a future `claude-opus-5-6` is unknown (`None`) and rules stay silent.
   A miss is better than a false positive in v0.1, and a new model needs a new table row.
6. **Vertex id format.** COORDINATION.md's example `claude-...@vertex` isn't a real
   format. Google Cloud uses the bare id from the 4.6 generation on (`claude-opus-5-5`) and
   `@YYYYMMDD` for older snapshots (`claude-opus-4-5@20251101`). Both are handled.
7. **Bedrock ids.** The Messages-API Bedrock endpoint uses `anthropic.claude-opus-5-5` (4.7
   and later). The legacy InvokeModel ids are `anthropic.claude-...-v1:0` or
   `anthropic.claude-opus-4-6-v1`, with `global.`/`us.`/`eu.`/`jp.`/`apac.` profile prefixes.
   chatlas's default `us.anthropic.claude-sonnet-5` combines the two styles. The docs don't
   list it, but the patterns match it, so its traits come from `claude-sonnet-5`.

Not encoded, but worth a future rule: `thinking: {"type": "disabled"}` returns a 400 on
Opus 5.5, Sonnet 5.5, Fable 5.x and Mythos 5.x (thinking table; ha-llmvision#747), and on Opus 5 / Haiku 5.5 only at effort `xhigh`/`max`.

## Corpus coverage per rule

Python cases are in `tests/corpus/` (see its README). "Required" is what a v0.1 linter
following the contract must report; "optional" is a real bug the linter may report but
doesn't have to.

| Rule | Required | Optional (why it isn't required) |
|---|---|---|
| LDL001 sampling params | — | reward-bench#290 (model is a function argument), TruLens#2948 (request dict, model is `self.model_engine`) |
| LDL002 sampling kwargs, SDK 1.x | — | Phoenix#17026 (params arrive via `**kwargs`), reward-bench#290 (`anthropic` unpinned, not `>=1`) |
| LDL003 `content[0].text` | Phoenix (6 across its three cases; the fix is the still-open #16005), reward-bench#290 (3) | — |
| LDL004 forced tool_choice | — | Phoenix#17027 (`model=self.model`), TruLens#2948 (request dict) |
| LDL005 budget_tokens | — | chatlas#448 (config built in a dict, reaches `messages.create(**kwargs)`) |
| LDL006 low max_tokens | — | livekit/agents#7726 (default set in a dict; default model doesn't think) |

**What this shows:** in every real bug except `content[0].text`, the model or the
parameter reaches `messages.create()` through something v0.1 doesn't resolve:
`self.model` / `self._opts.model`, a class constant (`DEFAULT_MODEL_ENGINE`), a
call such as `log_model_default("us.anthropic.claude-sonnet-5")`, a request dict built
up key by key, or `**kwargs`. Against real code, v0.1 will catch LDL003 and little else.
The post-fix files and ha-llmvision (raw HTTP payloads full of `temperature`,
`budget_tokens` and `tool_choice` keys for a dozen providers) are mostly false-positive
checks. → LEAD: resolving attribute defaults (constructor default arguments, class
constants) and dict-literal `**kwargs` would turn several optional findings into required
ones.

Notes for the rules:

- **LDL002** has to read `[project.optional-dependencies]` too: phoenix-evals declares
  `anthropic>=1,<2` only there (`pyproject.toml` lines 38 and 50). reward-bench lists
  `anthropic` unpinned, and a fresh install gets 1.x, but the contract only covers
  "requires `anthropic>=1`". A case without its own `pyproject.toml` (chatlas,
  ha-llmvision) finds the llm-drift-lint `pyproject.toml` if a rule walks all the way up.
- **LDL003**: Phoenix guards the access with `if hasattr(response.content[0], "text")`.
  It doesn't crash, but it raises "unexpected content format" on a leading thinking
  block, so it still counts. The `hasattr(...)` line itself (93/120) isn't listed: the
  rule is about `.content[0].text`.
- **LDL004**: Phoenix#17027's fix still returns `{"type": "tool", ...}` from
  `_tool_choice()` for models that accept it, and chatlas's fix still builds
  `budget_tokens` for an int (LDL005). A dict literal that isn't the `tool_choice=` /
  `thinking=` argument must not be flagged.
- **LDL006**: livekit/agents#7726 measured empty replies at `max_tokens=1024` on Sonnet 5
  (5 of 5 direct API runs) and Haiku 5.5 (4 of 5). A threshold of "< 1024" wouldn't flag
  1024. → RULES: consider `<= 1024` or a higher threshold.

## Motivating PRs not in the corpus

The linter only parses Python, so these have no corpus case. Their claims are used
above as secondary sources.

| PR | Language | What it fixed |
|---|---|---|
| [patterns-ai-core/langchainrb#1055](https://github.com/patterns-ai-core/langchainrb/pull/1055) | Ruby | default `temperature: 0.0` sent to every model (LDL001) |
| [patterns-ai-core/langchainrb#1073](https://github.com/patterns-ai-core/langchainrb/pull/1073) | Ruby | default `max_tokens: 256` (LDL006) |
| [AllYourBot/hostedgpt#823](https://github.com/AllYourBot/hostedgpt/pull/823) | Ruby | `dig("content", 0, "text")` (LDL003) |
| [Tencent/WeKnora#4155](https://github.com/Tencent/WeKnora/pull/4155) | Go | catalog sent `temperature`/`top_p` to models that reject them (LDL001) |

## False-positive scan

`scripts/fp_scan.py` runs every registered rule over the repos in
`~/.cache/claude-scratch/repos/` (19 repos, about 12,800 non-test Python files) and writes
`~/.cache/claude-scratch/ldl-fp.txt`. On 2026-10-16, after merging main (LDL001–LDL006
registered, LDL003 still v1), it reported 0 findings. Rerun it after each rule change.

A throwaway `.content[0].text` probe (not committed) flagged
`libs/partners/anthropic/langchain_anthropic/llms.py:288` and `:325` in langchain
(`c088db9`). Those lines may be real LDL003 bugs; nobody has checked them yet.
