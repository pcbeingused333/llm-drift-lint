# Changelog

## 0.1.0 (2026-10-16)

First release. Six rules for the Anthropic Python SDK, each backed by a real upstream fix:

- LDL001 `anthropic-sampling-params`: non-default `temperature` / `top_p` / `top_k`, or `temperature` with `top_p`, on models that reject them.
- LDL002 `anthropic-sampling-kwargs-sdk1`: sampling parameters as keyword arguments on `anthropic>=1`.
- LDL003 `content-first-block-text`: `response.content[0].text` where block 0 can be a thinking block.
- LDL004 `forced-tool-choice`: `tool_choice` `tool`/`any` on models that reject forcing a tool.
- LDL005 `thinking-budget-tokens`: manual `budget_tokens` on models that take effort levels.
- LDL006 `low-max-tokens-thinking`: `max_tokens` ≤ 1024 on models that think by default.

Also: model-id resolution through constants, parameter defaults and `**params` dicts; project dependency versions from pyproject, requirements and uv/poetry lockfiles; text, JSON and GitHub annotation output; inline `# ldl: ignore[...]`; a pre-commit hook; a regression corpus of the real pre-/post-fix files.
