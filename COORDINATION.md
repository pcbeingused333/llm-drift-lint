# Coordination: 3 Claude sessions working in parallel

Project: `llm-drift-lint`, a static linter for LLM API calls that break when providers change model behavior. It's Alex's own project (MIT, github.com/pcbeingused333, not published yet). Background and the real bugs that motivate each rule: `~/projects/job-search/plan-4-puntos/03-articulo-claude5-drift.md`.

## Sessions and ownership

| Session | Owns (only edits these) | Works on branch / worktree |
|---|---|---|
| **LEAD** (the original session) | `src/llm_drift_lint/core.py`, `cli.py`, `reporting.py`, `pyproject.toml`, `README.md`, `tests/test_core.py`, `tests/test_cli.py`, CI | `main` in `~/projects/llm-drift-lint` |
| **RULES** | `src/llm_drift_lint/rules/*.py` (one module per rule) and `src/llm_drift_lint/rules/__init__.py`, `tests/rules/` | branch `rules` in worktree `~/projects/llm-drift-lint-rules` |
| **DATA** | `src/llm_drift_lint/models.py`, `tests/corpus/`, `tests/test_corpus.py`, `docs/rules-evidence.md` | branch `data` in worktree `~/projects/llm-drift-lint-data` |

Rules:
- Never edit files another session owns. If you need a change there (e.g. a new helper in `core.py`), write it under **Requests** below and keep going with a local workaround. The lead merges.
- Commit often on your own branch. **Only the lead merges into `main`.** Merge `main` into your branch when the lead announces a change below (no rebases, no force-push).
- Don't push to GitHub and don't publish anything: Alex decides that.
- Commit identity: `git config user.name "Alex Castillo"`, `user.email alex.castillog33@gmail.com`; end commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Python env: `cd <your worktree> && export PATH=~/tools/wsl-guard:$PATH && source ~/tools/wsl-guard/shell/guard.zsh && uv venv -q .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -q -e ".[dev]"`. Run `.venv/bin/python -m pytest -q` and `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests` before each commit.
- Write long command output to `~/.cache/claude-scratch/`, not `/tmp`.

## The contract (core.py, frozen for v0.1)

- `FileContext.parse(source, path)` → context or `None` on a syntax error. `ctx.calls(*suffixes)` yields `CallSite`s, in source order, whose dotted target (callee chain minus its root variable; `self.client.messages.create` → `client.messages.create`) ends with a suffix, e.g. `ANTHROPIC_MESSAGE_CALLS`, `OPENAI_CHAT_CALLS`, `OPENAI_RESPONSES_CALLS`.
- `CallSite`: `.target`, `.kwargs` (name → AST node), `.kwarg_literal(name)` (literal value, or a module-level constant, else `None`), `.model()` (resolved `model=` string or `None`), `.line`, `.col`, `.node`.
- A rule subclasses `Rule`, sets `id` (`LDL001`…), `name` (kebab-case), `summary`, `references` (URLs of docs and of the real PRs), implements `check(ctx) -> Iterator[Finding]`, and is decorated with `@register`. Build findings with `self.finding(site, ctx, message, hint)`.
- Project dependencies: `ctx.min_version("anthropic")` returns the lowest version the file's project runs with (lockfile version if any, else the declared lower bound from pyproject / requirements), or `None` when unknown. LDL002 should only fire when this is `>= (1,)`. `ctx.project.declares(name)` tells whether it's declared at all.
- Model knowledge lives **only** in `models.py`: rules call `models.traits(model_id)` and read `ModelTraits` fields. Never hard-code model names in a rule.
- When the model can't be resolved statically (`site.model() is None`), a model-dependent rule stays silent by default. **False positives are worse than misses** for v0.1.

## Planned rules (RULES session; ids are fixed)

| Id | Name | What it flags | Motivating real PRs |
|---|---|---|---|
| LDL001 | `anthropic-sampling-params` | `temperature` / `top_p` / `top_k` sent to a model whose traits reject them (`rejects_sampling`: "all" → any value; "non_default" → any value other than 1.0) | langchainrb#1055, Tencent/WeKnora#4155, truera/trulens#2948 |
| LDL002 | `anthropic-sampling-kwargs-sdk1` | `temperature`/`top_p`/`top_k` passed as **keyword arguments** to `messages.create()` when the project requires `anthropic>=1` (SDK 1.x removed them; send in `extra_body`) | Arize-ai/phoenix#17026 |
| LDL003 | `content-first-block-text` | `<response>.content[0].text` (and `["content"][0]["text"]`, `dig("content", 0, "text")`-style subscripts) on an Anthropic response: block 0 can be a thinking block | allenai/reward-bench#290, AllYourBot/hostedgpt#823 |
| LDL004 | `forced-tool-choice` | `tool_choice={"type": "tool"\|"any", ...}` sent to a model with `rejects_forced_tool_choice` | Arize-ai/phoenix#17027, valentinfrlch/ha-llmvision#747 |
| LDL005 | `thinking-budget-tokens` | `thinking={"type": "enabled", "budget_tokens": N}` sent to a model with `rejects_budget_tokens` | posit-dev/chatlas#448 |
| LDL006 | `low-max-tokens-thinking` | literal `max_tokens` < 1024 (threshold in the rule) sent to a `thinks_by_default` model without an explicit `thinking` field | langchainrb#1073 |

Each rule gets tests in `tests/rules/test_ldlNNN.py`: at least one positive case taken from the shape of the real bug, and negatives (unresolvable model, accepted model, value set by caller, etc.).

## DATA session deliverables

1. `models.py`: a complete, sourced `MODEL_TABLE` for current Claude models (Opus/Sonnet/Haiku/Fable/Mythos and their Bedrock/Vertex ids, e.g. `us.anthropic.claude-...`, `anthropic.claude-...`, `claude-...@vertex`). Start from what the upstream PRs above state (read their descriptions with `gh pr view`). Cite each entry in `SOURCES`. When two sources disagree or something is unknown, leave the trait at its default (`None`/`False`) and note it in `docs/rules-evidence.md`.
2. `tests/corpus/`: for each motivating PR, the **pre-fix** version of the relevant file(s) (fetched with `gh api` / raw.githubusercontent at the PR's base commit) plus `expected.json` listing the findings (rule id, line) a correct linter must report. Also a few **post-fix** files that must report nothing.
3. `tests/test_corpus.py`: runs every rule over the corpus and asserts the expected findings (it will fail until the RULES session lands rules: mark missing rule ids with `pytest.mark.xfail(strict=False)`).
4. A false-positive check: a script `scripts/fp_scan.py` that runs the linter over the big repos already cloned in `~/.cache/claude-scratch/repos/` (langchain, llama_index, langgraph, crewAI, openai-agents-python…) and writes the findings to `~/.cache/claude-scratch/ldl-fp.txt`, for review.

## Status log (append, newest last; one line each, with session name)

- LEAD: skeleton + contract committed on `main`. RULES and DATA: create your worktrees from `main`.

- LEAD: `ctx.min_version(pkg)` / `ctx.project` added (src/llm_drift_lint/project.py); requires-python is now >=3.11 (tomllib). Merge `main` into your branch.

## Requests (to the lead)

(none yet)
