# Regression corpus

Real files from the upstream pull requests that motivate each rule, at the commit
just before the fix (`*-pre`) and at the fix (`*-post`). `expected.json` lists the
findings a correct linter reports on each case; `tests/test_corpus.py` checks them.

Each case directory keeps the file's path inside its repository, so path-based
logic (e.g. finding the project's `pyproject.toml`) sees the original layout. Only
the files a fix touched are included, plus the `pyproject.toml` that declares the
`anthropic` dependency where one exists. Files are unmodified.

| Case | Source | Commit | License |
|---|---|---|---|
| `phoenix-17026-17027-pre` | [Arize-ai/phoenix#17026](https://github.com/Arize-ai/phoenix/pull/17026), [#17027](https://github.com/Arize-ai/phoenix/pull/17027) (base) | `d92a385` | Elastic License 2.0 |
| `phoenix-17026-post` | #17026 (head) | `0817cb7` | Elastic License 2.0 |
| `phoenix-17027-post` | #17027 (head) | `648478d` | Elastic License 2.0 |
| `reward-bench-290-pre` / `-post` | [allenai/reward-bench#290](https://github.com/allenai/reward-bench/pull/290) | `5b958c9` / `1352986` | Apache-2.0 |
| `trulens-2948-pre` / `-post` | [truera/trulens#2948](https://github.com/truera/trulens/pull/2948) | `7123b62` / `f1f5ee2` | MIT |
| `chatlas-448-pre` / `-post` | [posit-dev/chatlas#448](https://github.com/posit-dev/chatlas/pull/448) | `1746ec6` / `a06b46c` | MIT |
| `ha-llmvision-747-pre` / `-post` | [valentinfrlch/ha-llmvision#747](https://github.com/valentinfrlch/ha-llmvision/pull/747) | `68ecf5a` / `429dab1` | Apache-2.0 |

The files remain under their original licenses, copied in `licenses/` (Phoenix's
`IP_NOTICE` included). They are test data only: `ruff.toml` here excludes them
from linting and formatting, and nothing imports them.

The motivating PRs in Ruby or Go (patterns-ai-core/langchainrb#1055 and #1073,
AllYourBot/hostedgpt#823, Tencent/WeKnora#4155) aren't in the corpus, because the
linter only parses Python. `docs/rules-evidence.md` covers them.
