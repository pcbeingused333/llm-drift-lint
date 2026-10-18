# Hunt corpus

Real non-Python files for `scripts/hunt_text.py` (the linter itself only parses Python, so these
aren't in `tests/corpus`). Each case is the file a fix touched, at the commit before the fix
(`*-pre`) and with the fix (`*-post`), unmodified and at its path inside the repository.
`tests/test_hunt_text.py` checks what the script reports on each.

| Case | Source | Commit (pre / post) | License |
|---|---|---|---|
| `neurolink-gemini-index-pre` / `-post` | juspay/neurolink `parseSSEStream` keys tool calls by `tc.index` only | `20b0368` / `d6ab3e6` | MIT |
| `praisonai-gemini-index-pre` / `-post` | MervinPraison/PraisonAI `OpenAIProvider.streamText` skips deltas without `index` | `d27f032` / `675b8d0` | MIT |

The post-fix commits are local fixes that aren't published yet. The files stay under their
original licenses, copied in `licenses/`. They are test data only.
