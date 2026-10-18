"""scripts/hunt_text.py: the multi-language hunting heuristics. Each positive case has the shape of
a real bug (fabric#2270 in Go, cherry-studio#21939 in TS, neurolink in TS), the negatives the
shape of the fixes."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "hunt_text", Path(__file__).resolve().parents[1] / "scripts" / "hunt_text.py"
)
hunt_text = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hunt_text)


def hunt(src: str, name: str = "x.ts") -> list[tuple[str, int]]:
    return [(rule, line) for rule, line, _ in hunt_text.hunt_file(Path(name), src)]


GO_GATES = """
// These models reject non-default sampling parameters.
var samplingParamsDisallowedPrefixes = []string{
	"claude-opus-4-7",
	"claude-opus-5",
}

// Requests to these models use thinking.type=adaptive; others get budget_tokens.
var adaptiveThinkingPrefixes = []string{
	"claude-opus-5",
}

func modelUsesAdaptiveThinking(model string) bool {
	return slices.ContainsFunc(adaptiveThinkingPrefixes, func(p string) bool { return strings.HasPrefix(model, p) })
}
"""


def test_go_prefix_lists_missing_models():
    assert hunt(GO_GATES, "anthropic.go") == [("LDL008", 3), ("LDL009", 9)]


def test_complete_lists_are_silent():
    src = GO_GATES.replace(
        '"claude-opus-5",\n}',
        '"claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable",'
        ' "claude-mythos", "claude-opus-4-8",\n}',
        1,
    ).replace(
        '"claude-opus-5",\n}\n\nfunc',
        '"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",'
        ' "claude-haiku-5", "claude-fable", "claude-mythos-5",\n}\n\nfunc',
    )
    assert hunt(src, "anthropic.go") == []


def test_regex_gate_that_only_matches_opus_4_7():
    src = """
/** Check if model is Claude Opus 4.7. Rejects temperature/top_p/top_k. */
export const isClaude47SeriesModel = (id: string): boolean => /claude-opus-4[.-]7(?:-[\\w-]+)?$/i.test(id)
"""
    assert hunt(src) == [("LDL008", 3)]


def test_partial_gate_next_to_a_complete_one_is_silent():
    src = """
/** Rejects temperature/top_p/top_k. */
export const isClaude47SeriesModel = (id: string) => /claude-opus-4[.-]7/i.test(id)
/** Rejects temperature/top_p/top_k. */
export const isFixedSampling = (id: string) => /claude-(opus-4-[78]|(opus|sonnet|haiku)-5|fable|mythos)/i.test(id)
"""
    assert hunt(src) == []


def test_provider_check_is_not_a_model_gate():
    assert hunt('if (model.startsWith("claude")) { body.temperature = temperature }\n') == []


def test_stream_reducer_keyed_by_index():
    src = """
for (const tc of chunk.choices[0].delta.tool_calls ?? []) {
  let state = toolCalls.get(tc.index);
}
"""
    assert hunt(src) == [("LDL010", 3)]
    handled = src.replace("let state", "const key = tc.index ?? toolCalls.size; let state")
    assert hunt(handled) == []


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('const DEFAULT_MODEL = "gpt-3.5-turbo";', [("LDL007", 1)]),
        ('model: process.env.OPENAI_MODEL || "gpt-4",', [("LDL007", 1)]),
        ('const PRICES = { "gpt-4": { input: 30 } };', []),
        ('if (model === "gpt-4") return 8192;', []),
        ('// default model was "gpt-4"', []),
        ('const DEFAULT_MODEL = "gpt-4o";', []),
    ],
)
def test_shutdown_defaults(line, expected):
    assert hunt(line + "\n") == expected


HUNT_CORPUS = Path(__file__).parent / "hunt_corpus"
# Real files at the commit before / after each fix (see hunt_corpus/README.md).
HUNT_EXPECTED = {
    "neurolink-gemini-index-pre": [("LDL010", "src/lib/providers/openaiChatCompletionsClient.ts", 932)],
    "neurolink-gemini-index-post": [],
    "praisonai-gemini-index-pre": [("LDL010", "src/praisonai-ts/src/llm/providers/openai.ts", 108)],
    "praisonai-gemini-index-post": [],
}


@pytest.mark.parametrize("case", sorted(HUNT_EXPECTED))
def test_hunt_corpus(case):
    root = HUNT_CORPUS / case
    found = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        text = path.read_text(encoding="utf-8")
        found += [
            (rule, path.relative_to(root).as_posix(), line)
            for rule, line, _ in hunt_text.hunt_file(path, text)
        ]
    assert found == HUNT_EXPECTED[case]


def test_hunt_corpus_is_listed():
    dirs = {p.name for p in HUNT_CORPUS.iterdir() if p.is_dir() and p.name != "licenses"}
    assert dirs == set(HUNT_EXPECTED)


def test_drop_guard_without_fallback():
    src = """
for (const tc of delta.tool_calls) {
  if (tc.index !== undefined) {
    toolCalls[tc.index].function.arguments += tc.function.arguments;
  }
}
"""
    assert hunt(src) == [("LDL010", 3)]
    assert (
        hunt(src.replace("if (tc.index !== undefined) {", "const i = tc.index ?? toolCalls.length; {")) == []
    )
