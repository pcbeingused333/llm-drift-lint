"""Text heuristics for the LDL007-LDL010 bug patterns in any language (JS/TS, Go, PHP, Ruby, Java...).

The linter itself only parses Python. This script is a hunting aid: it prints *candidates* for a
human to read, and it has false positives by design. Never report its output without reading the
code.

    python -I scripts/hunt_text.py REPO [REPO...] > candidates.tsv

Output: one tab-separated line per candidate: pattern, path:line, note.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_drift_lint import models

EXTS = {
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".go",
    ".php",
    ".rb",
    ".java",
    ".kt",
    ".cs",
    ".rs",
    ".dart",
    ".swift",
    ".py",
}
SKIP_DIRS = {
    "node_modules",
    "vendor",
    "dist",
    "build",
    ".git",
    ".next",
    "third_party",
    "__pycache__",
    ".venv",
    "venv",
}
TEST_HINT = re.compile(
    r"(^|[/._-])(tests?|spec|__tests__|fixtures?|mocks?|e2e|testdata|examples?)([/._-]|$)", re.IGNORECASE
)
MAX_BYTES = 400_000

# LDL007 ------------------------------------------------------------------------------------------
_SHUTDOWN_IDS = sorted(models.OPENAI_SHUTDOWNS, key=len, reverse=True)
_SHUTDOWN_ALT = "|".join(map(re.escape, _SHUTDOWN_IDS))
_SHUTDOWN_RE = re.compile(r"""["'`](?:openai/)?(""" + _SHUTDOWN_ALT + r""")["'`]""")
_DEFAULT_CTX = re.compile(r"model|default", re.IGNORECASE)
_NOT_DEFAULT = re.compile(
    r"price|pricing|cost|token|encod|tiktoken|azure|context|limit|window|max_|maxT|rate|alias|deprecat|"
    r"retire|legacy|includes|startsWith|===?\s*['\"]|case\s+['\"]|==\s*['\"]",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"""["'`][\w.:/-]{2,}["'`]""")

# LDL008 / LDL009 ---------------------------------------------------------------------------------
_FAMILY = re.compile(r"claude|opus|sonnet|haiku|fable|mythos", re.IGNORECASE)
_Q = r"""["'`]([^"'`\n]{1,80})["'`]"""
_ARRAY_START = re.compile(r"(\w+)\W{0,40}?(?:=|:=|=>|:)\s*[^;\n]*?[\[{(]\s*(?://.*|#.*)?$")
_ONE_LINE_ARRAY = re.compile(r"(\w+)\W{0,40}?(?:=|:=|=>|:)\s*[^;\n]*?[\[{(]((?:\s*" + _Q + r"\s*,?)+)[\]})]")
_STRING_TEST = re.compile(
    r"\.?(startsWith|start_with\?|HasPrefix|str_starts_with|startswith|includes|include\?|Contains|"
    r"str_contains|contains|indexOf)\(\s*(?:\w+\s*,\s*)?" + _Q
)
_REGEX_LITERAL = re.compile(
    r"(?<![\w)\]])/((?:[^/\n\\]|\\.){4,200})/[gimsuy]*"  # JS /.../i
    r"|(?:MustCompile|Compile|compile|preg_match|match|search|Regex|RegExp)\(\s*(?:r|`|@)?[\"'`]((?:[^\"'`\n\\]|\\.){4,200})[\"'`]"
)
_SAMPLING_WORD = re.compile(r"temperature|top_?p\b|top_?k\b|topP|topK|sampling", re.IGNORECASE)
_ADAPTIVE_WORD = re.compile(r"adaptive", re.IGNORECASE)
_BUDGET_WORD = re.compile(r"budget_?tokens|budgetTokens|BudgetTokens", re.IGNORECASE)
_PREFIX_USE = re.compile(r"startsWith|HasPrefix|start_with|str_starts_with|startswith", re.IGNORECASE)


def _covers(items: list[str], mode: str):
    if mode == "regex":
        rxs = []
        for it in items:
            try:
                rxs.append(re.compile(it, re.IGNORECASE))
            except re.error:
                return None
        return lambda m: any(r.search(m) for r in rxs)
    if mode == "prefix":
        return lambda m: any(m.startswith(i.lower()) for i in items)
    return lambda m: any(i.lower() in m for i in items)


def gates(lines: list[str], text: str):
    """(first line, last line, name, items, mode) for each Claude model gate in the file."""
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _ONE_LINE_ARRAY.search(line)
        if m:
            items = re.findall(_Q, m.group(2))
            if any(_FAMILY.search(x) for x in items):
                name = m.group(1)
                mode = (
                    "prefix"
                    if re.search(re.escape(name) + r"[\s\S]{0,200}?" + _PREFIX_USE.pattern, text)
                    or _PREFIX_USE.search(name)
                    or "prefix" in name.lower()
                    else "substring"
                )
                yield i + 1, i + 1, name, items, mode
        elif m := _ARRAY_START.search(line):
            items, j = [], i + 1
            while j < n and j < i + 80:
                body = lines[j].strip()
                if not body or body.startswith(("//", "#", "/*", "*")):
                    j += 1
                    continue
                found = re.findall(_Q, body)
                rest = re.sub(_Q, "", body).strip(" ,")
                if found and rest in ("", ")", "]", "}", "),", "],", "},"):
                    items += found
                    j += 1
                    continue
                break
            if items and any(_FAMILY.search(x) for x in items):
                name = m.group(1)
                mode = (
                    "prefix"
                    if re.search(re.escape(name) + r"[\s\S]{0,200}?" + _PREFIX_USE.pattern, text)
                    or "prefix" in name.lower()
                    else "substring"
                )
                yield i + 1, j, name, items, mode
                i = j
                continue
        tests = [(meth, val) for meth, val in _STRING_TEST.findall(line) if _FAMILY.search(val)]
        if tests:
            j, allt = i, list(tests)
            while j + 1 < n and (
                more := [(a, b) for a, b in _STRING_TEST.findall(lines[j + 1]) if _FAMILY.search(b)]
            ):
                allt += more
                j += 1
            mode = "prefix" if all(_PREFIX_USE.search(a) for a, _ in allt) else "substring"
            yield i + 1, j + 1, "", [b for _, b in allt], mode
            i = j + 1
            continue
        for a, b in _REGEX_LITERAL.findall(line):
            rx = a or b
            if _FAMILY.search(rx) and re.search(r"\d", rx):
                yield i + 1, i + 1, "", [rx], "regex"
        i += 1


_COMMENT = ("//", "#", "/*", "*", '"""', "'''")


def _purpose_context(lines: list[str], first: int, last: int, name: str) -> str:
    """The gate's name, its lines, and the comment block (or doc comment) right above it, also when
    the gate sits a few lines into a function (``const isX = (m) => {`` / ``const id = ...``)."""
    above, j, code, comments = [], first - 2, 0, 0
    while j >= 0 and len(above) < 8 and lines[j].strip():
        text = lines[j].strip()
        if text.startswith(_COMMENT):
            above.append(text)
            comments += 1
        elif comments or code == 3:
            break
        else:
            above.append(text)
            code += 1
        j -= 1
    return "\n".join([name, *reversed(above), *lines[first - 1 : last]])


def _trait_models(trait) -> list[str]:
    return [m for m in models.CLAUDE_IDS if (t := models.traits(m)) is not None and trait(t)]


SAMPLING_REJECTERS = _trait_models(lambda t: t.rejects_sampling is not None)
BUDGET_REJECTERS = _trait_models(lambda t: t.rejects_budget_tokens)


# LDL010 ------------------------------------------------------------------------------------------
_TOOL_CALLS = re.compile(r"tool_calls|toolCalls|ToolCalls")
_DELTA = re.compile(r"delta", re.IGNORECASE)
_INDEX_KEY = re.compile(
    r"\[\s*(?:\$?\w+(?:\.|->))+[iI]ndex\s*\]"  # calls[tc.index], $calls[$tc->index]
    r"|\[\s*\$?\w+\[\s*['\"]index['\"]\s*\]\s*\]"  # calls[tc["index"]], $calls[$tc['index']]
    r"|\.(?:get|has|set)\(\s*\w+(?:\?)?\.index\b"  # map.get(tc.index)
    r"|\*\w+\.Index\b"  # Go: *tc.Index
    r"|\[\s*int\(\s*\*?\w+\.Index\s*\)\s*\]"
)
_HANDLES_MISSING = re.compile(
    r"index\s*(?:===?|!==?)\s*(?:undefined|null|nil)|index\s*\?\?|\.Index\s*(?:==|!=)\s*nil|"
    r"index\s+is\s+(?:not\s+)?None|isset\(\s*\$\w+(?:\[['\"]index['\"]\]|->index)|"
    r"typeof\s+\w+\.index|['\"]index['\"]\s+in\s+|has(?:Own)?Property\(\s*['\"]index|index\s*\|\|",
    re.IGNORECASE,
)


def iter_files(root: Path):
    for path in root.rglob("*"):
        if path.suffix not in EXTS or not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(p in SKIP_DIRS for p in rel.parts) or TEST_HINT.search(str(rel)):
            continue
        if path.name.endswith((".min.js", ".d.ts")) or path.stat().st_size > MAX_BYTES:
            continue
        yield path


def hunt_file(path: Path, text: str):
    lines = text.splitlines()

    # LDL007: a shutdown id on a line that reads like a default.
    for i, line in enumerate(lines, 1):
        m = _SHUTDOWN_RE.search(line)
        if not m or not _DEFAULT_CTX.search(line) or _NOT_DEFAULT.search(line):
            continue
        if len(_QUOTED.findall(line)) > 2 or line.lstrip().startswith(("//", "#", "*", "<!--")):
            continue
        s = models.OPENAI_SHUTDOWNS[m.group(1).lower()]
        yield "LDL007", i, f"{m.group(1)} (shutdown {s.date} -> {s.replacement})"

    # LDL008 / LDL009: a Claude model gate for sampling / adaptive thinking that misses models.
    # A gate is skipped when another gate in the file for the same purpose covers every model it
    # misses (the old partial helper often stays around for another use after the fix).
    found: dict[str, list[tuple[int, str, list[str]]]] = {"LDL008": [], "LDL009": []}
    complete: dict[str, set[str]] = {"LDL008": set(), "LDL009": set()}
    for first, last, name, items, mode in gates(lines, text):
        covers = _covers(items, mode)
        # A gate on the provider ("claude", "anthropic/") isn't a model gate.
        if covers is None or not any(re.search(r"\d", i) for i in items if _FAMILY.search(i)):
            continue
        ctx = _purpose_context(lines, first, last, name)
        for rule, word, rejecters, need in (
            ("LDL008", _SAMPLING_WORD, SAMPLING_REJECTERS, None),
            ("LDL009", _ADAPTIVE_WORD, BUDGET_REJECTERS, _BUDGET_WORD),
        ):
            if not word.search(ctx) or (need is not None and not need.search(text)):
                continue
            covered = [m for m in rejecters if covers(m)]
            missing = [m for m in rejecters if not covers(m)]
            complete[rule] |= set(covered)
            if covered and missing:
                found[rule].append((first, f"{name or mode}: covers {','.join(covered)}", missing))
    for rule, hits in found.items():
        for first, note, missing in hits:
            still = [m for m in missing if m not in complete[rule]]
            if still:
                yield rule, first, f"{note}; misses {','.join(still)}"

    # LDL010: a stream reducer keyed by tool_calls[].index with no missing-index handling.
    if _TOOL_CALLS.search(text) and _DELTA.search(text) and not _HANDLES_MISSING.search(text):
        for i, line in enumerate(lines, 1):
            if _INDEX_KEY.search(line):
                yield "LDL010", i, line.strip()[:120]
                break


def _near(lines: list[str], lineno: int, rx: re.Pattern, span: int) -> bool:
    lo, hi = max(0, lineno - 1 - span), min(len(lines), lineno + span)
    return any(rx.search(lines[j]) for j in range(lo, hi))


def main(argv: list[str]) -> int:
    for arg in argv:
        root = Path(arg)
        for path in iter_files(root):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for rule, line, note in hunt_file(path, text):
                print(f"{rule}\t{path}:{line}\t{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
