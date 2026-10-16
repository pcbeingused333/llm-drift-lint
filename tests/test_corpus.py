"""Runs every registered rule over tests/corpus and checks the findings listed in expected.json.

Required findings of a rule that isn't registered yet are xfail; once the rule lands they must pass.
Any finding expected.json doesn't list (as required or optional) is a false positive.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from llm_drift_lint.core import FileContext, all_rules

CORPUS = Path(__file__).parent / "corpus"
EXPECTED = json.loads((CORPUS / "expected.json").read_text(encoding="utf-8"))
CASES = {case["id"]: case for case in EXPECTED["cases"]}
RULES = all_rules()
REGISTERED = {rule.id for rule in RULES}
PLANNED = {f"LDL00{n}" for n in range(1, 7)}


@functools.cache
def lint_case(case_id: str) -> tuple[tuple[str, str, int], ...]:
    """(rule id, path relative to the case directory, line) for every finding in the case."""
    case_dir = CORPUS / case_id
    found = []
    for path in sorted(case_dir.rglob("*.py")):
        # The real path, so rules that look for the project's pyproject.toml find the case's own.
        ctx = FileContext.parse(path.read_text(encoding="utf-8"), path)
        assert ctx is not None, f"corpus file doesn't parse: {path}"
        for rule in RULES:
            found.extend((f.rule_id, path.relative_to(case_dir).as_posix(), f.line) for f in rule.check(ctx))
    return tuple(found)


def matches(expected: dict, finding: tuple[str, str, int]) -> bool:
    rule_id, path, line = finding
    return (
        rule_id == expected["rule"]
        and path == expected["path"]
        and expected["line"] <= line <= expected["end_line"]
    )


def required_params():
    for case in EXPECTED["cases"]:
        for exp in case["findings"]:
            if exp["status"] != "required":
                continue
            marks = []
            if exp["rule"] not in REGISTERED:
                marks.append(pytest.mark.xfail(reason=f"{exp['rule']} isn't registered yet", strict=False))
            yield pytest.param(case["id"], exp, id=f"{case['id']}:{exp['rule']}:{exp['line']}", marks=marks)


@pytest.mark.parametrize("case_id, expected", list(required_params()))
def test_required_finding(case_id, expected):
    found = lint_case(case_id)
    assert any(matches(expected, f) for f in found), (
        f"{expected['rule']} not reported at {expected['path']}:{expected['line']} ({expected['why']})"
    )


@pytest.mark.parametrize("case_id", list(CASES))
def test_no_unexpected_findings(case_id):
    expected = CASES[case_id]["findings"]
    unexpected = [f for f in lint_case(case_id) if not any(matches(e, f) for e in expected)]
    assert not unexpected, f"false positives in {case_id}: {unexpected}"


def test_expected_json_is_consistent():
    case_dirs = {p.name for p in CORPUS.iterdir() if p.is_dir() and p.name != "licenses"}
    assert set(CASES) == {name for name in case_dirs if not name.startswith(".")}  # skip .ruff_cache
    for case in EXPECTED["cases"]:
        assert case["kind"] in {"pre-fix", "post-fix"}, case["id"]
        assert case["prs"] and case["commit"], case["id"]
        for exp in case["findings"]:
            assert exp["rule"] in PLANNED, exp
            assert exp["status"] in {"required", "optional"}, exp
            assert exp["why"], exp
            path = CORPUS / case["id"] / exp["path"]
            assert path.is_file(), path
            n_lines = len(path.read_text(encoding="utf-8").splitlines())
            assert 1 <= exp["line"] <= exp["end_line"] <= n_lines, exp
