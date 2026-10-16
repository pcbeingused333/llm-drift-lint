"""Find LLM API calls that break when providers change model behavior."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from .core import FileContext, Finding, all_rules
from .reporting import FORMATTERS, suppressed

DEFAULT_EXCLUDE = ("tests", "test", ".venv", "venv", "node_modules", "build", "dist", ".git", "site-packages")


def iter_files(root: Path, exclude: tuple[str, ...]) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*.py")):
        if not any(part in exclude for part in path.relative_to(root).parts):
            yield path


def lint_source(source: str, path: Path, rule_ids: set[str] | None = None) -> list[Finding]:
    ctx = FileContext.parse(source, path)
    if ctx is None:
        return []
    lines = source.splitlines()
    findings = []
    for rule in all_rules():
        if rule_ids is not None and rule.id not in rule_ids:
            continue
        findings.extend(f for f in rule.check(ctx) if not suppressed(f, lines))
    return findings


def lint_paths(
    paths: list[Path], rule_ids: set[str] | None = None, exclude: tuple[str, ...] = DEFAULT_EXCLUDE
) -> list[Finding]:
    findings: list[Finding] = []
    for root in paths:
        for path in iter_files(root, exclude):
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            findings.extend(lint_source(source, path, rule_ids))
    return sorted(findings, key=lambda f: (str(f.path), f.line, f.col, f.rule_id))


def _ids(value: str | None) -> set[str]:
    return {v.strip().upper() for v in value.split(",") if v.strip()} if value else set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-drift-lint", description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(".")])
    parser.add_argument("--select", help="comma-separated rule ids to run (default: all)")
    parser.add_argument("--ignore", help="comma-separated rule ids to skip")
    parser.add_argument("--format", choices=sorted(FORMATTERS), default="text")
    parser.add_argument("--list-rules", action="store_true", help="print the rules and exit")
    parser.add_argument("--include-tests", action="store_true", help="also scan test directories")
    parser.add_argument("--exit-zero", action="store_true", help="exit 0 even when there are findings")
    args = parser.parse_args(argv)

    rules = all_rules()
    if args.list_rules:
        for rule in rules:
            print(f"{rule.id}  {rule.name:32}  {rule.summary}")
        return 0

    known = {r.id for r in rules}
    unknown = (_ids(args.select) | _ids(args.ignore)) - known
    if unknown:
        parser.error(f"unknown rule id(s): {', '.join(sorted(unknown))}")
    selected = (_ids(args.select) or known) - _ids(args.ignore)

    exclude = () if args.include_tests else DEFAULT_EXCLUDE
    findings = lint_paths(args.paths, selected, exclude)
    output = FORMATTERS[args.format](findings)
    if output:
        print(output)
    if args.format == "text":
        print(f"{len(findings)} finding(s)", file=sys.stderr)
    return 0 if args.exit_zero or not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
