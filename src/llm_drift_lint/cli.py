from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from .core import FileContext, Finding, all_rules

DEFAULT_EXCLUDE = ("tests", "test", ".venv", "venv", "node_modules", "build", "dist", ".git")


def iter_files(root: Path, exclude: tuple[str, ...]) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*.py")):
        if not any(part in exclude for part in path.relative_to(root).parts):
            yield path


def lint_paths(paths: list[Path], select: set[str] | None = None, exclude=DEFAULT_EXCLUDE) -> list[Finding]:
    rules = [r for r in all_rules() if select is None or r.id in select]
    findings: list[Finding] = []
    for root in paths:
        for path in iter_files(root, exclude):
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            ctx = FileContext.parse(source, path)
            if ctx is None:
                continue
            for rule in rules:
                findings.extend(rule.check(ctx))
    return sorted(findings, key=lambda f: (str(f.path), f.line, f.col, f.rule_id))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-drift-lint", description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(".")])
    parser.add_argument("--select", help="comma-separated rule ids to run")
    parser.add_argument("--list-rules", action="store_true")
    parser.add_argument("--include-tests", action="store_true")
    args = parser.parse_args(argv)

    if args.list_rules:
        for rule in all_rules():
            print(f"{rule.id}  {rule.name:32}  {rule.summary}")
        return 0

    select = set(args.select.split(",")) if args.select else None
    exclude = () if args.include_tests else DEFAULT_EXCLUDE
    findings = lint_paths(args.paths, select, exclude)
    for f in findings:
        print(f"{f.path}:{f.line}:{f.col + 1}: {f.rule_id} {f.message}")
        if f.hint:
            print(f"    hint: {f.hint}")
    print(f"{len(findings)} finding(s)", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
