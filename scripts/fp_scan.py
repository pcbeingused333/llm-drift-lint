"""Run the linter over large cloned repos and write every finding for manual false-positive review.

Usage: python scripts/fp_scan.py [--repos DIR] [--out FILE] [--select LDL001,...] [--include-tests] [NAME ...]

By default it scans every repository under ~/.cache/claude-scratch/repos/ (or the NAMEs given) and
writes ~/.cache/claude-scratch/ldl-fp.txt: a summary per repo and rule, then each finding with its
source line. Each finding is either a real bug or a false positive to fix in the rule.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from llm_drift_lint.cli import DEFAULT_EXCLUDE, iter_files, lint_paths
from llm_drift_lint.core import all_rules

SCRATCH = Path.home() / ".cache" / "claude-scratch"


def repo_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "no git"
    return out.stdout.strip()


def source_line(path: Path, line: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="repo directory names (default: all)")
    parser.add_argument("--repos", type=Path, default=SCRATCH / "repos")
    parser.add_argument("--out", type=Path, default=SCRATCH / "ldl-fp.txt")
    parser.add_argument("--select", help="comma-separated rule ids to run")
    parser.add_argument("--include-tests", action="store_true")
    args = parser.parse_args(argv)

    repos = [args.repos / n for n in args.names] if args.names else sorted(args.repos.iterdir())
    repos = [r for r in repos if r.is_dir()]
    if not repos:
        print(f"no repositories under {args.repos}", file=sys.stderr)
        return 2
    select = set(args.select.split(",")) if args.select else None
    exclude = () if args.include_tests else DEFAULT_EXCLUDE
    rule_ids = [r.id for r in all_rules() if select is None or r.id in select]

    header = [
        f"llm-drift-lint false-positive scan, {time.strftime('%Y-%m-%d %H:%M')}",
        f"rules: {', '.join(rule_ids) or '(none registered)'}; tests {'included' if args.include_tests else 'excluded'}",
        "",
        f"{'repo':24} {'commit':10} {'files':>6} {'findings':>8}  by rule",
    ]
    body: list[str] = []
    total: Counter[str] = Counter()
    for repo in repos:
        n_files = sum(1 for _ in iter_files(repo, exclude))
        findings = lint_paths([repo], select, exclude)
        by_rule = Counter(f.rule_id for f in findings)
        total.update(by_rule)
        rules = ", ".join(f"{rule} {n}" for rule, n in sorted(by_rule.items()))
        header.append(f"{repo.name:24} {repo_commit(repo):10} {n_files:>6} {len(findings):>8}  {rules}")
        if findings:
            body += ["", f"== {repo.name}"]
        for f in findings:
            body.append(f"{f.path.relative_to(repo)}:{f.line}:{f.col + 1}: {f.rule_id} {f.message}")
            body.append(f"    {source_line(f.path, f.line)}")
    header += ["", "total: " + (", ".join(f"{r} {n}" for r, n in sorted(total.items())) or "0 findings")]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    print(f"{sum(total.values())} finding(s) in {len(repos)} repo(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
