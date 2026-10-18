"""Output formats and inline suppressions."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from .core import Finding

# `# ldl: ignore` silences every rule on that line; `# ldl: ignore[LDL001, LDL004]` only those.
_SUPPRESS = re.compile(r"#\s*ldl:\s*ignore(?:\[(?P<ids>[A-Z0-9,\s]+)\])?", re.IGNORECASE)


def suppressed(finding: Finding, lines: list[str]) -> bool:
    """Whether the finding's line (or the line above it, for long calls) carries a suppression comment."""
    for idx in (finding.line - 1, finding.line - 2):
        if 0 <= idx < len(lines):
            m = _SUPPRESS.search(lines[idx])
            if m and idx == finding.line - 2 and not lines[idx].lstrip().startswith("#"):
                continue  # a trailing comment on the previous statement doesn't count
            if m:
                ids = m.group("ids")
                if ids is None or finding.rule_id in {i.strip().upper() for i in ids.split(",")}:
                    return True
    return False


def format_text(findings: Iterable[Finding]) -> str:
    out = []
    for f in findings:
        level = " (warning)" if f.severity == "warning" else ""
        out.append(f"{f.path}:{f.line}:{f.col + 1}: {f.rule_id}{level} {f.message}")
        if f.hint:
            out.append(f"    hint: {f.hint}")
        if f.references:
            out.append(f"    see: {f.references[0]}")
    return "\n".join(out)


def format_json(findings: Iterable[Finding]) -> str:
    return json.dumps(
        [
            {
                "rule": f.rule_id,
                "severity": f.severity,
                "path": str(f.path),
                "line": f.line,
                "column": f.col + 1,
                "message": f.message,
                "hint": f.hint,
                "references": list(f.references),
            }
            for f in findings
        ],
        indent=2,
    )


def _escape_gh(value: str, prop: bool = False) -> str:
    value = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if prop:
        value = value.replace(":", "%3A").replace(",", "%2C")
    return value


def format_github(findings: Iterable[Finding]) -> str:
    """GitHub Actions workflow commands, which show up as annotations on the PR diff
    (``::error`` or ``::warning``, after the finding's severity)."""
    out = []
    for f in findings:
        message = f.message + (f" {f.hint}" if f.hint else "")
        out.append(
            f"::{f.severity} file={_escape_gh(str(f.path), True)},line={f.line},col={f.col + 1},"
            f"title={_escape_gh(f.rule_id, True)}::{_escape_gh(message)}"
        )
    return "\n".join(out)


FORMATTERS = {"text": format_text, "json": format_json, "github": format_github}
