import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_drift_lint import core
from llm_drift_lint.cli import lint_source, main
from llm_drift_lint.core import FileContext, Finding, Rule
from llm_drift_lint.reporting import format_github


class _ZeroTemperature(Rule):
    id = "LDL900"
    name = "test-zero-temperature"
    summary = "test rule: flags temperature=0 on messages.create"
    references = ("https://example.com/pr/1",)

    def check(self, ctx: FileContext) -> Iterator[Finding]:
        for site in ctx.calls("messages.create"):
            if site.kwarg_literal("temperature") == 0:
                yield self.finding(site, ctx, "temperature=0 sent", "drop it")


@pytest.fixture(autouse=True)
def only_test_rule(monkeypatch):
    monkeypatch.setattr(core, "_REGISTRY", {"LDL900": _ZeroTemperature})
    monkeypatch.setattr(core, "all_rules", lambda: [_ZeroTemperature()])
    from llm_drift_lint import cli

    monkeypatch.setattr(cli, "all_rules", lambda: [_ZeroTemperature()])


BAD = "client.messages.create(model='m', temperature=0)\n"


def test_lint_source_reports_and_suppressions_apply():
    assert [f.rule_id for f in lint_source(BAD, Path("a.py"))] == ["LDL900"]
    assert lint_source(BAD.rstrip() + "  # ldl: ignore\n", Path("a.py")) == []
    assert lint_source(BAD.rstrip() + "  # ldl: ignore[LDL900]\n", Path("a.py")) == []
    assert len(lint_source(BAD.rstrip() + "  # ldl: ignore[LDL001]\n", Path("a.py"))) == 1


def test_suppression_on_the_line_above_only_as_a_standalone_comment():
    assert lint_source("# ldl: ignore[LDL900]\n" + BAD, Path("a.py")) == []
    assert len(lint_source("x = 1  # ldl: ignore\n" + BAD, Path("a.py"))) == 1


def write(tmp_path: Path, rel: str, text: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_exit_codes_and_test_dirs_skipped(tmp_path, capsys):
    write(tmp_path, "tests/test_x.py", BAD)
    assert main([str(tmp_path)]) == 0
    assert main([str(tmp_path), "--include-tests"]) == 1
    write(tmp_path, "pkg/mod.py", BAD)
    assert main([str(tmp_path)]) == 1
    assert main([str(tmp_path), "--exit-zero"]) == 0
    assert main([str(tmp_path), "--ignore", "LDL900"]) == 0


def test_json_format(tmp_path, capsys):
    write(tmp_path, "pkg/mod.py", BAD)
    main([str(tmp_path), "--format", "json"])
    (item,) = json.loads(capsys.readouterr().out)
    assert item["rule"] == "LDL900" and item["line"] == 1 and item["column"] == 1
    assert item["references"] == ["https://example.com/pr/1"]


def test_github_format_escapes():
    f = Finding("LDL900", Path("a,b.py"), 3, 0, "50% off:\nnext", "hint")
    assert format_github([f]) == "::warning file=a%2Cb.py,line=3,col=1,title=LDL900::50%25 off:%0Anext hint"


def test_unknown_rule_id_is_an_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path), "--select", "LDL999"])
    assert exc.value.code == 2


def test_list_rules(capsys):
    assert main(["--list-rules"]) == 0
    assert "LDL900" in capsys.readouterr().out
