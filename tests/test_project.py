from pathlib import Path

import pytest

from llm_drift_lint.core import FileContext
from llm_drift_lint.project import _project_for_dir, lower_bound


@pytest.fixture(autouse=True)
def clear_cache():
    _project_for_dir.cache_clear()
    yield
    _project_for_dir.cache_clear()


@pytest.mark.parametrize(
    "spec,expected",
    [
        (">=1.0", (1, 0)),
        (">=0.40,<2", (0, 40)),
        ("~=1.13", (1, 13)),
        ("==1.13.0", (1, 13, 0)),
        ("^0.41.0", (0, 41, 0)),
        ("<2", None),
        ("", None),
    ],
)
def test_lower_bound(spec, expected):
    assert lower_bound(spec) == expected


def ctx_in(tmp_path: Path, rel: str = "pkg/mod.py") -> FileContext:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n")
    ctx = FileContext.parse(path.read_text(), path)
    assert ctx is not None
    return ctx


def test_pyproject_dependencies_and_extras(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="p"\ndependencies=["openai>=1.40"]\n'
        '[project.optional-dependencies]\napi=["Anthropic[bedrock]>=1.2,<2"]\n'
    )
    ctx = ctx_in(tmp_path)
    assert ctx.min_version("anthropic") == (1, 2)
    assert ctx.min_version("openai") == (1, 40)
    assert ctx.min_version("requests") is None


def test_lockfile_wins_over_declared_range(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="p"\ndependencies=["anthropic"]\n')
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "anthropic"\nversion = "1.13.0"\n')
    assert ctx_in(tmp_path).min_version("anthropic") == (1, 13, 0)


def test_unversioned_dependency_is_unknown(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="p"\ndependencies=["anthropic"]\n')
    ctx = ctx_in(tmp_path)
    assert ctx.min_version("anthropic") is None
    assert ctx.project.declares("anthropic")


def test_requirements_pin(tmp_path):
    (tmp_path / "requirements.txt").write_text("# deps\nanthropic==0.49.0\n-e .\n")
    assert ctx_in(tmp_path).min_version("anthropic") == (0, 49, 0)


def test_poetry_dependencies(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.poetry.dependencies]\npython="^3.11"\nanthropic={version="^1.0", optional=true}\n'
    )
    assert ctx_in(tmp_path).min_version("anthropic") == (1, 0)


def test_nearest_project_root_is_used(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="outer"\ndependencies=["anthropic>=0.30"]\n')
    (tmp_path / "integrations" / "x").mkdir(parents=True)
    (tmp_path / "integrations" / "x" / "pyproject.toml").write_text(
        '[project]\nname="inner"\ndependencies=["anthropic>=1"]\n'
    )
    assert ctx_in(tmp_path, "integrations/x/src/mod.py").min_version("anthropic") == (1,)
    assert ctx_in(tmp_path, "other/mod.py").min_version("anthropic") == (0, 30)


def test_string_source_has_no_project():
    ctx = FileContext.parse("x = 1")
    assert ctx is not None and ctx.min_version("anthropic") is None
