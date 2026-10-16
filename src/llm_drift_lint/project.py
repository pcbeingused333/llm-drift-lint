"""What a file's project says about its dependencies (pyproject, requirements files, lockfiles)."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_PROJECT_MARKERS = ("pyproject.toml", "setup.cfg", "setup.py", "requirements.txt", "uv.lock", "poetry.lock")
_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([^;#]*)")
_VERSION = re.compile(r"\d+(?:\.\d+)*")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_version(text: str) -> tuple[int, ...] | None:
    m = _VERSION.search(text)
    return tuple(int(p) for p in m.group(0).split(".")) if m else None


def lower_bound(specifier: str) -> tuple[int, ...] | None:
    """Smallest version a specifier allows, from its >=, >, ~=, == or ^ clauses (None if unbounded).

    ``>1`` is treated like ``>=1``: it is only used to ask whether 0.x is excluded.
    """
    bounds = []
    for clause in re.split(r"[,|]", specifier):
        clause = clause.strip()
        m = re.match(r"(>=|>|~=|===|==|\^|~)\s*(.+)", clause)
        if m and (v := _parse_version(m.group(2))) is not None:
            bounds.append(v)
    return max(bounds) if bounds else None


@dataclass
class ProjectInfo:
    root: Path | None
    # normalized name -> specifier text as declared (may be "")
    declared: dict[str, str] = field(default_factory=dict)
    # normalized name -> exact version from a lockfile or an == pin
    locked: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def min_version(self, name: str) -> tuple[int, ...] | None:
        """The lowest version of ``name`` this project can run with: the locked version if there is a
        lockfile entry, else the declared lower bound. None when unknown or unbounded."""
        key = _normalize(name)
        if key in self.locked:
            return self.locked[key]
        if key in self.declared:
            return lower_bound(self.declared[key])
        return None

    def declares(self, name: str) -> bool:
        key = _normalize(name)
        return key in self.declared or key in self.locked


def _read_requirements(path: Path, info: ProjectInfo) -> None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith(("#", "-")):
            continue
        if m := _REQ_LINE.match(line):
            name, spec = _normalize(m.group(1)), m.group(2).strip()
            info.declared.setdefault(name, spec)
            if spec.startswith("==") and (v := _parse_version(spec)) is not None:
                info.locked.setdefault(name, v)


def _read_pyproject(path: Path, info: ProjectInfo) -> None:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return
    project = data.get("project", {})
    reqs: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        reqs.extend(extra)
    for group in data.get("dependency-groups", {}).values():
        reqs.extend(r for r in group if isinstance(r, str))
    for req in reqs:
        if m := _REQ_LINE.match(req):
            info.declared.setdefault(_normalize(m.group(1)), m.group(2).strip())
    poetry = data.get("tool", {}).get("poetry", {})
    for table in [poetry.get("dependencies", {})] + [
        g.get("dependencies", {}) for g in poetry.get("group", {}).values()
    ]:
        for name, spec in table.items():
            if isinstance(spec, dict):
                spec = spec.get("version", "")
            info.declared.setdefault(_normalize(name), str(spec))


def _read_lock(path: Path, info: ProjectInfo) -> None:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return
    for pkg in data.get("package", []):
        if (v := _parse_version(str(pkg.get("version", "")))) is not None:
            info.locked[_normalize(pkg.get("name", ""))] = v


@lru_cache(maxsize=256)
def _project_for_dir(directory: Path) -> ProjectInfo:
    for d in (directory, *directory.parents):
        if any((d / marker).exists() for marker in _PROJECT_MARKERS):
            info = ProjectInfo(d)
            for lock in ("uv.lock", "poetry.lock"):
                if (d / lock).exists():
                    _read_lock(d / lock, info)
            if (d / "pyproject.toml").exists():
                _read_pyproject(d / "pyproject.toml", info)
            for req in sorted(d.glob("requirements*.txt")):
                _read_requirements(req, info)
            return info
    return ProjectInfo(None)


def project_for(path: Path) -> ProjectInfo:
    """Project info for the nearest directory above ``path`` that looks like a project root."""
    try:
        directory = path.resolve().parent
    except OSError:
        return ProjectInfo(None)
    return _project_for_dir(directory)
