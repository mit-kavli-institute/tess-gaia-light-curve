#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox"]
# ///
"""
Development tasks. Run `nox -l` to list sessions and `nox -s <session>` to run one.

Sessions use uv when it is installed, which also picks up the MIT-Kavli package index configured
under `[tool.uv]` in pyproject.toml (needed for `pyticdb`).
"""

from pathlib import Path
import zipfile

import nox


nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv|virtualenv"

PYTHON_VERSIONS = ["3.11", "3.12"]


@nox.session
def lint(session: nox.Session) -> None:
    """Run all pre-commit hooks on all files."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


@nox.session(python=PYTHON_VERSIONS[0])
def typecheck(session: nox.Session) -> None:
    """Type check the package with mypy."""
    session.install("-e", ".[typing]")
    session.run("mypy", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite. Needs Docker for the pyticdb database fixtures."""
    session.install("-e", ".[dev]")
    session.run("pytest", *session.posargs)


@nox.session(python=PYTHON_VERSIONS[0], default=False)
def build(session: nox.Session) -> None:
    """Build the sdist and wheel, and check the wheel ships its package data."""
    dist_dir = Path("dist")
    for old_dist in dist_dir.glob("*"):
        old_dist.unlink()
    session.install("build")
    session.run("python", "-m", "build")
    (wheel,) = dist_dir.glob("*.whl")
    with zipfile.ZipFile(wheel) as wheel_zip:
        if "tglc/utils/data/median_mask.fits" not in wheel_zip.namelist():
            session.error(f"{wheel.name} is missing tglc/utils/data/median_mask.fits")


if __name__ == "__main__":
    nox.main()
