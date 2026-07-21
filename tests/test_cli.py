"""
Tests for the tglc.util.cli module, which provides helper functions for creating command line
TGLC scripts.
"""

from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys

import pytest

from tglc import cli
from tglc.apertures import APERTURE_NAMES


@contextmanager
def tmp_chdir(path):
    """Change directory for the duration of the context manager being open."""
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        # Reload the cli module to re-compute tglc-data-dir default
        importlib.reload(cli)
        yield
    finally:
        os.chdir(original_dir)
        # Reload the cli module to re-compute tglc-data-dir default
        importlib.reload(cli)


def test_base_parser_has_expected_arguments():
    parser = cli.command_base_parser
    args = parser.parse_args(["-o", "1"])

    assert isinstance(args.orbit, int)
    assert args.ccd is None
    assert args.cutout is None
    assert isinstance(args.nprocs, int)
    assert isinstance(args.replace, bool)
    assert isinstance(args.debug, bool)
    assert args.logfile is None
    assert isinstance(args.enable_runtime_warnings, bool)
    assert isinstance(args.tglc_data_dir, Path)


def test_tglc_data_dir_finds_current(tmp_path: Path):
    tglc_data_dir = tmp_path / "tglc-data"
    tglc_data_dir.mkdir()

    with tmp_chdir(tglc_data_dir):
        args = cli.command_base_parser.parse_args(["-o", "1"])
        assert args.tglc_data_dir == tglc_data_dir


def test_tglc_data_dir_finds_parent(tmp_path: Path):
    tglc_data_dir = tmp_path / "tglc-data"
    working_directory = tglc_data_dir / "sector0080"
    working_directory.mkdir(parents=True)

    with tmp_chdir(working_directory):
        args = cli.command_base_parser.parse_args(["-o", "1"])
        assert args.tglc_data_dir == tglc_data_dir


def test_tglc_data_dir_falls_back_to_cwd(tmp_path: Path):
    with tmp_chdir(tmp_path):
        args = cli.command_base_parser.parse_args(["-o", "1"])
        assert args.tglc_data_dir == tmp_path


def parse_args(monkeypatch: pytest.MonkeyPatch, *argv: str):
    """Run cli.parse_tglc_args with the given command line arguments."""
    monkeypatch.setattr(sys, "argv", ["tglc", *argv])
    return cli.parse_tglc_args()


def test_lightcurves_includes_all_apertures_by_default(monkeypatch: pytest.MonkeyPatch):
    args = parse_args(monkeypatch, "lightcurves", "-o", "1")
    assert args.apertures == list(APERTURE_NAMES)


def test_lightcurves_accepts_aperture_subset(monkeypatch: pytest.MonkeyPatch):
    args = parse_args(monkeypatch, "lightcurves", "-o", "1", "--apertures", "primary")
    assert args.apertures == ["primary"]


def test_apertures_normalized_to_canonical_order_and_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
):
    args = parse_args(
        monkeypatch, "lightcurves", "-o", "1", "--apertures", "large", "primary", "primary"
    )
    assert args.apertures == ["primary", "large"]


def test_invalid_aperture_rejected(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(SystemExit):
        parse_args(monkeypatch, "lightcurves", "-o", "1", "--apertures", "medium")


def test_all_command_accepts_apertures(monkeypatch: pytest.MonkeyPatch):
    args = parse_args(monkeypatch, "all", "-o", "1", "--apertures", "primary")
    assert args.apertures == ["primary"]
