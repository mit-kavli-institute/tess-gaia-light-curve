"""
Tests for the tglc.util.cli module, which provides helper functions for creating command line
TGLC scripts.
"""

from contextlib import contextmanager
import importlib
import os
from pathlib import Path

import pytest

from tglc import cli


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


def test_cutouts_parser_accepts_ffi_cbv_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The cutouts subcommand exposes --ffi-cbv-dir; the path is parsed as a Path."""
    cbv_dir = tmp_path / "cbvs"
    monkeypatch.setattr("sys.argv", ["tglc", "cutouts", "-o", "42", "--ffi-cbv-dir", str(cbv_dir)])
    args = cli.parse_tglc_args()
    assert args.tglc_command == "cutouts"
    assert args.ffi_cbv_dir == cbv_dir


def test_cutouts_parser_ffi_cbv_dir_default_none(monkeypatch):
    monkeypatch.setattr("sys.argv", ["tglc", "cutouts", "-o", "42"])
    args = cli.parse_tglc_args()
    assert args.ffi_cbv_dir is None
