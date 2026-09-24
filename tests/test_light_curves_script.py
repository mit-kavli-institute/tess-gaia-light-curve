"""Tests for the `tglc lightcurves` script (`tglc.scripts.light_curves`)."""

import argparse
import logging
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from tglc.epsf import EPSF
from tglc.io import write_cutout_fits, write_epsf_fits
from tglc.scripts.light_curves import (
    get_requested_tic_ids,
    make_light_curves_main,
    read_tic_id_file,
)

from .synthetic_data import make_synthetic_cutout, make_synthetic_epsf


ORBIT = 185  # make_synthetic_cutout's orbit; contained in sector 89


def test_read_tic_id_file_one_id_per_line(tmp_path: Path):
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("500001\n500002\n500003\n")

    assert read_tic_id_file(tic_id_file) == [500001, 500002, 500003]


def test_read_tic_id_file_ignores_comments_and_blank_lines(tmp_path: Path):
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("# M dwarf targets\n\n500001  # bright one\n\n  500002\n")

    assert read_tic_id_file(tic_id_file) == [500001, 500002]


def test_read_tic_id_file_accepts_comma_separated_ids(tmp_path: Path):
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("500001, 500002\n500003,500004\n")

    assert read_tic_id_file(tic_id_file) == [500001, 500002, 500003, 500004]


def test_read_tic_id_file_deduplicates_preserving_order(tmp_path: Path):
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("500003\n500001\n500003\n")

    assert read_tic_id_file(tic_id_file) == [500003, 500001]


def test_read_tic_id_file_empty(tmp_path: Path):
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("# nothing here\n")

    assert read_tic_id_file(tic_id_file) == []


def test_read_tic_id_file_rejects_non_integer_entries(tmp_path: Path):
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("500001\nTIC 500002\n")

    with pytest.raises(ValueError, match="Invalid TIC ID 'TIC'"):
        read_tic_id_file(tic_id_file)


def test_get_requested_tic_ids_without_any_request():
    args = argparse.Namespace(tic=None, tic_file=None)

    assert get_requested_tic_ids(args) is None


def test_get_requested_tic_ids_from_command_line_only():
    args = argparse.Namespace(tic=[500001, 500002], tic_file=None)

    assert get_requested_tic_ids(args) == [500001, 500002]


def test_get_requested_tic_ids_combines_command_line_and_file(tmp_path: Path):
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("500002\n500003\n")
    args = argparse.Namespace(tic=[500001, 500002], tic_file=tic_id_file)

    assert get_requested_tic_ids(args) == [500001, 500002, 500003]


def test_get_requested_tic_ids_warns_for_empty_file(tmp_path: Path, caplog):
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("")
    args = argparse.Namespace(tic=None, tic_file=tic_id_file)

    assert get_requested_tic_ids(args) == []
    assert "No TIC IDs found" in caplog.text


def _fake_spacecraft_position(orbit, time, ephemerides_directory):
    """Stand-in for the JPL Horizons query, as in tests/test_light_curve.py."""
    return np.zeros((len(np.atleast_1d(time.tdb.jd)), 3)) * u.au


def _make_light_curve_tree(tmp_path: Path) -> Path:
    """Build a Manifest-shaped orbit directory with one cutout FITS and its matching ePSF."""
    ccd_directory = tmp_path / f"orbit-{ORBIT}" / "ffi" / "cam1" / "ccd1"
    source_directory = ccd_directory / "source"
    epsf_directory = ccd_directory / "epsf"
    for directory in (source_directory, epsf_directory):
        directory.mkdir(parents=True)

    cutout = make_synthetic_cutout()
    write_cutout_fits(cutout, source_directory / "source_0_0.fits")
    epsf = EPSF(
        make_synthetic_epsf(n_cadences=len(cutout.time)),
        psf_size=11,
        oversample=2,
        orbit=cutout.orbit,
        sector=cutout.sector,
        camera=cutout.camera,
        ccd=cutout.ccd,
        cutout_x=0,
        cutout_y=0,
    )
    write_epsf_fits(epsf, epsf_directory / "epsf_0_0.fits")
    return ccd_directory / "LC"


def _light_curve_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    settings = {
        "tglc_data_dir": tmp_path,
        "orbit": ORBIT,
        "ccd": [(1, 1)],
        "cutout": None,
        "nprocs": 1,
        "replace": False,
        "tic": None,
        "tic_file": None,
        "light_curve_max_magnitude": None,
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


def _light_curve_tic_ids(light_curve_directory: Path) -> list[int]:
    return sorted(int(file.stem) for file in light_curve_directory.glob("*.h5"))


def test_make_light_curves_main_produces_all_targets(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    light_curve_directory = _make_light_curve_tree(tmp_path)

    make_light_curves_main(_light_curve_args(tmp_path))

    # Star 4 at x=10.5 rounds to 10, outside the size - 2.5 = 9.5 bound
    assert _light_curve_tic_ids(light_curve_directory) == [500001, 500002, 500003]


def test_make_light_curves_main_applies_magnitude_limit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    light_curve_directory = _make_light_curve_tree(tmp_path)

    make_light_curves_main(_light_curve_args(tmp_path, light_curve_max_magnitude=12.0))

    assert _light_curve_tic_ids(light_curve_directory) == [500001, 500002]


def test_make_light_curves_main_adds_tic_file_targets_to_magnitude_limit(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    light_curve_directory = _make_light_curve_tree(tmp_path)
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("500003\n")

    make_light_curves_main(
        _light_curve_args(tmp_path, tic_file=tic_id_file, light_curve_max_magnitude=12.0)
    )

    assert _light_curve_tic_ids(light_curve_directory) == [500001, 500002, 500003]


def test_make_light_curves_main_tic_file_alone_restricts_targets(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    light_curve_directory = _make_light_curve_tree(tmp_path)
    tic_id_file = tmp_path / "targets.txt"
    tic_id_file.write_text("500002\n500003\n")

    make_light_curves_main(_light_curve_args(tmp_path, tic_file=tic_id_file))

    assert _light_curve_tic_ids(light_curve_directory) == [500002, 500003]


def test_make_light_curves_main_warns_about_tic_ids_never_found(
    tmp_path: Path, monkeypatch, caplog
):
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    light_curve_directory = _make_light_curve_tree(tmp_path)
    tic_id_file = tmp_path / "targets.txt"
    # 500004 is in the cutout catalog but outside the pixel bounds; 999999 isn't in it at all
    tic_id_file.write_text("500002\n500004\n999999\n")

    with caplog.at_level(logging.DEBUG):
        make_light_curves_main(_light_curve_args(tmp_path, tic_file=tic_id_file))

    assert _light_curve_tic_ids(light_curve_directory) == [500002]
    assert "2 of 3 requested TIC IDs were not found" in caplog.text
    assert "500004, 999999" in caplog.text
