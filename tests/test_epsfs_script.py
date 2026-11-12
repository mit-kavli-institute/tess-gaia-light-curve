"""Tests for the `tglc epsfs` script (`tglc.scripts.epsfs`)."""

import argparse
import logging
from pathlib import Path

import numpy as np

from tglc.epsf import EPSF
from tglc.io import write_cutout_fits
from tglc.scripts.epsfs import make_epsfs_main, read_source_and_fit_and_save_epsf

from .synthetic_data import make_synthetic_cutout


ORBIT = 185  # make_synthetic_cutout's orbit
N_CADENCES = 6
SCATTERED_LIGHT_CADENCES = (1, 2, 4)


def _make_source_tree(tmp_path: Path, scattered_light_cutouts: tuple[str, ...]) -> Path:
    """
    Build a Manifest-shaped orbit directory with cutouts 0_0 and 0_1.

    Cutouts named in `scattered_light_cutouts` get a negative median flux in
    `SCATTERED_LIGHT_CADENCES`. Returns the CCD directory.
    """
    ccd_directory = tmp_path / f"orbit-{ORBIT}" / "ffi" / "cam1" / "ccd1"
    source_directory = ccd_directory / "source"
    source_directory.mkdir(parents=True)
    for name in ("0_0", "0_1"):
        cutout = make_synthetic_cutout(n_cadences=N_CADENCES)
        if name in scattered_light_cutouts:
            cutout.flux[list(SCATTERED_LIGHT_CADENCES)] -= 500.0
        write_cutout_fits(cutout, source_directory / f"source_{name}.fits")
    return ccd_directory


def _epsf_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    settings = {
        "tglc_data_dir": tmp_path,
        "orbit": ORBIT,
        "ccd": [(1, 1)],
        "cutout": None,
        "nprocs": 1,
        "replace": False,
        "psf_size": 3,
        "oversample": 1,
        "edge_compression_factor": 1e-4,
        "uncertainty_power": 1.4,
        "no_gpu": True,
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


def _fit_and_save(source_file: Path, epsf_file: Path, replace: bool = False):
    return read_source_and_fit_and_save_epsf(
        (source_file, epsf_file),
        replace=replace,
        psf_size=3,
        oversample_factor=1,
        edge_compression_factor=1e-4,
        flux_uncertainty_power=1.4,
        use_gpu=False,
    )


def test_read_source_and_fit_and_save_epsf_returns_unfit_cadence_summary(tmp_path: Path):
    ccd_directory = _make_source_tree(tmp_path, scattered_light_cutouts=("0_1",))
    epsf_file = tmp_path / "epsf_0_1.fits"

    summary = _fit_and_save(ccd_directory / "source" / "source_0_1.fits", epsf_file)

    assert summary == ("source_0_1.fits", 3, N_CADENCES, 1, 4)
    failed = np.nonzero(EPSF.from_fits(epsf_file).failed_cadence_mask)[0]
    assert tuple(failed) == SCATTERED_LIGHT_CADENCES


def test_read_source_and_fit_and_save_epsf_summary_when_all_cadences_fit(tmp_path: Path):
    ccd_directory = _make_source_tree(tmp_path, scattered_light_cutouts=())
    epsf_file = tmp_path / "epsf_0_0.fits"

    summary = _fit_and_save(ccd_directory / "source" / "source_0_0.fits", epsf_file)

    assert summary == ("source_0_0.fits", 0, N_CADENCES, -1, -1)


def test_read_source_and_fit_and_save_epsf_returns_none_for_existing_file(tmp_path: Path):
    ccd_directory = _make_source_tree(tmp_path, scattered_light_cutouts=())
    source_file = ccd_directory / "source" / "source_0_0.fits"
    epsf_file = tmp_path / "epsf_0_0.fits"
    _fit_and_save(source_file, epsf_file)

    assert _fit_and_save(source_file, epsf_file) is None
    assert _fit_and_save(source_file, epsf_file, replace=True) is not None


def test_make_epsfs_main_logs_unfit_cadences(tmp_path: Path, caplog):
    ccd_directory = _make_source_tree(tmp_path, scattered_light_cutouts=("0_1",))

    with caplog.at_level(logging.INFO, logger="tglc.scripts.epsfs"):
        make_epsfs_main(_epsf_args(tmp_path))

    assert sorted(file.name for file in (ccd_directory / "epsf").glob("*.fits")) == [
        "epsf_0_0.fits",
        "epsf_0_1.fits",
    ]
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "source_0_1.fits: 3 of 6 cadences have no ePSF fit" in warnings[0]
    assert "cadence indices 1-4" in warnings[0]
    assert "ePSFs for 1-1: 3 cadences without a fit across 1 of 2 cutouts" in caplog.text


def test_make_epsfs_main_is_quiet_when_all_cadences_fit(tmp_path: Path, caplog):
    _make_source_tree(tmp_path, scattered_light_cutouts=())

    with caplog.at_level(logging.INFO, logger="tglc.scripts.epsfs"):
        make_epsfs_main(_epsf_args(tmp_path))

    assert "no ePSF fit" not in caplog.text
    assert "without a fit" not in caplog.text
