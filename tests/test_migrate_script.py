"""Tests for the `tglc migrate` script (`tglc.scripts.migrate`)."""

import argparse
import logging
from pathlib import Path
import pickle

from astropy.io import fits
from astropy.table import QTable
import numpy as np

from tglc.io import read_cutout_fits, write_cutout_fits
from tglc.proper_motion import catalog_is_propagated
from tglc.scripts.migrate import _load_catalogs, migrate_main
from tglc.utils.constants import DEFAULT_FILTER_MARGIN, get_orbit_midtime
from tglc.utils.manifest import Manifest

from .synthetic_data import (
    make_legacy_synthetic_cutout,
    make_synthetic_ccd_catalogs,
    make_synthetic_epsf,
)


ORBIT = 185  # make_synthetic_cutout's orbit; contained in sector 89


def _make_migration_tree(
    tmp_path: Path, *, with_catalogs: bool = True, propagated_catalogs: bool = True
) -> tuple[Path, Path]:
    """Build a Manifest-shaped orbit directory with one legacy pickle and one ePSF .npy."""
    ffi_directory = tmp_path / f"orbit-{ORBIT}" / "ffi"
    catalog_directory = ffi_directory / "catalogs"
    source_directory = ffi_directory / "cam1" / "ccd1" / "source"
    epsf_directory = ffi_directory / "cam1" / "ccd1" / "epsf"
    for directory in (catalog_directory, source_directory, epsf_directory):
        directory.mkdir(parents=True)

    if with_catalogs:
        gaia_catalog, tic_catalog = make_synthetic_ccd_catalogs(propagate=propagated_catalogs)
        gaia_catalog.write(catalog_directory / "Gaia_cam1_ccd1.ecsv", format="ascii.ecsv")
        tic_catalog.write(catalog_directory / "TIC_cam1_ccd1.ecsv", format="ascii.ecsv")

    pkl_path = source_directory / "source_0_0.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(make_legacy_synthetic_cutout(), fp, pickle.HIGHEST_PROTOCOL)
    npy_path = epsf_directory / "epsf_0_0.npy"
    np.save(npy_path, make_synthetic_epsf())
    return pkl_path, npy_path


def _migrate_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    settings = {
        "tglc_data_dir": tmp_path,
        "orbit": ORBIT,
        "ccd": [(1, 1)],
        "cutout": None,
        "nprocs": 1,
        "replace": False,
        "delete_original": False,
        "psf_size": 11,
        "oversample": 2,
        "filter_margin": DEFAULT_FILTER_MARGIN,
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


def test_migrate_main_migrates_cutouts_and_epsfs(tmp_path: Path):
    pkl_path, npy_path = _make_migration_tree(tmp_path)

    migrate_main(_migrate_args(tmp_path))

    source_fits = pkl_path.with_suffix(".fits")
    assert source_fits.is_file()
    assert npy_path.with_suffix(".fits").is_file()
    # The migrated cutout carries re-derived catalogs: PM epochs and the *_ref columns.
    assert fits.getheader(source_fits)["PMEPOCH"] is not None
    loaded = read_cutout_fits(source_fits)
    assert len(loaded.gaia) > 0
    assert "ra_ref" in loaded.gaia.colnames
    assert f"sector_{loaded.sector}_x_ref" in loaded.gaia.colnames


def test_migrate_main_multiprocessing(tmp_path: Path):
    """nprocs > 1 uses worker processes: work items and config must survive pickling.

    Under a "spawn" start method the workers' catalog cache starts empty, also exercising
    the read-from-path fallback ("fork" workers inherit the parent's pre-loaded cache).
    """
    pkl_path, npy_path = _make_migration_tree(tmp_path)

    migrate_main(_migrate_args(tmp_path, nprocs=2))

    source_fits = pkl_path.with_suffix(".fits")
    assert source_fits.is_file()
    assert npy_path.with_suffix(".fits").is_file()
    assert fits.getheader(source_fits)["PMEPOCH"] is not None


def test_migrate_main_upgrades_old_format_gaia_catalog(tmp_path: Path):
    """An old-format Gaia ECSV is propagated once per CCD and rewritten on disk."""
    pkl_path, _ = _make_migration_tree(tmp_path, propagated_catalogs=False)
    gaia_catalog_file = tmp_path / f"orbit-{ORBIT}" / "ffi" / "catalogs" / "Gaia_cam1_ccd1.ecsv"
    assert not catalog_is_propagated(QTable.read(gaia_catalog_file))

    migrate_main(_migrate_args(tmp_path))

    # The catalog file is rewritten in the propagated format with the orbit mid-time epoch.
    upgraded = QTable.read(gaia_catalog_file)
    assert catalog_is_propagated(upgraded)
    assert upgraded.meta["pm_orbit"] == ORBIT
    assert upgraded.meta["pm_epoch"] == float(get_orbit_midtime(ORBIT).jyear)
    # The migrated cutout carries the catalog's epoch.
    source_fits = pkl_path.with_suffix(".fits")
    assert fits.getheader(source_fits)["PMEPOCH"] == float(get_orbit_midtime(ORBIT).jyear)


def test_migrate_main_missing_catalogs_skips_cutouts_but_migrates_epsfs(tmp_path: Path):
    pkl_path, npy_path = _make_migration_tree(tmp_path, with_catalogs=False)

    migrate_main(_migrate_args(tmp_path))

    assert not pkl_path.with_suffix(".fits").exists()
    assert pkl_path.is_file()  # the pickle is untouched, not consumed
    assert npy_path.with_suffix(".fits").is_file()


def test_load_catalogs_warns_with_catalogs_command(tmp_path: Path, caplog):
    manifest = Manifest(tmp_path, orbit=ORBIT)

    with caplog.at_level(logging.WARNING, logger="tglc.scripts.migrate"):
        assert _load_catalogs(manifest, 1, 1) is None

    assert any("tglc catalogs" in record.message for record in caplog.records)


def test_migrate_main_redoes_stale_fits_without_replace(tmp_path: Path):
    pkl_path, npy_path = _make_migration_tree(tmp_path)
    # Simulate the old naive migration: a FITS sibling without PMEPOCH (the legacy
    # cutout has no pm attributes, so write_cutout_fits omits the keyword).
    source_fits = pkl_path.with_suffix(".fits")
    with pkl_path.open("rb") as fp:
        write_cutout_fits(pickle.load(fp), source_fits)
    assert "PMEPOCH" not in fits.getheader(source_fits)

    migrate_main(_migrate_args(tmp_path))

    assert fits.getheader(source_fits)["PMEPOCH"] is not None


def test_migrate_main_skips_current_fits_without_replace(tmp_path: Path):
    pkl_path, npy_path = _make_migration_tree(tmp_path)
    migrate_main(_migrate_args(tmp_path))
    source_fits = pkl_path.with_suffix(".fits")
    first_migration = source_fits.stat().st_mtime_ns

    migrate_main(_migrate_args(tmp_path))

    assert source_fits.stat().st_mtime_ns == first_migration
    assert npy_path.with_suffix(".fits").is_file()


def test_migrate_main_redoes_fits_with_different_filter_margin(tmp_path: Path):
    """Changing --filter-margin re-migrates existing FITS files without --replace."""
    pkl_path, npy_path = _make_migration_tree(tmp_path)
    migrate_main(_migrate_args(tmp_path, filter_margin=0.0))
    source_fits = pkl_path.with_suffix(".fits")
    assert fits.getheader(source_fits)["FILTMARG"] == 0.0

    migrate_main(_migrate_args(tmp_path, filter_margin=6.0))
    assert fits.getheader(source_fits)["FILTMARG"] == 6.0

    # Re-running with the matching margin skips the now-current file.
    unchanged = source_fits.stat().st_mtime_ns
    migrate_main(_migrate_args(tmp_path, filter_margin=6.0))
    assert source_fits.stat().st_mtime_ns == unchanged
