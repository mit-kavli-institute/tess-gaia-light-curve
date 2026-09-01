"""Tests for `tglc.io` FITS read/write and legacy migration helpers."""

from pathlib import Path
import pickle

from astropy.table import MaskedColumn
import numpy as np

from tglc.ffi import FFICutout, Source
from tglc.io import (
    EPSF_BACKGROUND_COLUMNS,
    migrate_cutout_pickle,
    migrate_epsf_npy,
    read_cutout_fits,
    read_epsf_fits,
    write_cutout_fits,
    write_epsf_fits,
)

from .synthetic_data import make_synthetic_cutout, make_synthetic_epsf


# ---------------------------------------------------------------------
# Cutout roundtrip
# ---------------------------------------------------------------------


def test_write_cutout_fits_roundtrip(tmp_path: Path):
    cutout = make_synthetic_cutout()
    fits_path = tmp_path / "source_0_0.fits"

    write_cutout_fits(cutout, fits_path)
    assert fits_path.is_file()

    loaded = read_cutout_fits(fits_path)
    assert isinstance(loaded, FFICutout)

    for attr in (
        "size",
        "orbit",
        "sector",
        "camera",
        "ccd",
        "ccd_x",
        "ccd_y",
        "exposure",
        "cutout_x",
        "cutout_y",
    ):
        assert getattr(loaded, attr) == getattr(cutout, attr), attr

    np.testing.assert_array_equal(loaded.flux, cutout.flux)
    np.testing.assert_array_equal(loaded.time, cutout.time)
    np.testing.assert_array_equal(loaded.cadence, cutout.cadence)
    np.testing.assert_array_equal(loaded.quality, cutout.quality)

    assert len(loaded.gaia) == len(cutout.gaia)
    assert len(loaded.tic) == len(cutout.tic)
    np.testing.assert_array_equal(loaded.tic["TIC"], cutout.tic["TIC"])


def test_cutout_fits_strap_mask_roundtrip(tmp_path: Path):
    cutout = make_synthetic_cutout()
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    loaded = read_cutout_fits(fits_path)
    assert isinstance(loaded.mask, np.ma.MaskedArray)
    np.testing.assert_array_equal(loaded.mask.data, cutout.mask.data)
    np.testing.assert_array_equal(loaded.mask.mask, cutout.mask.mask)


def test_cutout_fits_masked_gaia_columns(tmp_path: Path):
    cutout = make_synthetic_cutout()
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    loaded = read_cutout_fits(fits_path)

    pmra = loaded.gaia["pmra"]
    pmdec = loaded.gaia["pmdec"]
    assert isinstance(pmra, MaskedColumn)
    assert isinstance(pmdec, MaskedColumn)
    np.testing.assert_array_equal(pmra.mask, cutout.gaia["pmra"].mask)
    np.testing.assert_array_equal(pmdec.mask, cutout.gaia["pmdec"].mask)


def test_cutout_fits_designation_string_compares_to_str(tmp_path: Path):
    """Guards the bytes-vs-str pitfall: comparison against an f-string must work."""
    cutout = make_synthetic_cutout()
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    loaded = read_cutout_fits(fits_path)
    gaia3_id = 1002
    matches = np.nonzero(loaded.gaia["designation"] == f"Gaia DR3 {gaia3_id}")[0]
    assert matches.size == 1


def test_cutout_fits_wcs_roundtrip(tmp_path: Path):
    cutout = make_synthetic_cutout()
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    loaded = read_cutout_fits(fits_path)
    test_pixels = np.array([[5.0, 5.0], [10.0, 7.5]])
    np.testing.assert_allclose(
        loaded.wcs.all_pix2world(test_pixels, 0),
        cutout.wcs.all_pix2world(test_pixels, 0),
        atol=1e-9,
    )


def test_cutout_fits_empty_gaia(tmp_path: Path):
    cutout = make_synthetic_cutout()
    cutout.gaia = cutout.gaia[:0]
    cutout.tic = cutout.tic[:0]
    fits_path = tmp_path / "source_0_0.fits"

    write_cutout_fits(cutout, fits_path)
    loaded = read_cutout_fits(fits_path)
    assert len(loaded.gaia) == 0
    assert len(loaded.tic) == 0


# ---------------------------------------------------------------------
# ePSF roundtrip
# ---------------------------------------------------------------------


def test_write_epsf_fits_roundtrip(tmp_path: Path):
    epsf = make_synthetic_epsf()
    fits_path = tmp_path / "epsf_0_0.fits"

    write_epsf_fits(
        fits_path,
        epsf,
        psf_size=11,
        oversample=2,
        orbit=185,
        sector=89,
        camera=1,
        ccd=1,
        cutout_x=0,
        cutout_y=0,
    )
    assert fits_path.is_file()

    loaded, metadata = read_epsf_fits(fits_path)
    np.testing.assert_array_equal(loaded, epsf)
    assert metadata == {
        "psf_size": 11,
        "oversample": 2,
        "n_background": len(EPSF_BACKGROUND_COLUMNS),
        "orbit": 185,
        "sector": 89,
        "camera": 1,
        "ccd": 1,
        "cutout_x": 0,
        "cutout_y": 0,
        "background_columns": EPSF_BACKGROUND_COLUMNS,
    }


# ---------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------


def test_migrate_cutout_pickle(tmp_path: Path):
    cutout = make_synthetic_cutout()
    pkl_path = tmp_path / "source_0_0.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(cutout, fp, pickle.HIGHEST_PROTOCOL)

    fits_path = migrate_cutout_pickle(pkl_path)
    assert fits_path == pkl_path.with_suffix(".fits")
    assert fits_path.is_file()
    assert pkl_path.is_file()  # default does NOT delete original

    loaded = read_cutout_fits(fits_path)
    np.testing.assert_array_equal(loaded.flux, cutout.flux)
    np.testing.assert_array_equal(loaded.mask.data, cutout.mask.data)
    np.testing.assert_array_equal(loaded.mask.mask, cutout.mask.mask)


def test_migrate_cutout_pickle_legacy_source_class(tmp_path: Path):
    """Old pickles reference tglc.ffi.Source by name; the alias keeps load() working."""
    cutout = make_synthetic_cutout()
    # Simulate the legacy class name in the pickle stream by writing via the alias.
    assert Source is FFICutout

    pkl_path = tmp_path / "source_legacy.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(cutout, fp, pickle.HIGHEST_PROTOCOL)

    fits_path = migrate_cutout_pickle(pkl_path, delete_original=True)
    assert not pkl_path.exists()
    assert fits_path.is_file()


def test_migrate_cutout_pickle_sets_cutout_xy(tmp_path: Path):
    """Legacy pickles predate cutout_x/cutout_y; callers can supply them from the file name."""
    cutout = make_synthetic_cutout()
    del cutout.cutout_x
    del cutout.cutout_y
    pkl_path = tmp_path / "source_3_5.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(cutout, fp, pickle.HIGHEST_PROTOCOL)

    fits_path = migrate_cutout_pickle(pkl_path, cutout_x=3, cutout_y=5)

    loaded = read_cutout_fits(fits_path)
    assert loaded.cutout_x == 3
    assert loaded.cutout_y == 5


def test_migrate_epsf_npy(tmp_path: Path):
    epsf = make_synthetic_epsf()
    npy_path = tmp_path / "epsf_0_0.npy"
    np.save(npy_path, epsf)

    fits_path = migrate_epsf_npy(
        npy_path,
        psf_size=11,
        oversample=2,
        orbit=185,
        sector=89,
        camera=1,
        ccd=1,
        cutout_x=0,
        cutout_y=0,
    )
    assert fits_path == npy_path.with_suffix(".fits")
    assert npy_path.is_file()  # default does NOT delete original

    loaded, metadata = read_epsf_fits(fits_path)
    np.testing.assert_array_equal(loaded, epsf)
    assert metadata["psf_size"] == 11
    assert metadata["orbit"] == 185


def test_migrate_epsf_npy_delete_original(tmp_path: Path):
    epsf = make_synthetic_epsf()
    npy_path = tmp_path / "epsf_0_0.npy"
    np.save(npy_path, epsf)

    fits_path = migrate_epsf_npy(
        npy_path,
        psf_size=11,
        oversample=2,
        orbit=185,
        sector=89,
        camera=1,
        ccd=1,
        cutout_x=0,
        cutout_y=0,
        delete_original=True,
    )
    assert not npy_path.exists()
    assert fits_path.is_file()
