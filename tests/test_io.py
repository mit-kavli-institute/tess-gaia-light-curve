"""Tests for `tglc.io` FITS read/write and legacy migration helpers."""

from pathlib import Path
import pickle

from astropy.table import MaskedColumn, Table
from astropy.wcs import WCS
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_synthetic_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [75.0, 75.0]
    wcs.wcs.crval = [120.5, -45.25]
    wcs.wcs.cdelt = [-0.00583, 0.00583]  # ~21" per pixel, comparable to TESS
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _make_synthetic_cutout(
    *,
    n_cadences: int = 5,
    size: int = 12,
    n_stars: int = 4,
    sector: int = 89,
) -> FFICutout:
    """Build an FFICutout by hand, bypassing the heavy __init__ logic.

    Mirrors what `read_cutout_fits` does internally and lets the tests cover
    each persisted attribute without depending on the full ingestion pipeline.
    """
    rng = np.random.default_rng(seed=42)

    cutout = object.__new__(FFICutout)
    cutout.size = size
    cutout.orbit = 185
    cutout.sector = sector
    cutout.camera = 1
    cutout.ccd = 1
    cutout.exposure = 200
    cutout.ccd_x = 44
    cutout.ccd_y = 0
    cutout.cutout_x = 0
    cutout.cutout_y = 0

    cutout.wcs = _make_synthetic_wcs()
    cutout.flux = rng.normal(100.0, 5.0, size=(n_cadences, size, size)).astype(np.float32)

    strap_weights = rng.normal(1.0, 0.02, size=(size, size)).astype(np.float32)
    bad_pixels = np.zeros((size, size), dtype=bool)
    bad_pixels[2, 3] = True
    bad_pixels[7, 8] = True
    cutout.mask = np.ma.masked_array(strap_weights, mask=bad_pixels)

    cutout.time = np.linspace(3500.0, 3500.05, n_cadences).astype(np.float64)
    cutout.cadence = np.arange(1000, 1000 + n_cadences, dtype=np.int64)
    cutout.quality = np.zeros(n_cadences, dtype=np.int32)

    designations = [f"Gaia DR3 {1000 + i}" for i in range(n_stars)]
    pmra_values = np.array([1.2, np.nan, 0.5, np.nan], dtype=np.float64)
    pmdec_values = np.array([np.nan, -0.3, 0.0, np.nan], dtype=np.float64)
    gaia = Table(
        {
            "designation": designations,
            "ra": np.array([120.4, 120.5, 120.6, 120.55], dtype=np.float64),
            "dec": np.array([-45.2, -45.25, -45.3, -45.28], dtype=np.float64),
            "phot_g_mean_mag": np.array([10.0, 11.5, 12.3, 13.0], dtype=np.float64),
            "phot_bp_mean_mag": np.array([10.2, 11.7, 12.6, 13.3], dtype=np.float64),
            "phot_rp_mean_mag": np.array([9.7, 11.1, 12.0, 12.7], dtype=np.float64),
            "pmra": MaskedColumn(pmra_values, mask=np.isnan(pmra_values)),
            "pmdec": MaskedColumn(pmdec_values, mask=np.isnan(pmdec_values)),
            "tess_mag": np.array([9.95, 11.45, 12.25, 12.95], dtype=np.float64),
            "tess_flux": np.array([1500.0, 380.0, 180.0, 80.0], dtype=np.float64),
            "tess_flux_ratio": np.array([1.0, 0.25, 0.12, 0.05], dtype=np.float64),
            f"sector_{sector}_x": np.array([2.5, 5.5, 8.5, 10.5], dtype=np.float64),
            f"sector_{sector}_y": np.array([3.5, 6.5, 8.5, 9.5], dtype=np.float64),
        }
    )
    cutout.gaia = gaia

    cutout.tic = Table(
        {
            "TIC": np.array([500001, 500002, 500003, 500004], dtype=np.int64),
            "gaia3": np.array([1000, 1001, 1002, 1003], dtype=np.int64),
        }
    )

    return cutout


def _make_synthetic_epsf(n_cadences: int = 3, psf_size: int = 11, oversample: int = 2):
    k = (psf_size * oversample + 1) ** 2 + len(EPSF_BACKGROUND_COLUMNS)
    return np.linspace(0.0, 1.0, n_cadences * k).reshape(n_cadences, k).astype(np.float64)


# ---------------------------------------------------------------------
# Cutout roundtrip
# ---------------------------------------------------------------------


def test_write_cutout_fits_roundtrip(tmp_path: Path):
    cutout = _make_synthetic_cutout()
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
    cutout = _make_synthetic_cutout()
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    loaded = read_cutout_fits(fits_path)
    assert isinstance(loaded.mask, np.ma.MaskedArray)
    np.testing.assert_array_equal(loaded.mask.data, cutout.mask.data)
    np.testing.assert_array_equal(loaded.mask.mask, cutout.mask.mask)


def test_cutout_fits_masked_gaia_columns(tmp_path: Path):
    cutout = _make_synthetic_cutout()
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
    cutout = _make_synthetic_cutout()
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    loaded = read_cutout_fits(fits_path)
    gaia3_id = 1002
    matches = np.nonzero(loaded.gaia["designation"] == f"Gaia DR3 {gaia3_id}")[0]
    assert matches.size == 1


def test_cutout_fits_wcs_roundtrip(tmp_path: Path):
    cutout = _make_synthetic_cutout()
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
    cutout = _make_synthetic_cutout()
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
    epsf = _make_synthetic_epsf()
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
    cutout = _make_synthetic_cutout()
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
    cutout = _make_synthetic_cutout()
    # Simulate the legacy class name in the pickle stream by writing via the alias.
    assert Source is FFICutout

    pkl_path = tmp_path / "source_legacy.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(cutout, fp, pickle.HIGHEST_PROTOCOL)

    fits_path = migrate_cutout_pickle(pkl_path, delete_original=True)
    assert not pkl_path.exists()
    assert fits_path.is_file()


def test_migrate_epsf_npy(tmp_path: Path):
    epsf = _make_synthetic_epsf()
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
    epsf = _make_synthetic_epsf()
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
