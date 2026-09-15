"""Tests for `tglc.io` FITS read/write and legacy migration helpers."""

from pathlib import Path
import pickle

from astropy.io import fits
from astropy.table import MaskedColumn
from astropy.time import Time
import numpy as np
import pytest

from tglc.epsf import EPSF, EPSF_BACKGROUND_COLUMNS
from tglc.ffi import FFICutout, Source
from tglc.io import (
    migrate_cutout_pickle,
    migrate_epsf_npy,
    read_cutout_fits,
    read_epsf_fits,
    write_cutout_fits,
    write_epsf_fits,
)

from .synthetic_data import (
    make_constructed_cutout,
    make_legacy_synthetic_cutout,
    make_synthetic_ccd_catalogs,
    make_synthetic_cutout,
    make_synthetic_epsf,
    make_synthetic_gaia_catalog,
    make_synthetic_tic_catalog,
    strip_cutout_to_legacy_schema,
)


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
        "pm_epoch",
        "pm_reference_epoch",
    ):
        assert getattr(loaded, attr) == getattr(cutout, attr), attr

    np.testing.assert_array_equal(loaded.flux, cutout.flux)
    np.testing.assert_array_equal(loaded.time, cutout.time)
    np.testing.assert_array_equal(loaded.cadence, cutout.cadence)
    np.testing.assert_array_equal(loaded.quality, cutout.quality)

    assert len(loaded.gaia) == len(cutout.gaia)
    assert loaded.gaia.colnames == cutout.gaia.colnames
    assert len(loaded.tic) == len(cutout.tic)
    np.testing.assert_array_equal(loaded.tic["TIC"], cutout.tic["TIC"])


def test_cutout_fits_exposure_roundtrips_as_float(tmp_path: Path):
    """Fractional TICA EXPTIME values (e.g. 158.4) must survive the roundtrip exactly."""
    cutout = make_synthetic_cutout()
    cutout.exposure = 158.4
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    loaded = read_cutout_fits(fits_path)
    assert isinstance(loaded.exposure, float)
    assert loaded.exposure == 158.4


def test_read_cutout_fits_promotes_legacy_truncated_exposure(tmp_path: Path):
    """Files written before the float fix stored int(EXPTIME); reads recover the exact value."""
    cutout = make_synthetic_cutout()  # sector 89: effective exposure 158.4
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)
    # Simulate a legacy file, which stored the truncated integer 158.
    fits.setval(fits_path, "EXPOSURE", value=158)

    loaded = read_cutout_fits(fits_path)
    assert loaded.exposure == 158.4


def test_read_cutout_fits_keeps_exposure_not_matching_truncation(tmp_path: Path):
    """Values that aren't the sector's truncated effective exposure pass through unchanged."""
    cutout = make_synthetic_cutout()
    assert cutout.exposure == 200  # synthetic value, not int(158.4)
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    loaded = read_cutout_fits(fits_path)
    assert loaded.exposure == 200


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


def test_cutout_fits_roundtrip_preserves_pm_epoch_and_positions(tmp_path: Path):
    """Constructor-built cutouts persist their propagated positions and PM epochs."""
    gaia = make_synthetic_gaia_catalog(
        ra=[120.5, 120.45],
        dec=[-45.25, -45.2],
        pmra=[1000.0, -250.0],
        pmdec=[-100.0, 500.0],
        g_mag=[10.0, 11.0],
    )
    cutout = make_constructed_cutout(gaia)
    fits_path = tmp_path / "source_0_0.fits"

    write_cutout_fits(cutout, fits_path)

    header = fits.getheader(fits_path)
    assert header["PMEPOCH"] == pytest.approx(cutout.pm_epoch)
    assert header["PMREFEP"] == pytest.approx(cutout.pm_reference_epoch)

    loaded = read_cutout_fits(fits_path)
    assert loaded.pm_epoch == pytest.approx(cutout.pm_epoch)
    assert loaded.pm_reference_epoch == pytest.approx(cutout.pm_reference_epoch)
    np.testing.assert_array_equal(loaded.star_positions, cutout.star_positions)
    # Both the propagated and reference-epoch coordinate columns survive the roundtrip.
    for name in (
        "ra",
        "dec",
        "ra_ref",
        "dec_ref",
        f"sector_{cutout.sector}_x_ref",
        f"sector_{cutout.sector}_y_ref",
    ):
        np.testing.assert_array_equal(
            np.asarray(loaded.gaia[name]), np.asarray(cutout.gaia[name]), err_msg=name
        )


def test_read_cutout_fits_without_pm_epoch_is_none(tmp_path: Path):
    """Files written before PM propagation lack the keywords; the reader yields None."""
    cutout = make_synthetic_cutout()
    del cutout.pm_epoch
    del cutout.pm_reference_epoch
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    header = fits.getheader(fits_path)
    assert "PMEPOCH" not in header
    assert "PMREFEP" not in header

    loaded = read_cutout_fits(fits_path)
    assert loaded.pm_epoch is None
    assert loaded.pm_reference_epoch is None


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


def _make_synthetic_epsf_product(**metadata) -> EPSF:
    kwargs = {
        "psf_size": 11,
        "oversample": 2,
        "orbit": 185,
        "sector": 89,
        "camera": 1,
        "ccd": 1,
        "cutout_x": 0,
        "cutout_y": 0,
    }
    kwargs.update(metadata)
    return EPSF(make_synthetic_epsf(), **kwargs)


def _assert_provenance_keywords(header: fits.Header):
    assert header["ORIGIN"] == "MIT/TSO"
    assert header["CREATOR"] == "tglc"
    assert isinstance(header["PROCVER"], str) and header["PROCVER"]
    # DATE must parse as a FITS-format timestamp.
    Time(header["DATE"], format="fits")


def test_cutout_fits_provenance_keywords(tmp_path: Path):
    cutout = make_synthetic_cutout()
    fits_path = tmp_path / "source_0_0.fits"
    write_cutout_fits(cutout, fits_path)

    _assert_provenance_keywords(fits.getheader(fits_path))


def test_epsf_fits_provenance_keywords(tmp_path: Path):
    epsf = _make_synthetic_epsf_product()
    fits_path = tmp_path / "epsf_0_0.fits"
    write_epsf_fits(epsf, fits_path)

    _assert_provenance_keywords(fits.getheader(fits_path))


def test_write_epsf_fits_roundtrip(tmp_path: Path):
    epsf = _make_synthetic_epsf_product()
    fits_path = tmp_path / "epsf_0_0.fits"

    write_epsf_fits(epsf, fits_path)
    assert fits_path.is_file()

    loaded = read_epsf_fits(fits_path)
    assert isinstance(loaded, EPSF)
    np.testing.assert_array_equal(loaded.array, epsf.array)
    for attr in (
        "psf_size",
        "oversample",
        "orbit",
        "sector",
        "camera",
        "ccd",
        "cutout_x",
        "cutout_y",
        "background_columns",
    ):
        assert getattr(loaded, attr) == getattr(epsf, attr), attr
    assert loaded.background_columns == EPSF_BACKGROUND_COLUMNS


def test_epsf_to_fits_from_fits_roundtrip(tmp_path: Path):
    epsf = _make_synthetic_epsf_product(cutout_x=1, cutout_y=2)
    fits_path = tmp_path / "epsf_1_2.fits"

    epsf.to_fits(fits_path)
    loaded = EPSF.from_fits(fits_path)
    np.testing.assert_array_equal(loaded.array, epsf.array)
    assert (loaded.cutout_x, loaded.cutout_y) == (1, 2)


# ---------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------


def test_migrate_cutout_pickle(tmp_path: Path):
    cutout = make_legacy_synthetic_cutout()
    gaia_catalog, tic_catalog = make_synthetic_ccd_catalogs()
    pkl_path = tmp_path / "source_0_0.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(cutout, fp, pickle.HIGHEST_PROTOCOL)

    fits_path = migrate_cutout_pickle(pkl_path, gaia_catalog=gaia_catalog, tic_catalog=tic_catalog)
    assert fits_path == pkl_path.with_suffix(".fits")
    assert fits_path.is_file()
    assert pkl_path.is_file()  # default does NOT delete original

    loaded = read_cutout_fits(fits_path)
    np.testing.assert_array_equal(loaded.flux, cutout.flux)
    np.testing.assert_array_equal(loaded.mask.data, cutout.mask.data)
    np.testing.assert_array_equal(loaded.mask.mask, cutout.mask.mask)
    # The catalog tables are re-derived, so the migrated file carries the PM epochs and
    # the full current schema rather than the legacy pickle's stale tables.
    assert fits.getheader(fits_path)["PMEPOCH"] is not None
    assert len(loaded.gaia) > 0
    assert "ra_ref" in loaded.gaia.colnames
    assert f"sector_{loaded.sector}_x_ref" in loaded.gaia.colnames


def test_migrate_cutout_pickle_legacy_source_class(tmp_path: Path):
    """Old pickles reference tglc.ffi.Source by name; the alias keeps load() working."""
    cutout = make_legacy_synthetic_cutout()
    gaia_catalog, tic_catalog = make_synthetic_ccd_catalogs()
    # Simulate the legacy class name in the pickle stream by writing via the alias.
    assert Source is FFICutout

    pkl_path = tmp_path / "source_legacy.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(cutout, fp, pickle.HIGHEST_PROTOCOL)

    fits_path = migrate_cutout_pickle(
        pkl_path, gaia_catalog=gaia_catalog, tic_catalog=tic_catalog, delete_original=True
    )
    assert not pkl_path.exists()
    assert fits_path.is_file()


def test_migrate_cutout_pickle_recovers_truncated_exposure(tmp_path: Path):
    """Legacy pickles stored int(EXPTIME); the migrated FITS file carries the exact value."""
    cutout = make_legacy_synthetic_cutout()  # sector 89: effective exposure 158.4
    cutout.exposure = 158
    gaia_catalog, tic_catalog = make_synthetic_ccd_catalogs()
    pkl_path = tmp_path / "source_0_0.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(cutout, fp, pickle.HIGHEST_PROTOCOL)

    fits_path = migrate_cutout_pickle(pkl_path, gaia_catalog=gaia_catalog, tic_catalog=tic_catalog)

    assert fits.getval(fits_path, "EXPOSURE") == 158.4
    assert read_cutout_fits(fits_path).exposure == 158.4


def test_migrate_cutout_pickle_sets_cutout_xy(tmp_path: Path):
    """Legacy pickles predate cutout_x/cutout_y; callers can supply them from the file name."""
    cutout = make_legacy_synthetic_cutout()
    del cutout.cutout_x
    del cutout.cutout_y
    gaia_catalog, tic_catalog = make_synthetic_ccd_catalogs()
    pkl_path = tmp_path / "source_3_5.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(cutout, fp, pickle.HIGHEST_PROTOCOL)

    fits_path = migrate_cutout_pickle(
        pkl_path, gaia_catalog=gaia_catalog, tic_catalog=tic_catalog, cutout_x=3, cutout_y=5
    )

    loaded = read_cutout_fits(fits_path)
    assert loaded.cutout_x == 3
    assert loaded.cutout_y == 5


def test_migrate_cutout_pickle_rederives_catalogs(tmp_path: Path):
    """Migration re-derives the catalog tables exactly as a fresh construction would."""
    pmra_values = np.array([1000.0, np.nan], dtype=np.float64)
    pmdec_values = np.array([0.0, np.nan], dtype=np.float64)
    gaia_catalog = make_synthetic_gaia_catalog(
        ra=[120.5, 120.45],
        dec=[-45.25, -45.2],
        pmra=MaskedColumn(pmra_values, mask=np.isnan(pmra_values)),
        pmdec=MaskedColumn(pmdec_values, mask=np.isnan(pmdec_values)),
        g_mag=[10.0, 11.0],
    )
    tic_catalog = make_synthetic_tic_catalog(ra=(120.5, 120.45), dec=(-45.25, -45.2))
    oracle = make_constructed_cutout(gaia_catalog, tic_catalog)
    legacy = strip_cutout_to_legacy_schema(make_constructed_cutout(gaia_catalog, tic_catalog))
    pkl_path = tmp_path / "source_0_0.pkl"
    with pkl_path.open("wb") as fp:
        pickle.dump(legacy, fp, pickle.HIGHEST_PROTOCOL)

    fits_path = migrate_cutout_pickle(pkl_path, gaia_catalog=gaia_catalog, tic_catalog=tic_catalog)

    header = fits.getheader(fits_path)
    assert header["PMEPOCH"] == pytest.approx(2026.0)
    assert header["PMREFEP"] == pytest.approx(2016.0)

    loaded = read_cutout_fits(fits_path)
    assert loaded.pm_epoch == pytest.approx(oracle.pm_epoch)
    assert loaded.pm_reference_epoch == oracle.pm_reference_epoch
    assert loaded.gaia.colnames == oracle.gaia.colnames
    for name in oracle.gaia.colnames:
        if name == "designation":
            assert list(loaded.gaia[name]) == list(oracle.gaia[name])
        else:
            np.testing.assert_allclose(
                np.asarray(loaded.gaia[name], dtype=np.float64),
                np.asarray(oracle.gaia[name], dtype=np.float64),
                err_msg=name,
            )
    np.testing.assert_array_equal(loaded.gaia["pmra"].mask, oracle.gaia["pmra"].mask)
    np.testing.assert_array_equal(loaded.gaia["pmdec"].mask, oracle.gaia["pmdec"].mask)
    np.testing.assert_array_equal(loaded.tic["TIC"], oracle.tic["TIC"])
    np.testing.assert_array_equal(loaded.tic["gaia3"], oracle.tic["gaia3"])


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

    loaded = read_epsf_fits(fits_path)
    np.testing.assert_array_equal(loaded.array, epsf)
    assert loaded.psf_size == 11
    assert loaded.orbit == 185


def test_migrate_epsf_npy_rejects_mismatched_shape(tmp_path: Path):
    npy_path = tmp_path / "epsf_0_0.npy"
    np.save(npy_path, np.zeros((3, 10)))

    with pytest.raises(ValueError, match="expected"):
        migrate_epsf_npy(
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
    assert not npy_path.with_suffix(".fits").exists()


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
