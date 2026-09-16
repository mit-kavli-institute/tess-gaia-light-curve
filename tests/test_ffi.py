"""Tests for the :class:`tglc.ffi.FFICutout` wrapper class."""

import inspect
import warnings

from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn
from astropy.time import Time
import astropy.units as u
from erfa.core import ErfaWarning
import numpy as np
import pytest

from tglc.ffi import FFICutout, ffi
from tglc.utils.constants import DEFAULT_FILTER_MARGIN
from tglc.utils.proper_motion import propagate_gaia_catalog

from .synthetic_data import (
    make_constructed_cutout,
    make_synthetic_cutout,
    make_synthetic_gaia_catalog,
    make_synthetic_wcs,
)


# Epochs used by make_synthetic_gaia_catalog's default propagation: Julian year 2026.0,
# exactly 10 years after the Gaia DR3 reference epoch J2016.0. make_constructed_cutout's
# median cadence, TJD 4041.5, matches (2026.0), though the epoch now comes from the
# catalog meta rather than the cadence times.
GAIA_REFERENCE_EPOCH = Time(2016.0, format="jyear", scale="tdb")
OBSERVATION_EPOCH = Time(4041.5, format="tjd", scale="tdb")


def _oracle_local_position(wcs, ra, dec, pmra, pmdec):
    """Independently propagate one star and map it to cutout-local pixels (ccd_x=44)."""
    coordinate = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=GAIA_REFERENCE_EPOCH,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        moved = coordinate.apply_space_motion(new_obstime=OBSERVATION_EPOCH)
        pixel_x, pixel_y = wcs.world_to_pixel(moved)
    return float(pixel_x) - 44.0, float(pixel_y)


def _oracle_propagated_radec(ra, dec, pmra, pmdec):
    """Independently propagate one star and return its observation-epoch RA/Dec in degrees."""
    coordinate = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=GAIA_REFERENCE_EPOCH,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        moved = coordinate.apply_space_motion(new_obstime=OBSERVATION_EPOCH)
    return float(moved.ra.deg), float(moved.dec.deg)


def test_ffi_cutout_repr():
    cutout = make_synthetic_cutout()
    assert repr(cutout) == (
        "<FFICutout orbit-185 cam1-ccd1 cutout (0, 0) size=12 cadences=5 gaia=4 tic=4>"
    )


def test_ffi_cutout_repr_legacy_missing_cutout_indices():
    """Legacy pickles predate cutout_x/cutout_y; repr must not raise on them."""
    cutout = make_synthetic_cutout()
    del cutout.cutout_x
    del cutout.cutout_y
    assert "cutout (-1, -1)" in repr(cutout)


def test_ffi_cutout_star_positions():
    cutout = make_synthetic_cutout()

    star_positions = cutout.star_positions

    assert star_positions.shape == (4, 2)
    np.testing.assert_array_equal(
        star_positions,
        np.array(
            [cutout.gaia[f"sector_{cutout.sector}_x"], cutout.gaia[f"sector_{cutout.sector}_y"]]
        ).T,
    )


# ---------------------------------------------------------------------
# FFICutout.__init__: proper-motion propagation and catalog handling
# ---------------------------------------------------------------------


def test_init_propagates_proper_motion_matches_skycoord():
    gaia = make_synthetic_gaia_catalog(
        ra=[120.5, 120.45],
        dec=[-45.25, -45.2],
        pmra=[1000.0, 0.0],
        pmdec=[0.0, 1000.0],
        g_mag=[10.0, 11.0],  # distinct magnitudes keep the tess_mag sort order deterministic
    )

    cutout = make_constructed_cutout(gaia)

    assert len(cutout.gaia) == 2
    positions = cutout.star_positions
    expected = [
        _oracle_local_position(cutout.wcs, 120.5, -45.25, 1000.0, 0.0),
        _oracle_local_position(cutout.wcs, 120.45, -45.2, 0.0, 1000.0),
    ]
    np.testing.assert_allclose(positions, expected, atol=1e-6)

    # 1000 mas/yr over the 10-year baseline is 10 arcsec ~ 0.476 px in this WCS. The
    # tolerance discriminates no propagation (0 px) and the legacy double-cos(dec)
    # error (0.335 px at this declination) from the correct displacement.
    unpropagated = _oracle_local_position(cutout.wcs, 120.5, -45.25, 0.0, 0.0)
    assert abs(positions[0][0] - unpropagated[0]) == pytest.approx(0.476, abs=0.02)
    assert abs(positions[0][1] - unpropagated[1]) < 0.01

    # ra/dec keep their names but hold the propagated coordinates; the catalog
    # (reference epoch) values move to the *_ref columns.
    for i, (ra, dec, pmra, pmdec) in enumerate(
        [(120.5, -45.25, 1000.0, 0.0), (120.45, -45.2, 0.0, 1000.0)]
    ):
        assert cutout.gaia["ra_ref"][i] == ra
        assert cutout.gaia["dec_ref"][i] == dec
        expected_ra, expected_dec = _oracle_propagated_radec(ra, dec, pmra, pmdec)
        assert cutout.gaia["ra"][i] == pytest.approx(expected_ra, abs=1e-10)
        assert cutout.gaia["dec"][i] == pytest.approx(expected_dec, abs=1e-10)

    # The *_ref pixel columns hold the un-propagated positions.
    ref_positions = np.array(
        [
            cutout.gaia[f"sector_{cutout.sector}_x_ref"],
            cutout.gaia[f"sector_{cutout.sector}_y_ref"],
        ]
    ).T
    expected_ref = [
        _oracle_local_position(cutout.wcs, 120.5, -45.25, 0.0, 0.0),
        _oracle_local_position(cutout.wcs, 120.45, -45.2, 0.0, 0.0),
    ]
    np.testing.assert_allclose(ref_positions, expected_ref, atol=1e-6)
    assert abs(ref_positions[0][0] - positions[0][0]) == pytest.approx(0.476, abs=0.02)


def test_init_fully_populated_pm_catalog_does_not_crash():
    """Regression test: catalogs with no missing PM values crashed on `.mask` access."""
    gaia = make_synthetic_gaia_catalog(
        ra=[120.5, 120.45],
        dec=[-45.25, -45.2],
        pmra=[10.0, -5.0],
        pmdec=[3.0, 8.0],
        g_mag=[10.0, 11.0],
    )

    cutout = make_constructed_cutout(gaia)

    assert len(cutout.gaia) == 2
    assert isinstance(cutout.gaia["pmra"], MaskedColumn)
    assert isinstance(cutout.gaia["pmdec"], MaskedColumn)
    assert not cutout.gaia["pmra"].mask.any()
    assert not cutout.gaia["pmdec"].mask.any()


def test_init_partially_masked_pm_stays_at_catalog_position():
    gaia = make_synthetic_gaia_catalog(
        ra=[120.5, 120.45],
        dec=[-45.25, -45.2],
        pmra=MaskedColumn([1000.0, np.nan], mask=[False, True]),
        pmdec=MaskedColumn([0.0, np.nan], mask=[False, True]),
        g_mag=[10.0, 11.0],
    )

    cutout = make_constructed_cutout(gaia)

    positions = cutout.star_positions
    moved = _oracle_local_position(cutout.wcs, 120.5, -45.25, 1000.0, 0.0)
    stationary = _oracle_local_position(cutout.wcs, 120.45, -45.2, 0.0, 0.0)
    np.testing.assert_allclose(positions[0], moved, atol=1e-6)
    np.testing.assert_allclose(positions[1], stationary, atol=1e-6)
    # The missing PM survives as the NaN-masked MaskedColumn read_cutout_fits expects.
    assert bool(cutout.gaia["pmra"].mask[1])
    assert np.isnan(np.asarray(cutout.gaia["pmra"])[1])
    # The masked-PM star stays at its catalog position in both coordinate systems.
    assert cutout.gaia["ra"][1] == pytest.approx(cutout.gaia["ra_ref"][1], abs=1e-9)
    assert cutout.gaia["dec"][1] == pytest.approx(cutout.gaia["dec_ref"][1], abs=1e-9)
    for axis in ("x", "y"):
        assert cutout.gaia[f"sector_{cutout.sector}_{axis}"][1] == pytest.approx(
            cutout.gaia[f"sector_{cutout.sector}_{axis}_ref"][1], abs=1e-6
        )


def test_init_zero_pm_star_position_unchanged():
    gaia = make_synthetic_gaia_catalog(ra=[120.5], dec=[-45.25], pmra=[0.0], pmdec=[0.0])

    cutout = make_constructed_cutout(gaia)

    raw_x, raw_y = cutout.wcs.world_to_pixel(SkyCoord(120.5 * u.deg, -45.25 * u.deg))
    np.testing.assert_allclose(
        cutout.star_positions[0], [float(raw_x) - 44.0, float(raw_y)], atol=1e-6
    )
    # With zero PM the propagated and reference-epoch columns agree.
    assert cutout.gaia["ra"][0] == pytest.approx(cutout.gaia["ra_ref"][0], abs=1e-9)
    assert cutout.gaia["dec"][0] == pytest.approx(cutout.gaia["dec_ref"][0], abs=1e-9)
    for axis in ("x", "y"):
        assert cutout.gaia[f"sector_{cutout.sector}_{axis}"][0] == pytest.approx(
            cutout.gaia[f"sector_{cutout.sector}_{axis}_ref"][0], abs=1e-6
        )


def test_init_all_pm_masked_does_not_crash():
    gaia = make_synthetic_gaia_catalog(
        ra=[120.5, 120.45],
        dec=[-45.25, -45.2],
        pmra=MaskedColumn([np.nan, np.nan], mask=[True, True]),
        pmdec=MaskedColumn([np.nan, np.nan], mask=[True, True]),
        g_mag=[10.0, 11.0],
    )

    cutout = make_constructed_cutout(gaia)

    assert len(cutout.gaia) == 2
    for i, (ra, dec) in enumerate([(120.5, -45.25), (120.45, -45.2)]):
        np.testing.assert_allclose(
            cutout.star_positions[i],
            _oracle_local_position(cutout.wcs, ra, dec, 0.0, 0.0),
            atol=1e-6,
        )
    assert cutout.gaia["pmra"].mask.all()


def test_init_spatial_filter_uses_propagated_positions():
    wcs = make_synthetic_wcs()
    outside = wcs.pixel_to_world(43.2, 74.0)  # 0.8 px outside the window's low-x edge
    inside = wcs.pixel_to_world(44.8, 74.0)  # 0.8 px inside
    # ~4.8 px over the 10-year baseline; find the sign that moves stars toward +x.
    pm_magnitude = 10000.0
    ra_out, dec_out = outside.ra.deg, outside.dec.deg
    ra_in, dec_in = inside.ra.deg, inside.dec.deg
    if _oracle_local_position(wcs, ra_out, dec_out, pm_magnitude, 0.0)[0] > 0:
        pm_toward_positive_x = pm_magnitude
    else:
        pm_toward_positive_x = -pm_magnitude

    gaia = make_synthetic_gaia_catalog(
        ra=[ra_out, ra_out, ra_in],
        dec=[dec_out, dec_out, dec_in],
        # Star 0 starts outside and PM carries it in; star 1 starts at the same place and
        # moves further out; star 2 starts inside and PM carries it out.
        pmra=[pm_toward_positive_x, -pm_toward_positive_x, -pm_toward_positive_x],
        pmdec=[0.0, 0.0, 0.0],
        g_mag=[10.0, 11.0, 12.0],
    )

    cutout = make_constructed_cutout(gaia)

    assert list(cutout.gaia["designation"]) == ["Gaia DR3 9000"]
    np.testing.assert_allclose(
        cutout.star_positions[0],
        _oracle_local_position(wcs, ra_out, dec_out, pm_toward_positive_x, 0.0),
        atol=1e-6,
    )


def test_init_filter_margin_admits_halo_stars():
    """A positive filter margin admits stars just outside the cutout window."""
    wcs = make_synthetic_wcs()
    # make_constructed_cutout: size=150, window [44, 194] x [0, 150] in CCD pixels.
    halo_low = wcs.pixel_to_world(39.0, 74.0)  # cutout-local (-5, 74)
    halo_high = wcs.pixel_to_world(120.0, 155.0)  # cutout-local (76, 155)
    inside = wcs.pixel_to_world(100.0, 75.0)  # cutout-local (56, 75)
    gaia = make_synthetic_gaia_catalog(
        ra=[halo_low.ra.deg, halo_high.ra.deg, inside.ra.deg],
        dec=[halo_low.dec.deg, halo_high.dec.deg, inside.dec.deg],
        pmra=[0.0, 0.0, 0.0],
        pmdec=[0.0, 0.0, 0.0],
        g_mag=[10.0, 11.0, 12.0],
    )

    narrow = make_constructed_cutout(gaia)  # helper pins filter_margin=0.0
    assert list(narrow.gaia["designation"]) == ["Gaia DR3 9002"]
    assert narrow.filter_margin == 0.0

    wide = make_constructed_cutout(gaia, filter_margin=6.0)
    assert list(wide.gaia["designation"]) == ["Gaia DR3 9000", "Gaia DR3 9001", "Gaia DR3 9002"]
    assert wide.filter_margin == 6.0
    assert wide.gaia[f"sector_{wide.sector}_x"][0] == pytest.approx(-5.0, abs=1e-3)
    assert wide.gaia[f"sector_{wide.sector}_y"][1] == pytest.approx(155.0, abs=1e-3)


def test_filter_margin_defaults_to_six():
    assert DEFAULT_FILTER_MARGIN == 6.0
    for function in (FFICutout.__init__, FFICutout.derive_catalogs, ffi):
        assert (
            inspect.signature(function).parameters["filter_margin"].default == DEFAULT_FILTER_MARGIN
        ), function.__qualname__


def test_init_empty_gaia_selection():
    gaia = make_synthetic_gaia_catalog(ra=[125.0], dec=[-40.0], pmra=[0.0], pmdec=[0.0])

    cutout = make_constructed_cutout(gaia)

    assert len(cutout.gaia) == 0
    assert cutout.star_positions.shape == (0, 2)
    # The pre/post-propagation column schema is stable even for empty selections.
    for name in (
        "ra",
        "dec",
        "ra_ref",
        "dec_ref",
        f"sector_{cutout.sector}_x",
        f"sector_{cutout.sector}_y",
        f"sector_{cutout.sector}_x_ref",
        f"sector_{cutout.sector}_y_ref",
    ):
        assert name in cutout.gaia.colnames


def test_init_records_propagation_epochs():
    gaia = make_synthetic_gaia_catalog(ra=[120.5], dec=[-45.25], pmra=[0.0], pmdec=[0.0])

    cutout = make_constructed_cutout(gaia)

    assert cutout.pm_epoch == pytest.approx(2026.0, abs=1e-9)
    assert cutout.pm_reference_epoch == 2016.0


def test_init_ref_epoch_flows_from_catalog_meta():
    gaia = make_synthetic_gaia_catalog(
        ra=[120.5], dec=[-45.25], pmra=[0.0], pmdec=[0.0], propagate=False
    )
    gaia["ref_epoch"] = np.array([2015.5]) * u.yr
    propagate_gaia_catalog(gaia, 2026.0)

    cutout = make_constructed_cutout(gaia)

    assert cutout.pm_reference_epoch == pytest.approx(2015.5)


def test_init_rejects_unpropagated_catalog():
    gaia = make_synthetic_gaia_catalog(
        ra=[120.5], dec=[-45.25], pmra=[0.0], pmdec=[0.0], propagate=False
    )

    with pytest.raises(ValueError, match="not proper-motion propagated"):
        make_constructed_cutout(gaia)
