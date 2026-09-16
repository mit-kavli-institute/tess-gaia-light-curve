"""Tests for :mod:`tglc.proper_motion`: en-masse Gaia proper-motion propagation."""

import warnings

from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, QTable
from astropy.time import Time
import astropy.units as u
from erfa.core import ErfaWarning
from hypothesis import example, given, settings, strategies as st
from hypothesis.extra import numpy as npst
import numpy as np
import pytest

from tglc.proper_motion import (
    _MAS_YR_TO_RAD,
    catalog_is_propagated,
    load_propagated_gaia_catalog,
    propagate_coordinates,
    propagate_gaia_catalog,
    propagate_gaia_catalog_for_orbit,
    write_gaia_catalog_ecsv,
)
from tglc.utils.constants import get_orbit_midtime


def _oracle_propagate(ra, dec, pmra, pmdec, epoch_jyear, reference_epoch_jyear) -> SkyCoord:
    """Independently propagate stars with astropy's ERFA-backed `apply_space_motion`."""
    coordinates = SkyCoord(
        ra=np.atleast_1d(ra) * u.deg,
        dec=np.atleast_1d(dec) * u.deg,
        pm_ra_cosdec=np.atleast_1d(pmra) * u.mas / u.yr,
        pm_dec=np.atleast_1d(pmdec) * u.mas / u.yr,
        obstime=Time(reference_epoch_jyear, format="jyear", scale="tdb"),
    )
    with warnings.catch_warnings():
        # ERFA warns that propagating without distance/RV overrides the distance;
        # proper-motion-only propagation is intended here.
        warnings.simplefilter("ignore", ErfaWarning)
        return coordinates.apply_space_motion(
            new_obstime=Time(epoch_jyear, format="jyear", scale="tdb")
        )


RA = st.floats(min_value=0.0, max_value=360.0, exclude_max=True)
FULL_DEC = st.floats(min_value=-90.0, max_value=90.0)
# The astropy oracle converts pm_ra_cosdec to pm_ra by dividing by cos(dec), so its own
# precision degrades at the poles; oracle comparisons stay slightly away from them.
ORACLE_DEC = st.floats(min_value=-89.99, max_value=89.99)
PROPER_MOTION = st.floats(min_value=-11_000.0, max_value=11_000.0)  # Barnard's star is ~10.4"/yr
EPOCH = st.floats(min_value=1990.0, max_value=2046.0)


@st.composite
def _star_field(draw, dec_strategy=FULL_DEC):
    n = draw(st.integers(min_value=1, max_value=16))
    ra = draw(npst.arrays(np.float64, n, elements=RA))
    dec = draw(npst.arrays(np.float64, n, elements=dec_strategy))
    pmra = draw(npst.arrays(np.float64, n, elements=PROPER_MOTION))
    pmdec = draw(npst.arrays(np.float64, n, elements=PROPER_MOTION))
    return ra, dec, pmra, pmdec


# `deadline=None` on all property tests: the first call JIT-compiles the numba kernel, which
# takes far longer than any single example is allowed by the default deadline.


@settings(deadline=None)
@given(
    ra=RA,
    dec=ORACLE_DEC,
    pmra=PROPER_MOTION,
    pmdec=PROPER_MOTION,
    epoch=EPOCH,
    reference_epoch=EPOCH,
)
# Barnard's star: the largest known proper motion.
@example(ra=269.45, dec=4.69, pmra=-802.8, pmdec=10362.5, epoch=2026.0, reference_epoch=2016.0)
# RA wraparound across 0/360.
@example(ra=359.99995, dec=0.0, pmra=11_000.0, pmdec=0.0, epoch=2046.0, reference_epoch=1990.0)
# Pole-adjacent, both directions, forward and backward in time.
@example(ra=180.0, dec=89.99, pmra=11_000.0, pmdec=11_000.0, epoch=2046.0, reference_epoch=1990.0)
@example(ra=0.0, dec=-89.99, pmra=-11_000.0, pmdec=-11_000.0, epoch=1990.0, reference_epoch=2046.0)
# Zero proper motion and zero elapsed time.
@example(ra=120.5, dec=-45.25, pmra=0.0, pmdec=0.0, epoch=2026.0, reference_epoch=2016.0)
@example(ra=120.5, dec=-45.25, pmra=1000.0, pmdec=1000.0, epoch=2016.0, reference_epoch=2016.0)
def test_propagate_coordinates_matches_astropy_oracle(ra, dec, pmra, pmdec, epoch, reference_epoch):
    new_ra, new_dec = propagate_coordinates([ra], [dec], [pmra], [pmdec], epoch, reference_epoch)
    oracle = _oracle_propagate(ra, dec, pmra, pmdec, epoch, reference_epoch)
    separation = SkyCoord(ra=new_ra * u.deg, dec=new_dec * u.deg).separation(oracle)
    assert separation.to_value(u.uas).max() < 20.0


@settings(deadline=None)
@given(stars=_star_field(), epoch=EPOCH, reference_epoch=EPOCH)
def test_propagated_coordinates_stay_within_bounds(stars, epoch, reference_epoch):
    ra, dec, pmra, pmdec = stars
    new_ra, new_dec = propagate_coordinates(ra, dec, pmra, pmdec, epoch, reference_epoch)
    assert np.isfinite(new_ra).all()
    assert np.isfinite(new_dec).all()
    assert ((0.0 <= new_ra) & (new_ra < 360.0)).all()
    assert ((-90.0 <= new_dec) & (new_dec <= 90.0)).all()


@settings(deadline=None)
@given(
    stars=_star_field(dec_strategy=st.floats(min_value=-60.0, max_value=60.0)),
    epoch=EPOCH,
    reference_epoch=EPOCH,
)
def test_propagation_round_trip_returns_near_start(stars, epoch, reference_epoch):
    ra, dec, pmra, pmdec = stars
    forward_ra, forward_dec = propagate_coordinates(ra, dec, pmra, pmdec, epoch, reference_epoch)
    back_ra, back_dec = propagate_coordinates(
        forward_ra, forward_dec, pmra, pmdec, reference_epoch, epoch
    )
    separation = SkyCoord(ra=back_ra * u.deg, dec=back_dec * u.deg).separation(
        SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    )
    # The round trip is only approximate: the return leg applies the proper motion in the
    # tangent basis at the displaced position, so the residual is second-order in the total
    # displacement (empirically < 6 * theta^2 for |dec| <= 60; a sign or basis error in the
    # kernel would instead leave a first-order residual of ~2 * theta).
    displacement = np.hypot(pmra, pmdec) * _MAS_YR_TO_RAD * abs(epoch - reference_epoch)
    assert (separation.to_value(u.rad) <= 6.0 * displacement**2 + 1e-12).all()


@settings(deadline=None)
@given(stars=_star_field(), epoch=EPOCH, reference_epoch=EPOCH)
def test_zero_proper_motion_passes_through_bit_exact(stars, epoch, reference_epoch):
    ra, dec, _, _ = stars
    zeros = np.zeros_like(ra)
    new_ra, new_dec = propagate_coordinates(ra, dec, zeros, zeros, epoch, reference_epoch)
    np.testing.assert_array_equal(new_ra, ra)
    np.testing.assert_array_equal(new_dec, dec)


def test_missing_proper_motions_stay_at_catalog_positions():
    ra = np.array([10.0, 20.0, 30.0, 40.0])
    dec = np.array([-45.0, 0.0, 45.0, 80.0])
    pmra = MaskedColumn(
        [1000.0, np.nan, 1000.0, 1000.0], mask=[False, False, True, False], unit=u.mas / u.yr
    )
    pmdec = MaskedColumn(
        [1000.0, 1000.0, 1000.0, np.nan], mask=[False, False, False, False], unit=u.mas / u.yr
    )
    new_ra, new_dec = propagate_coordinates(ra, dec, pmra, pmdec, 2026.0, 2016.0)
    # Stars with NaN or masked values in either component stay at their catalog positions.
    np.testing.assert_array_equal(new_ra[1:], ra[1:])
    np.testing.assert_array_equal(new_dec[1:], dec[1:])
    assert new_ra[0] != ra[0]
    assert new_dec[0] != dec[0]


def test_empty_input():
    empty = np.array([], dtype=np.float64)
    new_ra, new_dec = propagate_coordinates(empty, empty, empty, empty, 2026.0, 2016.0)
    assert len(new_ra) == 0
    assert len(new_dec) == 0


def test_quantity_inputs_are_converted_to_expected_units():
    # 1 arcsec/yr == 1000 mas/yr; both spellings must propagate identically.
    from_quantities = propagate_coordinates(
        [120.5] * u.deg,
        [-45.25] * u.deg,
        [1.0] * u.arcsec / u.yr,
        [-0.5] * u.arcsec / u.yr,
        2026.0,
        2016.0,
    )
    from_plain_values = propagate_coordinates([120.5], [-45.25], [1000.0], [-500.0], 2026.0, 2016.0)
    np.testing.assert_array_equal(from_quantities[0], from_plain_values[0])
    np.testing.assert_array_equal(from_quantities[1], from_plain_values[1])


# ---------------------------------------------------------------------
# Catalog table propagation and the ECSV write/upgrade path
# ---------------------------------------------------------------------


def _make_unpropagated_catalog(**extra_columns) -> QTable:
    """A minimal old-format Gaia catalog table with the production column order."""
    catalog = QTable(
        {
            "designation": ["Gaia DR3 9000", "Gaia DR3 9001", "Gaia DR3 9002"],
            "phot_g_mean_mag": np.array([10.0, 11.0, 12.0]),
            "phot_bp_mean_mag": np.array([10.3, 11.3, 12.3]),
            "phot_rp_mean_mag": np.array([9.7, 10.7, 11.7]),
            "ra": np.array([120.5, 120.6, 120.7]) * u.deg,
            "dec": np.array([-45.25, -45.3, -45.35]) * u.deg,
            "pmra": MaskedColumn(
                [1000.0, np.nan, 0.0], mask=[False, True, False], unit=u.mas / u.yr
            ),
            "pmdec": MaskedColumn(
                [-500.0, 0.0, 0.0], mask=[False, False, False], unit=u.mas / u.yr
            ),
        }
    )
    for name, values in extra_columns.items():
        catalog[name] = values
    return catalog


def _assert_matches_oracle(catalog, index, ra, dec, pmra, pmdec):
    oracle = _oracle_propagate(
        ra, dec, pmra, pmdec, catalog.meta["pm_epoch"], catalog.meta["pm_reference_epoch"]
    )
    separation = SkyCoord(ra=catalog["ra"][index], dec=catalog["dec"][index]).separation(oracle)
    assert separation.to_value(u.uas).max() < 20.0


def test_propagate_gaia_catalog():
    catalog = _make_unpropagated_catalog()

    propagate_gaia_catalog(catalog, 2026.0)

    assert catalog.meta["pm_epoch"] == 2026.0
    # Gaia DR3 default reference epoch
    assert catalog.meta["pm_reference_epoch"] == 2016.0
    # ra/dec keep their names but hold propagated positions; the catalog positions move to the
    # *_ref columns, inserted immediately after dec to mirror the cutout table column order.
    assert catalog.colnames.index("ra_ref") == catalog.colnames.index("dec") + 1
    assert catalog.colnames.index("dec_ref") == catalog.colnames.index("ra_ref") + 1
    assert catalog["ra"].unit == u.deg
    assert catalog["ra_ref"].unit == u.deg
    np.testing.assert_array_equal(catalog["ra_ref"].to_value(u.deg), [120.5, 120.6, 120.7])
    np.testing.assert_array_equal(catalog["dec_ref"].to_value(u.deg), [-45.25, -45.3, -45.35])
    _assert_matches_oracle(catalog, 0, 120.5, -45.25, 1000.0, -500.0)
    # Stars with masked or zero proper motions stay at their catalog positions.
    assert catalog["ra"][1] == catalog["ra_ref"][1]
    assert catalog["dec"][1] == catalog["dec_ref"][1]
    assert catalog["ra"][2] == catalog["ra_ref"][2]
    # The input masks are untouched.
    np.testing.assert_array_equal(np.ma.getmaskarray(catalog["pmra"]), [False, True, False])


def test_propagate_gaia_catalog_ref_epoch_column_overrides_default():
    catalog = _make_unpropagated_catalog(ref_epoch=np.array([2020.0, 2020.0, 2020.0]))

    propagate_gaia_catalog(catalog, 2026.0)

    assert catalog.meta["pm_reference_epoch"] == 2020.0
    _assert_matches_oracle(catalog, 0, 120.5, -45.25, 1000.0, -500.0)


def test_propagate_gaia_catalog_twice_raises():
    catalog = _make_unpropagated_catalog()
    propagate_gaia_catalog(catalog, 2026.0)
    with pytest.raises(ValueError, match="already"):
        propagate_gaia_catalog(catalog, 2026.0)


def test_propagate_gaia_catalog_for_orbit():
    catalog = _make_unpropagated_catalog()

    propagate_gaia_catalog_for_orbit(catalog, 185)

    assert catalog.meta["pm_orbit"] == 185
    assert catalog.meta["pm_epoch"] == float(get_orbit_midtime(185).jyear)
    assert catalog.meta["pm_reference_epoch"] == 2016.0


def test_catalog_is_propagated():
    catalog = _make_unpropagated_catalog()
    assert not catalog_is_propagated(catalog)
    propagate_gaia_catalog(catalog, 2026.0)
    assert catalog_is_propagated(catalog)


def test_write_gaia_catalog_ecsv_round_trip(tmp_path):
    catalog = _make_unpropagated_catalog()
    propagate_gaia_catalog(catalog, 2026.0)
    path = tmp_path / "Gaia_cam1_ccd1.ecsv"

    write_gaia_catalog_ecsv(catalog, path)

    assert not list(tmp_path.glob("*.tmp"))
    read_back = QTable.read(path)
    assert catalog_is_propagated(read_back)
    assert read_back.meta["pm_epoch"] == 2026.0
    assert read_back.meta["pm_reference_epoch"] == 2016.0
    assert read_back.colnames == catalog.colnames
    np.testing.assert_array_equal(read_back["ra"].to_value(u.deg), catalog["ra"].to_value(u.deg))
    np.testing.assert_array_equal(
        read_back["ra_ref"].to_value(u.deg), catalog["ra_ref"].to_value(u.deg)
    )
    np.testing.assert_array_equal(np.ma.getmaskarray(read_back["pmra"]), [False, True, False])


def test_load_propagated_gaia_catalog_upgrades_and_rewrites_old_format(tmp_path):
    path = tmp_path / "Gaia_cam1_ccd1.ecsv"
    write_gaia_catalog_ecsv(_make_unpropagated_catalog(), path)

    loaded = load_propagated_gaia_catalog(path, 185)

    assert catalog_is_propagated(loaded)
    assert loaded.meta["pm_orbit"] == 185
    assert loaded.meta["pm_epoch"] == float(get_orbit_midtime(185).jyear)
    _assert_matches_oracle(loaded, 0, 120.5, -45.25, 1000.0, -500.0)
    # The file is rewritten in the new format so the upgrade only ever happens once.
    on_disk = QTable.read(path)
    assert catalog_is_propagated(on_disk)
    np.testing.assert_array_equal(on_disk["ra"].to_value(u.deg), loaded["ra"].to_value(u.deg))
    assert not list(tmp_path.glob("*.tmp"))


def test_load_propagated_gaia_catalog_leaves_new_format_untouched(tmp_path):
    catalog = _make_unpropagated_catalog()
    propagate_gaia_catalog(catalog, 2026.0)
    path = tmp_path / "Gaia_cam1_ccd1.ecsv"
    write_gaia_catalog_ecsv(catalog, path)
    content_before = path.read_bytes()

    loaded = load_propagated_gaia_catalog(path, 185)

    # A new-format file is returned as-is: no re-propagation, no rewrite.
    assert loaded.meta["pm_epoch"] == 2026.0
    assert "pm_orbit" not in loaded.meta
    assert path.read_bytes() == content_before
