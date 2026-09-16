"""
Proper-motion propagation of Gaia catalog positions to a TESS observation epoch.

Positions are propagated once, en masse, for a whole orbit/camera/CCD Gaia catalog when the
catalog is generated (``tglc catalogs``): the catalog's ``ra``/``dec`` columns hold positions at
the orbit mid-time (recorded as ``pm_epoch`` in the table meta, in Julian years) and the original
catalog positions move to ``ra_ref``/``dec_ref`` (at ``pm_reference_epoch``). Old-format catalog
files without propagated positions are upgraded transparently by `load_propagated_gaia_catalog`,
which propagates once per CCD and rewrites the file on disk.

This module must stay importable without the ``pyticdb`` extra (no ``tglc.databases`` /
``tglc.scripts.catalogs`` imports), because ``tglc.ffi`` and ``tglc migrate`` depend on it.
"""

import logging
import os
from pathlib import Path

from astropy.table import QTable, Table
import astropy.units as u
from numba import float64, jit, prange
import numpy as np

from tglc.utils.constants import get_orbit_midtime


logger = logging.getLogger(__name__)


GAIA_DR3_REFERENCE_EPOCH = 2016.0
"""Julian year that Gaia DR3 catalog positions are referred to."""

_DEG_TO_RAD = np.pi / 180.0
_MAS_YR_TO_RAD = np.pi / (180.0 * 3600.0 * 1000.0)


@jit(
    float64[:, :](float64[:], float64[:], float64[:], float64[:], float64),
    nogil=True,
    parallel=True,
)
def _propagate_unit_vectors(ra_deg, dec_deg, pmra_mas_yr, pmdec_mas_yr, dt_years):
    """
    Fast JIT-compiled, multithreaded proper-motion propagation of (ra, dec) by dt_years.

    Each star's unit vector is displaced by dt * (pmra * e_alpha + pmdec * e_delta) — its proper
    motion converted to radians along the local east/north tangent basis — and the direction of
    the result is converted back to spherical coordinates. This matches ERFA's pmsafe/starpm
    propagation (position + velocity * dt, renormalized), which is what astropy's
    `SkyCoord.apply_space_motion` computes for catalogs with zero parallax and radial velocity.
    `pmra` must be mu_alpha* = mu_alpha * cos(delta), the Gaia convention. Returns a (2, n) array
    of propagated (ra, dec) in degrees, with ra in [0, 360). Stars with zero proper motion (which
    includes stars whose missing proper motions were replaced with 0) pass through bit-exact.
    """
    result = np.empty((2, ra_deg.shape[0]))
    for i in prange(ra_deg.shape[0]):
        displacement_ra = pmra_mas_yr[i] * _MAS_YR_TO_RAD * dt_years
        displacement_dec = pmdec_mas_yr[i] * _MAS_YR_TO_RAD * dt_years
        if displacement_ra == 0.0 and displacement_dec == 0.0:
            result[0, i] = ra_deg[i]
            result[1, i] = dec_deg[i]
            continue
        alpha = ra_deg[i] * _DEG_TO_RAD
        delta = dec_deg[i] * _DEG_TO_RAD
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        # Unit vector p = (cos d cos a, cos d sin a, sin d) displaced along the tangent basis
        # e_alpha = (-sin a, cos a, 0), e_delta = (-sin d cos a, -sin d sin a, cos d).
        x = (
            cos_delta * cos_alpha
            - displacement_ra * sin_alpha
            - displacement_dec * sin_delta * cos_alpha
        )
        y = (
            cos_delta * sin_alpha
            + displacement_ra * cos_alpha
            - displacement_dec * sin_delta * sin_alpha
        )
        z = sin_delta + displacement_dec * cos_delta
        new_alpha = np.arctan2(y, x)
        if new_alpha < 0.0:
            new_alpha += 2.0 * np.pi
        new_ra = new_alpha / _DEG_TO_RAD
        if new_ra >= 360.0:
            new_ra -= 360.0
        result[0, i] = new_ra
        # arctan2(z, hypot(x, y)) is scale-invariant (no renormalization needed) and, unlike
        # arcsin(z / norm), keeps full precision at the poles.
        result[1, i] = np.arctan2(z, np.hypot(x, y)) / _DEG_TO_RAD
    return result


def _values_as_float64(values, unit) -> np.ndarray:
    """Coerce a column/array to a plain float64 array of values in `unit`."""
    if hasattr(values, "to_value"):
        values = values.to_value(unit)
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def propagate_coordinates(
    ra, dec, pmra, pmdec, epoch_jyear: float, reference_epoch_jyear: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate sky positions from `reference_epoch_jyear` to `epoch_jyear` by their proper motions.

    Parameters
    ----------
    ra, dec : array-like or Quantity
        Positions at the reference epoch, in degrees.
    pmra, pmdec : array-like, Quantity, or masked column
        Proper motions in mas/yr, with ``pmra`` = mu_alpha* = mu_alpha * cos(delta) (the Gaia
        convention). Stars with masked or non-finite proper motions keep their input positions
        bit-exact.
    epoch_jyear, reference_epoch_jyear : float
        Target and reference epochs in Julian years.

    Returns
    -------
    ra, dec : np.ndarray
        Propagated positions in degrees, with ra in [0, 360).
    """
    ra_deg = _values_as_float64(ra, u.deg)
    dec_deg = _values_as_float64(dec, u.deg)
    pmra_mas_yr = _values_as_float64(pmra, u.mas / u.yr)
    pmdec_mas_yr = _values_as_float64(pmdec, u.mas / u.yr)
    pm_missing = (
        np.ma.getmaskarray(pmra)
        | np.ma.getmaskarray(pmdec)
        | ~np.isfinite(pmra_mas_yr)
        | ~np.isfinite(pmdec_mas_yr)
    )
    propagated = _propagate_unit_vectors(
        ra_deg,
        dec_deg,
        np.where(pm_missing, 0.0, pmra_mas_yr),
        np.where(pm_missing, 0.0, pmdec_mas_yr),
        float(epoch_jyear) - float(reference_epoch_jyear),
    )
    return propagated[0], propagated[1]


def propagate_gaia_catalog(
    gaia_catalog, epoch_jyear: float, *, reference_epoch_jyear: float | None = None
) -> None:
    """
    Propagate a Gaia catalog table's positions to `epoch_jyear` in place.

    ``ra``/``dec`` are overwritten with the propagated positions and the original catalog
    positions are inserted as ``ra_ref``/``dec_ref`` immediately after ``dec``, mirroring the
    column order of cutout catalog tables. The epochs used are recorded in the table meta as
    ``pm_epoch`` and ``pm_reference_epoch`` (Julian years).

    When `reference_epoch_jyear` is not given, it is taken as the median of the catalog's
    ``ref_epoch`` column when present, and J2016.0 (Gaia DR3) otherwise.

    Raises `ValueError` if the catalog is already propagated.
    """
    if "pm_epoch" in gaia_catalog.meta or "ra_ref" in gaia_catalog.colnames:
        raise ValueError("Gaia catalog is already proper-motion propagated")
    if reference_epoch_jyear is None:
        if "ref_epoch" in gaia_catalog.colnames:
            reference_epoch_jyear = float(
                np.median(np.asarray(gaia_catalog["ref_epoch"], dtype=np.float64))
            )
        else:
            # Gaia DR3 positions are referred to J2016.0.
            reference_epoch_jyear = GAIA_DR3_REFERENCE_EPOCH

    ra_ref = _values_as_float64(gaia_catalog["ra"], u.deg)
    dec_ref = _values_as_float64(gaia_catalog["dec"], u.deg)
    propagated_ra, propagated_dec = propagate_coordinates(
        gaia_catalog["ra"],
        gaia_catalog["dec"],
        gaia_catalog["pmra"],
        gaia_catalog["pmdec"],
        epoch_jyear,
        reference_epoch_jyear,
    )
    gaia_catalog["ra"] = propagated_ra * u.deg
    gaia_catalog["dec"] = propagated_dec * u.deg
    dec_index = gaia_catalog.colnames.index("dec")
    gaia_catalog.add_column(ra_ref * u.deg, name="ra_ref", index=dec_index + 1)
    gaia_catalog.add_column(dec_ref * u.deg, name="dec_ref", index=dec_index + 2)
    gaia_catalog.meta["pm_epoch"] = float(epoch_jyear)
    gaia_catalog.meta["pm_reference_epoch"] = float(reference_epoch_jyear)


def propagate_gaia_catalog_for_orbit(gaia_catalog, orbit: int) -> None:
    """Propagate a Gaia catalog table in place to the mid-time of a TESS orbit."""
    propagate_gaia_catalog(gaia_catalog, float(get_orbit_midtime(orbit).jyear))
    gaia_catalog.meta["pm_orbit"] = int(orbit)


def catalog_is_propagated(gaia_catalog) -> bool:
    """Whether a Gaia catalog table already holds proper-motion-propagated positions."""
    return (
        "pm_epoch" in gaia_catalog.meta
        and "pm_reference_epoch" in gaia_catalog.meta
        and "ra_ref" in gaia_catalog.colnames
        and "dec_ref" in gaia_catalog.colnames
    )


def write_gaia_catalog_ecsv(catalog, path: Path) -> None:
    """
    Write a catalog table to `path` as ECSV, atomically.

    Astropy's fast ascii writer doesn't work with ecsv by default, but we can write the header
    from an empty slice (which carries the meta) and then append the data to get an equivalent
    file. The file is written to a temporary name and renamed into place so concurrent readers
    never see a partial file.
    """
    # The fast ascii writer requires Column objects, so view a QTable's Quantity columns as
    # unit-carrying Columns for writing.
    catalog = Table(catalog, copy=False)
    temporary_file = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    catalog[:0].write(temporary_file, format="ascii.ecsv", overwrite=True)
    with open(temporary_file, "a") as output:
        catalog.write(output, format="ascii.fast_no_header", delimiter=" ", strip_whitespace=False)
    temporary_file.replace(path)


def load_propagated_gaia_catalog(path: Path, orbit: int) -> QTable:
    """
    Read a Gaia catalog ECSV, upgrading old-format files to proper-motion-propagated form.

    Old-format catalogs (without propagated positions) are propagated to the orbit mid-time and
    rewritten on disk, so the upgrade cost is paid once per catalog file rather than once per
    cutout. Call this once per CCD in the parent process before creating worker pools.
    """
    gaia_catalog = QTable.read(path)
    if catalog_is_propagated(gaia_catalog):
        return gaia_catalog
    logger.info(f"Upgrading Gaia catalog to proper-motion-propagated format: {path}")
    propagate_gaia_catalog_for_orbit(gaia_catalog, orbit)
    write_gaia_catalog_ecsv(gaia_catalog, path)
    return gaia_catalog
