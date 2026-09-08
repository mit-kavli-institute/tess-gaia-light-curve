"""Aperture photometry for TGLC light curves with 3 apertures.

The individual measurement and normalization steps are exposed as pure functions operating on
plain numpy arrays so that experiments and diagnostics can run any step in isolation (e.g., to
inspect the raw, pre-normalization aperture flux). :func:`get_aperture_photometry` composes
the measurement steps, and :func:`normalize_photometry` adds the normalized flux and magnitude
products on top.
"""

from astropy import units as u
from astropy.table import QTable
import numpy as np
from scipy.ndimage import center_of_mass

from tglc.utils.constants import (
    TESS_PIXEL_SATURATION_LEVEL,
    convert_tess_flux_to_tess_magnitude,
    convert_tess_magnitude_to_tess_flux,
)


def get_aperture_limits(
    aperture_size: int, x: int, y: int, top_limit: int, right_limit: int
) -> tuple[int, int, int, int]:
    """Get (bottom, top, left, right) limits for a square aperture centered at (x, y).

    The window is silently clamped to the image bounds; whether clamping occurred (i.e., whether
    the aperture covers fewer than ``aperture_size**2`` pixels) is not returned.
    """
    bottom = max(0, y - aperture_size // 2)
    top = min(top_limit, y + aperture_size // 2 + 1)
    left = max(0, x - aperture_size // 2)
    right = min(right_limit, x + aperture_size // 2 + 1)
    return bottom, top, left, right


def measure_aperture_flux(
    images: np.ndarray, aperture_limits: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Measure the raw per-cadence flux summed over an aperture window.

    This is the un-normalized aperture flux in electrons, before saturation masking and before
    the local-background normalization applied by :func:`normalize_photometry`.

    Parameters
    ----------
    images : array
        3D ``(t, n, m)`` time series of images, in electrons.
    aperture_limits : tuple[int, int, int, int]
        ``(bottom, top, left, right)`` aperture window, as returned by
        :func:`get_aperture_limits`.

    Returns
    -------
    flux : array
        1D ``(t,)`` array of NaN-ignoring aperture sums.
    """
    bottom, top, left, right = aperture_limits
    return np.nansum(images[:, bottom:top, left:right], axis=(1, 2))


def measure_aperture_centroids(
    images: np.ndarray, aperture_limits: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Measure per-cadence flux-weighted centroids of an aperture window.

    Parameters
    ----------
    images : array
        3D ``(t, n, m)`` time series of images.
    aperture_limits : tuple[int, int, int, int]
        ``(bottom, top, left, right)`` aperture window, as returned by
        :func:`get_aperture_limits`.

    Returns
    -------
    centroids : array
        2D ``(t, 2)`` array of centroids in **row-major (y, x) order**, in the coordinate
        system of the full images (window offsets applied). NaN pixels propagate to NaN
        centroids.
    """
    bottom, top, left, right = aperture_limits
    centroids = np.array([center_of_mass(image[bottom:top, left:right]) for image in images])
    centroids[:, 0] += bottom
    centroids[:, 1] += left
    return centroids


def get_saturation_mask(
    flux: np.ndarray, aperture_size: int, exposure_time_seconds: float
) -> np.ndarray:
    """
    Flag cadences whose aperture flux implies a saturated 2-second integration.

    Points are considered saturated if any of the 2 second integrations would have been saturated
    over the course of the exposure, using the saturation level of 200,000 e- given in the TESS
    Instrument Handbook, p. 37.

    Parameters
    ----------
    flux : array
        1D ``(t,)`` raw aperture flux in electrons, from :func:`measure_aperture_flux`.
    aperture_size : int
        Nominal side length of the square aperture. Note the threshold uses the nominal
        ``aperture_size**2`` pixel count even when :func:`get_aperture_limits` clamped the
        window to fewer pixels at an image edge (historical behavior).
    exposure_time_seconds : float
        Exposure time per cadence in seconds.

    Returns
    -------
    is_saturated : array
        1D ``(t,)`` boolean array flagging saturated cadences.
    """
    return flux > (
        TESS_PIXEL_SATURATION_LEVEL.to_value(u.electron)
        * (aperture_size**2)
        * exposure_time_seconds
        / 2.0
    )


def get_expected_total_flux(tmag: float, exposure_time_seconds: float) -> float:
    """Expected total flux in electrons for a star of the given TESS magnitude per cadence."""
    return (
        convert_tess_magnitude_to_tess_flux(tmag).to_value(u.electron / u.second)
        * exposure_time_seconds
    )


def get_flux_portion_in_aperture(
    flux_portion: np.ndarray, aperture_limits: tuple[int, int, int, int]
) -> float:
    """Portion of the target star's flux falling inside the aperture window."""
    bottom, top, left, right = aperture_limits
    return np.nansum(flux_portion[bottom:top, left:right])


def get_local_background(
    flux: np.ndarray, quality_flags: np.ndarray, expected_aperture_flux: float
) -> float:
    """
    Local background level: the average amount of aperture flux above the expected amount.

    Parameters
    ----------
    flux : array
        1D ``(t,)`` aperture flux in electrons. Expected to already have saturated cadences
        set to NaN.
    quality_flags : array
        Per-cadence quality flags; only cadences flagged 0 contribute to the median.
    expected_aperture_flux : float
        Catalog-predicted flux in the aperture, in electrons.

    Returns
    -------
    local_background : float
        ``nanmedian(flux[quality_flags == 0]) - expected_aperture_flux``. NaN if there are no
        good-flagged, non-NaN cadences.
    """
    return np.nanmedian(flux[quality_flags == 0]) - expected_aperture_flux


def normalize_aperture_flux(flux: np.ndarray, local_background: float) -> np.ndarray:
    """
    Normalize aperture flux by subtracting the local background level.

    The normalization shifts the whole series by a single scalar so that the good-cadence median
    lands on the catalog-predicted aperture flux; absolute photometry does not survive it.

    Parameters
    ----------
    flux : array
        1D ``(t,)`` raw aperture flux in electrons. Not modified.
    local_background : float
        Background level from :func:`get_local_background`. If NaN (no good cadences), no
        subtraction is performed.

    Returns
    -------
    normalized_flux : array
        New array with the background subtracted and non-positive values set to NaN (which
        prevents runtime warnings when converting to magnitude; the clip applies to the
        returned flux values as well).
    """
    flux = flux.copy()
    if not np.isnan(local_background):
        flux -= local_background
    flux[flux <= 0] = np.nan
    return flux


def get_aperture_photometry(
    images: np.ndarray,
    aperture_size: int,
    x: int,
    y: int,
    exposure_time: u.Quantity,
    column_name_prefix: str = "",
) -> QTable:
    """
    Extract raw aperture photometry from a time series of images.

    The measurement phase only: aperture sums, flux-weighted centroids, and saturation masking.
    No normalization is applied; see :func:`normalize_photometry` for the normalized flux and
    magnitude columns.

    Saturated points are removed; see :func:`get_saturation_mask`.

    Parameters
    ----------
    images : array_like
        3 dimensional array with time as first dimension and image cutouts as remaining dimensions.
    aperture_size : int
        Side length of square aperture to use.
    x, y : int
        Aperture center coordinates in images.
    exposure_time : u.Quantity (time)
        Exposure time for each light curve value. Used to determine saturated points.
    column_name_prefix : str
        Prefix inserted into column names. Default is no prefix.

    Returns
    -------
    photometry_data : QTable
        Table with raw aperture photometry, in the coordinate system of the images.

        Columns:
        - `"{column_name_prefix}raw_flux"`: Un-normalized total flux value in aperture, or NaN if
          saturated. No background shift or non-positive clip is applied.
        - `"{column_name_prefix}centroid_x"`: X coordinate in image of flux-weighted aperture centroid
        - `"{column_name_prefix}centroid_y"`: Y coordinate in image of flux-weighted aperture centroid

        Metadata:
        - `"{column_name_prefix}aperture_limits"`: The (bottom, top, left, right) aperture window
          used, derived by :func:`get_aperture_limits` (silently clamped at image edges).
          :func:`normalize_photometry` reads it to crop the flux-portion map consistently.
    """
    aperture_limits = get_aperture_limits(aperture_size, x, y, images.shape[1], images.shape[2])
    exposure_time_seconds = exposure_time.to_value(u.second)

    flux = measure_aperture_flux(images, aperture_limits)
    centroids = measure_aperture_centroids(images, aperture_limits)

    is_saturated = get_saturation_mask(flux, aperture_size, exposure_time_seconds)
    flux[is_saturated] = np.nan
    centroids[is_saturated, :] = np.nan
    centroids = centroids * u.pixel

    return QTable(
        {
            f"{column_name_prefix}raw_flux": flux * u.electron,
            f"{column_name_prefix}centroid_x": centroids[:, 1],
            f"{column_name_prefix}centroid_y": centroids[:, 0],
        },
        meta={f"{column_name_prefix}aperture_limits": aperture_limits},
    )


def normalize_photometry(
    photometry: QTable,
    quality_flags: np.ndarray,
    tmag: float,
    exposure_time: u.Quantity,
    flux_portion: np.ndarray,
    column_name_prefix: str = "",
    inplace: bool = False,
) -> QTable:
    """
    Add normalized flux and magnitude columns to raw aperture photometry.

    The normalization phase: shifts the raw aperture flux by a single scalar (the "local
    background") so that its good-cadence median lands on the flux expected in the aperture for
    the target's TESS magnitude, based on the reference flux of 15,000 e-/s for a star of TESS
    magnitude 10 given in the TESS Instrument Handbook, p. 37. The shift is additive and
    recorded in the table metadata, so `flux = raw_flux - local_background` (except where the
    non-positive clip produced NaN) and absolute photometry is recoverable from `raw_flux`.

    See <https://archive.stsci.edu/missions/tess/doc/TESS_Instrument_Handbook_v0.1.pdf#page=38>.

    Parameters
    ----------
    photometry : QTable
        Raw aperture photometry from :func:`get_aperture_photometry` (with the same
        `column_name_prefix`); its `raw_flux` column and `aperture_limits` metadata are read.
    quality_flags : array_like[int]
        Quality flags for the cadences, where 0 indicates a good value. Only good cadences
        contribute to the normalization level.
    tmag : float
        TESS magnitude of target star.
    exposure_time : u.Quantity (time)
        Exposure time for each light curve value.
    flux_portion : array_like
        Proportion of flux in each pixel of the images the photometry was measured from. Should
        be a 2D array with shape matching the images, and entries that sum to 1.
    column_name_prefix : str
        Prefix inserted into column names. Default is no prefix.
    inplace : bool
        If `True`, add the new columns and metadata to `photometry` itself; if `False` (the
        default), operate on a copy. The resulting table is returned either way.

    Returns
    -------
    photometry_data : QTable
        The input table (or a copy of it) with normalization products added.

        Columns added:
        - `"{column_name_prefix}flux"`: Normalized total flux value in aperture. NaN where
          saturated or where the normalized value was non-positive (clipped to prevent runtime
          warnings converting to magnitude).
        - `"{column_name_prefix}magnitude"`: Normalized magnitude value for aperture, or NaN if
          saturated

        Metadata added:
        - `"{column_name_prefix}local_background"`: Local background flux level used in
          normalization.
    """
    if not inplace:
        photometry = photometry.copy()
    raw_flux = photometry[f"{column_name_prefix}raw_flux"].to_value(u.electron)
    aperture_limits = photometry.meta[f"{column_name_prefix}aperture_limits"]
    exposure_time_seconds = exposure_time.to_value(u.second)

    flux_portion_in_aperture = get_flux_portion_in_aperture(flux_portion, aperture_limits)
    expected_aperture_flux = (
        get_expected_total_flux(tmag, exposure_time_seconds) * flux_portion_in_aperture
    )
    local_background = get_local_background(raw_flux, quality_flags, expected_aperture_flux)
    flux = normalize_aperture_flux(raw_flux, local_background) * u.electron

    photometry[f"{column_name_prefix}flux"] = flux
    photometry[f"{column_name_prefix}magnitude"] = convert_tess_flux_to_tess_magnitude(
        flux / flux_portion_in_aperture / exposure_time
    )
    photometry.meta[f"{column_name_prefix}local_background"] = local_background * u.electron
    return photometry
