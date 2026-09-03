"""Light curve extraction functionality."""

from collections.abc import Generator
import logging
from math import ceil, floor
from pathlib import Path
from typing import NamedTuple

from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
from astropy.table import QTable, hstack
from astropy.time import Time
import astropy.units as u
import numpy as np

from tglc.aperture_light_curve import ApertureLightCurve, ApertureLightCurveMetadata
from tglc.aperture_photometry import get_expected_total_flux, get_normalized_aperture_photometry
from tglc.epsf import EPSF
from tglc.ffi import FFICutout
from tglc.utils.constants import TESSJD, apply_barycentric_correction  # noqa: F401 for tjd format
from tglc.utils.tess_ephemeris import get_tess_spacecraft_position


logger = logging.getLogger(__name__)


LIGHT_CURVE_APERTURES = (("primary", 3), ("small", 1), ("large", 5))
"""(name, side length) of the square apertures extracted for every light curve."""


class CutoutWindow(NamedTuple):
    """Edge-clamped pixel window of a cutout within a larger image."""

    left: int
    right: int
    bottom: int
    top: int

    @property
    def shape(self) -> tuple[int, int]:
        """Numpy-order (height, width) shape of the window."""
        return (self.top - self.bottom, self.right - self.left)


def get_cutout_window(
    target_x: float, target_y: float, image_shape: tuple[int, int], cutout_size: int = 5
) -> CutoutWindow:
    """
    Get the pixel window of a square cutout centered on the target's nearest pixel.

    Parameters
    ----------
    target_x, target_y : float
        Coordinates of the target in the image. The window is centered on the nearest pixel
        (builtin `round`, i.e. banker's rounding).
    image_shape : tuple[int, int]
        Numpy-order `(height, width)` shape of the image; the window is clamped to it, so
        cutouts near the image edge may be smaller than `cutout_size`.
    cutout_size : int
        Side length of the square cutout in pixels.

    Returns
    -------
    window : CutoutWindow
        Edge-clamped window bounds.
    """
    left = max(0, round(target_x) - floor(cutout_size / 2))
    right = min(image_shape[1], round(target_x) + ceil(cutout_size / 2))
    bottom = max(0, round(target_y) - floor(cutout_size / 2))
    top = min(image_shape[0], round(target_y) + ceil(cutout_size / 2))
    return CutoutWindow(left=left, right=right, bottom=bottom, top=top)


def get_design_matrix_rows_for_window(window: CutoutWindow, image_width: int) -> np.ndarray:
    """
    Get the rows of a full-image design matrix corresponding to a cutout window's pixels.

    Parameters
    ----------
    window : CutoutWindow
        Cutout window within the image.
    image_width : int
        Width (number of columns) of the image the design matrix was built for, which is the
        stride between rows in the flattened row-major pixel order. (The historical inline
        formula used the image height instead, which is equivalent only because TGLC images
        are square.)

    Returns
    -------
    rows : array
        1D integer array of design-matrix row indices for the window's pixels, in the same
        order as the flattened cutout.
    """
    cutout_x, cutout_y = np.meshgrid(
        np.arange(window.left, window.right), np.arange(window.bottom, window.top)
    )
    return (cutout_x + cutout_y * image_width).flatten()


def make_target_design_matrix(
    epsf: EPSF,
    cutout_shape: tuple[int, int],
    target_x_in_cutout: float,
    target_y_in_cutout: float,
    target_flux_ratio: float,
) -> np.ndarray:
    """
    Make a PSF-only design matrix modeling a single target star in a cutout.

    Delegates to `EPSF.make_design_matrix` with no background strap mask, so the result has
    only the `epsf.n_psf_parameters` PSF columns. Note this is a numba-JIT compiled call
    executed once per target.

    Parameters
    ----------
    epsf : EPSF
        Fitted ePSF whose `psf_size` and `oversample` define the PSF model grid.
    cutout_shape : tuple[int, int]
        Numpy-order `(height, width)` shape of the cutout.
    target_x_in_cutout, target_y_in_cutout : float
        Coordinates of the target in the cutout frame.
    target_flux_ratio : float
        Ratio of flux from the target star to the maximum flux from any star in the image.

    Returns
    -------
    target_design_matrix : array
        Design matrix with shape `(height * width, epsf.n_psf_parameters)`.
    """
    target_design_matrix, _ = epsf.make_design_matrix(
        cutout_shape,
        np.array([[target_x_in_cutout, target_y_in_cutout]]),
        np.array([target_flux_ratio]),
    )
    return target_design_matrix


def make_field_design_matrix(
    full_design_matrix_for_cutout: np.ndarray, target_design_matrix: np.ndarray
) -> np.ndarray:
    """
    Make a design matrix modeling everything in a cutout *except* the target star.

    Subtracts the target's PSF-only design matrix from the leading PSF columns of the full
    matrix. The background columns are untouched, so **when evaluated with the full ePSF
    parameter array the result models the field stars AND the background**, not field stars
    alone.

    Parameters
    ----------
    full_design_matrix_for_cutout : array
        Rows of the full-image design matrix for the cutout's pixels, with shape
        `(n_pixels, k)`. Not modified.
    target_design_matrix : array
        PSF-only design matrix for the target star from :func:`make_target_design_matrix`,
        with shape `(n_pixels, n_psf_parameters)`.

    Returns
    -------
    field_design_matrix : array
        Copy of `full_design_matrix_for_cutout` with the target subtracted from the PSF
        columns.
    """
    field_design_matrix = full_design_matrix_for_cutout.copy()
    field_design_matrix[:, : target_design_matrix.shape[1]] -= target_design_matrix
    return field_design_matrix


def evaluate_epsf_model(
    design_matrix: np.ndarray, parameters: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    """
    Evaluate a per-cadence forward model of an image from a design matrix and ePSF parameters.

    Parameters
    ----------
    design_matrix : array
        Design matrix with shape `(height * width, n)`.
    parameters : array
        Per-cadence model parameters with shape `(t, n)` — e.g. `epsf.array` (full model),
        `epsf.psf_parameters` (PSF only), or `epsf.background_parameters` (background only,
        with the matching design-matrix columns).
    image_shape : tuple[int, int]
        Numpy-order `(height, width)` shape to reshape each modeled image to.

    Returns
    -------
    model : array
        Modeled image time series with shape `(t, height, width)`.
    """
    return np.dot(design_matrix, parameters.T).T.reshape(parameters.shape[0], *image_shape)


def get_psf_portion(target_psf_model: np.ndarray) -> np.ndarray:
    """
    Get the portion of the target star's flux falling in each pixel of a cutout.

    Note (historical behavior): the time axis is collapsed, giving a single static portion map
    for all cadences, and the map is normalized to the *cutout* total — i.e. it assumes 100% of
    the star's flux falls inside the cutout window.

    Parameters
    ----------
    target_psf_model : array
        Per-cadence forward model of the target star alone, with shape `(t, height, width)`,
        from :func:`evaluate_epsf_model` with `epsf.psf_parameters`.

    Returns
    -------
    psf_portion : array
        2D `(height, width)` array on the cutout grid whose entries sum to 1, as required by
        `tglc.aperture_photometry.get_normalized_aperture_photometry`.
    """
    return np.nansum(target_psf_model, axis=0) / np.nansum(target_psf_model)


def get_epsf_flux_fraction(target_psf_model: np.ndarray, expected_total_flux: float) -> np.ndarray:
    """
    Per-cadence fraction of a target's catalog-expected flux captured by the fitted ePSF.

    The ePSF fit absorbs TESS's optical throughput loss multiplicatively: toward the camera
    edge, the fitted PSF's integrated response per unit catalog flux is systematically lower
    than near the center. Because the fit's star flux ratios are normalized per cutout (to the
    brightest star in the cutout), raw fitted-PSF totals are not comparable across cutouts;
    dividing the modeled target flux by the catalog-expected flux anchors the value, making it
    comparable across cutouts and cameras. This is the multiplicative counterpart of the
    additive ``local_background`` offset recorded by
    `tglc.aperture_photometry.get_normalized_aperture_photometry`.

    Parameters
    ----------
    target_psf_model : array
        Per-cadence forward model of the target star alone, with shape `(t, height, width)`,
        from :func:`evaluate_epsf_model` with `epsf.psf_parameters`.
    expected_total_flux : float
        Catalog-expected total flux of the target in electrons per cadence, from
        `tglc.aperture_photometry.get_expected_total_flux`.

    Returns
    -------
    epsf_flux_fraction : array
        1D dimensionless array with one entry per cadence. Cadences whose ePSF fit failed
        (rows of NaN) yield 0 from the NaN-ignoring sum.
    """
    return np.nansum(target_psf_model, axis=(1, 2)) / expected_total_flux


def get_cutout_for_light_curve(
    flux: np.ndarray,
    epsf: EPSF,
    full_design_matrix: np.ndarray,
    target_x: float,
    target_y: float,
    target_flux_ratio: float,
    cutout_size: int = 5,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    Make a decontaminated flux cutout suitable for light curve extraction.

    Composes the module's cutout steps: crop a window around the target, forward-model
    everything except the target (field stars **and** background), subtract that model from the
    raw cutout, and compute the target's PSF portion map. Call the step functions directly to
    obtain intermediates such as the raw cutout, the field model, or the per-cadence target PSF
    model.

    Parameters
    ----------
    flux : array
        Time series of images, with shape `(t, n, m)`.
    epsf : EPSF
        Best-fit PSF and background parameters. Its `psf_size` and `oversample` are used to create
        a design matrix specific to the target star.
    full_design_matrix : array
        Design matrix modeling all stars in the image (and background), with shape `(n * m, k)`.
    target_x, target_y : float
        Coordinates of target in images.
    target_flux_ratio : float
        Ratio of flux from target star to max flux from any star in the image, according to Gaia
        catalog data. Used to create a design matrix specifc to the target star.
    cutout_size : int
        Side length of the square cutout in pixels. Cutout may be smaller if target star is near the
        edge of the images.

    Returns
    -------
    cutout, target_x_cutout, target_y_cutout, psf_portions : tuple[array, float, float, array]
        Tuple containing a time series of cutout images, with shape `(t, cutout_size, cutout_size)`
        (last two dimensions may differ for cutouts near image edges), the coordinates of the target
        star within the cutout, and the portion of the ePSF contained in each pixel of the cutout.
    """
    window = get_cutout_window(target_x, target_y, flux.shape[1:], cutout_size)
    target_x_in_cutout = target_x - window.left
    target_y_in_cutout = target_y - window.bottom
    cutout_flux = flux[:, window.bottom : window.top, window.left : window.right]

    # We need a design matrix that models everything in the cutout *except* the target star. To do
    # this, we get the relevant part of the complete design matrix and a design matrix for
    # *just* the target star, and subtract the target design matrix from the complete design matrix.
    # Note: it would be simpler to do this for the entire image and then get the cutout at the end,
    # but that's *much* slower because the matrices involved are huge.
    full_design_matrix_for_cutout = full_design_matrix[
        get_design_matrix_rows_for_window(window, flux.shape[2])
    ]
    target_design_matrix = make_target_design_matrix(
        epsf, window.shape, target_x_in_cutout, target_y_in_cutout, target_flux_ratio
    )
    field_design_matrix = make_field_design_matrix(
        full_design_matrix_for_cutout, target_design_matrix
    )
    # The field model includes the background model, so this subtraction removes both.
    cutout_field_model = evaluate_epsf_model(field_design_matrix, epsf.array, window.shape)
    decontaminated_cutout_flux = cutout_flux - cutout_field_model

    cutout_target_psf = evaluate_epsf_model(target_design_matrix, epsf.psf_parameters, window.shape)
    psf_portion_in_cutout = get_psf_portion(cutout_target_psf)

    return decontaminated_cutout_flux, target_x_in_cutout, target_y_in_cutout, psf_portion_in_cutout


def get_high_background_cadence_mask(epsf: EPSF) -> np.ndarray:
    """
    Flag cadences with an outlying background level in the fitted ePSF.

    Cadences deviating from the median by at least 1 MAD-standard-deviation are flagged. The
    flagged cadences are excluded from the photometric normalization in
    `tglc.aperture_photometry.get_normalized_aperture_photometry`.

    Parameters
    ----------
    epsf : EPSF
        Fitted ePSF whose background parameters are examined.

    Returns
    -------
    high_background_points : array
        1D boolean array with one entry per cadence.
    """
    # Use the model's flat background level to determine points that should be ignored during
    # normalization in photometry
    # NOTE: "y_strap" preserves the historical column choice (epsf[:, -6]), but the comment above
    # suggests the "flat" column was intended. Changing it alters photometry normalization, so it
    # is tracked as a follow-up investigation rather than fixed here (issue #19).
    flat_background = epsf.background_parameter("y_strap")
    return np.abs(flat_background - np.nanmedian(flat_background)) >= mad_std(
        flat_background, ignore_nan=True
    )


def get_background_model(
    epsf: EPSF, full_design_matrix: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    """
    Forward-model the background for every cadence.

    Parameters
    ----------
    epsf : EPSF
        Fitted ePSF providing the per-cadence background parameters.
    full_design_matrix : array
        Full-image design matrix built with a background strap mask, whose last
        `epsf.n_background` columns are the background contributions.
    image_shape : tuple[int, int]
        Numpy-order `(height, width)` shape of the image.

    Returns
    -------
    model_background : array
        Modeled background with shape `(t, height, width)`.
    """
    return evaluate_epsf_model(
        full_design_matrix[:, -epsf.n_background :], epsf.background_parameters, image_shape
    )


def get_background_outlier_mask(background_light_curve: np.ndarray) -> np.ndarray:
    """
    Flag cadences whose background level is an outlier (at least 5 MAD-standard-deviations).

    Parameters
    ----------
    background_light_curve : array
        1D per-cadence background level at a target's position.

    Returns
    -------
    background_outliers : array
        1D boolean array with one entry per cadence.
    """
    # NOTE: mad_std is deliberately called without ignore_nan=True to preserve historical
    # behavior: any NaN cadence (e.g. from a failed ePSF fit) makes the threshold NaN and the
    # whole mask False. Tracked as a follow-up in issue #20.
    return np.abs(background_light_curve - np.nanmedian(background_light_curve)) >= 5 * mad_std(
        background_light_curve
    )


def generate_light_curves(
    source: FFICutout,
    epsf: EPSF,
    ephemerides_directory: Path,
    tic_ids: list[int] | None = None,
) -> Generator[ApertureLightCurve, None, None]:
    """
    Generator function that yields aperture light curves extracted from the source cutout.

    Parameters
    ----------
    source : FFICutout
        Cutout including flux data and positions of stars in the flux images.
    epsf : EPSF
        Best-fit PSF and background parameters fit for `source`. Its `psf_size` and `oversample`
        are used to construct design matrices.
    ephemerides_directory : Path
        Directory containing cached TESS spacecraft ephemeris files, used for barycentric time
        corrections.
    tic_ids : list[int] | None
        Optional list of TIC IDs that should have light curves made. If specified, all other targets
        will be ignored. By default, all targets in the source TIC catalog have light curves made.

    Yields
    ------
    light_curve : ApertureLightCurve
        Aperture light curves extracted from the source cutout with the ePSF parameters given.

    Raises
    ------
    ValueError
        If the ePSF's identifying metadata or cadence count doesn't match `source`, indicating a
        mispaired source/ePSF file combination.
    """
    if not epsf.matches_cutout(source):
        raise ValueError(
            "ePSF does not match source cutout: ePSF is for orbit "
            f"{epsf.orbit} {epsf.camera}-{epsf.ccd} cutout ({epsf.cutout_x}, {epsf.cutout_y}) "
            f"with {epsf.n_cadences} cadences, source cutout is orbit "
            f"{source.orbit} {source.camera}-{source.ccd} cutout "
            f"({source.cutout_x}, {source.cutout_y}) with {source.flux.shape[0]} cadences"
        )

    tic_match_table = source.tic
    if tic_ids is not None:
        tic_match_table = tic_match_table[np.isin(tic_match_table["TIC"], tic_ids)]
    if len(tic_match_table) == 0:
        logger.debug("No targets found, skipping light curve generation")
        return
    logger.debug(f"Making light curves for {tic_match_table} targets")

    star_positions = source.star_positions
    design_matrix, _ = epsf.make_design_matrix(
        source.flux.shape[1:],
        star_positions,
        source.gaia["tess_flux_ratio"].data,
        source.mask.data,
    )

    high_background_points = get_high_background_cadence_mask(epsf)

    # These are used for all light curves
    model_background = get_background_model(epsf, design_matrix, source.flux.shape[1:])
    time = Time(source.time, format="tjd", scale="tdb")
    tess_spacecraft_position = get_tess_spacecraft_position(
        source.orbit, time, ephemerides_directory
    )

    nearest_pixel_x = np.round(star_positions[:, 0]).astype(int)
    nearest_pixel_y = np.round(star_positions[:, 1]).astype(int)
    # Targets outside these bounds have too little data to make light curves
    pixel_left_bound = 1.5
    pixel_right_bound = source.size - 2.5
    pixel_bottom_bound = 1.5
    pixel_top_bound = source.size - 2.5

    for tic_id, gaia3_id in tic_match_table:
        try:
            i = np.nonzero(source.gaia["designation"] == f"Gaia DR3 {gaia3_id}")[0][0]
        except IndexError:
            logger.debug(f"No Gaia catalog entry found for TIC {tic_id}/Gaia DR3 {gaia3_id}")
            continue

        if not (
            (pixel_left_bound <= nearest_pixel_x[i] <= pixel_right_bound)
            and (pixel_bottom_bound <= nearest_pixel_y[i] <= pixel_top_bound)
        ):
            continue

        # Compose the cutout steps directly (rather than calling get_cutout_for_light_curve) so
        # the target PSF model cube is available for the ePSF flux fraction below.
        window = get_cutout_window(
            star_positions[i][0], star_positions[i][1], source.flux.shape[1:], cutout_size=5
        )
        star_x = star_positions[i][0] - window.left
        star_y = star_positions[i][1] - window.bottom
        target_design_matrix = make_target_design_matrix(
            epsf, window.shape, star_x, star_y, source.gaia["tess_flux_ratio"].data[i]
        )
        field_design_matrix = make_field_design_matrix(
            design_matrix[get_design_matrix_rows_for_window(window, source.flux.shape[2])],
            target_design_matrix,
        )
        # The field model includes the background model, so this subtraction removes both.
        light_curve_cutout = source.flux[
            :, window.bottom : window.top, window.left : window.right
        ] - evaluate_epsf_model(field_design_matrix, epsf.array, window.shape)
        cutout_target_psf = evaluate_epsf_model(
            target_design_matrix, epsf.psf_parameters, window.shape
        )
        psf_portions = get_psf_portion(cutout_target_psf)
        epsf_flux_fraction = get_epsf_flux_fraction(
            cutout_target_psf, get_expected_total_flux(source.gaia["tess_mag"][i], source.exposure)
        )

        sky_coord = SkyCoord(source.gaia["ra"][i], source.gaia["dec"][i], unit="deg")
        time_btjd = apply_barycentric_correction(time, sky_coord, tess_spacecraft_position)
        aperture_photometry_data = [
            get_normalized_aperture_photometry(
                light_curve_cutout,
                np.array(source.quality) | high_background_points,
                aperture_size,
                round(star_x),
                round(star_y),
                source.gaia["tess_mag"][i],
                source.exposure * u.second,
                psf_portions,
                column_name_prefix=f"{aperture_name}_aperture_",
            )
            for aperture_name, aperture_size in LIGHT_CURVE_APERTURES
        ]
        # Shift centroids from the cutout frame into CCD coordinates. This re-derives the cutout
        # origin, including the nearest-pixel rounding offset relative to the window edge.
        for (aperture_name, _), table in zip(
            LIGHT_CURVE_APERTURES, aperture_photometry_data, strict=False
        ):
            table[f"{aperture_name}_aperture_centroid_x"] += (
                source.ccd_x + nearest_pixel_x[i] - star_x
            ) * u.pixel
            table[f"{aperture_name}_aperture_centroid_y"] += (
                source.ccd_y + nearest_pixel_y[i] - star_y
            ) * u.pixel

        # Background light curve is the background level at the star's location
        background_light_curve = model_background[:, nearest_pixel_y[i], nearest_pixel_x[i]]
        background_quality_flags = get_background_outlier_mask(background_light_curve)

        target_ccd_x = star_positions[i][0] + source.ccd_x
        target_ccd_y = star_positions[i][1] + source.ccd_y

        light_curve_meta = ApertureLightCurveMetadata(
            tic_id=tic_id,
            orbit=source.orbit,
            sector=source.sector,
            camera=source.camera,
            ccd=source.ccd,
            ccd_x=target_ccd_x,
            ccd_y=target_ccd_y,
            sky_coord=sky_coord,
            tess_magnitude=source.gaia["tess_mag"][i],
            exposure_time=source.exposure * u.second,
            primary_aperture_local_background=aperture_photometry_data[0].meta[
                "primary_aperture_local_background"
            ],
            small_aperture_local_background=aperture_photometry_data[1].meta[
                "small_aperture_local_background"
            ],
            large_aperture_local_background=aperture_photometry_data[2].meta[
                "large_aperture_local_background"
            ],
        )

        base_light_curve = QTable(
            {
                "time": time_btjd,
                "cadence": source.cadence,
                # Use 2 for background quality flags to avoid conflicting with QLP quality flags
                "quality_flag": background_quality_flags.astype(int) * 2,
                "background_flux": background_light_curve,
                "epsf_flux_fraction": epsf_flux_fraction,
            }
        )

        light_curve = ApertureLightCurve(
            hstack(aperture_photometry_data + [base_light_curve]), meta=light_curve_meta
        )
        yield light_curve
