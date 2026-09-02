"""ePSF fitting functions and the fitted-ePSF data product class."""

import logging
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING

from numba import jit
import numpy as np

from tglc.utils._optional_deps import HAS_CUPY


if TYPE_CHECKING:
    from tglc.ffi import FFICutout


logger = logging.getLogger(__name__)


EPSF_BACKGROUND_COLUMNS = (
    "y_strap",
    "x_strap",
    "flat_strap",
    "x_gradient",
    "y_gradient",
    "flat",
)
"""Names of the 6 background columns appended after the ePSF parameters.

The order matches the construction in :func:`make_tglc_design_matrix`.
"""


@jit
def get_xy_coordinates_centered_at_zero(shape: tuple[int, int]):
    """
    Returns coordinates for an array with the given shape with (0, 0) at the center of the array.

    Returns
    -------
    x, y : tuple[array, array]
        X and Y coordinates.
    """
    x_coordinates = np.arange(shape[1]) - (shape[1] - 1) / 2
    y_coordinates = np.arange(shape[0]) - (shape[0] - 1) / 2
    return np.repeat(x_coordinates, shape[0]).reshape(shape[::-1]).T, np.repeat(
        y_coordinates, shape[1]
    ).reshape(shape)


@jit
def make_tglc_design_matrix(
    image_shape: tuple[int, int],
    psf_shape_pixels: tuple[int, int],
    oversample_factor: int,
    star_positions: np.ndarray,
    star_flux_ratios: np.ndarray,
    background_strap_mask: np.ndarray | None = None,
    edge_compression_scale_factor: float | None = None,
):
    """
    Construct the TGLC design matrix from equation (3) of Han & Brandt, 2023.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of image (FFI cutout) that will be used as observed data.
    psf_shape_pixels : tuple[int, int]
        Extent of ePSF array in pixels.
    oversample_factor : int
        Factor by which to oversample the ePSF compared to image pixels.
    star_positions : array
        Positions of stars in image with shape (n, 2). The first column is `x` and the second column
        is `y`. Same order as `star_flux_ratios`.
    star_flux_ratios : array
        Ratio of flux from each star to maximum flux from any star, where flux is calculated using
        catalog brightness for each star. Shape (n,) and same order as `star_positions`.
    background_strap_mask : array | None
        Mask giving the background strap values for each pixel. If omitted or set to `None`, no
        columns for background modeling are added to the design matrix.
    edge_compression_scale_factor : float | None
        Scale factor used when forcing edges of ePSF to 0. This is only needed during fitting (not
        forward modeling) and produces extra rows in the output. If omitted or set to `None`, no
        extra rows are added to the design matrix. If included, `background_strap_mask` must also be
        given.

    Returns
    -------
    design_matrix, regularization_extension_size : tuple[array, int]
        Design matrix and amount that observed vectors need to be extended by for regularization
        during fitting. If `edge_compression_scale_factor` is `None`, then
        `regularization_extension_size` will be `0`.
    """
    oversampled_psf_shape = (
        psf_shape_pixels[0] * oversample_factor + 1,
        psf_shape_pixels[1] * oversample_factor + 1,
    )
    # epsf_contributions_to_pixels[iy, ix, py, px] is the contribution of point (px, py) in the
    # oversampled PSF to pixel (ix, iy) in the image.
    epsf_contributions_to_pixels = np.zeros(
        (image_shape[0], image_shape[1], oversampled_psf_shape[0], oversampled_psf_shape[1])
    )
    pixels_in_epsf_x = (
        np.arange(psf_shape_pixels[1], dtype=np.int64) - (psf_shape_pixels[1] - 1) // 2
    )
    pixels_in_epsf_y = (
        np.arange(psf_shape_pixels[0], dtype=np.int64) - (psf_shape_pixels[0] - 1) // 2
    )
    for (x, y), flux_ratio in zip(star_positions, star_flux_ratios):  # noqa: B905 (for JIT)
        nearest_pixel_x, nearest_pixel_y = (round(x), round(y))
        for pixel_x in pixels_in_epsf_x + nearest_pixel_x:
            if pixel_x < 0 or pixel_x >= image_shape[1]:
                continue
            for pixel_y in pixels_in_epsf_y + nearest_pixel_y:
                if pixel_y < 0 or pixel_y >= image_shape[0]:
                    continue
                # Get the coordinate of the nearest pixel center in coordinates of the PSF grid,
                # with the bottom left PSF point at (0, 0) and distance 1 between adjacent PSF
                # points.
                pixel_psf_x, pixel_psf_y = (
                    (pixel_x - x) * oversample_factor + oversampled_psf_shape[1] // 2,
                    (pixel_y - y) * oversample_factor + oversampled_psf_shape[0] // 2,
                )
                # The four closest PSF points are bilinearly interpolated to give the PSF model
                # value of the pixel, and their coordinates are given by rounding the pixel center
                # coordinates up and down. The contribution from each pixel is the weight it is
                # given in the bilinear interpolation, which is the product of the distances in the
                # x and y directions. We further weight the contribution in importance by the flux
                # ratio of the current star.
                for psf_x, psf_y in [
                    (floor(pixel_psf_x), floor(pixel_psf_y)),
                    (floor(pixel_psf_x), ceil(pixel_psf_y)),
                    (ceil(pixel_psf_x), floor(pixel_psf_y)),
                    (ceil(pixel_psf_x), ceil(pixel_psf_y)),
                ]:
                    # Naively, the interpolation weight is:
                    #   np.abs(pixel_psf_x - psf_x) * np.abs(pixel_psf_y - psf_y)
                    # If the pixel lies on a PSF pixel boundary, one of these terms will vanish. But
                    # that actually means we are only interpolating between two pixel centers on a
                    # line, instead of four on a square. Those points will get double counted
                    # because ceil and floor will give the same result, so we use 0.5 as the weight
                    # to correct that.
                    x_interpolation_weight = np.abs(pixel_psf_x - psf_x) or 0.5
                    y_interpolation_weight = np.abs(pixel_psf_y - psf_y) or 0.5
                    epsf_contributions_to_pixels[pixel_y, pixel_x, psf_y, psf_x] += (
                        flux_ratio * x_interpolation_weight * y_interpolation_weight
                    )

    design_matrix = epsf_contributions_to_pixels.reshape(
        image_shape[0] * image_shape[1],
        oversampled_psf_shape[0] * oversampled_psf_shape[1],
    )
    if background_strap_mask is not None:
        # To calculate the linear gradients, we need the x and y coordinates of each pixel.
        image_pixel_xs, image_pixel_ys = get_xy_coordinates_centered_at_zero(image_shape)
        # background_contributions_to_pixels[iy, ix, b] is the contribution of background parameter
        # b to pixel (ix, iy) in the image.
        background_contribution_to_pixels = np.stack(
            (
                # This order is for historical compatibility
                background_strap_mask * image_pixel_ys,  # y-dependent background straps
                background_strap_mask * image_pixel_xs,  # x-dependent background straps
                background_strap_mask,  # flat background straps
                image_pixel_xs,  # x component of linear gradient => use y coordinate of each point
                image_pixel_ys,  # y component of linear gradient => use x coordinate of each point
                np.ones(image_shape),  # flat background level => same contribution to each point
            ),
            axis=-1,
        )

        # Construct the full design matrix by flattening image coordinates.
        design_matrix = np.hstack(
            (
                design_matrix,
                background_contribution_to_pixels.reshape(image_shape[0] * image_shape[1], -1),
            )
        )

    regularization_extension_size = 0
    if edge_compression_scale_factor is not None:
        # With the current set up, the flat background level could be partly fitted in the ePSF by
        # having a constant background level:
        # [[10 11 10]               [[0 1 0]
        #  [11 13 11]   instead of   [1 3 1]
        #  [10 11 10]]               [0 1 0]]
        # In the case shown here, the background level should be 10 higher than whatever was fitted.
        # To implement this, add rows to the design matrix that pick out a specific PSFpoint and give
        # it a weight based on its distance to the center of the PSF. The vector of observations should
        # have an appropriate number of zeros appended to it at fitting time.
        psf_point_x, psf_point_y = get_xy_coordinates_centered_at_zero(oversampled_psf_shape)
        psf_distance_from_center_weight = edge_compression_scale_factor * (
            1
            - np.exp(
                -0.5
                * (
                    (psf_point_x / psf_shape_pixels[1]) ** 4
                    + (psf_point_y / psf_shape_pixels[0]) ** 4
                )
            )
        )
        edge_compression_block = np.hstack(
            (
                np.diag(
                    psf_distance_from_center_weight.reshape(
                        oversampled_psf_shape[0] * oversampled_psf_shape[1]
                    )
                ),
                np.zeros(
                    (
                        oversampled_psf_shape[0] * oversampled_psf_shape[1],
                        background_contribution_to_pixels.shape[-1],
                    )
                ),
            )
        )
        design_matrix = np.vstack((design_matrix, edge_compression_block))
        regularization_extension_size = oversampled_psf_shape[0] * oversampled_psf_shape[1]

    return design_matrix, regularization_extension_size


def get_default_epsf_flux_mask(flux: np.ndarray, base_flux_mask: np.ndarray) -> np.ndarray:
    """
    Default pixel mask used by :func:`fit_epsf`: the base mask plus dim pixels.

    Pixels dimmer than 0.8 times the (NaN-ignoring) median flux are masked in addition to the
    pixels flagged in `base_flux_mask`, discarding roughly the sky-dominated portion of the image
    from the fit.

    Parameters
    ----------
    flux : array
        2D array of observed flux values for one cadence.
    base_flux_mask : array[bool]
        2D mask array indicating bad (e.g., saturated) pixels.

    Returns
    -------
    flux_mask : array[bool]
        2D boolean mask of pixels to exclude from the fit.

    Notes
    -----
    NaN flux pixels are *not* masked by the brightness cut (``NaN < x`` is False); they are
    excluded from the median by ``nanmedian`` but remain unmasked, so a NaN pixel propagates
    into the fit's normal equations (historical behavior).

    Works on numpy or cupy inputs: ``np.nanmedian`` dispatches to cupy via the array-function
    protocol, and the element-wise operators are array-module-agnostic.
    """
    return base_flux_mask | (flux < 0.8 * np.nanmedian(flux))


def fit_epsf(
    design_matrix: np.ndarray,
    flux: np.ndarray,
    base_flux_mask: np.ndarray,
    flux_uncertainty_power: float,
    regularization_dimensions: int,
    *,
    flux_mask: np.ndarray | None = None,
):
    """
    Find the best-fit ePSF parameters given a design matrix and observed flux values.

    Uses `xp.linalg.lstsq` where `xp` is numpy or cupy depending on the whether `design_matrix` is
    on the CPU or GPU.

    Parameters
    ----------
    design_matrix : array
        2D matrix with shape `(m + r, n)` where `m` is the number of pixels in image, `r` is the
        number of extra dimensions used for regularization, and `n` is the number of parameters in
        the ePSF model.
    flux : array
        2D array of observed flux values with shape `(a, b)` where `a * b == m`.
    base_flux_mask : array[bool]
        2D mask array indicating bad (e.g., saturated) pixels.
    flux_uncertainty_power : float
        Power of pixel value used as observational uncertainty in fit. <1 emphasizes contributions
        from dimmer stars, 1 means all contributions are equal.
    regularization_dimensions : int
        Number of extra dimensions used for regularization. Must be added to observed vector.
    flux_mask : array[bool], optional
        Complete pixel mask to use for the fit. If `None` (the default), it is computed by
        :func:`get_default_epsf_flux_mask`, which adds pixels dimmer than 0.8 times the median
        flux to `base_flux_mask`. If provided, `base_flux_mask` is ignored; the mask must match
        `flux`'s shape and should live on the same array module (numpy/cupy) as
        `flux`/`design_matrix`.

    Returns
    -------
    epsf_parameters : array
        Array of best-fit ePSF parameters.
    """
    flux_uncertainty_scale = 1 / (np.abs(flux) ** flux_uncertainty_power)
    if flux_mask is None:
        flux_mask = get_default_epsf_flux_mask(flux, base_flux_mask)

    # Set up observed vector accounting for regularization
    observed_vector = np.hstack((flux.flatten(), np.zeros(regularization_dimensions)))
    uncertainty_scale = np.hstack(
        (flux_uncertainty_scale.flatten(), np.ones(regularization_dimensions))
    )
    mask = np.hstack((flux_mask.flatten(), np.zeros(regularization_dimensions, dtype=bool)))

    if HAS_CUPY:
        import cupy as cp

        xp = cp.get_array_module(design_matrix, flux)
    else:
        xp = np

    A = (design_matrix * uncertainty_scale[:, np.newaxis])[~mask]
    b = (observed_vector * uncertainty_scale)[~mask]

    try:
        # Solve the normal equation instead of running a least squares fit directly. This is much
        # faster because `alpha` is the size of the number of dimensions of the PSF model, which is
        # much smaller than the number of dimensions of the observed flux. In the usual case, this
        # amounts to solving a 535-dimesnional linear equation instead of running a least squares
        # fit on a 23029x535 matrix.
        # Using the normal equation is valid because A has *many* more rows than columns, so the
        # chance that A has linearly dependent columns is negligible.
        alpha = A.T @ A
        beta = A.T @ b
        return xp.linalg.solve(alpha, beta)
    except xp.linalg.LinAlgError:
        # Just in case - this is useful eg for testing
        return xp.linalg.lstsq(A, b)[0]


def fit_epsf_for_source(
    source: "FFICutout",
    psf_size: int,
    oversample_factor: int,
    edge_compression_factor: float,
    flux_uncertainty_power: float,
    use_gpu: bool = True,
):
    """
    Fit an ePSF for each cadence in an :class:`FFICutout`.

    Parameters
    ----------
    source : FFICutout
        FFI cutout with observed flux, star positions, and star brightnesses.
    psf_size : int
        Side length of ePSF in pixels.
    oversample_factor : int
        Factor by which to oversample the ePSF compared to image pixels.
    flux_uncertainty_power : float
        Power of pixel value used as observational uncertainty in ePSF fit. <1 emphasizes
        contributions from dimmer stars, 1 means all contributions are equal.
    use_gpu : bool
        If `True`, use `cupy` to run the ePSF parameter fit on the GPU. Requires `cupy` to be
        installed and at least one CUDA device to be available.

    Returns
    -------
    epsf : array
        2D array where first dimension corresponds to cadences in `source` and second dimension
        contains the best-fit ePSF parameters per cadence.
    """
    logger.debug(
        f"Fitting ePSF for source in {source.camera}-{source.ccd} at {source.ccd_x}, {source.ccd_y}"
    )
    star_positions = np.array(
        [source.gaia[f"sector_{source.sector}_x"], source.gaia[f"sector_{source.sector}_y"]]
    ).T
    design_matrix, regularization_extension_size = make_tglc_design_matrix(
        source.flux.shape[1:],
        (psf_size, psf_size),
        oversample_factor,
        star_positions,
        source.gaia["tess_flux_ratio"].data,
        source.mask.data,
        edge_compression_factor,
    )
    flux = source.flux
    # Mask out saturated pixels as a base
    base_flux_mask = source.mask.mask

    if use_gpu and HAS_CUPY:
        import cupy as cp

        design_matrix = cp.asarray(design_matrix)
        flux = cp.asarray(flux)
        base_flux_mask = cp.asarray(base_flux_mask)
        xp = cp
    else:
        xp = np

    e_psf = xp.zeros((flux.shape[0], design_matrix.shape[1]))
    # JIT-ing this loop using numba did not give much performance benefit. Maybe vectorizing would?
    for i in range(flux.shape[0]):
        try:
            # fit_epsf will automatically use the appropriate lstsq method.
            e_psf[i] = fit_epsf(
                design_matrix,
                flux[i],
                base_flux_mask,
                flux_uncertainty_power,
                regularization_extension_size,
            )
        except np.linalg.LinAlgError as e:
            logger.warning(f"Error while fitting ePSF: {e}")
            e_psf[i] = np.nan
    if xp != np:
        e_psf = e_psf.get()
    return e_psf


class EPSF:
    """
    Fitted-ePSF data product: per-cadence best-fit ePSF and background parameters for one FFI
    cutout, bundled with the metadata needed to interpret and trace the parameter array.
    """

    def __init__(
        self,
        array: np.ndarray,
        *,
        psf_size: int,
        oversample: int,
        orbit: int,
        sector: int,
        camera: int,
        ccd: int,
        cutout_x: int,
        cutout_y: int,
        background_columns: tuple[str, ...] = EPSF_BACKGROUND_COLUMNS,
    ):
        """
        Parameters
        ----------
        array : array
            2D ``(t, k)`` array of best-fit ePSF + background parameters, where ``t`` is the
            number of cadences and ``k == parameter_count(psf_size, oversample)``. Coerced to a
            float64 numpy array; cadences whose fit failed are rows of NaN.
        psf_size : int
            Side length in pixels of the square PSF model.
        oversample : int
            Factor by which the PSF is oversampled relative to image pixels.
        orbit, sector, camera, ccd : int
            TESS identifiers for the cutout this ePSF was fit for.
        cutout_x, cutout_y : int
            Cutout grid indices matching the originating :class:`FFICutout`. ``-1`` is a legacy
            "not set" sentinel.
        background_columns : tuple[str, ...]
            Names of the background parameter columns appended after the PSF parameters. Defaults
            to :data:`EPSF_BACKGROUND_COLUMNS`.
        """
        array = np.asarray(array, dtype=np.float64)
        expected_columns = self.parameter_count(psf_size, oversample, len(background_columns))
        if array.ndim != 2 or array.shape[1] != expected_columns:
            raise ValueError(
                f"ePSF array has shape {array.shape}; expected (n_cadences, {expected_columns}) "
                f"for psf_size={psf_size}, oversample={oversample}, and {len(background_columns)} "
                "background parameters"
            )
        self.array = array
        self.psf_size = int(psf_size)
        self.oversample = int(oversample)
        self.orbit = int(orbit)
        self.sector = int(sector)
        self.camera = int(camera)
        self.ccd = int(ccd)
        self.cutout_x = int(cutout_x)
        self.cutout_y = int(cutout_y)
        self.background_columns = tuple(background_columns)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} orbit-{self.orbit} cam{self.camera}-ccd{self.ccd} "
            f"cutout ({self.cutout_x}, {self.cutout_y}) "
            f"psf_size={self.psf_size} oversample={self.oversample} cadences={self.n_cadences}>"
        )

    @staticmethod
    def parameter_count(
        psf_size: int, oversample: int, n_background: int = len(EPSF_BACKGROUND_COLUMNS)
    ) -> int:
        """Total number of fit parameters (``k``) for the given ePSF configuration."""
        return (psf_size * oversample + 1) ** 2 + n_background

    @property
    def n_cadences(self) -> int:
        """Number of cadences (``t``) the ePSF was fit for."""
        return self.array.shape[0]

    @property
    def n_parameters(self) -> int:
        """Total number of fit parameters (``k``)."""
        return self.array.shape[1]

    @property
    def n_background(self) -> int:
        """Number of background parameter columns."""
        return len(self.background_columns)

    @property
    def oversampled_psf_shape(self) -> tuple[int, int]:
        """Shape of the oversampled PSF model grid."""
        return (self.psf_size * self.oversample + 1, self.psf_size * self.oversample + 1)

    @property
    def n_psf_parameters(self) -> int:
        """Number of PSF model parameters (points in the oversampled PSF grid)."""
        return (self.psf_size * self.oversample + 1) ** 2

    @property
    def psf_parameters(self) -> np.ndarray:
        """View of the PSF model parameters, with shape ``(t, n_psf_parameters)``."""
        return self.array[:, : self.n_psf_parameters]

    @property
    def background_parameters(self) -> np.ndarray:
        """View of the background parameters, with shape ``(t, n_background)``."""
        return self.array[:, self.n_psf_parameters :]

    @property
    def failed_cadence_mask(self) -> np.ndarray:
        """Boolean array flagging cadences whose ePSF fit failed (recorded as rows of NaN)."""
        return np.isnan(self.array).any(axis=1)

    def background_parameter(self, name: str) -> np.ndarray:
        """
        Get the time series of a single background parameter by name.

        Parameters
        ----------
        name : str
            One of the names in ``background_columns``.

        Returns
        -------
        parameter : array
            1D array of the parameter's best-fit value per cadence, with shape ``(t,)``.
        """
        try:
            column = self.background_columns.index(name)
        except ValueError:
            raise ValueError(
                f"Unknown background parameter {name!r}: valid names are {self.background_columns}"
            ) from None
        return self.array[:, self.n_psf_parameters + column]

    def make_design_matrix(
        self,
        image_shape: tuple[int, int],
        star_positions: np.ndarray,
        star_flux_ratios: np.ndarray,
        background_strap_mask: np.ndarray | None = None,
        edge_compression_scale_factor: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Construct a TGLC design matrix matching this ePSF's ``psf_size`` and ``oversample``.

        Delegates to :func:`make_tglc_design_matrix`; see its docstring for parameter and return
        value details.
        """
        return make_tglc_design_matrix(
            image_shape,
            (self.psf_size, self.psf_size),
            self.oversample,
            star_positions,
            star_flux_ratios,
            background_strap_mask,
            edge_compression_scale_factor,
        )

    def matches_cutout(self, cutout: "FFICutout") -> bool:
        """
        Check whether this ePSF was fit for the given :class:`FFICutout`.

        Compares the TESS identifiers, the cutout grid indices (ignoring the legacy ``-1``
        "not set" sentinel on either side), and the cadence counts.
        """
        if (self.orbit, self.sector, self.camera, self.ccd) != (
            cutout.orbit,
            cutout.sector,
            cutout.camera,
            cutout.ccd,
        ):
            return False
        for own_index, cutout_index in [
            (self.cutout_x, cutout.cutout_x),
            (self.cutout_y, cutout.cutout_y),
        ]:
            if own_index != -1 and cutout_index != -1 and own_index != cutout_index:
                return False
        return self.n_cadences == cutout.flux.shape[0]

    @classmethod
    def from_cutout_fit(
        cls,
        cutout: "FFICutout",
        *,
        psf_size: int,
        oversample: int,
        edge_compression_factor: float,
        flux_uncertainty_power: float,
        use_gpu: bool = True,
    ) -> "EPSF":
        """
        Fit an ePSF for each cadence of an :class:`FFICutout`.

        Delegates the fit to :func:`fit_epsf_for_source` and harvests the identifying metadata
        from the cutout; see :func:`fit_epsf_for_source` for parameter details.
        """
        array = fit_epsf_for_source(
            cutout,
            psf_size,
            oversample,
            edge_compression_factor,
            flux_uncertainty_power,
            use_gpu=use_gpu,
        )
        return cls(
            array,
            psf_size=psf_size,
            oversample=oversample,
            orbit=cutout.orbit,
            sector=cutout.sector,
            camera=cutout.camera,
            ccd=cutout.ccd,
            cutout_x=cutout.cutout_x,
            cutout_y=cutout.cutout_y,
        )

    @classmethod
    def from_fits(cls, path: Path) -> "EPSF":
        """Read an :class:`EPSF` from a FITS file written by :func:`tglc.io.write_epsf_fits`."""
        # Local import: breaks the epsf <-> io cycle.
        from tglc.io import read_epsf_fits

        return read_epsf_fits(path)

    def to_fits(self, path: Path) -> None:
        """Write this :class:`EPSF` to a FITS file via :func:`tglc.io.write_epsf_fits`."""
        # Local import: breaks the epsf <-> io cycle.
        from tglc.io import write_epsf_fits

        write_epsf_fits(self, path)
