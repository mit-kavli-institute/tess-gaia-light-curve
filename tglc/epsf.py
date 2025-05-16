"""ePSF helper functions."""

from math import ceil, floor
from typing import Optional

from numba import jit
import numpy as np

from tglc.util._optional_deps import HAS_CUPY


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
    background_strap_mask: Optional[np.ndarray] = None,
    edge_compression_scale_factor: Optional[float] = None,
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
        extra rows are added to the design matrix.

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
    for (x, y), flux_ratio in zip(star_positions, star_flux_ratios):
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
                    (floor(pixel_psf_x), floor(pixel_psf_y)),  # left down
                    (floor(pixel_psf_x), ceil(pixel_psf_y)),  # right down
                    (ceil(pixel_psf_x), floor(pixel_psf_y)),  # left up
                    (ceil(pixel_psf_x), ceil(pixel_psf_y)),  # right up
                ]:
                    x_interpolation = np.abs(pixel_psf_x - psf_x) or 1.0
                    y_interpolation = np.abs(pixel_psf_y - psf_y) or 1.0
                    epsf_contributions_to_pixels[pixel_y, pixel_x, psf_y, psf_x] = (
                        flux_ratio * np.abs(x_interpolation * y_interpolation)
                    )

    design_matrix = epsf_contributions_to_pixels.reshape(
        image_shape[0] * image_shape[1],
        oversampled_psf_shape[0] * oversampled_psf_shape[1],
    )
    if background_strap_mask is not None:
        # To calculate the linear gradients, we need the x and y coordinates of each pixel.
        image_pixel_xs, image_pixel_ys = get_xy_coordinates_centered_at_zero(image_shape)
        # background_contributions_to_pixels[iy, ix, b] is the contribution of background parameter b to
        # pixel (ix, iy) in the image.
        background_contribution_to_pixels = np.stack(
            (
                np.ones(image_shape),  # flat background level => same contribution to each point
                image_pixel_ys,  # y component of linear gradient => use x coordinate of each point
                image_pixel_xs,  # x component of linear gradient => use y coordinate of each point
                background_strap_mask,  # flat contribution of background straps
                background_strap_mask
                * image_pixel_ys,  # y-dependent contribution of background straps
                background_strap_mask
                * image_pixel_xs,  # x-dependent contribution of background straps
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


def fit_epsf(
    design_matrix: np.ndarray,
    flux: np.ndarray,
    base_flux_mask: np.ndarray,
    flux_uncertainty_power: float,
    regularization_dimensions: int,
):
    """
    Find the best-fit ePSF parameters given a design matrix and observed flux values.

    Uses `xp.linalg.lstsq` where `xp` is numpy or cupy by default. If `design_matrix` is sparse,
    uses `lsmr` from `cupyx.scipy.sparse.linalg` or `scipy.sparse.linalg`.

    Parameters
    ----------
    design_matrix : array
        2D matrix with shape `(m + r, n)` where `m` is the number of pixels in image, `r` is the
        number of extra dimensions used for regularization, and `n` is the number of parameters in
        the ePSF model.
    flux : array
        2D array of observed flux values with shape `(a, b)` where `a * b == m`.
    base_flux_mask : array[bool]
        2D mask array indicating bad (e.g., saturated) pixels. Pixels lower than 0.8 times the
        median flux are masked in addition.
    flux_uncertainty_power : float
        Power of pixel value used as observational uncertainty in fit. <1 emphasizes contributions
        from dimmer stars, 1 means all contributions are equal.
    regularization_dimensions : int
        Number of extra dimensions used for regularization. Must be added to observed vector.

    Returns
    -------
    epsf_parameters : array
        Array of best-fit ePSF parameters.
    """
    flux_uncertainty_scale = 1 / (np.abs(flux) ** flux_uncertainty_power)
    flux_mask = base_flux_mask | (flux < 0.8 * np.nanmedian(flux))

    # Set up observed vector accounting for regularization
    observed_vector = np.hstack((flux.flatten(), np.zeros(regularization_dimensions)))
    uncertainty_scale = np.hstack(
        (flux_uncertainty_scale.flatten(), np.ones(regularization_dimensions))
    )
    mask = np.hstack((flux_mask.flatten(), np.zeros(regularization_dimensions, dtype=np.bool)))

    if HAS_CUPY:
        import cupy as cp

        xp = cp.get_array_module(design_matrix, flux)
        if xp == cp:
            from cupyx.scipy import sparse
            from cupyx.scipy.sparse import linalg  # noqa: F401
        else:
            from scipy import sparse
    else:
        from scipy import sparse

        xp = np

    if sparse.issparse(design_matrix):
        result, _istop, _itn, _normr, _normar, _norma, _conda, _normx = sparse.linalg.lsmr(
            (design_matrix.multiply(uncertainty_scale[:, np.newaxis])).tocsr()[~mask],
            (observed_vector * uncertainty_scale)[~mask],
        )
    else:
        result, _residuals, _rank, _s = xp.linalg.lstsq(
            (design_matrix * uncertainty_scale[:, np.newaxis])[~mask],
            (observed_vector * uncertainty_scale)[~mask],
        )
    return result
