"""ePSF helper functions."""

from math import ceil, floor

import numpy as np


def get_xy_coordinates_centered_at_zero(shape: tuple[int, int]):
    return np.meshgrid(
        np.arange(shape[1]) - (shape[1] - 1) / 2, np.arange(shape[0]) - (shape[0] - 1) / 2
    )


def make_tglc_design_matrix(
    image_shape: tuple[int, int],
    psf_shape_pixels: tuple[int, int],
    oversample_factor: int,
    star_positions: np.ndarray,
    background_strap_mask: np.ndarray,
    edge_compression_scale_factor: float,
):
    """
    Construct the TGLC design matrix from equation (3) of Han & Brandt, 2023.

    Returns
    -------
    design_matrix, regularization_zeros : tuple[array, int]
        Design matrix and number of zeros that should be appended to observed vectors for
        regularization during fitting.
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
    pixels_in_psf_x = np.arange(psf_shape_pixels[1]) - (psf_shape_pixels[1] - 1) / 2
    pixels_in_epsf_y = np.arange(psf_shape_pixels[0]) - (psf_shape_pixels[0] - 1) / 2
    for x, y in star_positions:
        nearest_pixel_x, nearest_pixel_y = (round(x), round(y))
        for pixel_x in (pixels_in_psf_x + nearest_pixel_x).astype(int):
            for pixel_y in (pixels_in_epsf_y + nearest_pixel_y).astype(int):
                # Get the coordinate of the nearest pixel center in coordinates of the PSF grid, with
                # the bottom left PSF point at (0, 0) and distance 1 between adjacent PSF points.
                pixel_psf_x, pixel_psf_y = (
                    (pixel_x - x) / oversample_factor + oversampled_psf_shape[1] // 2,
                    (pixel_y - y) / oversample_factor + oversampled_psf_shape[0] // 2,
                )
                # The four closest PSF points are interpolated to give the PSF model value of the pixel,
                # and their coordinates are given by rounding the pixel center coordinates up and down. The
                # contribution of each PSF point is given by the product of the differences in x and y.
                for psf_x, psf_y in [
                    (floor(pixel_psf_x), floor(pixel_psf_y)),  # left down
                    (floor(pixel_psf_x), ceil(pixel_psf_y)),  # right down
                    (ceil(pixel_psf_x), floor(pixel_psf_y)),  # left up
                    (ceil(pixel_psf_x), ceil(pixel_psf_y)),  # right up
                ]:
                    # TODO explain or give better names
                    x_interpolation = np.abs(pixel_psf_x - psf_x) or 1.0
                    y_interpolation = np.abs(pixel_psf_y - psf_y) or 1.0
                    epsf_contributions_to_pixels[pixel_y, pixel_x, psf_y, psf_x] = np.abs(
                        x_interpolation * y_interpolation
                    )

    # To calculate the linear gradients, we need the x and y coordinates of each pixel.
    image_pixel_xs, image_pixel_ys = get_xy_coordinates_centered_at_zero(image_shape)
    # background_contributions_to_pixels[iy, ix, b] is the contribution of background parameter b to
    # pixel (ix, iy) in the image.
    background_contribution_to_pixels = np.stack(
        [
            np.ones(image_shape),  # flat background level => same contribution to each point
            image_pixel_ys,  # y component of linear gradient => use x coordinate of each point
            image_pixel_xs,  # x component of linear gradient => use y coordinate of each point
            background_strap_mask,  # flat contribution of background straps
            background_strap_mask * image_pixel_ys,  # y-dependent contribution of background straps
            background_strap_mask * image_pixel_xs,  # x-dependent contribution of background straps
        ]
    )

    # Construct the full design matrix by flattening image coordinates.
    design_matrix = np.hstack(
        [
            epsf_contributions_to_pixels.reshape(
                image_shape[0] * image_shape[1], oversampled_psf_shape[0] * oversampled_psf_shape[1]
            ),
            background_contribution_to_pixels.reshape(image_shape[0] * image_shape[1], -1),
        ]
    )

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
            * ((psf_point_x / psf_shape_pixels[1]) ** 4 + (psf_point_y / psf_shape_pixels[0]) ** 4)
        )
    )
    edge_compression_block = np.hstack(
        [
            np.diag(
                psf_distance_from_center_weight.reshape(
                    oversampled_psf_shape[0] * oversampled_psf_shape[1]
                )
            ),
            np.zeros(
                (
                    oversampled_psf_shape[0] * oversampled_psf_shape[1],
                    background_contribution_to_pixels.shape[0],
                )
            ),
        ]
    )

    return (
        np.vstack([design_matrix, edge_compression_block]),
        oversampled_psf_shape[0] * oversampled_psf_shape[1],
    )


# @guvectorize([(float64[:, :], float64[:], float64, float64, float64[:])], ("(m,n),(n),(),(m)"))
def fit_psf(
    design_matrix: np.ndarray,
    flux: np.ndarray,
    pixel_weight_power: float,
    result: np.ndarray,
):
    uncertainty_scale = 1 / (flux**pixel_weight_power)
    return np.linalg.lstsq(design_matrix * uncertainty_scale, flux * uncertainty_scale)
