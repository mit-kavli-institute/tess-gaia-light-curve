import numpy as np
import pytest

from tglc.epsf import get_xy_coordinates_centered_at_zero, make_tglc_design_matrix


@pytest.mark.parametrize("shape", [(11, 11), (150, 150), (20, 10), (10, 20)])
def test_get_xy_coordinates_centered_at_zero(shape: tuple[int, int]):
    # get_xy_coordinates_centered_at_zero basically reimplements np.meshgrid for compatibility with
    # numba, so we test that it matches np.meshgrid for a few different shapes.
    meshgrid_x, meshgrid_y = np.meshgrid(
        np.arange(shape[1]) - (shape[1] - 1) / 2, np.arange(shape[0]) - (shape[0] - 1) / 2
    )

    test_x, test_y = get_xy_coordinates_centered_at_zero(shape)

    np.testing.assert_array_equal(test_x, meshgrid_x)
    np.testing.assert_array_equal(test_y, meshgrid_y)


def test_make_tglc_design_matrix():
    # Actual expected values
    image_shape = (150, 150)
    psf_shape_pixels = (11, 11)
    oversample_factor = 2
    star_positions = np.array([[10, 10]])
    star_flux_ratios = np.array([1])
    background_strap_mask = np.zeros(image_shape)
    edge_compression_scale_factor = 1e-4

    design_matrix, n_normalization_zeros = make_tglc_design_matrix(
        image_shape,
        psf_shape_pixels,
        oversample_factor,
        star_positions,
        star_flux_ratios,
        background_strap_mask,
        edge_compression_scale_factor,
    )
    assert design_matrix.shape[0] == 150 * 150 + 23 * 23
    assert design_matrix.shape[1] == 23 * 23 + 6
    assert n_normalization_zeros == 23 * 23

    simple_psf = np.ones((23, 23))
    simple_model_vector = np.hstack([simple_psf.reshape(23 * 23), np.zeros(6)])
    ones_around_star = np.zeros((150, 150))
    for i in range(-5, 6):
        for j in range(-5, 6):
            ones_around_star[10 + i, 10 + j] = 1
    modelled_image = np.dot(design_matrix, simple_model_vector)[:22500].reshape(150, 150)
    assert np.all(modelled_image == ones_around_star)
