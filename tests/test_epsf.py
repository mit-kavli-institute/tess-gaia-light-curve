import numpy as np
import pytest

from tglc.epsf import fit_epsf, get_xy_coordinates_centered_at_zero, make_tglc_design_matrix
from tglc.utils._optional_deps import HAS_CUPY


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

    design_matrix, regularization_extension_size = make_tglc_design_matrix(
        image_shape, psf_shape_pixels, oversample_factor, star_positions, star_flux_ratios
    )

    assert design_matrix.shape == (150 * 150, 23 * 23)
    assert regularization_extension_size == 0
    np.testing.assert_equal(design_matrix[:, -6:], 0)


def test_make_tglc_design_matrix_with_background():
    # Actual expected values
    image_shape = (150, 150)
    psf_shape_pixels = (11, 11)
    oversample_factor = 2
    star_positions = np.array([[10, 10]])
    star_flux_ratios = np.array([1])
    background_strap_mask = np.zeros(image_shape)

    design_matrix, regularization_extension_size = make_tglc_design_matrix(
        image_shape,
        psf_shape_pixels,
        oversample_factor,
        star_positions,
        star_flux_ratios,
        background_strap_mask,
    )

    assert design_matrix.shape == (150 * 150, 23 * 23 + 6)
    assert regularization_extension_size == 0


def test_make_tglc_design_matrix_with_edge_compression():
    # Actual expected values
    image_shape = (150, 150)
    psf_shape_pixels = (11, 11)
    oversample_factor = 2
    star_positions = np.array([[10, 10]])
    star_flux_ratios = np.array([1])
    background_strap_mask = np.zeros(image_shape)
    edge_compression_scale_factor = 1e-4

    design_matrix, regularization_extension_size = make_tglc_design_matrix(
        image_shape,
        psf_shape_pixels,
        oversample_factor,
        star_positions,
        star_flux_ratios,
        background_strap_mask,
        edge_compression_scale_factor,
    )

    assert design_matrix.shape == (150 * 150 + 23 * 23, 23 * 23 + 6)
    assert regularization_extension_size == 23 * 23


def test_make_tglc_design_matrix_models_image():
    # Actual expected values
    image_shape = (150, 150)
    psf_shape_pixels = (11, 11)
    oversample_factor = 2
    star_positions = np.array([[10, 10]])
    star_flux_ratios = np.array([1])

    design_matrix, _ = make_tglc_design_matrix(
        image_shape, psf_shape_pixels, oversample_factor, star_positions, star_flux_ratios
    )

    simple_psf = np.ones((23, 23))
    no_background_model = simple_psf.reshape(-1)
    ones_around_star = np.zeros((150, 150))
    for i in range(-5, 6):
        for j in range(-5, 6):
            ones_around_star[10 + i, 10 + j] = 1
    modeled_image = np.dot(design_matrix, no_background_model).reshape(150, 150)
    np.testing.assert_equal(modeled_image, ones_around_star)


def test_fit_epsf():
    # Actual expected values
    image_shape = (150, 150)
    psf_shape_pixels = (11, 11)
    oversample_factor = 2
    star_positions = np.array([[10, 10]])
    star_flux_ratios = np.array([1])
    star_flux_ratios = 1 - np.arange(len(star_positions)) / (len(star_positions) * 0.9)
    background_strap_mask = np.zeros(image_shape)
    edge_compression_scale_factor = 1e-4

    design_matrix, regularization_extension_size = make_tglc_design_matrix(
        image_shape,
        psf_shape_pixels,
        oversample_factor,
        star_positions,
        star_flux_ratios,
        background_strap_mask,
        edge_compression_scale_factor,
    )

    # Can't use zeros because we do 1/(flux ** uncertainty_power) during the fit
    observed_flux = np.ones((150, 150))
    for i in range(-5, 6):
        for j in range(-5, 6):
            observed_flux[10 + i, 10 + j] += 1

    base_flux_mask = np.zeros((150, 150), dtype=bool)

    epsf = fit_epsf(
        design_matrix,
        observed_flux,
        base_flux_mask,
        flux_uncertainty_power=1.4,
        regularization_dimensions=regularization_extension_size,
    )
    # ePSF should be all zeros and ones
    np.testing.assert_allclose(
        np.where(~np.isclose(epsf[:-6], 0, atol=1e-6), epsf[:-6] - 1, epsf[:-6]), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(epsf[-6:-1], 0.0, atol=1e-6)
    np.testing.assert_allclose(epsf[-1], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        np.dot(design_matrix, epsf)[: 150 * 150].reshape(150, 150), observed_flux, atol=1e-6
    )


@pytest.mark.skipif(not HAS_CUPY, reason="cupy is required")
def test_fit_epsf_with_cupy_design_matrix():
    import cupy as cp

    # Actual expected values
    image_shape = (150, 150)
    psf_shape_pixels = (11, 11)
    oversample_factor = 2
    star_positions = np.array([[10, 10]])
    star_flux_ratios = np.array([1])
    star_flux_ratios = 1 - np.arange(len(star_positions)) / (len(star_positions) * 0.9)
    background_strap_mask = np.zeros(image_shape)
    edge_compression_scale_factor = 1e-4

    design_matrix, regularization_extension_size = make_tglc_design_matrix(
        image_shape,
        psf_shape_pixels,
        oversample_factor,
        star_positions,
        star_flux_ratios,
        background_strap_mask,
        edge_compression_scale_factor,
    )
    design_matrix = cp.asarray(design_matrix)

    # Can't use zeros because we do 1/(flux ** uncertainty_power) during the fit
    observed_flux = np.ones((150, 150))
    for i in range(-5, 6):
        for j in range(-5, 6):
            observed_flux[10 + i, 10 + j] += 1

    base_flux_mask = np.zeros((150, 150), dtype=bool)

    epsf = fit_epsf(
        design_matrix,
        observed_flux,
        base_flux_mask,
        flux_uncertainty_power=1.4,
        regularization_dimensions=regularization_extension_size,
    )
    # ePSF should be all zeros and ones
    np.testing.assert_allclose(
        np.where(~np.isclose(epsf[:-6], 0, atol=1e-6), epsf[:-6] - 1, epsf[:-6]), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(epsf[-6:-1], 0.0, atol=1e-6)
    np.testing.assert_allclose(epsf[-1], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        np.dot(design_matrix, epsf)[: 150 * 150].reshape(150, 150), observed_flux, atol=1e-6
    )
