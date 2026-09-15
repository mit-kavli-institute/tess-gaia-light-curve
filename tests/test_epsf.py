import numpy as np
import pytest

from tglc.epsf import (
    EPSF,
    EPSF_BACKGROUND_COLUMNS,
    fit_epsf,
    get_default_epsf_flux_mask,
    get_xy_coordinates_centered_at_zero,
    make_tglc_design_matrix,
)
from tglc.utils._optional_deps import HAS_CUPY

from .synthetic_data import make_synthetic_cutout, make_synthetic_epsf


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


# ---------------------------------------------------------------------
# EPSF class
# ---------------------------------------------------------------------


def _epsf_metadata(**overrides) -> dict:
    metadata = {
        "psf_size": 11,
        "oversample": 2,
        "orbit": 185,
        "sector": 89,
        "camera": 1,
        "ccd": 1,
        "cutout_x": 0,
        "cutout_y": 0,
    }
    metadata.update(overrides)
    return metadata


def test_epsf_parameter_count():
    assert EPSF.parameter_count(11, 2) == 23 * 23 + 6
    assert EPSF.parameter_count(3, 1) == 4 * 4 + 6
    assert EPSF.parameter_count(11, 2, n_background=0) == 23 * 23


def test_epsf_constructor_validates_shape():
    with pytest.raises(ValueError, match=r"expected \(n_cadences, 535\).*psf_size=11"):
        EPSF(np.zeros((3, 100)), **_epsf_metadata())
    with pytest.raises(ValueError, match="expected"):
        EPSF(np.zeros(535), **_epsf_metadata())  # 1D array rejected


def test_epsf_constructor_coerces_types():
    epsf = EPSF(make_synthetic_epsf().astype(np.float32), **_epsf_metadata(orbit=np.int64(185)))
    assert epsf.array.dtype == np.float64
    assert isinstance(epsf.orbit, int)


def test_epsf_repr():
    epsf = EPSF(make_synthetic_epsf(), **_epsf_metadata())
    assert repr(epsf) == (
        "<EPSF orbit-185 cam1-ccd1 cutout (0, 0) psf_size=11 oversample=2 cadences=3>"
    )


def test_epsf_shape_properties():
    epsf = EPSF(make_synthetic_epsf(), **_epsf_metadata())
    assert epsf.n_cadences == 3
    assert epsf.n_parameters == 535
    assert epsf.n_psf_parameters == 23 * 23
    assert epsf.oversampled_psf_shape == (23, 23)
    assert epsf.n_background == 6
    assert epsf.background_columns == EPSF_BACKGROUND_COLUMNS


def test_epsf_parameter_views_share_memory():
    epsf = EPSF(make_synthetic_epsf(), **_epsf_metadata())
    assert epsf.psf_parameters.shape == (3, 23 * 23)
    assert epsf.background_parameters.shape == (3, 6)
    assert np.shares_memory(epsf.psf_parameters, epsf.array)
    assert np.shares_memory(epsf.background_parameters, epsf.array)
    np.testing.assert_array_equal(
        np.hstack((epsf.psf_parameters, epsf.background_parameters)), epsf.array
    )


def test_epsf_background_parameter_by_name():
    epsf = EPSF(make_synthetic_epsf(), **_epsf_metadata())
    np.testing.assert_array_equal(epsf.background_parameter("y_strap"), epsf.array[:, -6])
    np.testing.assert_array_equal(epsf.background_parameter("flat"), epsf.array[:, -1])
    with pytest.raises(ValueError, match="y_strap.*flat"):
        epsf.background_parameter("not_a_column")


def test_epsf_failed_cadence_mask():
    array = make_synthetic_epsf()
    array[1] = np.nan
    epsf = EPSF(array, **_epsf_metadata())
    np.testing.assert_array_equal(epsf.failed_cadence_mask, [False, True, False])


def test_epsf_make_design_matrix_matches_direct_call():
    image_shape = (12, 12)
    star_positions = np.array([[5.0, 5.0]])
    star_flux_ratios = np.array([1.0])
    strap_mask = np.zeros(image_shape)
    epsf = EPSF(
        np.zeros((2, EPSF.parameter_count(3, 1))), **_epsf_metadata(psf_size=3, oversample=1)
    )

    for mask, edge_compression in [(None, None), (strap_mask, None), (strap_mask, 1e-4)]:
        method_matrix, method_extension = epsf.make_design_matrix(
            image_shape, star_positions, star_flux_ratios, mask, edge_compression
        )
        direct_matrix, direct_extension = make_tglc_design_matrix(
            image_shape, (3, 3), 1, star_positions, star_flux_ratios, mask, edge_compression
        )
        np.testing.assert_array_equal(method_matrix, direct_matrix)
        assert method_extension == direct_extension


def test_epsf_matches_cutout():
    cutout = make_synthetic_cutout()  # orbit=185, sector=89, camera=1, ccd=1, 5 cadences
    array = make_synthetic_epsf(n_cadences=5)

    assert EPSF(array, **_epsf_metadata()).matches_cutout(cutout)
    # -1 is the legacy "not set" sentinel and should not fail the match
    assert EPSF(array, **_epsf_metadata(cutout_x=-1, cutout_y=-1)).matches_cutout(cutout)
    assert not EPSF(array, **_epsf_metadata(ccd=2)).matches_cutout(cutout)
    assert not EPSF(array, **_epsf_metadata(cutout_x=3)).matches_cutout(cutout)
    # Cadence count mismatch
    assert not EPSF(make_synthetic_epsf(n_cadences=4), **_epsf_metadata()).matches_cutout(cutout)


def test_epsf_from_cutout_fit():
    cutout = make_synthetic_cutout()
    epsf = EPSF.from_cutout_fit(
        cutout,
        psf_size=3,
        oversample=1,
        edge_compression_factor=1e-4,
        flux_uncertainty_power=1.4,
        use_gpu=False,
    )
    assert epsf.array.shape == (cutout.flux.shape[0], EPSF.parameter_count(3, 1))
    assert (epsf.orbit, epsf.sector, epsf.camera, epsf.ccd) == (185, 89, 1, 1)
    assert (epsf.cutout_x, epsf.cutout_y) == (0, 0)
    assert epsf.matches_cutout(cutout)


# ---------------------------------------------------------------------
# Characterization tests pinning pre-refactor fit_epsf masking behavior
# ---------------------------------------------------------------------


def _small_fit_problem():
    image_shape = (12, 12)
    design_matrix, regularization_extension_size = make_tglc_design_matrix(
        image_shape, (3, 3), 1, np.array([[5.0, 5.0]]), np.array([1.0]), np.zeros(image_shape), 1e-4
    )
    flux = np.ones(image_shape)
    flux[4:7, 4:7] += 1.0
    return design_matrix, regularization_extension_size, flux


def test_fit_epsf_implicit_low_flux_mask_matches_manual_mask():
    """Pins the implicit flux < 0.8 * nanmedian(flux) pixel cut inside fit_epsf.

    Passing the same cut as the base mask must be a no-op, because fit_epsf unions the
    base mask with the implicit cut. Any change to the implicit mask semantics breaks this.
    """
    design_matrix, regularization_extension_size, flux = _small_fit_problem()
    flux[0:2, 0:2] = 0.1  # distinctly below 0.8 * median
    manual_mask = flux < 0.8 * np.nanmedian(flux)

    default_result = fit_epsf(
        design_matrix, flux, np.zeros(flux.shape, dtype=bool), 1.4, regularization_extension_size
    )
    manual_result = fit_epsf(design_matrix, flux, manual_mask, 1.4, regularization_extension_size)

    np.testing.assert_array_equal(default_result, manual_result)


def test_fit_epsf_nan_pixel_is_not_masked():
    """Pins that NaN flux pixels are NOT excluded by the implicit mask (NaN < x is False).

    An unmasked NaN pixel propagates through the normal equations and produces all-NaN
    parameters for the cadence; masking the pixel explicitly restores a finite fit.
    """
    design_matrix, regularization_extension_size, flux = _small_fit_problem()
    flux[0, 0] = np.nan

    nan_result = fit_epsf(
        design_matrix, flux, np.zeros(flux.shape, dtype=bool), 1.4, regularization_extension_size
    )
    assert np.isnan(nan_result).all()

    mask_nan_pixel = np.zeros(flux.shape, dtype=bool)
    mask_nan_pixel[0, 0] = True
    masked_result = fit_epsf(
        design_matrix, flux, mask_nan_pixel, 1.4, regularization_extension_size
    )
    assert np.isfinite(masked_result).all()


def test_get_default_epsf_flux_mask():
    flux = np.array([[1.0, 1.0, 0.5], [1.0, np.nan, 1.0], [2.0, 1.0, 0.1]])
    base_flux_mask = np.zeros(flux.shape, dtype=bool)
    base_flux_mask[0, 0] = True

    mask = get_default_epsf_flux_mask(flux, base_flux_mask)

    # median of non-NaN values is 1.0; pixels < 0.8 are masked, base mask is unioned in,
    # and the NaN pixel is NOT masked (NaN < x is False)
    np.testing.assert_array_equal(
        mask, [[True, False, True], [False, False, False], [False, False, True]]
    )


def test_fit_epsf_explicit_flux_mask_matches_default():
    design_matrix, regularization_extension_size, flux = _small_fit_problem()
    base_flux_mask = np.zeros(flux.shape, dtype=bool)

    default_result = fit_epsf(
        design_matrix, flux, base_flux_mask, 1.4, regularization_extension_size
    )
    explicit_result = fit_epsf(
        design_matrix,
        flux,
        base_flux_mask,
        1.4,
        regularization_extension_size,
        flux_mask=get_default_epsf_flux_mask(flux, base_flux_mask),
    )

    np.testing.assert_array_equal(default_result, explicit_result)


def test_fit_epsf_custom_flux_mask_changes_fit():
    design_matrix, regularization_extension_size, flux = _small_fit_problem()
    base_flux_mask = np.zeros(flux.shape, dtype=bool)
    # Mask the star's brightest pixels: the fit must differ from the default
    custom_mask = np.zeros(flux.shape, dtype=bool)
    custom_mask[4:7, 4:7] = True

    default_result = fit_epsf(
        design_matrix, flux, base_flux_mask, 1.4, regularization_extension_size
    )
    custom_result = fit_epsf(
        design_matrix,
        flux,
        base_flux_mask,
        1.4,
        regularization_extension_size,
        flux_mask=custom_mask,
    )

    assert not np.array_equal(default_result, custom_result)
