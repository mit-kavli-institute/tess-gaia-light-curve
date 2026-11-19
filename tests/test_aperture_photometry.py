"""
Tests for the tglc.aperture_photometry module, which provides a function for doing aperture photometry
on image cutouts.
"""

import warnings

from astropy import units as u
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst
import numpy as np
import pytest
from scipy.ndimage import center_of_mass

from tglc.aperture_photometry import (
    get_flux_portion_in_aperture,
    get_local_background,
    get_normalized_aperture_photometry,
    get_saturation_mask,
    measure_aperture_centroids,
    measure_aperture_flux,
    normalize_aperture_flux,
)


def test_get_normalized_aperture_photometry():
    # Sample data is for a tmag=10 source that distributes its light exactly evenly over the 25
    # pixels in the image cutout
    images = np.ones((5, 5, 5)) * 15_000 * 200 / (5 * 5)
    flux_portion = np.ones((5, 5)) / (5 * 5)
    quality_flags = np.zeros(5, dtype=int)

    photometry_data = get_normalized_aperture_photometry(
        images, quality_flags, 3, 2, 2, 10, 200 * u.second, flux_portion=flux_portion
    )
    assert np.all(photometry_data["magnitude"] == 10)
    assert np.all(photometry_data["centroid_x"] == 2.0 * u.pixel)
    assert np.all(photometry_data["centroid_y"] == 2.0 * u.pixel)
    assert photometry_data.meta["local_background"] == 0 * u.electron


def test_get_normalized_aperture_photometry_with_bottom_heavy_image():
    # Sample data is for a tmag=10 source that distributes its light exactly evenly over the bottom
    # 3x5 region of the images.
    light = np.ones((5, 3, 5)) * 15_000 * 200 / (5 * 3)
    images = np.pad(light, [(0, 0), (0, 2), (0, 0)])
    flux_portion = np.pad(np.ones((3, 5)) / (5 * 3), [(0, 2), (0, 0)])
    quality_flags = np.zeros(5, dtype=int)

    photometry_data = get_normalized_aperture_photometry(
        images, quality_flags, 3, 2, 2, 10, 200 * u.second, flux_portion=flux_portion
    )
    assert np.all(photometry_data["magnitude"] == 10)
    assert np.all(photometry_data["centroid_x"] == 2.0 * u.pixel)
    assert np.all(photometry_data["centroid_y"] == 1.5 * u.pixel)
    assert photometry_data.meta["local_background"] == 0 * u.electron


def test_get_normalized_aperture_photometry_with_left_heavy_image():
    # Sample data is for a tmag=10 source that distributes its light exactly evenly over the left
    # 5x3 region of the images.
    light = np.ones((5, 5, 3)) * 15_000 * 200 / (5 * 3)
    images = np.pad(light, [(0, 0), (0, 0), (0, 2)])
    flux_portion = np.pad(np.ones((5, 3)) / (5 * 3), [(0, 0), (0, 2)])
    quality_flags = np.zeros(5, dtype=int)

    photometry_data = get_normalized_aperture_photometry(
        images, quality_flags, 3, 2, 2, 10, 200 * u.second, flux_portion=flux_portion
    )
    assert np.all(photometry_data["magnitude"] == 10)
    assert np.all(photometry_data["centroid_x"] == 1.5 * u.pixel)
    assert np.all(photometry_data["centroid_y"] == 2.0 * u.pixel)
    assert photometry_data.meta["local_background"] == 0 * u.electron


def test_get_normalized_aperture_photometr_with_local_background():
    # Sample data is for a tmag=10 source that distributes its light exactly evenly over the 25
    # pixels in the image cutout, and there is some "local background" above the expected flux amount
    images = np.ones((5, 5, 5)) * 15_000 * 200 / (5 * 5) + 47
    flux_portion = np.ones((5, 5)) / (5 * 5)
    quality_flags = np.zeros(5, dtype=int)

    photometry_data = get_normalized_aperture_photometry(
        images, quality_flags, 3, 2, 2, 10, 200 * u.second, flux_portion=flux_portion
    )
    assert np.all(photometry_data["magnitude"] == 10)
    assert np.all(photometry_data["centroid_x"] == 2.0 * u.pixel)
    assert np.all(photometry_data["centroid_y"] == 2.0 * u.pixel)
    assert photometry_data.meta["local_background"] == 47 * 9 * u.electron


def test_get_normalized_aperture_photometr_with_fully_saturated_first_image():
    # Sample data is for a tmag=10 source that distributes its light exactly evenly over the 25
    # pixels in the image cutout, and there is some "local background" above the expected flux amount
    images = np.ones((5, 5, 5)) * 15_000 * 200 / (5 * 5)
    images[0, :, :] += 2e5 * 200 / 2  # total electrons from 200s to saturate a pixel in 2s
    flux_portion = np.ones((5, 5)) / (5 * 5)
    quality_flags = np.zeros(5, dtype=int)

    photometry_data = get_normalized_aperture_photometry(
        images, quality_flags, 3, 2, 2, 10, 200 * u.second, flux_portion=flux_portion
    )
    assert np.isnan(photometry_data["magnitude"][0])
    assert np.all(photometry_data["magnitude"][1:] == 10.0)
    assert np.isnan(photometry_data["centroid_x"][0])
    assert np.all(photometry_data["centroid_x"][1:] == 2.0 * u.pixel)
    assert np.isnan(photometry_data["centroid_y"][0])
    assert np.all(photometry_data["centroid_y"][1:] == 2.0 * u.pixel)
    assert photometry_data.meta["local_background"] == 0 * u.electron


def test_get_normalized_aperture_photometry_with_large_aperture():
    # Sample data is for a tmag=10 source that distributes its light exactly evenly over the 25
    # pixels in the image cutout
    images = np.ones((5, 5, 5)) * 15_000 * 200 / (5 * 5)
    flux_portion = np.ones((5, 5)) / (5 * 5)
    quality_flags = np.zeros(5, dtype=int)

    photometry_data = get_normalized_aperture_photometry(
        images, quality_flags, 5, 2, 2, 10, 200 * u.second, flux_portion=flux_portion
    )
    assert np.all(photometry_data["magnitude"] == 10)
    assert np.all(photometry_data["centroid_x"] == 2.0 * u.pixel)
    assert np.all(photometry_data["centroid_y"] == 2.0 * u.pixel)
    assert photometry_data.meta["local_background"] == 0 * u.electron


def test_get_normalized_aperture_photometry_with_small_aperture():
    # Sample data is for a tmag=10 source that distributes its light exactly evenly over the 25
    # pixels in the image cutout
    images = np.ones((5, 5, 5)) * 15_000 * 200 / (5 * 5)
    flux_portion = np.ones((5, 5)) / (5 * 5)
    quality_flags = np.zeros(5, dtype=int)

    photometry_data = get_normalized_aperture_photometry(
        images, quality_flags, 1, 2, 2, 10, 200 * u.second, flux_portion=flux_portion
    )
    assert np.all(photometry_data["magnitude"] == 10)
    assert np.all(photometry_data["centroid_x"] == 2.0 * u.pixel)
    assert np.all(photometry_data["centroid_y"] == 2.0 * u.pixel)
    assert photometry_data.meta["local_background"] == 0 * u.electron


def test_get_normalized_aperture_photometry_with_star_near_edge():
    # Sample data is for a tmag=10 source that distributes its light exactly evenly over the 25
    # pixels in the image cutout
    images = np.ones((5, 5, 5)) * 15_000 * 200 / (5 * 5)
    flux_portion = np.ones((5, 5)) / (5 * 5)
    quality_flags = np.zeros(5, dtype=int)

    photometry_data = get_normalized_aperture_photometry(
        images, quality_flags, 3, 0, 0, 10, 200 * u.second, flux_portion=flux_portion
    )
    assert np.all(photometry_data["magnitude"] == 10)
    assert np.all(photometry_data["centroid_x"] == 0.5 * u.pixel)
    assert np.all(photometry_data["centroid_y"] == 0.5 * u.pixel)
    assert photometry_data.meta["local_background"] == 0 * u.electron


def test_get_normalized_aperture_photometry_with_colname_prefix():
    # Sample data is for a tmag=10 source that distributes its light exactly evenly over the 25
    # pixels in the image cutout
    images = np.ones((5, 5, 5)) * 15_000 * 200 / (5 * 5)
    flux_portion = np.ones((5, 5)) / (5 * 5)
    quality_flags = np.zeros(5, dtype=int)

    photometry_data = get_normalized_aperture_photometry(
        images,
        quality_flags,
        3,
        0,
        0,
        10,
        200 * u.second,
        flux_portion=flux_portion,
        column_name_prefix="column_prefix",
    )
    for name in photometry_data.colnames:
        assert name.startswith("column_prefix")
    assert "column_prefixlocal_background" in photometry_data.meta


# ---------------------------------------------------------------------
# Characterization tests pinning pre-refactor behavior
# ---------------------------------------------------------------------
# Golden values generated against 92df5c1 (pre-refactor) with
# np.array2string(..., floatmode="unique"). Regenerating them defeats the purpose --
# they pin the behavior the explicit-steps refactor must preserve bit-for-bit.


def test_get_normalized_aperture_photometry_characterization():
    """Pins normalization with flagged cadences, a saturated cadence, and the flux <= 0 clip."""
    star_flux = 15_000.0 * 200.0
    images = np.full((5, 5, 5), star_flux / 25 + 100.0)
    images[0, 2, 1] += 500.0  # centroid asymmetry on cadence 0
    images[1, :, :] = 1.0  # flagged cadence, driven <= 0 by normalization -> NaN clip
    images[4, :, :] = 5.0e7  # good cadence above the saturation threshold -> NaN
    quality_flags = np.array([0, 1, 0, 2, 0])
    flux_portion = np.full((5, 5), 1 / 25)

    photometry_data = get_normalized_aperture_photometry(
        images, quality_flags, 3, 2, 2, 10.0, 200 * u.second, flux_portion=flux_portion
    )

    np.testing.assert_array_equal(
        photometry_data["flux"].value, [1080250.0, np.nan, 1079750.0, 1079750.0, np.nan]
    )
    np.testing.assert_array_equal(
        photometry_data["magnitude"],
        [9.999748701259206, np.nan, 10.000251356918534, 10.000251356918534, np.nan],
    )
    np.testing.assert_array_equal(
        photometry_data["centroid_x"].value, [1.9995376363972628, 2.0, 2.0, 2.0, np.nan]
    )
    np.testing.assert_array_equal(photometry_data["centroid_y"].value, [2.0, 2.0, 2.0, 2.0, np.nan])
    assert photometry_data.meta["local_background"] == 1150.0 * u.electron


def test_get_normalized_aperture_photometry_characterization_edge_clamped():
    """Pins the silently edge-clamped aperture window and its interaction with normalization."""
    star_flux = 15_000.0 * 200.0
    images = np.full((3, 5, 5), star_flux / 25 + 50.0)
    images[1, 0, 1] += 200.0
    flux_portion = np.full((5, 5), 1 / 25)

    photometry_data = get_normalized_aperture_photometry(
        images, np.zeros(3, dtype=int), 3, 0, 0, 10.0, 200 * u.second, flux_portion=flux_portion
    )

    np.testing.assert_array_equal(photometry_data["flux"].value, [480000.0, 480200.0, 480000.0])
    np.testing.assert_array_equal(photometry_data["magnitude"], [10.0, 9.999547704136447, 10.0])
    np.testing.assert_array_equal(
        photometry_data["centroid_x"].value, [0.5, 0.5002081598667777, 0.5]
    )
    np.testing.assert_array_equal(
        photometry_data["centroid_y"].value, [0.5, 0.4997918401332223, 0.5]
    )
    assert photometry_data.meta["local_background"] == 200.0 * u.electron


# ---------------------------------------------------------------------
# Step-function unit tests
# ---------------------------------------------------------------------


def test_measure_aperture_flux_is_raw_nansum():
    images = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    images[0, 1, 1] = np.nan
    limits = (1, 3, 1, 3)

    flux = measure_aperture_flux(images, limits)

    np.testing.assert_array_equal(
        flux, [np.nansum(images[0, 1:3, 1:3]), np.nansum(images[1, 1:3, 1:3])]
    )


def test_measure_aperture_centroids_row_major_with_offsets():
    images = np.zeros((1, 5, 5))
    images[0, 3, 2] = 1.0  # all flux in one pixel: centroid = that pixel
    limits = (2, 5, 1, 4)

    centroids = measure_aperture_centroids(images, limits)

    # Row-major (y, x), offset back into full-image coordinates
    np.testing.assert_array_equal(centroids, [[3.0, 2.0]])


def _oracle_aperture_centroids(images, aperture_limits):
    """Per-cadence `scipy.ndimage.center_of_mass`, the definition the vectorized code must match."""
    bottom, top, left, right = aperture_limits
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centroids = np.array([center_of_mass(image[bottom:top, left:right]) for image in images])
    centroids[:, 0] += bottom
    centroids[:, 1] += left
    return centroids


@st.composite
def _image_stacks_with_aperture_limits(draw):
    n_cadences = draw(st.integers(1, 6))
    height = draw(st.integers(1, 7))
    width = draw(st.integers(1, 7))
    images = draw(
        npst.arrays(
            np.float64,
            (n_cadences, height, width),
            # Strictly positive flux keeps the centroid well conditioned (no cancellation in the
            # normalizing sum), so agreement can be asserted to tight relative tolerance.
            elements=st.floats(min_value=1e-3, max_value=1e7),
        )
    )
    bottom = draw(st.integers(0, height - 1))
    top = draw(st.integers(bottom + 1, height))
    left = draw(st.integers(0, width - 1))
    right = draw(st.integers(left + 1, width))
    return images, (bottom, top, left, right)


@given(_image_stacks_with_aperture_limits())
def test_measure_aperture_centroids_matches_center_of_mass(images_and_limits):
    images, limits = images_and_limits

    centroids = measure_aperture_centroids(images, limits)

    assert centroids.shape == (images.shape[0], 2)
    np.testing.assert_allclose(
        centroids, _oracle_aperture_centroids(images, limits), rtol=1e-12, atol=0
    )


@given(
    npst.arrays(
        np.float64,
        st.tuples(st.integers(1, 6), st.integers(1, 5), st.integers(1, 5)),
        elements=st.floats(min_value=-1e7, max_value=1e7),
    )
)
def test_measure_aperture_centroids_matches_center_of_mass_for_signed_flux(images):
    """Decontaminated cutouts contain negative pixels, where the normalizing sum can nearly cancel.

    The agreement tolerance is therefore scaled by the conditioning of each cadence's flux sum,
    `sum(|flux|) / |sum(flux)|`; cadences whose sum cancels to zero are not finite in either
    implementation.
    """
    limits = (0, images.shape[1], 0, images.shape[2])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centroids = measure_aperture_centroids(images, limits)
    oracle = _oracle_aperture_centroids(images, limits)

    total = images.sum(axis=(1, 2))
    finite = total != 0
    np.testing.assert_array_equal(np.isfinite(centroids), np.isfinite(oracle))
    condition = np.abs(images).sum(axis=(1, 2))[finite] / np.abs(total[finite])
    tolerance = 1e-12 * condition[:, np.newaxis] * max(images.shape[1:])
    assert (np.abs(centroids[finite] - oracle[finite]) <= tolerance).all()


def test_measure_aperture_centroids_nan_pixel_only_affects_its_cadence_and_window():
    rng = np.random.default_rng(seed=7)
    images = rng.uniform(1.0, 10.0, size=(3, 5, 5))
    limits = (1, 4, 1, 4)
    clean_centroids = measure_aperture_centroids(images, limits)

    images[0, 0, 0] = np.nan  # outside the window: no effect
    images[1, 2, 2] = np.nan  # inside the window: that cadence's centroid is NaN
    centroids = measure_aperture_centroids(images, limits)

    assert np.isnan(centroids[1]).all()
    np.testing.assert_array_equal(centroids[[0, 2]], clean_centroids[[0, 2]])


def test_measure_aperture_centroids_zero_flux_window_is_not_finite():
    images = np.ones((2, 4, 4))
    images[1] = 0.0
    limits = (0, 3, 1, 4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centroids = measure_aperture_centroids(images, limits)

    np.testing.assert_array_equal(centroids[0], [1.0, 2.0])
    assert not np.isfinite(centroids[1]).any()


def test_measure_aperture_centroids_float32_images():
    rng = np.random.default_rng(seed=11)
    images = rng.uniform(1.0, 1e4, size=(4, 6, 6)).astype(np.float32)
    limits = (1, 6, 0, 5)

    centroids = measure_aperture_centroids(images, limits)

    assert centroids.dtype == np.float64
    np.testing.assert_allclose(centroids, _oracle_aperture_centroids(images, limits), rtol=1e-6)


def test_get_saturation_mask_uses_nominal_aperture_area():
    # Threshold for aperture_size=3, 200 s exposure: 200_000 * 9 * 200 / 2 = 1.8e8
    flux = np.array([1.7e8, 1.9e8])

    np.testing.assert_array_equal(get_saturation_mask(flux, 3, 200.0), [False, True])
    # The nominal area is used even for a clamped edge window covering fewer pixels:
    # the same flux against the same aperture_size gives the same threshold regardless
    # of how many pixels the window actually covers.
    np.testing.assert_array_equal(get_saturation_mask(flux, 1, 200.0), [True, True])


def test_get_local_background_excludes_flagged_cadences_and_nan():
    flux = np.array([100.0, 500.0, 102.0, np.nan, 104.0])
    quality_flags = np.array([0, 1, 0, 0, 0])

    # Median over good, non-NaN cadences {100, 102, 104} = 102
    assert get_local_background(flux, quality_flags, 100.0) == 2.0


def test_get_local_background_nan_when_no_good_cadences():
    flux = np.array([np.nan, 500.0])
    quality_flags = np.array([0, 1])

    with pytest.warns(RuntimeWarning, match="All-NaN"):
        assert np.isnan(get_local_background(flux, quality_flags, 100.0))


def test_normalize_aperture_flux_copies_subtracts_and_clips():
    flux = np.array([100.0, 5.0, 200.0])

    normalized = normalize_aperture_flux(flux, 50.0)

    np.testing.assert_array_equal(normalized, [50.0, np.nan, 150.0])  # 5 - 50 <= 0 -> NaN
    np.testing.assert_array_equal(flux, [100.0, 5.0, 200.0])  # input not mutated


def test_normalize_aperture_flux_nan_background_is_noop_except_clip():
    flux = np.array([100.0, -1.0])

    normalized = normalize_aperture_flux(flux, np.nan)

    np.testing.assert_array_equal(normalized, [100.0, np.nan])


def test_get_flux_portion_in_aperture():
    flux_portion = np.full((4, 4), 1 / 16)

    assert get_flux_portion_in_aperture(flux_portion, (0, 2, 0, 2)) == 4 / 16
