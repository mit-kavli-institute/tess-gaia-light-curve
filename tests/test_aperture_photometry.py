"""
Tests for the tglc.aperture_photometry module, which provides a function for doing aperture photometry
on image cutouts.
"""

from astropy import units as u
import numpy as np

from tglc.aperture_photometry import get_normalized_aperture_photometry


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
