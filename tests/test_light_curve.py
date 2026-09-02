"""Tests for `tglc.light_curve`, including characterization tests pinning pre-refactor behavior.

Golden values were generated against 92df5c1 (pre-refactor) with
`np.array2string(..., floatmode="unique")`. Regenerating them defeats the purpose -- they pin
the behavior the explicit-steps refactor must preserve bit-for-bit.
"""

from math import ceil, floor
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from tglc.epsf import EPSF
from tglc.light_curve import (
    CutoutWindow,
    evaluate_epsf_model,
    generate_light_curves,
    get_background_model,
    get_background_outlier_mask,
    get_cutout_for_light_curve,
    get_cutout_window,
    get_design_matrix_rows_for_window,
    get_high_background_cadence_mask,
    get_psf_portion,
    make_field_design_matrix,
    make_target_design_matrix,
)

from .synthetic_data import make_synthetic_cutout, make_synthetic_epsf


def _make_cutout_and_epsf():
    cutout = make_synthetic_cutout()  # 12x12, 5 cadences, 4 stars
    epsf = EPSF(
        make_synthetic_epsf(n_cadences=5),
        psf_size=11,
        oversample=2,
        orbit=185,
        sector=89,
        camera=1,
        ccd=1,
        cutout_x=0,
        cutout_y=0,
    )
    return cutout, epsf


def _make_full_design_matrix(cutout, epsf):
    star_positions = np.array(
        [cutout.gaia[f"sector_{cutout.sector}_x"], cutout.gaia[f"sector_{cutout.sector}_y"]]
    ).T
    design_matrix, _ = epsf.make_design_matrix(
        cutout.flux.shape[1:],
        star_positions,
        cutout.gaia["tess_flux_ratio"].data,
        cutout.mask.data,
    )
    return design_matrix, star_positions


def _legacy_get_cutout_for_light_curve(
    flux, epsf, full_design_matrix, target_x, target_y, target_flux_ratio, cutout_size=5
):
    """Verbatim copy of the pre-refactor implementation (92df5c1).

    Includes the historical design-matrix row stride based on the image height
    (`flux.shape[1]`), which is only equivalent to the row-major stride for square images.
    """
    points_in_oversampled_psf = epsf.n_psf_parameters
    cutout_left = max(0, round(target_x) - floor(cutout_size / 2))
    cutout_right = min(flux.shape[2], round(target_x) + ceil(cutout_size / 2))
    cutout_bottom = max(0, round(target_y) - floor(cutout_size / 2))
    cutout_top = min(flux.shape[1], round(target_y) + ceil(cutout_size / 2))
    cutout_shape = (cutout_top - cutout_bottom, cutout_right - cutout_left)
    target_x_in_cutout = target_x - cutout_left
    target_y_in_cutout = target_y - cutout_bottom
    cutout_flux = flux[:, cutout_bottom:cutout_top, cutout_left:cutout_right]

    cutout_x, cutout_y = np.meshgrid(
        np.arange(cutout_left, cutout_right), np.arange(cutout_bottom, cutout_top)
    )
    cutout_coordinate_rows_in_design_matrix = (cutout_x + cutout_y * flux.shape[1]).flatten()
    full_design_matrix_for_cutout = full_design_matrix[cutout_coordinate_rows_in_design_matrix]

    target_design_matrix_for_cutout, _ = epsf.make_design_matrix(
        cutout_shape,
        np.array([[target_x_in_cutout, target_y_in_cutout]]),
        np.array([target_flux_ratio]),
    )

    field_design_matrix_for_cutout = full_design_matrix_for_cutout.copy()
    field_design_matrix_for_cutout[:, :points_in_oversampled_psf] -= target_design_matrix_for_cutout
    cutout_field_model = np.dot(field_design_matrix_for_cutout, epsf.array.T).T.reshape(
        flux.shape[0], *cutout_shape
    )
    decontaminated_cutout_flux = cutout_flux - cutout_field_model

    cutout_target_psf = np.dot(target_design_matrix_for_cutout, epsf.psf_parameters.T).T.reshape(
        flux.shape[0], *cutout_shape
    )
    psf_portion_in_cutout = np.nansum(cutout_target_psf, axis=0) / np.nansum(cutout_target_psf)

    return decontaminated_cutout_flux, target_x_in_cutout, target_y_in_cutout, psf_portion_in_cutout


@pytest.mark.parametrize("star_index", [0, 3], ids=["interior_star", "edge_clamped_star"])
def test_get_cutout_for_light_curve_characterization(star_index):
    cutout, epsf = _make_cutout_and_epsf()
    design_matrix, star_positions = _make_full_design_matrix(cutout, epsf)
    args = (
        cutout.flux,
        epsf,
        design_matrix,
        star_positions[star_index][0],
        star_positions[star_index][1],
        cutout.gaia["tess_flux_ratio"].data[star_index],
    )

    result = get_cutout_for_light_curve(*args)
    legacy_result = _legacy_get_cutout_for_light_curve(*args)

    for new_value, legacy_value in zip(result, legacy_result, strict=True):
        np.testing.assert_array_equal(new_value, legacy_value)
    # psf_portion contract required by get_normalized_aperture_photometry
    psf_portion = result[3]
    assert psf_portion.shape == result[0].shape[1:]
    np.testing.assert_allclose(np.nansum(psf_portion), 1.0)


def _fake_spacecraft_position(orbit, time, ephemerides_directory):
    # Zero position -> zero barycentric light-time correction, so time_btjd == source.time
    return np.zeros((len(np.atleast_1d(time.tdb.jd)), 3)) * u.au


def test_generate_light_curves_characterization(monkeypatch):
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    cutout, epsf = _make_cutout_and_epsf()

    light_curves = list(generate_light_curves(cutout, epsf, Path("/nonexistent")))

    # Star 4 at x=10.5 rounds to 10, outside the size - 2.5 = 9.5 bound
    assert [light_curve.meta["tic_id"] for light_curve in light_curves] == [500001, 500002, 500003]

    light_curve = light_curves[0]
    np.testing.assert_array_equal(light_curve["time"].value, cutout.time)
    np.testing.assert_array_equal(light_curve["cadence"], cutout.cadence)
    np.testing.assert_array_equal(light_curve["quality_flag"], [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(
        light_curve["background_flux"],
        [
            -1.6148926257507605,
            -3.2431652242556024,
            -4.871437822760444,
            -6.499710421265286,
            -8.127983019770129,
        ],
    )
    np.testing.assert_array_equal(
        light_curve["primary_aperture_flux"].value,
        [
            1130853.4208847038,
            1130856.8277028569,
            1130898.8318949724,
            1130904.391262869,
            1130901.4221197183,
        ],
    )
    np.testing.assert_array_equal(
        light_curve["primary_aperture_magnitude"],
        [9.950043598389602, 9.950040327496865, 9.95, 9.949994662659474, 9.949997513219143],
    )
    np.testing.assert_array_equal(
        light_curve["primary_aperture_centroid_x"].value,
        [
            45.51360489522273,
            45.4803138107138,
            45.48642484931261,
            45.490718298902465,
            45.49650680393563,
        ],
    )
    np.testing.assert_array_equal(
        light_curve["primary_aperture_centroid_y"].value,
        [
            4.505145996637938,
            4.478288482170659,
            4.511170455683255,
            4.500872892719402,
            4.495941784997689,
        ],
    )
    np.testing.assert_array_equal(
        light_curve["small_aperture_magnitude"],
        [9.950032374765069, 9.950079058481862, 9.949967678645303, 9.95, 9.95000810091382],
    )
    np.testing.assert_array_equal(
        light_curve["large_aperture_magnitude"],
        [9.950031176713066, 9.950022034783288, 9.95, 9.949989191407335, 9.949978838176177],
    )
    assert light_curve.meta["primary_aperture_local_background"] == -1129937.174836132 * u.electron
    assert light_curve.meta["small_aperture_local_background"] == -125548.63178431898 * u.electron
    assert light_curve.meta["large_aperture_local_background"] == -3138756.963741101 * u.electron


def test_generate_light_curves_tic_ids_filter(monkeypatch):
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    cutout, epsf = _make_cutout_and_epsf()

    light_curves = list(generate_light_curves(cutout, epsf, Path("/nonexistent"), tic_ids=[500002]))

    assert len(light_curves) == 1
    assert light_curves[0].meta["tic_id"] == 500002


def test_generate_light_curves_rejects_mismatched_epsf():
    cutout, _ = _make_cutout_and_epsf()
    mismatched_epsf = EPSF(
        make_synthetic_epsf(n_cadences=5),
        psf_size=11,
        oversample=2,
        orbit=186,
        sector=89,
        camera=1,
        ccd=1,
        cutout_x=0,
        cutout_y=0,
    )

    with pytest.raises(ValueError, match="does not match"):
        next(generate_light_curves(cutout, mismatched_epsf, Path("/nonexistent")))


# ---------------------------------------------------------------------
# Cutout step-function unit tests
# ---------------------------------------------------------------------


def test_get_cutout_window_interior():
    window = get_cutout_window(5.5, 6.5, (12, 12))

    # Banker's rounding: round(5.5) == 6, round(6.5) == 6
    assert window == CutoutWindow(left=4, right=9, bottom=4, top=9)
    assert window.shape == (5, 5)


def test_get_cutout_window_edge_clamped():
    window = get_cutout_window(10.5, 9.5, (12, 12))

    # round(10.5) == 10: right edge clamps to the image width
    assert window == CutoutWindow(left=8, right=12, bottom=8, top=12)
    assert window.shape == (4, 4)

    low_window = get_cutout_window(0.5, 0.5, (12, 12))
    # round(0.5) == 0: left/bottom clamp to 0
    assert low_window == CutoutWindow(left=0, right=3, bottom=0, top=3)


def test_get_design_matrix_rows_for_window():
    window = CutoutWindow(left=1, right=3, bottom=2, top=4)

    rows = get_design_matrix_rows_for_window(window, 12)

    np.testing.assert_array_equal(rows, [25, 26, 37, 38])


def test_get_design_matrix_rows_for_window_matches_legacy_height_stride_formula():
    """On square images, the correct width stride equals the historical height-based formula."""
    image_shape = (12, 12)
    window = CutoutWindow(left=4, right=9, bottom=4, top=9)
    cutout_x, cutout_y = np.meshgrid(
        np.arange(window.left, window.right), np.arange(window.bottom, window.top)
    )
    legacy_rows = (cutout_x + cutout_y * image_shape[0]).flatten()  # height used as stride

    np.testing.assert_array_equal(
        get_design_matrix_rows_for_window(window, image_shape[1]), legacy_rows
    )


def test_make_target_design_matrix_matches_direct_call():
    _, epsf = _make_cutout_and_epsf()

    target_design_matrix = make_target_design_matrix(epsf, (5, 5), 2.5, 2.5, 1.0)

    direct, _ = epsf.make_design_matrix((5, 5), np.array([[2.5, 2.5]]), np.array([1.0]))
    np.testing.assert_array_equal(target_design_matrix, direct)
    assert target_design_matrix.shape == (25, epsf.n_psf_parameters)


def test_make_field_design_matrix_subtracts_psf_columns_only():
    full = np.arange(12.0).reshape(3, 4)  # 2 PSF columns + 2 background columns
    target = np.ones((3, 2))

    field = make_field_design_matrix(full, target)

    np.testing.assert_array_equal(field[:, :2], full[:, :2] - 1.0)
    np.testing.assert_array_equal(field[:, 2:], full[:, 2:])  # background columns untouched
    np.testing.assert_array_equal(full, np.arange(12.0).reshape(3, 4))  # input not mutated


def test_evaluate_epsf_model():
    design_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    parameters = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 cadences

    model = evaluate_epsf_model(design_matrix, parameters, (2, 2))

    np.testing.assert_array_equal(
        model,
        np.dot(design_matrix, parameters.T).T.reshape(2, 2, 2),
    )
    assert model.shape == (2, 2, 2)


def test_get_psf_portion_collapses_time_and_normalizes():
    model = np.zeros((2, 2, 2))
    model[0] = [[1.0, 1.0], [1.0, 1.0]]
    model[1] = [[3.0, 1.0], [np.nan, 1.0]]

    portion = get_psf_portion(model)

    assert portion.shape == (2, 2)
    np.testing.assert_allclose(np.nansum(portion), 1.0)
    np.testing.assert_array_equal(portion, [[4 / 9, 2 / 9], [1 / 9, 2 / 9]])


# ---------------------------------------------------------------------
# Background step-function unit tests
# ---------------------------------------------------------------------


def test_get_high_background_cadence_mask_uses_y_strap_column():
    """Quirk pin for issue #19: the mask is driven by the y_strap column, not flat."""
    _, epsf = _make_cutout_and_epsf()
    epsf.array[:, :] = 1.0
    # y_strap (column -6) outlier on cadence 1; flat (column -1) outlier on cadence 3
    epsf.array[:, -6] = [1.05, 100.0, 1.0, 1.0, 1.1]
    epsf.array[:, -1] = [1.0, 1.0, 1.0, 100.0, 1.0]

    mask = get_high_background_cadence_mask(epsf)

    np.testing.assert_array_equal(mask, [False, True, False, False, False])


def test_get_background_model_matches_legacy_expression():
    cutout, epsf = _make_cutout_and_epsf()
    design_matrix, _ = _make_full_design_matrix(cutout, epsf)

    model = get_background_model(epsf, design_matrix, cutout.flux.shape[1:])

    legacy = np.dot(design_matrix[:, -epsf.n_background :], epsf.background_parameters.T).T.reshape(
        cutout.flux.shape
    )
    np.testing.assert_array_equal(model, legacy)


def test_get_background_outlier_mask():
    background = np.array([10.0, 10.1, 9.9, 500.0, 10.0])

    np.testing.assert_array_equal(
        get_background_outlier_mask(background), [False, False, False, True, False]
    )


def test_get_background_outlier_mask_nan_disables_all_flags():
    """Quirk pin for issue #20: one NaN cadence makes mad_std NaN and every flag False."""
    background = np.array([10.0, 10.1, 9.9, 500.0, np.nan])

    np.testing.assert_array_equal(get_background_outlier_mask(background), [False] * 5)
