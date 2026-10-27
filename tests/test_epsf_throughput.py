"""Tests for `tglc.epsf_throughput` and the timeless scalars recorded in HDF5 light curves."""

from pathlib import Path

from astropy.table import Table
import astropy.units as u
import h5py
from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from tglc.aperture_photometry import (
    get_aperture_limits,
    get_expected_total_flux,
    get_flux_portion_in_aperture,
    measure_aperture_flux,
)
from tglc.epsf import EPSF, get_reference_flux
from tglc.epsf_throughput import EPSFThroughput, read_hdf5_raw_flux
from tglc.io import write_epsf_fits
from tglc.light_curve import (
    LIGHT_CURVE_APERTURES,
    evaluate_epsf_model,
    generate_light_curves,
    get_cutout_for_light_curve,
    make_target_design_matrix,
)
from tglc.utils.constants import convert_tess_flux_to_tess_magnitude

from .synthetic_data import make_synthetic_cutout, make_synthetic_epsf


PSF_SIZE = 11
OVERSAMPLE = 2
THROUGHPUT_KWARGS = {
    "reference_flux": 15_000 * u.electron / u.second,
    "exposure_time": 200 * u.second,
    "ccd_x": 0,
    "ccd_y": 0,
    "cutout_size": 150,
}


def _make_epsf(array: np.ndarray, **overrides) -> EPSF:
    identifiers = {"orbit": 185, "sector": 89, "camera": 1, "ccd": 1, "cutout_x": 0, "cutout_y": 0}
    identifiers.update(overrides)
    return EPSF(array, psf_size=PSF_SIZE, oversample=OVERSAMPLE, **identifiers)


def _make_cutout_and_epsf(n_cadences: int = 5):
    cutout = make_synthetic_cutout(n_cadences=n_cadences)
    # The synthetic flux ratios are rounded; make them exactly consistent with the magnitudes, as
    # FFICutout.derive_catalogs does, so the target magnitude cancels out of the flux fraction.
    tess_mag = np.asarray(cutout.gaia["tess_mag"])
    cutout.gaia["tess_flux_ratio"] = 10 ** (-(tess_mag - tess_mag.min()) / 2.5)
    return cutout, _make_epsf(make_synthetic_epsf(n_cadences=n_cadences))


def _fake_spacecraft_position(orbit, time, ephemerides_directory):
    return np.zeros((len(np.atleast_1d(time.tdb.jd)), 3)) * u.au


#######################################################################
#### Flux fraction ####################################################
#######################################################################


def test_from_cutout_uses_brightest_star_and_cutout_scalars():
    cutout, epsf = _make_cutout_and_epsf()
    throughput = EPSFThroughput.from_cutout(epsf, cutout)
    assert throughput.exposure_time == 200 * u.second
    assert (throughput.ccd_x, throughput.ccd_y) == (44, 0)
    assert throughput.reference_flux_per_cadence == pytest.approx(
        get_expected_total_flux(9.95, 200)
    )


def test_get_reference_flux_is_flux_ratio_one_catalog_flux():
    cutout, _ = _make_cutout_and_epsf()
    # Any star's catalog flux divided by its flux ratio gives the same reference flux.
    for tess_mag, flux_ratio in zip(
        cutout.gaia["tess_mag"], cutout.gaia["tess_flux_ratio"], strict=True
    ):
        assert get_reference_flux(cutout.gaia) == pytest.approx(
            get_expected_total_flux(tess_mag, 1.0) / flux_ratio, rel=1e-12
        )


def test_from_epsf_matches_from_cutout_for_fitted_epsf():
    cutout, _ = _make_cutout_and_epsf()
    cutout.flux = cutout.flux.astype(np.float64)
    epsf = EPSF.from_cutout_fit(
        cutout,
        psf_size=PSF_SIZE,
        oversample=OVERSAMPLE,
        edge_compression_factor=1e-6,
        flux_uncertainty_power=1.4,
        use_gpu=False,
    )
    assert epsf.reference_flux == get_reference_flux(cutout.gaia)
    assert (epsf.exposure, epsf.ccd_x, epsf.ccd_y, epsf.cutout_size) == (200, 44, 0, 12)
    x, y = cutout.star_positions[0]
    np.testing.assert_array_equal(
        EPSFThroughput.from_epsf(epsf).flux_fraction(x, y),
        EPSFThroughput.from_cutout(epsf, cutout).flux_fraction(x, y),
    )


def test_from_epsf_rejects_epsf_without_recorded_scalars():
    _, epsf = _make_cutout_and_epsf()
    with pytest.raises(
        ValueError, match="does not record reference_flux, exposure, ccd_x, ccd_y, cutout_size"
    ):
        EPSFThroughput.from_epsf(epsf)


def test_from_fits_needs_only_the_epsf_file(tmp_path):
    cutout, epsf = _make_cutout_and_epsf()
    recorded = _make_epsf(
        epsf.array,
        reference_flux=get_reference_flux(cutout.gaia),
        exposure=200,
        ccd_x=44,
        ccd_y=0,
        cutout_size=12,
    )
    write_epsf_fits(recorded, tmp_path / "epsf.fits")
    x, y = cutout.star_positions[2]
    np.testing.assert_allclose(
        EPSFThroughput.from_fits(tmp_path / "epsf.fits").flux_fraction(x, y),
        EPSFThroughput.from_cutout(epsf, cutout).flux_fraction(x, y),
        rtol=1e-15,
    )


def test_from_cutout_rejects_mismatched_epsf():
    cutout, _ = _make_cutout_and_epsf()
    with pytest.raises(ValueError, match="does not match"):
        EPSFThroughput.from_cutout(_make_epsf(make_synthetic_epsf(5), camera=2), cutout)


@pytest.mark.parametrize("star_index", range(4))
def test_flux_fraction_matches_full_footprint_target_model(star_index):
    # Each oracle uses the star's own magnitude and flux ratio, while `flux_fraction` uses only the
    # cutout's reference star, so this also checks that the target magnitude cancels.
    cutout, epsf = _make_cutout_and_epsf()
    throughput = EPSFThroughput.from_cutout(epsf, cutout)
    x, y = cutout.star_positions[star_index]

    # Oracle: model the target alone in an image large enough to hold its whole footprint. The
    # even shift keeps the rounding of the half-pixel synthetic positions unchanged.
    shift = 14
    image_shape = (40, 40)
    target_model = evaluate_epsf_model(
        make_target_design_matrix(
            epsf, image_shape, x + shift, y + shift, cutout.gaia["tess_flux_ratio"][star_index]
        ),
        epsf.psf_parameters,
        image_shape,
    )
    expected = target_model.sum(axis=(1, 2)) / get_expected_total_flux(
        cutout.gaia["tess_mag"][star_index], cutout.exposure
    )

    np.testing.assert_allclose(throughput.flux_fraction(x, y), expected, rtol=1e-12)


@pytest.mark.parametrize("psf_size", [9, 11])
@pytest.mark.parametrize("position", [(2.5, 3.5), (7.5, 0.5), (-1.5, 12.5), (4.25, 6.75)])
def test_flux_fraction_keeps_half_pixel_footprint(psf_size, position):
    # Exact half-pixel positions are rounded to even, so the footprint the pipeline models depends
    # on the parity of the nearest pixel; the flux fraction must use the same footprint.
    rng = np.random.default_rng(seed=psf_size)
    array = rng.uniform(0, 1, (2, EPSF.parameter_count(psf_size, OVERSAMPLE)))
    epsf = EPSF(
        array, psf_size=psf_size, oversample=OVERSAMPLE, orbit=1, sector=1, camera=1, ccd=1,
        cutout_x=0, cutout_y=0,
    )  # fmt: skip
    throughput = EPSFThroughput(epsf, **THROUGHPUT_KWARGS)
    x, y = position
    shift = 20
    image_shape = (50, 50)
    target_model = evaluate_epsf_model(
        make_target_design_matrix(epsf, image_shape, x + shift, y + shift, 1.0),
        epsf.psf_parameters,
        image_shape,
    )
    np.testing.assert_allclose(
        throughput.flux_fraction(x, y),
        target_model.sum(axis=(1, 2)) / throughput.reference_flux_per_cadence,
        rtol=1e-12,
    )


@settings(deadline=None)
@given(
    x=st.floats(-10, 200, allow_nan=False),
    y=st.floats(-10, 200, allow_nan=False),
    level=st.floats(1e-3, 1e6),
)
def test_flux_fraction_of_flat_psf_counts_whole_footprint(x, y, level):
    # Each pixel's bilinear weights sum to 1, so an unclamped footprint of a flat PSF totals
    # level * psf_size**2 wherever the star is.
    array = np.zeros((2, EPSF.parameter_count(PSF_SIZE, OVERSAMPLE)))
    array[:, : (PSF_SIZE * OVERSAMPLE + 1) ** 2] = level
    throughput = EPSFThroughput(
        _make_epsf(array),
        **THROUGHPUT_KWARGS,
    )
    np.testing.assert_allclose(
        throughput.flux_fraction(x, y),
        level * PSF_SIZE**2 / throughput.reference_flux_per_cadence,
        rtol=1e-12,
    )


@settings(deadline=None)
@given(
    x=st.floats(0, 150, allow_nan=False),
    y=st.floats(0, 150, allow_nan=False),
    scale=st.floats(1e-3, 1e3),
)
def test_flux_fraction_is_linear_in_psf_parameters(x, y, scale):
    _, epsf = _make_cutout_and_epsf()
    kwargs = THROUGHPUT_KWARGS
    scaled_epsf = _make_epsf(epsf.array * scale)
    np.testing.assert_allclose(
        EPSFThroughput(scaled_epsf, **kwargs).flux_fraction(x, y),
        scale * EPSFThroughput(epsf, **kwargs).flux_fraction(x, y),
        rtol=1e-12,
    )


def test_flux_fraction_is_nan_for_failed_cadences():
    cutout, epsf = _make_cutout_and_epsf()
    epsf.array[2] = np.nan
    fraction = EPSFThroughput.from_cutout(epsf, cutout).flux_fraction(*cutout.star_positions[0])
    assert np.isnan(fraction[2])
    assert np.isfinite(np.delete(fraction, 2)).all()


def test_flux_fraction_at_ccd_subtracts_cutout_origin():
    cutout, epsf = _make_cutout_and_epsf()
    throughput = EPSFThroughput.from_cutout(epsf, cutout)
    x, y = cutout.star_positions[1]
    np.testing.assert_array_equal(
        throughput.flux_fraction_at_ccd(x + cutout.ccd_x, y + cutout.ccd_y),
        throughput.flux_fraction(x, y),
    )


def _make_injected_throughput_cutout(throughput: np.ndarray):
    """Noise-free cutout whose stars deliver `throughput[t]` of their catalog flux, plus its fit.

    The flux is rendered with the pipeline's own design matrix from a unit-total Gaussian PSF on the
    oversampled grid, scaled to the brightest star's flux, over a flat background.
    """
    rng = np.random.default_rng(seed=23)
    size, n_stars = 40, 40
    cutout = make_synthetic_cutout(n_cadences=len(throughput), size=size)
    tess_mag = rng.uniform(9, 13, n_stars)
    cutout.gaia = Table(
        {
            "designation": [f"Gaia DR3 {i}" for i in range(n_stars)],
            "ra": np.full(n_stars, 120.5),
            "dec": np.full(n_stars, -45.25),
            "tess_mag": tess_mag,
            "tess_flux_ratio": 10 ** (-(tess_mag - tess_mag.min()) / 2.5),
            "sector_89_x": rng.uniform(-3, size + 2, n_stars),
            "sector_89_y": rng.uniform(-3, size + 2, n_stars),
        }
    )
    cutout.tic = Table({"TIC": np.arange(n_stars), "gaia3": np.arange(n_stars)})
    grid = (np.arange(PSF_SIZE * OVERSAMPLE + 1) - PSF_SIZE) / OVERSAMPLE
    gaussian = np.exp(-0.5 * (grid[:, None] ** 2 + grid[None, :] ** 2) / 1.2**2)
    gaussian /= gaussian.sum() / OVERSAMPLE**2
    reference_flux = get_expected_total_flux(tess_mag.min(), cutout.exposure)
    true_parameters = np.array(
        [
            np.concatenate([loss * reference_flux * gaussian.flatten(), [0, 0, 0, 0, 0, 100.0]])
            for loss in throughput
        ]
    )
    cutout.mask = np.ma.masked_array(cutout.mask.data[:size, :size], mask=False)
    true_epsf = _make_epsf(true_parameters)
    design_matrix, _ = true_epsf.make_design_matrix(
        (size, size), cutout.star_positions, cutout.gaia["tess_flux_ratio"].data, cutout.mask.data
    )
    cutout.flux = (true_parameters @ design_matrix.T).reshape(len(throughput), size, size)
    fit_epsf = EPSF.from_cutout_fit(
        cutout,
        psf_size=PSF_SIZE,
        oversample=OVERSAMPLE,
        edge_compression_factor=1e-6,
        flux_uncertainty_power=1.4,
        use_gpu=False,
    )
    return cutout, true_epsf, fit_epsf


def test_flux_fraction_recovers_injected_throughput():
    """An ePSF fit to stars delivering a fraction `s` of their catalog flux measures `s`."""
    cutout, true_epsf, fit_epsf = _make_injected_throughput_cutout(np.array([0.8]))
    x, y = 20.3, 19.8
    fitted = EPSFThroughput.from_cutout(fit_epsf, cutout).flux_fraction(x, y)
    truth = EPSFThroughput.from_cutout(true_epsf, cutout).flux_fraction(x, y)
    np.testing.assert_allclose(truth, 0.8, rtol=1e-2)
    # The edge-compression regularization biases the fit slightly.
    np.testing.assert_allclose(fitted, truth, rtol=5e-3)


#######################################################################
#### Corrections ######################################################
#######################################################################


def test_correction_removes_time_varying_throughput(monkeypatch):
    """Dividing the un-normalized flux by f(t) flattens a throughput drift.

    Dividing the normalized flux instead counts the median deficit twice: it inflates the light
    curve by about 1 / f and over-corrects the drift into a reversed trend.
    """
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    throughput = np.array([0.7, 0.8, 0.9])
    cutout, _, fit_epsf = _make_injected_throughput_cutout(throughput)
    light_curve = next(iter(generate_light_curves(cutout, fit_epsf, Path("/nonexistent"))))

    corrected = EPSFThroughput.from_epsf(fit_epsf).correct_light_curve(light_curve)

    np.testing.assert_allclose(corrected["epsf_flux_fraction"], throughput, rtol=5e-3)
    normalized_flux = light_curve["primary_aperture_flux"].to_value(u.electron)
    assert np.ptp(normalized_flux) / np.mean(normalized_flux) > 0.15  # the drift is visible
    corrected_flux = corrected["primary_aperture_flux"].to_value(u.electron)
    # What remains is the fit's ~0.1% bias (see test_flux_fraction_recovers_injected_throughput).
    assert np.ptp(corrected_flux) / np.mean(corrected_flux) < 5e-3
    # The corrected flux sits at the catalog-expected aperture flux, up to the light curve's 5x5
    # flux-portion approximation (a constant, target-dependent factor of a few percent).
    expected_aperture_flux = (
        get_expected_total_flux(light_curve.meta["tess_magnitude"], cutout.exposure)
        * light_curve.meta["primary_aperture_flux_portion"]
    )
    np.testing.assert_allclose(corrected_flux, expected_aperture_flux, rtol=0.1)


@given(
    raw_flux=st.floats(1.0, 1e8),
    fraction=st.floats(0.05, 2.0),
    flux_portion=st.floats(0.05, 1.0),
)
def test_corrected_magnitude_is_magnitude_of_corrected_flux(raw_flux, fraction, flux_portion):
    throughput = EPSFThroughput(_make_epsf(make_synthetic_epsf(1)), **THROUGHPUT_KWARGS)
    flux, magnitude = throughput._correct_aperture(
        np.array([raw_flux]) * u.electron, np.array([fraction]), flux_portion
    )
    np.testing.assert_allclose(flux.to_value(u.electron), raw_flux / fraction, rtol=1e-12)
    np.testing.assert_allclose(
        magnitude,
        convert_tess_flux_to_tess_magnitude(
            raw_flux / fraction / flux_portion / (200 * u.second) * u.electron
        ),
        rtol=0,
        atol=1e-12,
    )


def test_correct_flux_rejects_non_positive_fractions(caplog):
    flux = EPSFThroughput.correct_flux(np.array([10.0, 10.0, 10.0]), np.array([0.5, 0.0, -0.1]))
    np.testing.assert_array_equal(flux, [20.0, np.nan, np.nan])
    assert "2 cadences have a non-positive ePSF flux fraction" in caplog.text


@pytest.fixture
def light_curves(monkeypatch):
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    cutout, epsf = _make_cutout_and_epsf()
    return cutout, epsf, list(generate_light_curves(cutout, epsf, Path("/nonexistent")))


def _expected_correction(throughput, light_curve, aperture_name, fraction):
    raw_flux = (
        light_curve[f"{aperture_name}_aperture_flux"]
        + light_curve.meta[f"{aperture_name}_aperture_local_background"]
    )
    flux = raw_flux / fraction
    magnitude = convert_tess_flux_to_tess_magnitude(
        flux / light_curve.meta[f"{aperture_name}_aperture_flux_portion"] / throughput.exposure_time
    )
    return flux, magnitude


def test_correct_light_curve_divides_out_flux_fraction(light_curves):
    cutout, epsf, (light_curve, *_) = light_curves
    throughput = EPSFThroughput.from_cutout(epsf, cutout)

    corrected = throughput.correct_light_curve(light_curve)

    # The light curve's CCD position maps back to the target's cutout position.
    fraction = throughput.flux_fraction(*cutout.star_positions[0])
    np.testing.assert_allclose(corrected["epsf_flux_fraction"], fraction, rtol=1e-12)
    for aperture_name, _ in LIGHT_CURVE_APERTURES:
        flux, magnitude = _expected_correction(throughput, light_curve, aperture_name, fraction)
        np.testing.assert_allclose(corrected[f"{aperture_name}_aperture_flux"], flux, rtol=1e-12)
        np.testing.assert_allclose(
            corrected[f"{aperture_name}_aperture_magnitude"], magnitude, rtol=1e-12
        )
    # The input light curve is not modified.
    assert "epsf_flux_fraction" not in light_curve.colnames


def test_correct_light_curve_without_local_background_uses_flux_as_is(light_curves):
    cutout, epsf, (light_curve, *_) = light_curves
    throughput = EPSFThroughput.from_cutout(epsf, cutout)
    # A NaN local background means normalization found no good cadences and subtracted nothing.
    light_curve.meta["primary_aperture_local_background"] = np.nan * u.electron

    corrected = throughput.correct_light_curve(light_curve)

    np.testing.assert_allclose(
        corrected["primary_aperture_flux"],
        light_curve["primary_aperture_flux"] / corrected["epsf_flux_fraction"],
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("meta_update", "match"),
    [
        ({"ccd": 2}, "ePSF is for orbit 185 1-1"),
        ({"cutout_x": 1}, r"is for cutout \(1, 0\), but the ePSF is for cutout \(0, 0\)"),
        ({"ccd_x": 44 + 13.0}, "lies outside the ePSF's 12 px cutout"),
        ({"ccd_y": -0.5}, "lies outside the ePSF's 12 px cutout"),
    ],
)
def test_correct_light_curve_rejects_light_curve_from_another_cutout(
    light_curves, meta_update, match
):
    cutout, epsf, (light_curve, *_) = light_curves
    throughput = EPSFThroughput.from_cutout(epsf, cutout)
    light_curve.meta.update(meta_update)
    with pytest.raises(ValueError, match=match):
        throughput.correct_light_curve(light_curve)


def test_correct_light_curve_rejects_cadence_count_mismatch(light_curves):
    cutout, _, (light_curve, *_) = light_curves
    throughput = EPSFThroughput(
        _make_epsf(make_synthetic_epsf(n_cadences=6)),
        **{**THROUGHPUT_KWARGS, "ccd_x": cutout.ccd_x, "cutout_size": cutout.size},
    )
    with pytest.raises(ValueError, match="has 5 cadences, but the ePSF has 6"):
        throughput.correct_light_curve(light_curve)


def test_legacy_cutout_index_sentinel_skips_index_check(light_curves):
    cutout, epsf, (light_curve, *_) = light_curves
    legacy_epsf = _make_epsf(epsf.array, cutout_x=-1, cutout_y=-1)
    throughput = EPSFThroughput.from_cutout(legacy_epsf, cutout)
    light_curve.meta["cutout_x"] = 3
    throughput.correct_light_curve(light_curve)


#######################################################################
#### HDF5 scalars #####################################################
#######################################################################


def test_light_curve_records_flux_portion_and_cutout(light_curves):
    cutout, epsf, (light_curve, *_) = light_curves
    assert (light_curve.meta["cutout_x"], light_curve.meta["cutout_y"]) == (0, 0)
    design_matrix, _ = epsf.make_design_matrix(
        cutout.flux.shape[1:],
        cutout.star_positions,
        cutout.gaia["tess_flux_ratio"].data,
        cutout.mask.data,
    )
    x, y = cutout.star_positions[0]
    images, star_x, star_y, psf_portion = get_cutout_for_light_curve(
        cutout.flux, epsf, design_matrix, x, y, cutout.gaia["tess_flux_ratio"][0]
    )
    for aperture_name, aperture_size in LIGHT_CURVE_APERTURES:
        limits = get_aperture_limits(
            aperture_size, round(star_x), round(star_y), images.shape[1], images.shape[2]
        )
        assert light_curve.meta[f"{aperture_name}_aperture_flux_portion"] == (
            get_flux_portion_in_aperture(psf_portion, limits)
        )


def test_hdf5_scalars_invert_normalization(light_curves, tmp_path):
    cutout, epsf, (light_curve, *_) = light_curves
    path = tmp_path / "lc.h5"
    light_curve.write_hdf5(path)

    design_matrix, _ = epsf.make_design_matrix(
        cutout.flux.shape[1:],
        cutout.star_positions,
        cutout.gaia["tess_flux_ratio"].data,
        cutout.mask.data,
    )
    x, y = cutout.star_positions[0]
    images, star_x, star_y, _ = get_cutout_for_light_curve(
        cutout.flux, epsf, design_matrix, x, y, cutout.gaia["tess_flux_ratio"][0]
    )
    with h5py.File(path) as file:
        assert file.attrs["ExposureTime"] == 200
        assert file.attrs["ZeroPointFlux"] == 15_000
        assert (file.attrs["CutoutX"], file.attrs["CutoutY"]) == (0, 0)
        for aperture_name, aperture_size in LIGHT_CURVE_APERTURES:
            raw_flux = measure_aperture_flux(
                images,
                get_aperture_limits(
                    aperture_size, round(star_x), round(star_y), images.shape[1], images.shape[2]
                ),
            )
            # Adding back the background cancels most of the normalized flux's digits.
            np.testing.assert_allclose(
                read_hdf5_raw_flux(file, aperture_name).to_value(u.electron), raw_flux, rtol=1e-9
            )


def test_hdf5_adds_no_time_series(light_curves, tmp_path):
    _, _, (light_curve, *_) = light_curves
    path = tmp_path / "lc.h5"
    light_curve.write_hdf5(path)
    datasets = []
    with h5py.File(path) as file:
        file.visititems(
            lambda name, item: datasets.append(name) if isinstance(item, h5py.Dataset) else None
        )
    apertures = [
        f"LightCurve/AperturePhotometry/{name}Aperture/{dataset}"
        for name in ["Primary", "Small", "Large"]
        for dataset in ["RawMagnitude", "RawMagnitudeError", "X", "Y"]
    ]
    assert sorted(datasets) == sorted(
        [
            "LightCurve/BJD",
            "LightCurve/Cadence",
            "LightCurve/X",
            "LightCurve/Y",
            "LightCurve/QualityFlag",
            "LightCurve/Background/Value",
            "LightCurve/Background/Error",
            *apertures,
        ]
    )


def test_correct_hdf5_matches_correct_light_curve(light_curves, tmp_path):
    cutout, epsf, (light_curve, *_) = light_curves
    light_curve["quality_flag"][1] = 2
    path = tmp_path / "lc.h5"
    light_curve.write_hdf5(path)
    throughput = EPSFThroughput.from_cutout(epsf, cutout)

    from_file = throughput.correct_hdf5(path)
    in_memory = throughput.correct_light_curve(light_curve)

    assert from_file["time"].format == "tjd"
    assert from_file["time"].scale == "tdb"
    np.testing.assert_allclose(from_file["time"].tjd, in_memory["time"].tjd, rtol=1e-15)
    for column in ["cadence", "quality_flag", "background_flux"]:
        np.testing.assert_array_equal(from_file[column], in_memory[column])
    np.testing.assert_allclose(
        from_file["epsf_flux_fraction"], in_memory["epsf_flux_fraction"], rtol=1e-12
    )
    for aperture_name, _ in LIGHT_CURVE_APERTURES:
        np.testing.assert_allclose(
            from_file[f"{aperture_name}_aperture_flux"],
            in_memory[f"{aperture_name}_aperture_flux"],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            from_file[f"{aperture_name}_aperture_magnitude"],
            in_memory[f"{aperture_name}_aperture_magnitude"],
            rtol=0,
            atol=1e-9,
        )


@pytest.mark.parametrize(
    ("attribute", "group"),
    [
        ("ExposureTime", "/"),
        ("CutoutX", "/"),
        ("fluxportion", "LightCurve/AperturePhotometry/SmallAperture"),
    ],
)
def test_correct_hdf5_rejects_files_without_scalars(light_curves, tmp_path, attribute, group):
    cutout, epsf, (light_curve, *_) = light_curves
    path = tmp_path / "lc.h5"
    light_curve.write_hdf5(path)
    with h5py.File(path, "a") as file:
        del file[group].attrs[attribute]

    with pytest.raises(ValueError, match=f"does not record {attribute}"):
        EPSFThroughput.from_cutout(epsf, cutout).correct_hdf5(path)


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        ("Camera", 3, "is for orbit 185 3-1"),
        ("CutoutY", 2, r"is for cutout \(0, 2\)"),
    ],
)
def test_correct_hdf5_rejects_file_from_another_cutout(
    light_curves, tmp_path, attribute, value, match
):
    cutout, epsf, (light_curve, *_) = light_curves
    path = tmp_path / "lc.h5"
    light_curve.write_hdf5(path)
    with h5py.File(path, "a") as file:
        file.attrs[attribute] = value

    with pytest.raises(ValueError, match=match):
        EPSFThroughput.from_cutout(epsf, cutout).correct_hdf5(path)


def test_correct_hdf5_rejects_empty_file(light_curves, tmp_path):
    cutout, epsf, (light_curve, *_) = light_curves
    path = tmp_path / "lc.h5"
    light_curve.write_hdf5(path)
    with h5py.File(path, "a") as file:
        for name in ["BJD", "X", "Y"]:
            del file["LightCurve"][name]
            file["LightCurve"].create_dataset(name, data=np.zeros(0))

    with pytest.raises(ValueError, match="has 0 cadences, but the ePSF has 5"):
        EPSFThroughput.from_cutout(epsf, cutout).correct_hdf5(path)
