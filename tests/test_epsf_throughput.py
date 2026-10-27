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
from tglc.epsf import EPSF
from tglc.epsf_throughput import EPSFThroughput, read_hdf5_offset_flux
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
    assert throughput.reference_tmag == 9.95
    assert throughput.exposure_time == 200 * u.second
    assert (throughput.ccd_x, throughput.ccd_y) == (44, 0)
    assert throughput.reference_flux == pytest.approx(get_expected_total_flux(9.95, 200))


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
    throughput = EPSFThroughput(
        epsf, reference_tmag=10.0, exposure_time=200 * u.second, ccd_x=0, ccd_y=0
    )
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
        target_model.sum(axis=(1, 2)) / throughput.reference_flux,
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
        _make_epsf(array), reference_tmag=10.0, exposure_time=200 * u.second, ccd_x=0, ccd_y=0
    )
    np.testing.assert_allclose(
        throughput.flux_fraction(x, y), level * PSF_SIZE**2 / throughput.reference_flux, rtol=1e-12
    )


@settings(deadline=None)
@given(
    x=st.floats(0, 150, allow_nan=False),
    y=st.floats(0, 150, allow_nan=False),
    scale=st.floats(1e-3, 1e3),
)
def test_flux_fraction_is_linear_in_psf_parameters(x, y, scale):
    _, epsf = _make_cutout_and_epsf()
    kwargs = {"reference_tmag": 10.0, "exposure_time": 200 * u.second, "ccd_x": 0, "ccd_y": 0}
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


def test_flux_fraction_recovers_injected_throughput():
    """An ePSF fit to stars delivering a fraction `s` of their catalog flux measures `s`."""
    rng = np.random.default_rng(seed=23)
    size, n_stars, throughput_loss = 40, 40, 0.8
    cutout = make_synthetic_cutout(n_cadences=1, size=size)
    tess_mag = rng.uniform(9, 13, n_stars)
    cutout.gaia = Table(
        {
            "designation": [f"Gaia DR3 {i}" for i in range(n_stars)],
            "tess_mag": tess_mag,
            "tess_flux_ratio": 10 ** (-(tess_mag - tess_mag.min()) / 2.5),
            "sector_89_x": rng.uniform(-3, size + 2, n_stars),
            "sector_89_y": rng.uniform(-3, size + 2, n_stars),
        }
    )
    # A unit-total Gaussian PSF on the oversampled grid, scaled to the brightest star's flux.
    grid = (np.arange(PSF_SIZE * OVERSAMPLE + 1) - PSF_SIZE) / OVERSAMPLE
    gaussian = np.exp(-0.5 * (grid[:, None] ** 2 + grid[None, :] ** 2) / 1.2**2)
    gaussian /= gaussian.sum() / OVERSAMPLE**2
    reference_flux = get_expected_total_flux(tess_mag.min(), cutout.exposure)
    true_parameters = np.concatenate(
        [throughput_loss * reference_flux * gaussian.flatten(), [0, 0, 0, 0, 0, 100.0]]
    )
    true_epsf = _make_epsf(true_parameters[np.newaxis, :])
    design_matrix, _ = true_epsf.make_design_matrix(
        (size, size), cutout.star_positions, cutout.gaia["tess_flux_ratio"].data, cutout.mask.data
    )
    cutout.flux = (design_matrix @ true_parameters).reshape(1, size, size)
    cutout.mask = np.ma.masked_array(cutout.mask.data[:size, :size], mask=False)

    fit_epsf = EPSF.from_cutout_fit(
        cutout,
        psf_size=PSF_SIZE,
        oversample=OVERSAMPLE,
        edge_compression_factor=1e-6,
        flux_uncertainty_power=1.4,
        use_gpu=False,
    )
    x, y = 20.3, 19.8
    fitted = EPSFThroughput.from_cutout(fit_epsf, cutout).flux_fraction(x, y)
    truth = EPSFThroughput.from_cutout(true_epsf, cutout).flux_fraction(x, y)
    np.testing.assert_allclose(truth, throughput_loss, rtol=1e-2)
    # The edge-compression regularization biases the fit slightly.
    np.testing.assert_allclose(fitted, truth, rtol=5e-3)


#######################################################################
#### Corrections ######################################################
#######################################################################


@given(
    flux=st.floats(1.0, 1e8),
    fraction=st.floats(0.05, 2.0),
)
def test_flux_and_magnitude_corrections_agree(flux, fraction):
    magnitude = convert_tess_flux_to_tess_magnitude(flux * u.electron / u.second)
    corrected_flux = EPSFThroughput.correct_flux(flux, fraction)
    np.testing.assert_allclose(
        EPSFThroughput.correct_magnitude(magnitude, fraction),
        convert_tess_flux_to_tess_magnitude(corrected_flux * u.electron / u.second),
        rtol=0,
        atol=1e-12,
    )


@pytest.fixture
def light_curves(monkeypatch):
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    cutout, epsf = _make_cutout_and_epsf()
    return cutout, epsf, list(generate_light_curves(cutout, epsf, Path("/nonexistent")))


def test_correct_light_curve_divides_out_flux_fraction(light_curves):
    cutout, epsf, (light_curve, *_) = light_curves
    throughput = EPSFThroughput.from_cutout(epsf, cutout)

    corrected = throughput.correct_light_curve(light_curve)

    # The light curve's CCD position maps back to the target's cutout position.
    fraction = throughput.flux_fraction(*cutout.star_positions[0])
    np.testing.assert_allclose(corrected["epsf_flux_fraction"], fraction, rtol=1e-12)
    for aperture_name, _ in LIGHT_CURVE_APERTURES:
        np.testing.assert_allclose(
            corrected[f"{aperture_name}_aperture_flux"],
            light_curve[f"{aperture_name}_aperture_flux"] / fraction,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            corrected[f"{aperture_name}_aperture_magnitude"],
            light_curve[f"{aperture_name}_aperture_magnitude"] + 2.5 * np.log10(fraction),
            rtol=1e-12,
        )
    # The input light curve is not modified.
    assert "epsf_flux_fraction" not in light_curve.colnames


def test_correct_light_curve_rejects_other_ccd(light_curves):
    cutout, _, (light_curve, *_) = light_curves
    throughput = EPSFThroughput.from_cutout(_make_epsf(make_synthetic_epsf(5)), cutout)
    light_curve.meta["ccd"] = 2
    with pytest.raises(ValueError, match="ePSF is for orbit 185 1-1"):
        throughput.correct_light_curve(light_curve)


#######################################################################
#### HDF5 scalars #####################################################
#######################################################################


def test_light_curve_records_flux_portion(light_curves):
    cutout, epsf, (light_curve, *_) = light_curves
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
        for aperture_name, aperture_size in LIGHT_CURVE_APERTURES:
            aperture_group = file["LightCurve/AperturePhotometry"][
                f"{aperture_name.capitalize()}Aperture"
            ]
            offset_flux = read_hdf5_offset_flux(file, aperture_name)
            np.testing.assert_allclose(
                offset_flux, light_curve[f"{aperture_name}_aperture_flux"], rtol=1e-12
            )
            raw_flux = measure_aperture_flux(
                images,
                get_aperture_limits(
                    aperture_size, round(star_x), round(star_y), images.shape[1], images.shape[2]
                ),
            )
            # Adding back the background cancels most of the offset flux's digits.
            np.testing.assert_allclose(
                offset_flux.to_value(u.electron) + aperture_group.attrs["localbackground"],
                raw_flux,
                rtol=1e-9,
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
    path = tmp_path / "lc.h5"
    light_curve.write_hdf5(path)
    throughput = EPSFThroughput.from_cutout(epsf, cutout)

    from_file = throughput.correct_hdf5(path)
    in_memory = throughput.correct_light_curve(light_curve)

    np.testing.assert_array_equal(from_file["cadence"], light_curve["cadence"])
    np.testing.assert_allclose(
        from_file["epsf_flux_fraction"], in_memory["epsf_flux_fraction"], rtol=1e-12
    )
    for aperture_name, _ in LIGHT_CURVE_APERTURES:
        for column in [f"{aperture_name}_aperture_magnitude", f"{aperture_name}_aperture_flux"]:
            np.testing.assert_allclose(from_file[column], in_memory[column], rtol=1e-12)


def test_correct_hdf5_without_scalars_returns_magnitudes_only(light_curves, tmp_path):
    cutout, epsf, (light_curve, *_) = light_curves
    path = tmp_path / "lc.h5"
    light_curve.write_hdf5(path)
    with h5py.File(path, "a") as file:
        del file.attrs["ExposureTime"]

    table = EPSFThroughput.from_cutout(epsf, cutout).correct_hdf5(path)

    assert "primary_aperture_magnitude" in table.colnames
    assert "primary_aperture_flux" not in table.colnames


def test_correct_hdf5_rejects_other_ccd(light_curves, tmp_path):
    cutout, epsf, (light_curve, *_) = light_curves
    light_curve.meta["camera"] = 3
    path = tmp_path / "lc.h5"
    light_curve.write_hdf5(path)
    with pytest.raises(ValueError, match="is for orbit 185 3-1"):
        EPSFThroughput.from_cutout(epsf, cutout).correct_hdf5(path)
