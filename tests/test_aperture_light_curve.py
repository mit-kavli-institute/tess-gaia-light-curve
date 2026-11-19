"""Tests for `tglc.aperture_light_curve`: the HDF5 layout written by `ApertureLightCurve`."""

from pathlib import Path

import astropy.units as u
import h5py
import numpy as np

from tglc.aperture_light_curve import ApertureLightCurve
from tglc.epsf import EPSF
from tglc.light_curve import generate_light_curves

from .synthetic_data import make_synthetic_cutout, make_synthetic_epsf


def _fake_spacecraft_position(orbit, time, ephemerides_directory):
    """Stand-in for the JPL Horizons query, as in tests/test_light_curve.py."""
    return np.zeros((len(np.atleast_1d(time.tdb.jd)), 3)) * u.au


def _make_light_curve(monkeypatch) -> ApertureLightCurve:
    monkeypatch.setattr("tglc.light_curve.get_tess_spacecraft_position", _fake_spacecraft_position)
    cutout = make_synthetic_cutout()  # 12x12, 5 cadences, 4 stars
    epsf = EPSF(
        make_synthetic_epsf(n_cadences=len(cutout.time)),
        psf_size=11,
        oversample=2,
        orbit=cutout.orbit,
        sector=cutout.sector,
        camera=cutout.camera,
        ccd=cutout.ccd,
        cutout_x=0,
        cutout_y=0,
    )
    return next(generate_light_curves(cutout, epsf, Path("/nonexistent")))


def test_write_hdf5_target_position_columns(tmp_path: Path, monkeypatch):
    """`LightCurve/X` and `LightCurve/Y` repeat the target's CCD position for every cadence."""
    light_curve = _make_light_curve(monkeypatch)
    n_cadences = len(light_curve)
    output_file = tmp_path / "light_curve.h5"

    light_curve.write_hdf5(output_file)

    with h5py.File(output_file) as file:
        for name, meta_key in (("X", "ccd_x"), ("Y", "ccd_y")):
            dataset = file["LightCurve"][name]
            assert dataset.dtype == np.float64
            assert dataset.shape == (n_cadences,)
            np.testing.assert_array_equal(
                dataset[:], np.full(n_cadences, light_curve.meta[meta_key])
            )


def test_write_hdf5_layout(tmp_path: Path, monkeypatch):
    light_curve = _make_light_curve(monkeypatch)
    n_cadences = len(light_curve)
    output_file = tmp_path / "light_curve.h5"

    light_curve.write_hdf5(output_file)

    with h5py.File(output_file) as file:
        assert file.attrs["TIC ID"] == light_curve.meta["tic_id"]
        assert file.attrs["Orbit"] == light_curve.meta["orbit"]

        lc_group = file["LightCurve"]
        np.testing.assert_array_equal(lc_group["BJD"][:], light_curve.time.tjd)
        np.testing.assert_array_equal(lc_group["Cadence"][:], light_curve["cadence"])
        np.testing.assert_array_equal(lc_group["QualityFlag"][:], light_curve["quality_flag"])
        np.testing.assert_array_equal(
            lc_group["Background"]["Value"][:], light_curve["background_flux"]
        )
        assert lc_group["Background"]["Error"].shape == (n_cadences,)

        photometry_group = lc_group["AperturePhotometry"]
        assert sorted(photometry_group) == ["LargeAperture", "PrimaryAperture", "SmallAperture"]
        for aperture_name in ("Primary", "Small", "Large"):
            aperture_group = photometry_group[f"{aperture_name}Aperture"]
            column_prefix = f"{aperture_name.lower()}_aperture_"
            np.testing.assert_array_equal(
                aperture_group["RawMagnitude"][:], light_curve[f"{column_prefix}magnitude"].value
            )
            assert aperture_group["RawMagnitudeError"].shape == (n_cadences,)
            for dataset_name, column in (("X", "centroid_x"), ("Y", "centroid_y")):
                np.testing.assert_array_equal(
                    aperture_group[dataset_name][:],
                    light_curve[f"{column_prefix}{column}"].value.astype(np.float32),
                )
