"""Tests for `tglc.aperture_light_curve`: the HDF5 layout written by `ApertureLightCurve`."""

from pathlib import Path

from astropy.stats import mad_std
import astropy.units as u
import h5py
import numpy as np
import pytest

from tglc.aperture_light_curve import ApertureLightCurve
from tglc.light_curve import generate_light_curves

from .synthetic_data import make_synthetic_cutout_and_epsf


@pytest.fixture
def written_light_curve(
    tmp_path: Path, fake_spacecraft_position
) -> tuple[ApertureLightCurve, Path]:
    """A synthetic light curve and the HDF5 file `write_hdf5` produced for it."""
    cutout, epsf = make_synthetic_cutout_and_epsf()  # 12x12, 5 cadences, 4 stars
    light_curve = next(generate_light_curves(cutout, epsf, Path("/nonexistent")))
    output_file = tmp_path / "light_curve.h5"
    light_curve.write_hdf5(output_file)
    return light_curve, output_file


def _assert_dataset(dataset: h5py.Dataset, expected: np.ndarray, dtype: type):
    assert dataset.dtype == dtype
    np.testing.assert_array_equal(dataset[:], np.asarray(expected).astype(dtype))


def test_write_hdf5_attributes(written_light_curve):
    light_curve, output_file = written_light_curve
    meta = light_curve.meta

    with h5py.File(output_file) as file:
        assert dict(file.attrs) == {
            "TIC ID": meta["tic_id"],
            "Orbit": meta["orbit"],
            "Sector": meta["sector"],
            "Camera": meta["camera"],
            "CCD": meta["ccd"],
            "RA": meta["sky_coord"].ra.deg,
            "Dec": meta["sky_coord"].dec.deg,
            "BJDoffset": 2457000.0,
            "TessMag": meta["tess_magnitude"],
        }

        photometry_group = file["LightCurve"]["AperturePhotometry"]
        assert sorted(photometry_group) == ["LargeAperture", "PrimaryAperture", "SmallAperture"]
        for aperture_name, aperture_size in (("Primary", 3), ("Small", 1), ("Large", 5)):
            local_background = meta[f"{aperture_name.lower()}_aperture_local_background"]
            assert dict(photometry_group[f"{aperture_name}Aperture"].attrs) == {
                "name": f"TGLCAperture{aperture_name}",
                "description": f"{aperture_size}x{aperture_size} square",
                "localbackground": local_background.to_value(u.electron),
            }


def test_write_hdf5_datasets(written_light_curve):
    light_curve, output_file = written_light_curve
    n_cadences = len(light_curve)

    with h5py.File(output_file) as file:
        lc_group = file["LightCurve"]
        _assert_dataset(lc_group["BJD"], light_curve.time.tjd, np.float64)
        _assert_dataset(lc_group["Cadence"], light_curve["cadence"], np.int64)
        _assert_dataset(lc_group["QualityFlag"], light_curve["quality_flag"], np.int64)
        # The target's CCD position, repeated for every cadence
        _assert_dataset(lc_group["X"], np.full(n_cadences, light_curve.meta["ccd_x"]), np.float64)
        _assert_dataset(lc_group["Y"], np.full(n_cadences, light_curve.meta["ccd_y"]), np.float64)

        background_flux = light_curve["background_flux"]
        _assert_dataset(lc_group["Background"]["Value"], background_flux, np.float64)
        _assert_dataset(
            lc_group["Background"]["Error"],
            np.full(n_cadences, mad_std(background_flux)),
            np.float64,
        )

        for aperture_name in ("Primary", "Small", "Large"):
            aperture_group = lc_group["AperturePhotometry"][f"{aperture_name}Aperture"]
            column_prefix = f"{aperture_name.lower()}_aperture_"
            magnitude = light_curve[f"{column_prefix}magnitude"]
            _assert_dataset(aperture_group["RawMagnitude"], magnitude, np.float64)
            _assert_dataset(
                aperture_group["RawMagnitudeError"],
                np.full(n_cadences, mad_std(magnitude)),
                np.float32,
            )
            _assert_dataset(
                aperture_group["X"], light_curve[f"{column_prefix}centroid_x"].value, np.float32
            )
            _assert_dataset(
                aperture_group["Y"], light_curve[f"{column_prefix}centroid_y"].value, np.float32
            )
