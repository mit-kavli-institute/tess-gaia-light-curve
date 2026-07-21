"""
Tests for the tglc.aperture_light_curve module, which provides the light curve class holding
photometry data for one or more apertures.
"""

from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
import h5py
import numpy as np
import pytest

from tglc.aperture_light_curve import ApertureLightCurve, ApertureLightCurveMetadata
from tglc.apertures import APERTURE_NAMES


N_CADENCES = 5


def make_light_curve_data(apertures: list[str]) -> QTable:
    """Create a synthetic light curve table with columns for the given apertures."""
    data = {
        "time": Time(np.arange(N_CADENCES) + 2500.0, format="tjd", scale="tdb"),
        "cadence": np.arange(N_CADENCES),
        "quality_flag": np.zeros(N_CADENCES, dtype=int),
        "background_flux": np.linspace(10.0, 20.0, N_CADENCES) * u.electron,
    }
    for aperture_name in apertures:
        data[f"{aperture_name}_aperture_magnitude"] = np.full(N_CADENCES, 10.0)
        data[f"{aperture_name}_aperture_centroid_x"] = np.full(N_CADENCES, 2.0) * u.pixel
        data[f"{aperture_name}_aperture_centroid_y"] = np.full(N_CADENCES, 2.0) * u.pixel
    return QTable(data)


def make_light_curve_metadata(apertures: list[str]) -> ApertureLightCurveMetadata:
    """Create synthetic light curve metadata with local backgrounds for the given apertures."""
    return ApertureLightCurveMetadata(
        tic_id=1,
        orbit=185,
        sector=89,
        camera=1,
        ccd=1,
        ccd_x=100.0,
        ccd_y=200.0,
        sky_coord=SkyCoord(10.0, 20.0, unit="deg"),
        tess_magnitude=10.0,
        exposure_time=200 * u.second,
        **{
            f"{aperture_name}_aperture_local_background": 0 * u.electron
            for aperture_name in apertures
        },
    )


def test_all_apertures_by_default():
    all_apertures = list(APERTURE_NAMES)
    light_curve = ApertureLightCurve(
        make_light_curve_data(all_apertures), meta=make_light_curve_metadata(all_apertures)
    )
    assert light_curve.meta["apertures"] == ["primary", "small", "large"]


def test_apertures_normalized_to_canonical_order_and_deduplicated():
    all_apertures = list(APERTURE_NAMES)
    light_curve = ApertureLightCurve(
        make_light_curve_data(all_apertures),
        meta=make_light_curve_metadata(all_apertures),
        apertures=["large", "primary", "primary"],
    )
    assert light_curve.meta["apertures"] == ["primary", "large"]


def test_apertures_derived_from_columns_when_unspecified():
    light_curve = ApertureLightCurve(
        make_light_curve_data(["primary"]), meta=make_light_curve_metadata(["primary"])
    )
    assert light_curve.meta["apertures"] == ["primary"]


def test_unrecognized_aperture_rejected():
    with pytest.raises(ValueError, match="Unrecognized apertures"):
        ApertureLightCurve(
            make_light_curve_data(["primary"]),
            meta=make_light_curve_metadata(["primary"]),
            apertures=["primary", "medium"],
        )


def test_empty_aperture_list_rejected():
    with pytest.raises(ValueError, match="at least one aperture"):
        ApertureLightCurve(
            make_light_curve_data(["primary"]),
            meta=make_light_curve_metadata(["primary"]),
            apertures=[],
        )


def test_missing_aperture_columns_rejected():
    with pytest.raises(ValueError, match="small_aperture_magnitude"):
        ApertureLightCurve(
            make_light_curve_data(["primary"]),
            meta=make_light_curve_metadata(["primary", "small"]),
            apertures=["primary", "small"],
        )


def test_missing_aperture_local_background_rejected():
    with pytest.raises(ValueError, match="primary_aperture_local_background"):
        ApertureLightCurve(make_light_curve_data(["primary"]), meta=make_light_curve_metadata([]))


def test_write_hdf5_with_all_apertures(tmp_path: Path):
    all_apertures = list(APERTURE_NAMES)
    light_curve = ApertureLightCurve(
        make_light_curve_data(all_apertures), meta=make_light_curve_metadata(all_apertures)
    )
    output_file = tmp_path / "1.h5"
    light_curve.write_hdf5(output_file)

    with h5py.File(output_file, "r") as file:
        assert file.attrs["TIC ID"] == 1
        assert file.attrs["Orbit"] == 185
        assert file.attrs["Sector"] == 89
        assert file.attrs["Camera"] == 1
        assert file.attrs["CCD"] == 1
        assert file.attrs["RA"] == 10.0
        assert file.attrs["Dec"] == 20.0
        assert file.attrs["TessMag"] == 10.0

        lc_group = file["LightCurve"]
        for dataset_name in ["BJD", "Cadence", "X", "Y", "QualityFlag"]:
            assert lc_group[dataset_name].shape == (N_CADENCES,)
        for dataset_name in ["Value", "Error"]:
            assert lc_group["Background"][dataset_name].shape == (N_CADENCES,)

        photometry_group = lc_group["AperturePhotometry"]
        assert list(photometry_group) == ["LargeAperture", "PrimaryAperture", "SmallAperture"]
        for aperture_name, aperture_size in [
            ("Primary", 3),
            ("Small", 1),
            ("Large", 5),
        ]:
            aperture_group = photometry_group[f"{aperture_name}Aperture"]
            assert aperture_group.attrs["name"] == f"TGLCAperture{aperture_name}"
            assert aperture_group.attrs["description"] == f"{aperture_size}x{aperture_size} square"
            assert aperture_group.attrs["localbackground"] == 0.0
            for dataset_name in ["RawMagnitude", "RawMagnitudeError", "X", "Y"]:
                assert aperture_group[dataset_name].shape == (N_CADENCES,)


def test_write_hdf5_with_primary_aperture_only(tmp_path: Path):
    light_curve = ApertureLightCurve(
        make_light_curve_data(["primary"]),
        meta=make_light_curve_metadata(["primary"]),
        apertures=["primary"],
    )
    output_file = tmp_path / "1.h5"
    light_curve.write_hdf5(output_file)

    with h5py.File(output_file, "r") as file:
        lc_group = file["LightCurve"]
        for dataset_name in ["BJD", "Cadence", "X", "Y", "QualityFlag"]:
            assert lc_group[dataset_name].shape == (N_CADENCES,)
        for dataset_name in ["Value", "Error"]:
            assert lc_group["Background"][dataset_name].shape == (N_CADENCES,)

        photometry_group = lc_group["AperturePhotometry"]
        assert list(photometry_group) == ["PrimaryAperture"]
        aperture_group = photometry_group["PrimaryAperture"]
        assert aperture_group.attrs["name"] == "TGLCAperturePrimary"
        assert aperture_group.attrs["description"] == "3x3 square"
        for dataset_name in ["RawMagnitude", "RawMagnitudeError", "X", "Y"]:
            assert aperture_group[dataset_name].shape == (N_CADENCES,)
