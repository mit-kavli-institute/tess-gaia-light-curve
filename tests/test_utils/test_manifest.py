# ruff: noqa: B018 a lot of what we're testing is that attributes can be accessed without errors
from pathlib import Path

import pytest

from tglc.utils.manifest import Manifest, get_tic_id_shard_path


@pytest.mark.parametrize(
    "tic_id,expected",
    [
        (12345678901, Path("0/000/012/345/678")),
        (0, Path("0/000/000/000/000")),
        (1, Path("0/000/000/000/000")),
        (999, Path("0/000/000/000/000")),
        (1000, Path("0/000/000/000/001")),
        (9999999999999999, Path("9/999/999/999/999")),
    ],
)
def test_get_tic_id_shard_path(tic_id: int, expected: Path):
    assert get_tic_id_shard_path(tic_id) == expected


@pytest.mark.parametrize("tic_id", [-1, 10**16])
def test_get_tic_id_shard_path_rejects_out_of_range_ids(tic_id: int):
    with pytest.raises(ValueError, match="TIC ID"):
        get_tic_id_shard_path(tic_id)


def test_Manifest():
    m = Manifest(Path("/tglc-data"))
    assert m.tglc_data_dir == Path("/tglc-data")
    assert "/tglc-data" in str(m)


def test_Manifest_raises_errors_for_missing_attributes():
    m = Manifest(Path("/tglc-data"))
    with pytest.raises(RuntimeError):
        m.orbit
    with pytest.raises(RuntimeError):
        m.camera
    with pytest.raises(RuntimeError):
        m.ccd
    with pytest.raises(RuntimeError):
        m.cadence
    with pytest.raises(RuntimeError):
        m.tic_id
    with pytest.raises(RuntimeError):
        m.cutout_x
    with pytest.raises(RuntimeError):
        m.cutout_y


def test_Manifest_raises_error_for_property_requiring_missing_attribute():
    m = Manifest(Path("/tglc-data"))
    with pytest.raises(RuntimeError):
        m.orbit_directory


def test_Manifest_no_error_for_non_null_attribute():
    m = Manifest(Path("/tglc-data"))
    with pytest.raises(RuntimeError):
        m.orbit
    with pytest.raises(RuntimeError):
        m.orbit_directory
    m.orbit = 123
    assert m.orbit == 123
    m.orbit_directory


def test_Manifest_kitchen_sink_properties():
    m = Manifest(
        Path("/tglc-data"),
        orbit=9,
        camera=1,
        ccd=1,
        cadence=1,
        tic_id=1,
        cutout_x=1,
        cutout_y=1,
    )

    assert isinstance(m.orbit_directory, Path)
    assert "orbit-9/ffi" in str(m.orbit_directory)

    assert isinstance(m.catalog_directory, Path)
    assert "catalogs" in str(m.catalog_directory)

    assert isinstance(m.gaia_catalog_file, Path)
    assert "Gaia_cam1_ccd1.ecsv" in str(m.gaia_catalog_file)

    assert isinstance(m.tic_catalog_file, Path)
    assert "TIC_cam1_ccd1.ecsv" in str(m.tic_catalog_file)

    assert isinstance(m.camera_directory, Path)
    assert "cam1" in str(m.camera_directory)

    assert isinstance(m.ccd_directory, Path)
    assert "ccd1" in str(m.ccd_directory)

    assert isinstance(m.ffi_directory, Path)
    assert "ffi" in str(m.ffi_directory)

    assert m.tica_ffi_file_pattern == "hlsp_tica_tess_ffi_s0001-o1-*-cam1-ccd1_tess_v01_img.fits"

    assert isinstance(m.tica_ffi_file, Path)
    assert "hlsp_tica_tess_ffi_s0001-o1-00000001-cam1-ccd1_tess_v01_img.fits" in str(
        m.tica_ffi_file
    )

    assert isinstance(m.source_directory, Path)
    assert "source" in str(m.source_directory)

    assert isinstance(m.source_file, Path)
    assert "source_1_1.fits" in str(m.source_file)

    assert isinstance(m.source_file_legacy_pkl, Path)
    assert "source_1_1.pkl" in str(m.source_file_legacy_pkl)

    assert isinstance(m.epsf_directory, Path)
    assert "epsf" in str(m.epsf_directory)

    assert isinstance(m.epsf_file, Path)
    assert "epsf_1_1.fits" in str(m.epsf_file)

    assert isinstance(m.epsf_file_legacy_npy, Path)
    assert "epsf_1_1.npy" in str(m.epsf_file_legacy_npy)

    assert isinstance(m.light_curve_directory, Path)
    assert "LC" in str(m.light_curve_directory)

    assert isinstance(m.light_curve_file, Path)
    assert str(m.light_curve_file).endswith("LC/0/000/000/000/000/1.h5")
