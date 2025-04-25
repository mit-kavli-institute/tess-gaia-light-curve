"""
Tests for the tglc.util.tess_pointing module, which provides spacecraft and camera
pointings for each TESS sector.
"""

from importlib import resources

import astropy.units as u
import pytest

from tglc.util import data
from tglc.util.tess_pointings import TESS_SECTOR_POINTINGS, get_sector_camera_pointing


def test_pointing_data_exists():
    data_files = resources.files(data)
    assert (data_files / "Years1-8_pointings_long.csv").is_file()


def test_pointing_table_has_expected_sectors():
    for sector in range(1, 135):
        assert sector in TESS_SECTOR_POINTINGS["Sector"]


def test_pointing_table_has_expected_columns():
    # Basic columns
    for column_name in ["Sector", "Dates"]:
        assert column_name in TESS_SECTOR_POINTINGS.colnames
    # Pointing columns
    for pointing_type in ["Spacecraft", "Camera 1", "Camera 2", "Camera 3", "Camera 4"]:
        for column in ["RA", "Dec", "Roll"]:
            column_name = f"{pointing_type} {column}"
            assert column_name in TESS_SECTOR_POINTINGS.colnames
            assert TESS_SECTOR_POINTINGS[column_name].unit == u.deg


@pytest.mark.parametrize("bad_sector", [0, -1, 135, 150])
def test_get_sector_camera_pointing_rejects_invalid_sector(bad_sector: int):
    with pytest.raises(ValueError):
        get_sector_camera_pointing(bad_sector, 1)


@pytest.mark.parametrize("bad_camera", [0, -1, 5, 10])
def test_get_sector_camera_pointing_rejects_invalid_camera(bad_camera: int):
    with pytest.raises(ValueError):
        get_sector_camera_pointing(1, bad_camera)


@pytest.mark.parametrize(
    "sector,camera", [(1, 1), (1, 2), (1, 3), (1, 4), (44, 2), (96, 4), (134, 1)]
)
def test_get_sector_camera_pointing(sector: int, camera: int):
    coord = get_sector_camera_pointing(sector, camera)
    assert coord.ra.unit == u.deg
    assert coord.dec.unit == u.deg
