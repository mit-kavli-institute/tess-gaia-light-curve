"""
Tests for the tglc.util.tess_ephemeris module, which provides the TESS spacecraft
position.
"""

from astropy import units as u
from astropy.time import Time
import pytest

from tglc.util.tess_ephemeris import get_tess_spacecraft_position


@pytest.mark.parametrize(
    "sector,jd",
    [
        (1, 2458270.5),
        (5, 2458483.5),
        (6, 2458453.5),
        (19, 2458848.5),
        (20, 2458818.5),
        (32, 2459214.5),
        (33, 2459200.5),
        (45, 2459580.5),
        (46, 2459563.5),
        (59, 2460077.5),
        (60, 2459914.5),
        (73, 2460340.5),
        (74, 2460279.5),
        (87, 2460706.5),
        (88, 2460645.5),
        (101, 2461071.5),
    ],
)
def test_get_tess_spacecraft_position(sector: int, jd: float):
    position = get_tess_spacecraft_position(sector, Time(jd, format="jd", scale="tdb"))
    assert len(position) == 3
    assert position.unit.physical_type == u.au.physical_type


@pytest.mark.parametrize("bad_sector", [0, -1, 102, 150])
def test_get_tess_spacecraft_position_with_invalid_sector(bad_sector: int):
    with pytest.raises(ValueError):
        get_tess_spacecraft_position(bad_sector, 0.0)
