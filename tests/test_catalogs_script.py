"""Tests for the `tglc catalogs` script (`tglc.scripts.catalogs`)."""

import pytest


pytest.importorskip("pyticdb")

from tglc.scripts.catalogs import _get_camera_query_grid_centers


def test_get_camera_query_grid_centers_with_unknown_sector():
    with pytest.raises(ValueError, match="upgrade tesswcs"):
        _get_camera_query_grid_centers(99999, 1, 1)
