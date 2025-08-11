"""Tests for the fixtures that provide the pyticdb databases using docker containers."""

import pyticdb


def test_pyticdb_mock(pyticdb_databases):
    query_result = pyticdb.query_by_id(1227555, "id", "tmag")

    assert len(query_result) > 0


def test_gaia_mock(pyticdb_databases):
    query_result = pyticdb.query_by_id(
        5731121797626498944, "designation", "phot_g_mean_mag", database="gaia3", table="gaia_source"
    )

    assert len(query_result) > 0
