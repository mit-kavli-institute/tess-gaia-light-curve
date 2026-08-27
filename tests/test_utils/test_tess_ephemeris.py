"""
Tests for the tglc.utils.tess_ephemeris module, which provides the TESS spacecraft position
based on ephemerides fetched from JPL Horizons and cached per orbit.

By default no test in this module makes real network requests: the Horizons query is
monkeypatched with canned responses. The exception is the `network`-marked canary test, which
is deselected by default and contacts the real Horizons API when run with `pytest -m network`.
"""

from pathlib import Path

from astropy.time import Time
import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import requests

from tglc.utils import tess_ephemeris
from tglc.utils.tess_ephemeris import (
    EPHEMERIS_COLUMNS,
    HorizonsError,
    get_spacecraft_ephemeris,
    get_tess_spacecraft_position,
)

from ..sample_data import SAMPLE_DATA_DIRECTORY


TEST_ORBIT = 185
TEST_CACHE_FILE_NAME = f"tess_ephem_orbit-{TEST_ORBIT:04d}.csv"
HOURLY_STEP = 1.0 / 24.0


def make_ephemeris_dataframe(start_jd: float, stop_jd: float) -> pd.DataFrame:
    """Create a fake hourly ephemeris covering [start_jd, stop_jd]."""
    jd = np.arange(start_jd, stop_jd + HOURLY_STEP / 2, HOURLY_STEP)
    return pd.DataFrame(
        {
            "JDTDB": jd,
            "Calendar Date (TDB)": "A.D. 2025-Feb-12 00:00:00.0000",
            "X": np.linspace(-0.8, -0.7, len(jd)),
            "Y": np.linspace(0.5, 0.6, len(jd)),
            "Z": np.linspace(0.2, 0.3, len(jd)),
            "LT": 0.0057,
            "RG": 0.989,
            "RR": -1.1e-5,
        }
    )


def make_horizons_response(start_jd: float, stop_jd: float) -> str:
    """Create fake JPL Horizons vector table response text covering [start_jd, stop_jd]."""
    ephemeris = make_ephemeris_dataframe(start_jd, stop_jd)
    rows = [
        f"{row.JDTDB:.9f}, {row['Calendar Date (TDB)']}, {row.X:.16e}, {row.Y:.16e},"
        f" {row.Z:.16e}, {row.LT:.16e}, {row.RG:.16e}, {row.RR:.16e},"
        for _, row in ephemeris.iterrows()
    ]
    return (
        "API VERSION: 1.2\nAPI SOURCE: NASA/JPL Horizons API\n$$SOE\n"
        + "\n".join(rows)
        + ("\n$$EOE\nCoordinate system description:\n")
    )


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP status {self.status_code}")


@pytest.fixture
def fake_horizons_query(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace _query_horizons with a call-counting fake covering a generous padded span."""
    state = {"calls": 0, "start_jd": 2460715.0, "stop_jd": 2460730.0}

    def query(start_string: str, stop_string: str) -> str:
        state["calls"] += 1
        return make_horizons_response(state["start_jd"], state["stop_jd"])

    monkeypatch.setattr(tess_ephemeris, "_query_horizons", query)
    return state


def test_parse_horizons_response():
    ephemeris = tess_ephemeris._parse_horizons_response(
        make_horizons_response(2460718.5, 2460720.5)
    )
    assert list(ephemeris.columns) == EPHEMERIS_COLUMNS
    assert len(ephemeris) == 49
    assert ephemeris["JDTDB"].is_monotonic_increasing
    assert ephemeris["JDTDB"].dtype == np.float64
    assert ephemeris["X"].dtype == np.float64


def test_parse_horizons_response_empty_table_raises():
    with pytest.raises(HorizonsError, match="empty"):
        tess_ephemeris._parse_horizons_response("header\n$$SOE\n$$EOE\nfooter")


def test_get_spacecraft_ephemeris_creates_cache(fake_horizons_query: dict, tmp_path: Path):
    start = Time(2460720.0, format="jd", scale="tdb")
    stop = Time(2460725.0, format="jd", scale="tdb")
    ephemeris = get_spacecraft_ephemeris(TEST_ORBIT, start, stop, tmp_path)

    assert fake_horizons_query["calls"] == 1
    assert list(ephemeris.columns) == EPHEMERIS_COLUMNS
    cache_file = tmp_path / TEST_CACHE_FILE_NAME
    assert cache_file.is_file()
    cached_ephemeris = pd.read_csv(cache_file)
    assert list(cached_ephemeris.columns) == EPHEMERIS_COLUMNS
    assert cached_ephemeris["JDTDB"].to_numpy() == pytest.approx(ephemeris["JDTDB"].to_numpy())


def test_get_spacecraft_ephemeris_cache_hit_skips_query(fake_horizons_query: dict, tmp_path: Path):
    start = Time(2460720.0, format="jd", scale="tdb")
    stop = Time(2460725.0, format="jd", scale="tdb")
    get_spacecraft_ephemeris(TEST_ORBIT, start, stop, tmp_path)
    get_spacecraft_ephemeris(TEST_ORBIT, start, stop, tmp_path)
    assert fake_horizons_query["calls"] == 1


def test_get_spacecraft_ephemeris_refetches_short_cache(fake_horizons_query: dict, tmp_path: Path):
    # Seed a cache file that does not cover the requested span
    make_ephemeris_dataframe(2460700.0, 2460705.0).to_csv(
        tmp_path / TEST_CACHE_FILE_NAME, index=False
    )
    start = Time(2460720.0, format="jd", scale="tdb")
    stop = Time(2460725.0, format="jd", scale="tdb")
    ephemeris = get_spacecraft_ephemeris(TEST_ORBIT, start, stop, tmp_path)

    assert fake_horizons_query["calls"] == 1
    assert ephemeris["JDTDB"].iloc[0] <= start.jd
    assert stop.jd <= ephemeris["JDTDB"].iloc[-1]
    cached_ephemeris = pd.read_csv(tmp_path / TEST_CACHE_FILE_NAME)
    assert cached_ephemeris["JDTDB"].iloc[-1] >= stop.jd


def test_get_spacecraft_ephemeris_raises_for_incomplete_coverage(
    fake_horizons_query: dict, tmp_path: Path
):
    # Fake response covers less than the padded span requested from Horizons
    fake_horizons_query["start_jd"] = 2460720.0
    fake_horizons_query["stop_jd"] = 2460725.0
    start = Time(2460720.0, format="jd", scale="tdb")
    stop = Time(2460725.0, format="jd", scale="tdb")
    with pytest.raises(HorizonsError, match="does not cover the padded span"):
        get_spacecraft_ephemeris(TEST_ORBIT, start, stop, tmp_path)


def test_get_tess_spacecraft_position(fake_horizons_query: dict, tmp_path: Path):
    time = Time(np.linspace(2460720.0, 2460725.0, 100), format="jd", scale="tdb")
    position = get_tess_spacecraft_position(TEST_ORBIT, time, tmp_path)
    assert position.shape == (100, 3)
    assert position.unit.physical_type == u.au.physical_type
    # X values in the fake ephemeris increase linearly, so interpolation is exact
    ephemeris = make_ephemeris_dataframe(
        fake_horizons_query["start_jd"], fake_horizons_query["stop_jd"]
    )
    expected_x = np.interp(time.jd, ephemeris["JDTDB"], ephemeris["X"])
    assert position[:, 0].to_value(u.au) == pytest.approx(expected_x)


def test_get_tess_spacecraft_position_out_of_range_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Return an ephemeris that doesn't cover the requested times, bypassing the fetch-time
    # coverage checks, to prove interpolation refuses to extrapolate (np.interp would
    # silently clamp to the first/last samples)
    monkeypatch.setattr(
        tess_ephemeris,
        "get_spacecraft_ephemeris",
        lambda orbit, start, stop, ephemerides_directory: make_ephemeris_dataframe(
            2460721.0, 2460723.0
        ),
    )
    time = Time([2460720.0, 2460722.0, 2460725.0], format="jd", scale="tdb")
    with pytest.raises(ValueError, match="outside the coverage"):
        get_tess_spacecraft_position(TEST_ORBIT, time, tmp_path)


def test_query_horizons_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    state = {"calls": 0, "sleeps": []}
    response_text = make_horizons_response(2460718.0, 2460727.0)

    def fake_get(self, url, params=None, timeout=None):
        state["calls"] += 1
        if state["calls"] < 3:
            raise requests.ConnectionError("connection refused")
        return FakeResponse(response_text)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(tess_ephemeris.time_module, "sleep", state["sleeps"].append)

    assert tess_ephemeris._query_horizons("2025-02-12 00:00", "2025-02-21 00:00") == response_text
    assert state["calls"] == 3
    # Exponential backoff between attempts
    assert state["sleeps"] == [
        tess_ephemeris.RETRY_BACKOFF,
        tess_ephemeris.RETRY_BACKOFF * 2,
    ]


def test_query_horizons_retries_exhausted_raises(monkeypatch: pytest.MonkeyPatch):
    state = {"calls": 0}

    def fake_get(self, url, params=None, timeout=None):
        state["calls"] += 1
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(tess_ephemeris.time_module, "sleep", lambda seconds: None)

    with pytest.raises(HorizonsError, match="after"):
        tess_ephemeris._query_horizons("2025-02-12 00:00", "2025-02-21 00:00")
    assert state["calls"] == tess_ephemeris.MAX_RETRIES


def test_query_horizons_missing_markers_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        requests.Session,
        "get",
        lambda self, url, params=None, timeout=None: FakeResponse("No ephemeris for target"),
    )
    monkeypatch.setattr(tess_ephemeris.time_module, "sleep", lambda seconds: None)

    with pytest.raises(HorizonsError):
        tess_ephemeris._query_horizons("2025-02-12 00:00", "2025-02-21 00:00")


@pytest.mark.network
def test_fetch_real_horizons_ephemeris(tmp_path: Path):
    """
    Canary test against the real JPL Horizons API, verifying the query parameters are still
    accepted and the response format is still parseable. Interpolated positions are compared
    against the checked-in sample ephemeris to catch unit or reference frame regressions.
    """
    start = Time(2460722.0, format="jd", scale="tdb")
    stop = Time(2460723.0, format="jd", scale="tdb")
    ephemeris = get_spacecraft_ephemeris(TEST_ORBIT, start, stop, tmp_path)
    assert list(ephemeris.columns) == EPHEMERIS_COLUMNS
    # The fetched table covers the padded span (up to strftime minute truncation + step slack)
    padding_days = tess_ephemeris.EPHEMERIS_PADDING.to_value(u.day)
    assert ephemeris["JDTDB"].iloc[0] <= start.jd - padding_days + 0.1
    assert stop.jd + padding_days - 0.1 <= ephemeris["JDTDB"].iloc[-1]

    time = Time([2460722.2, 2460722.8], format="jd", scale="tdb")
    position = get_tess_spacecraft_position(TEST_ORBIT, time, tmp_path)

    sample_ephemeris = pd.read_csv(
        SAMPLE_DATA_DIRECTORY / "ephemerides" / f"tess_ephem_orbit-{TEST_ORBIT:04d}.csv"
    )
    expected = np.array(
        [np.interp(time.jd, sample_ephemeris["JDTDB"], sample_ephemeris[axis]) for axis in "XYZ"]
    ).T
    # The sample file holds a slightly older trajectory solution, so allow ~15,000 km. Unit
    # mistakes (km vs AU) or frame mistakes (ecliptic vs ICRF) are orders of magnitude larger.
    assert position.to_value(u.au) == pytest.approx(expected, abs=1e-4)
