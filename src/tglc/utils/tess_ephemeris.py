"""
Provides solar system coordinates for TESS throughout the mission.

Ephemerides are fetched on demand from the JPL Horizons API as solar system barycenter vector
tables and cached on disk per orbit. The Horizons query settings match the procedure historically
used to produce static ephemeris files for QLC/TGLC:

    Ephemeris Type      : Vector Table            (EPHEM_TYPE='VECTORS')
    Target Body         : TESS                    (COMMAND='-95')
    Coordinate Origin   : Solar System Barycenter (CENTER='500@0')
    Quantities code = 4 : position, LT, range, range-rate (VEC_TABLE='4')
    Reference frame     : ICRF                    (REF_SYSTEM='ICRF')
    Reference plane     : x-y axes of frame       (REF_PLANE='FRAME')
    Vector correction   : geometric states        (VEC_CORR='NONE')
    Output units        : AU and days             (OUT_UNITS='AU-D')
"""

import fcntl
import io
import logging
import os
from pathlib import Path
import time as time_module

from astropy.time import Time
import astropy.units as u
import numpy as np
import pandas as pd
import requests

from tglc import __version__ as tglc_version


logger = logging.getLogger(__name__)


HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
TESS_HORIZONS_ID = "-95"
SOLAR_SYSTEM_BARYCENTER = "500@0"
EPHEMERIS_PADDING = 2.0 * u.day
EPHEMERIS_STEP = 1.0 * u.hour
HTTP_TIMEOUT = 120.0
MAX_RETRIES = 4
RETRY_BACKOFF = 5.0
USER_AGENT = f"tglc/{tglc_version} (https://github.com/mit-kavli-institute/tess-gaia-light-curve)"
EPHEMERIS_COLUMNS = ["JDTDB", "Calendar Date (TDB)", "X", "Y", "Z", "LT", "RG", "RR"]


class HorizonsError(RuntimeError):
    """Raised when JPL Horizons is unreachable or returns an unusable response."""


def _ephemeris_cache_file(orbit: int, ephemerides_directory: Path) -> Path:
    """Get the path to the cached ephemeris file for a TESS orbit."""
    return ephemerides_directory / f"tess_ephem_orbit-{orbit:04d}.csv"


def _query_horizons(start_string: str, stop_string: str) -> str:
    """Fetch a TESS vector table from the JPL Horizons API as raw text, with retries."""
    params = {
        "format": "text",
        "COMMAND": f"'{TESS_HORIZONS_ID}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{SOLAR_SYSTEM_BARYCENTER}'",
        "START_TIME": f"'{start_string}'",
        "STOP_TIME": f"'{stop_string}'",
        "STEP_SIZE": f"'{round(EPHEMERIS_STEP.to_value(u.minute))}m'",
        "VEC_TABLE": "'4'",
        "REF_SYSTEM": "'ICRF'",
        "REF_PLANE": "'FRAME'",
        "VEC_CORR": "'NONE'",
        "OUT_UNITS": "'AU-D'",
        "CSV_FORMAT": "'YES'",
        "VEC_LABELS": "'NO'",
        "TIME_DIGITS": "'FRACSEC'",
    }

    last_exception: Exception | None = None
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = session.get(HORIZONS_API_URL, params=params, timeout=HTTP_TIMEOUT)
                response.raise_for_status()
                if "$$SOE" not in response.text or "$$EOE" not in response.text:
                    raise HorizonsError(
                        "Horizons response missing $$SOE/$$EOE markers for "
                        f"{start_string} -> {stop_string}. Response began:\n"
                        f"{response.text[:500].strip()}"
                    )
                return response.text
            except (requests.RequestException, HorizonsError) as exception:
                last_exception = exception
                if attempt < MAX_RETRIES:
                    sleep_seconds = RETRY_BACKOFF * 2 ** (attempt - 1)
                    logger.warning(
                        f"Horizons query {start_string} -> {stop_string} failed "
                        f"(attempt {attempt}/{MAX_RETRIES}): {exception}. "
                        f"Retrying in {sleep_seconds:.1f}s"
                    )
                    time_module.sleep(sleep_seconds)
    raise HorizonsError(
        f"Failed to fetch TESS ephemeris for {start_string} -> {stop_string} after "
        f"{MAX_RETRIES} attempts"
    ) from last_exception


def _parse_horizons_response(text: str) -> pd.DataFrame:
    """Extract the CSV block between $$SOE and $$EOE markers in a Horizons response."""
    body = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip()
    if not body:
        raise HorizonsError("Horizons returned an empty vector table")

    ephemeris = pd.read_csv(
        io.StringIO(body),
        header=None,
        # Rows end with a trailing comma, creating an extra empty column
        names=EPHEMERIS_COLUMNS + ["_trailing"],
        skipinitialspace=True,
    )
    return (
        ephemeris.drop(columns="_trailing")
        .drop_duplicates(subset="JDTDB")
        .sort_values("JDTDB")
        .reset_index(drop=True)
    )


def _fetch_spacecraft_ephemeris(start: Time, stop: Time) -> pd.DataFrame:
    """Fetch a TESS ephemeris covering [start - padding, stop + padding] from JPL Horizons."""
    padded_start = start - EPHEMERIS_PADDING
    padded_stop = stop + EPHEMERIS_PADDING
    # Horizons interprets vector table START/STOP epochs as TDB
    start_string = padded_start.tdb.strftime("%Y-%m-%d %H:%M")
    stop_string = padded_stop.tdb.strftime("%Y-%m-%d %H:%M")
    logger.info(f"Fetching TESS ephemeris from JPL Horizons: {start_string} -> {stop_string} TDB")
    ephemeris = _parse_horizons_response(_query_horizons(start_string, stop_string))

    # The fixed-step grid runs from START_TIME, so the last row may sit up to one step before
    # STOP_TIME; epochs are also truncated to whole minutes in the query.
    slack = (EPHEMERIS_STEP + 2 * u.minute).to_value(u.day)
    if (
        ephemeris["JDTDB"].iloc[0] > padded_start.tdb.jd + slack
        or ephemeris["JDTDB"].iloc[-1] < padded_stop.tdb.jd - slack
    ):
        raise HorizonsError(
            "Downloaded ephemeris does not cover the padded span "
            f"{start_string} -> {stop_string} TDB; the response may be truncated. If the stop "
            "time is far in the future, the predicted ephemeris may not extend that far yet."
        )
    return ephemeris


def get_spacecraft_ephemeris(
    orbit: int, start: Time, stop: Time, ephemerides_directory: Path
) -> pd.DataFrame:
    """
    Get a TESS solar system barycenter vector table covering [start, stop], cached per orbit.

    The cached file is treated as immutable because orbits are processed after their data spans
    have passed, when JPL serves the definitive trajectory. Delete the file to force a fresh
    download. If a cached file exists but does not cover the requested span, it is re-fetched
    and replaced. A lock file makes concurrent workers on the same host wait for a single
    download instead of each querying JPL Horizons.
    """
    ephemerides_directory.mkdir(parents=True, exist_ok=True)
    cache_file = _ephemeris_cache_file(orbit, ephemerides_directory)
    lock_file = cache_file.with_name(cache_file.name + ".lock")
    with lock_file.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache_file.is_file():
            ephemeris = pd.read_csv(cache_file)
            if (
                ephemeris["JDTDB"].iloc[0] <= start.tdb.jd
                and stop.tdb.jd <= ephemeris["JDTDB"].iloc[-1]
            ):
                logger.debug(f"Using cached TESS ephemeris {cache_file}")
                return ephemeris
            logger.warning(
                f"Cached TESS ephemeris {cache_file} does not cover the requested span, re-fetching"
            )
        ephemeris = _fetch_spacecraft_ephemeris(start, stop)
        temporary_file = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
        ephemeris.to_csv(temporary_file, index=False)
        temporary_file.replace(cache_file)
        logger.info(f"Cached {len(ephemeris)} TESS ephemeris rows for orbit {orbit}: {cache_file}")
    return ephemeris


def get_tess_spacecraft_position(
    orbit: int,
    time: Time,
    ephemerides_directory: Path,
) -> u.Quantity["length"]:  # noqa: F821
    """
    Get the TESS spacecraft position relative to the solar system barycenter at given timestamps.

    Returns an (N, 3) quantity in AU, interpolated from a per-orbit ephemeris fetched from JPL
    Horizons (cached in `ephemerides_directory`). Raises a `ValueError` if any timestamps fall
    outside the coverage of the ephemeris instead of extrapolating.
    """
    jd = np.atleast_1d(time.tdb.jd)
    start = Time(jd.min(), format="jd", scale="tdb")
    stop = Time(jd.max(), format="jd", scale="tdb")
    ephemeris = get_spacecraft_ephemeris(orbit, start, stop, ephemerides_directory)
    if jd.min() < ephemeris["JDTDB"].iloc[0] or jd.max() > ephemeris["JDTDB"].iloc[-1]:
        raise ValueError(
            f"Timestamps (JD {jd.min():.4f}-{jd.max():.4f}) are outside the coverage of the "
            f"TESS ephemeris for orbit {orbit} "
            f"(JD {ephemeris['JDTDB'].iloc[0]:.4f}-{ephemeris['JDTDB'].iloc[-1]:.4f}). Delete "
            f"{_ephemeris_cache_file(orbit, ephemerides_directory)} to force a fresh download."
        )
    spacecraft_x = np.interp(jd, ephemeris["JDTDB"], ephemeris["X"])
    spacecraft_y = np.interp(jd, ephemeris["JDTDB"], ephemeris["Y"])
    spacecraft_z = np.interp(jd, ephemeris["JDTDB"], ephemeris["Z"])
    return np.array([spacecraft_x, spacecraft_y, spacecraft_z]).T * u.au
