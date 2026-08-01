"""
Correct the barycentric timestamps of existing TGLC H5 light curves in place.

Light curves produced before commit d55ae79 carry BJD values computed with the old, buggy
ephemeris interpolation (see `audit_lightcurve_timing.py`). This script repairs them:

1. The true spacecraft times are recovered from the cutout `Source` pickles: `Source.cadence`
   and `Source.time` (FFI MIDTJD values) map each H5 `LightCurve/Cadence` entry back to its
   spacecraft timestamp. FFI timestamps differ slightly between cameras/CCDs, so files are
   grouped by (orbit, camera, ccd) and each group uses a Source pickle from the same CCD.
2. The barycentric correction is recomputed with the production code path
   (`get_tess_spacecraft_position` + `apply_barycentric_correction`), so corrected files are
   bit-identical to what the current pipeline would write.
3. Only the values of the `LightCurve/BJD` dataset are overwritten (same shape and dtype);
   no other datasets or attributes are touched. Files storing full JD (pre-fe6513a) are
   corrected in their own convention. The old BJD is never an input to the correction, so
   re-running is idempotent and even partially corrupted BJD datasets are fully repaired.

Groups whose Source pickle is missing are skipped and reported (`source_missing`) — the
correction never guesses spacecraft times. Regenerate Sources with `tglc cutouts` and re-run.

Note for network filesystems with broken POSIX locking: set `HDF5_USE_FILE_LOCKING=FALSE`.
Running two fixer invocations concurrently over overlapping inputs is not supported.

Subcommands
-----------
fix              Correct H5 files (directories or manifest lists) in place.
make-synthetic   Generate an offline synthetic dataset with known-wrong BJDs.
verify-synthetic Check a fixed synthetic dataset against expectations (exit code 0/1).
"""

import argparse
from collections import Counter
from datetime import UTC, datetime
from functools import partial
import hashlib
import json
import logging
from pathlib import Path
import pickle
import re
import sys

from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
# Make tglc (repo checkout) and the sibling audit module importable when running directly
sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import audit_lightcurve_timing as audit  # noqa: E402

from tglc import __version__ as tglc_version  # noqa: E402
from tglc.ffi import Source  # noqa: E402
from tglc.utils.constants import TESSJD, apply_barycentric_correction  # noqa: E402, F401
from tglc.utils.manifest import Manifest  # noqa: E402
from tglc.utils.mapping import pool_map_if_multiprocessing  # noqa: E402
from tglc.utils.tess_ephemeris import HorizonsError, get_tess_spacecraft_position  # noqa: E402


logger = logging.getLogger("fix_lightcurve_timing")

CAMERA_CCD_PATTERN = re.compile(r"cam(\d)/ccd(\d)")

FIX_RESULT_COLUMNS = [
    "path", "status", "tic_id", "orbit_attr", "camera_attr", "ccd_attr",
    "orbit_group", "camera_group", "ccd_group", "ra", "dec", "n_cadences",
    "bjd_format", "max_abs_delta_s", "mean_delta_s", "error",
]  # fmt: skip

SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------------------------
# Grouping and Source discovery
# ---------------------------------------------------------------------------------------------


def read_group_attributes(path: Path) -> tuple[int, int, int]:
    """Read (Orbit, Camera, CCD) from H5 attributes, with the legacy-layout fallback."""
    with h5py.File(path, "r") as file:
        attrs = file.attrs if "Orbit" in file.attrs else file["LightCurve"].attrs
        return int(attrs["Orbit"]), int(attrs["Camera"]), int(attrs["CCD"])


def group_paths_by_ccd(
    paths: list[Path],
) -> tuple[dict[tuple[int, int, int], list[Path]], list[Path]]:
    """
    Group paths by (orbit, camera, ccd) via path patterns, falling back to H5 attributes.

    Returns the groups plus a list of ungrouped paths (unreadable files with no recognizable
    path pattern), which are reported as read errors without any correction attempt.
    """
    groups: dict[tuple[int, int, int], list[Path]] = {}
    ungrouped = []
    needs_attributes = []
    for path in paths:
        posix_path = path.as_posix()
        orbit_match = audit.ORBIT_PATH_PATTERN.search(posix_path)
        camera_ccd_match = CAMERA_CCD_PATTERN.search(posix_path)
        if orbit_match and camera_ccd_match:
            group = (
                int(orbit_match.group(1)),
                int(camera_ccd_match.group(1)),
                int(camera_ccd_match.group(2)),
            )
            groups.setdefault(group, []).append(path)
        else:
            needs_attributes.append(path)
    if needs_attributes:
        logger.info(
            f"Reading H5 attributes from {len(needs_attributes)} files without standard paths"
        )
        for path in needs_attributes:
            try:
                groups.setdefault(read_group_attributes(path), []).append(path)
            except Exception:
                ungrouped.append(path)
    return groups, ungrouped


def find_source_directory(
    group: tuple[int, int, int], paths: list[Path], tglc_data_dir: Path | None
) -> Path:
    """Locate the Source pickle directory for an (orbit, camera, ccd) group."""
    orbit, camera, ccd = group
    if tglc_data_dir is not None:
        manifest = Manifest(tglc_data_dir, orbit=orbit, camera=camera, ccd=ccd)
        candidates = [manifest.source_directory]
    else:
        # Standard tree: .../orbit-N/ffi/camC/ccdD/{LC,source}/
        candidates = list(dict.fromkeys(path.parent.parent / "source" for path in paths))
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("source_*.pkl")):
            return candidate
    raise FileNotFoundError(
        f"No Source pickles found for orbit {orbit} camera {camera} CCD {ccd} "
        f"(searched: {', '.join(str(c) for c in candidates)})"
    )


def load_source_time_arrays(source_directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the cadence -> spacecraft TJD mapping from one Source pickle in the directory."""
    source_file = sorted(source_directory.glob("source_*.pkl"))[0]
    with source_file.open("rb") as source_pickle:
        source: Source = pickle.load(source_pickle)
    cadence = np.asarray(source.cadence, dtype=np.int64)
    tjd = np.asarray(source.time, dtype=np.float64)
    del source  # free the flux cube immediately
    if len(cadence) != len(tjd) or len(cadence) == 0:
        raise ValueError(f"Inconsistent cadence/time arrays in {source_file}")
    if np.any(np.diff(cadence) <= 0):
        order = np.argsort(cadence)
        cadence, tjd = cadence[order], tjd[order]
    return cadence, tjd


# ---------------------------------------------------------------------------------------------
# Per-file correction
# ---------------------------------------------------------------------------------------------


def fix_one_file(
    path: Path,
    group: tuple[int, int, int],
    source_cadence: np.ndarray,
    source_tjd: np.ndarray,
    position_au: np.ndarray,
    dry_run: bool,
) -> dict:
    """Recompute and overwrite one light curve's BJD dataset. Never raises."""
    row = dict.fromkeys(FIX_RESULT_COLUMNS)
    orbit, camera, ccd = group
    row.update(path=str(path), orbit_group=orbit, camera_group=camera, ccd_group=ccd)
    try:
        with h5py.File(path, "r" if dry_run else "r+") as file:
            attrs = file.attrs if "TIC ID" in file.attrs else file["LightCurve"].attrs
            row.update(
                tic_id=int(attrs["TIC ID"]),
                orbit_attr=int(attrs["Orbit"]),
                camera_attr=int(attrs["Camera"]),
                ccd_attr=int(attrs["CCD"]),
                ra=float(attrs["RA"]),
                dec=float(attrs["Dec"]),
            )
            if (row["orbit_attr"], row["camera_attr"], row["ccd_attr"]) != group:
                row.update(status="group_mismatch")
                return row

            cadence = np.asarray(file["LightCurve/Cadence"][:], dtype=np.int64)
            bjd_dataset = file["LightCurve/BJD"]
            old_bjd = np.asarray(bjd_dataset[:], dtype=np.float64)
            row.update(n_cadences=len(cadence))
            if len(cadence) == 0:
                row.update(status="empty")
                return row
            if len(old_bjd) != len(cadence):
                row.update(status="length_mismatch")
                return row

            indices = np.searchsorted(source_cadence, cadence)
            in_range = indices < len(source_cadence)
            matched = np.zeros(len(cadence), dtype=bool)
            matched[in_range] = source_cadence[indices[in_range]] == cadence[in_range]
            if not matched.all():
                row.update(
                    status="cadence_mismatch",
                    error=f"{np.count_nonzero(~matched)} cadences not found in Source",
                )
                return row

            legacy_jd = old_bjd.min() > 1e6
            row.update(bjd_format="jd" if legacy_jd else "tjd")
            spacecraft_time = Time(source_tjd[indices], format="tjd", scale="tdb")
            coordinate = SkyCoord(row["ra"], row["dec"], unit="deg")
            corrected = apply_barycentric_correction(
                spacecraft_time, coordinate, position_au[indices] * u.au
            )
            new_values = corrected.jd if legacy_jd else corrected.tjd

            delta_seconds = (new_values - old_bjd) * SECONDS_PER_DAY
            row.update(
                max_abs_delta_s=float(np.abs(delta_seconds).max()),
                mean_delta_s=float(delta_seconds.mean()),
            )
            if dry_run:
                row.update(status="dry_run")
                return row
            try:
                bjd_dataset[...] = new_values
            except Exception as exception:
                row.update(status="write_error", error=repr(exception))
                return row
            row.update(status="fixed")
    except Exception as exception:
        row.update(status="read_error", error=repr(exception))
    return row


def fix_chunk(
    paths: list[Path],
    group: tuple[int, int, int],
    source_cadence: np.ndarray,
    source_tjd: np.ndarray,
    position_au: np.ndarray,
    dry_run: bool,
) -> list[dict]:
    """Worker task: fix a chunk of files with shared per-group arrays."""
    return [
        fix_one_file(path, group, source_cadence, source_tjd, position_au, dry_run)
        for path in paths
    ]


# ---------------------------------------------------------------------------------------------
# Group driver
# ---------------------------------------------------------------------------------------------


def log_file_suffix(dry_run: bool) -> str:
    return ".dryrun.csv" if dry_run else ".csv"


def write_rows_csv(rows: list[dict], output_file: Path) -> None:
    results = pd.DataFrame(rows, columns=FIX_RESULT_COLUMNS).sort_values("path")
    temporary_file = output_file.with_name(output_file.name + ".tmp")
    results.to_csv(temporary_file, index=False)
    temporary_file.replace(output_file)


def write_group_error(
    results_dir: Path, group: tuple[int, int, int], status: str, message: str, dry_run: bool
) -> None:
    orbit, camera, ccd = group
    error_file = results_dir / (
        f"fix_orbit-{orbit:04d}_cam{camera}_ccd{ccd}.errors{log_file_suffix(dry_run)}"
    )
    pd.DataFrame(
        [{"orbit": orbit, "camera": camera, "ccd": ccd, "status": status, "error": message}]
    ).to_csv(error_file, index=False)
    logger.error(f"Orbit {orbit} cam {camera} ccd {ccd}: {status}: {message}")


def fix_group(group: tuple[int, int, int], paths: list[Path], args: argparse.Namespace) -> Counter:
    orbit, camera, ccd = group
    log_file = args.results_dir / (
        f"fix_orbit-{orbit:04d}_cam{camera}_ccd{ccd}{log_file_suffix(args.dry_run)}"
    )
    if log_file.exists() and not args.replace:
        logger.info(f"Orbit {orbit} cam {camera} ccd {ccd}: log exists, skipping "
                    f"({len(paths)} files)")  # fmt: skip
        return Counter(skipped_as_done=len(paths))

    try:
        source_directory = find_source_directory(group, paths, args.tglc_data_dir)
    except FileNotFoundError as exception:
        write_group_error(args.results_dir, group, "source_missing", str(exception), args.dry_run)
        return Counter(group_errors=1)
    try:
        source_cadence, source_tjd = load_source_time_arrays(source_directory)
    except Exception as exception:
        write_group_error(
            args.results_dir, group, "source_unreadable", repr(exception), args.dry_run
        )
        return Counter(group_errors=1)
    try:
        position = get_tess_spacecraft_position(
            orbit, Time(source_tjd, format="tjd", scale="tdb"), args.ephemerides_dir
        )
    except (HorizonsError, ValueError) as exception:
        write_group_error(
            args.results_dir, group, "ephemeris_unavailable", str(exception), args.dry_run
        )
        return Counter(group_errors=1)

    task = partial(
        fix_chunk,
        group=group,
        source_cadence=source_cadence,
        source_tjd=source_tjd,
        position_au=position.to_value(u.au),
        dry_run=args.dry_run,
    )
    chunks = [paths[i : i + args.chunk_size] for i in range(0, len(paths), args.chunk_size)]
    rows = []
    for chunk_rows in tqdm(
        pool_map_if_multiprocessing(
            task, chunks, nprocs=args.nprocs, pool_map_method="imap_unordered"
        ),
        total=len(chunks),
        desc=f"orbit {orbit} cam {camera} ccd {ccd}",
        unit="chunk",
    ):
        rows.extend(chunk_rows)
    write_rows_csv(rows, log_file)

    counters = Counter(row["status"] for row in rows)
    worst = max(
        (row for row in rows if row["max_abs_delta_s"] is not None),
        key=lambda row: row["max_abs_delta_s"],
        default=None,
    )
    if worst is not None:
        counters["_max_delta_s"] = worst["max_abs_delta_s"]
        logger.info(
            f"Orbit {orbit} cam {camera} ccd {ccd}: {counters.get('fixed', 0)} fixed, "
            f"{counters.get('dry_run', 0)} dry-run, max |delta| = "
            f"{worst['max_abs_delta_s']:.3f} s ({worst['path']})"
        )
    return counters


def run_fix(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.ephemerides_dir is None:
        args.ephemerides_dir = args.results_dir / "ephemerides"

    paths = audit.collect_input_paths(args.inputs)
    logger.info(f"Collected {len(paths)} unique H5 paths")
    groups, ungrouped = group_paths_by_ccd(paths)
    if args.orbit:
        groups = {group: files for group, files in groups.items() if group[0] in args.orbit}
    logger.info(f"Fixing {sum(map(len, groups.values()))} files across {len(groups)} "
                "(orbit, camera, ccd) groups")  # fmt: skip

    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_revision": audit.get_git_revision(),
        "tglc_version": tglc_version,
        "dry_run": args.dry_run,
        "ephemerides_dir": str(args.ephemerides_dir),
        "inputs": [str(path) for path in args.inputs],
        "n_files": len(paths),
        "n_groups": len(groups),
        "argv": sys.argv,
    }
    meta_name = "fix_meta.dryrun.json" if args.dry_run else "fix_meta.json"
    (args.results_dir / meta_name).write_text(json.dumps(metadata, indent=2))

    if ungrouped:
        write_rows_csv(
            [
                dict.fromkeys(FIX_RESULT_COLUMNS)
                | {
                    "path": str(path),
                    "status": "read_error",
                    "error": "could not determine (orbit, camera, ccd) from path or attributes",
                }  # fmt: skip
                for path in ungrouped
            ],
            args.results_dir / f"fix_ungrouped{log_file_suffix(args.dry_run)}",
        )
        logger.warning(f"{len(ungrouped)} files could not be grouped; see fix_ungrouped csv")

    totals = Counter(read_error=len(ungrouped))
    max_delta = 0.0
    for group in sorted(groups):
        counters = fix_group(group, groups[group], args)
        max_delta = max(max_delta, counters.pop("_max_delta_s", 0.0))
        totals.update(counters)

    logger.info("=== Fix summary ===")
    for status, count in sorted(totals.items()):
        logger.info(f"  {status}: {count:,}")
    logger.info(f"  max |delta| applied{' (dry run)' if args.dry_run else ''}: {max_delta:.3f} s")


# ---------------------------------------------------------------------------------------------
# Synthetic verification dataset
# ---------------------------------------------------------------------------------------------

SYNTHETIC_ORBIT = 185
SYNTHETIC_SECTOR = 89
SYNTHETIC_INJECTED_OFFSET_S = 0.5


def make_fake_ephemeris_cache(ephemerides_dir: Path, orbit: int, tjd: np.ndarray) -> None:
    """Write a smooth analytic per-orbit ephemeris cache covering the tjd span plus margins."""
    jd = np.arange(tjd.min() + audit.TJD_OFFSET - 3.0, tjd.max() + audit.TJD_OFFSET + 3.0, 1 / 24)
    phase = 2 * np.pi * (jd - jd[0]) / 365.25
    cache = pd.DataFrame(
        {
            "JDTDB": jd,
            "Calendar Date (TDB)": "synthetic",
            "X": -0.8 + 0.02 * np.sin(phase * 27),
            "Y": 0.55 + 0.02 * np.cos(phase * 27),
            "Z": 0.24 + 0.01 * np.sin(phase * 13),
            "LT": 0.0057,
            "RG": 1.0,
            "RR": 0.0,
        }
    )
    ephemerides_dir.mkdir(parents=True, exist_ok=True)
    cache.to_csv(ephemerides_dir / f"tess_ephem_orbit-{orbit:04d}.csv", index=False)


def expected_corrected_tjd(
    ephemerides_dir: Path, orbit: int, tjd: np.ndarray, ra: float, dec: float
) -> np.ndarray:
    """Independently compute the correct BTJD (plain numpy, no tglc correction code)."""
    cache = pd.read_csv(ephemerides_dir / f"tess_ephem_orbit-{orbit:04d}.csv")
    jd = tjd + audit.TJD_OFFSET
    position = np.array([np.interp(jd, cache["JDTDB"], cache[axis]) for axis in ["X", "Y", "Z"]]).T
    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    star = np.array(
        [np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)]
    )
    light_time_days = position.dot(star) * audit.AU_IN_LIGHTSECONDS / SECONDS_PER_DAY
    return tjd + light_time_days


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_make_synthetic(args: argparse.Namespace) -> None:
    """
    Build an offline synthetic dataset whose H5 BJDs are wrong by a known 0.5 s, alongside a
    synthetic Source pickle carrying the true spacecraft times and a fake ephemeris cache.
    """
    orbit = SYNTHETIC_ORBIT
    rng_tjd = np.linspace(3722.0, 3735.7, 240)
    source_cadence = np.arange(90000, 90000 + len(rng_tjd), dtype=np.int64)
    offset_days = SYNTHETIC_INJECTED_OFFSET_S / SECONDS_PER_DAY

    ephemerides_dir = args.outdir / "ephemerides"
    make_fake_ephemeris_cache(ephemerides_dir, orbit, rng_tjd)

    ccd_dir = args.outdir / "data" / f"orbit-{orbit}" / "ffi" / "cam1" / "ccd1"
    lc_dir = ccd_dir / "LC"
    source_dir = ccd_dir / "source"
    lc_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    source = Source.__new__(Source)
    source.cadence = source_cadence
    source.time = rng_tjd
    with (source_dir / "source_0_0.pkl").open("wb") as source_pickle:
        pickle.dump(source, source_pickle)

    expected_dir = args.outdir / "expected"
    expected_dir.mkdir(parents=True, exist_ok=True)
    expectations: dict[str, dict] = {}

    def make_wrong_lightcurve(tic_id, ra, dec, cadence_subset=None, camera=1):
        indices = np.arange(len(rng_tjd)) if cadence_subset is None else cadence_subset
        true_btjd = expected_corrected_tjd(ephemerides_dir, orbit, rng_tjd[indices], ra, dec)
        path = lc_dir / f"{tic_id}.h5"
        audit.make_synthetic_lightcurve(
            path, tic_id, orbit, SYNTHETIC_SECTOR, ra=ra, dec=dec,
            tjd=true_btjd + offset_days, cadence=source_cadence[indices], camera=camera,
        )  # fmt: skip
        np.save(expected_dir / f"{tic_id}.npy", true_btjd)
        return path

    # Standard file, full cadence set
    make_wrong_lightcurve(1000, ra=30.0, dec=-20.0)
    expectations["1000"] = {"status": "fixed", "expected": "1000.npy", "format": "tjd"}
    # Subset of cadences (every 3rd) exercises the searchsorted mapping
    make_wrong_lightcurve(1001, ra=200.0, dec=45.0, cadence_subset=np.arange(0, len(rng_tjd), 3))
    expectations["1001"] = {"status": "fixed", "expected": "1001.npy", "format": "tjd"}
    # Legacy full-JD file: shift stored values to JD after writing
    path = make_wrong_lightcurve(1002, ra=310.0, dec=5.0)
    with h5py.File(path, "r+") as file:
        file["LightCurve/BJD"][...] = file["LightCurve/BJD"][:] + audit.TJD_OFFSET
    expectations["1002"] = {"status": "fixed", "expected": "1002.npy", "format": "jd"}
    # Legacy attribute layout: attrs on the LightCurve group instead of the file root
    path = make_wrong_lightcurve(1003, ra=110.0, dec=-60.0)
    with h5py.File(path, "r+") as file:
        for key in list(file.attrs):
            file["LightCurve"].attrs[key] = file.attrs[key]
            del file.attrs[key]
    expectations["1003"] = {"status": "fixed", "expected": "1003.npy", "format": "tjd"}
    # Cadences that don't exist in the Source
    audit.make_synthetic_lightcurve(
        lc_dir / "1004.h5", 1004, orbit, SYNTHETIC_SECTOR, ra=10.0, dec=10.0,
        tjd=rng_tjd[:50], cadence=np.arange(500, 550),
    )  # fmt: skip
    expectations["1004"] = {"status": "cadence_mismatch"}
    # Camera attribute disagrees with the cam1 path
    audit.make_synthetic_lightcurve(
        lc_dir / "1005.h5", 1005, orbit, SYNTHETIC_SECTOR, ra=10.0, dec=10.0,
        tjd=rng_tjd, cadence=source_cadence, camera=2,
    )  # fmt: skip
    expectations["1005"] = {"status": "group_mismatch"}
    # Corrupt file
    (lc_dir / "1006.h5").write_bytes(b"this is not an HDF5 file")
    expectations["1006"] = {"status": "read_error"}

    # A second orbit with light curves but no Source pickles at all
    orphan_lc_dir = args.outdir / "data" / "orbit-186" / "ffi" / "cam1" / "ccd1" / "LC"
    orphan_lc_dir.mkdir(parents=True, exist_ok=True)
    audit.make_synthetic_lightcurve(
        orphan_lc_dir / "1100.h5", 1100, 186, SYNTHETIC_SECTOR, ra=10.0, dec=10.0,
        tjd=rng_tjd, cadence=source_cadence,
    )  # fmt: skip
    expectations["1100"] = {"status": "source_missing_group"}

    # Files the fixer must never modify
    hashes = {name: sha256_of(lc_dir / f"{name}.h5") for name in ["1004", "1005", "1006"]}
    hashes["1100"] = sha256_of(orphan_lc_dir / "1100.h5")
    (args.outdir / "hashes.json").write_text(json.dumps(hashes, indent=2))
    (args.outdir / "expectations.json").write_text(json.dumps(expectations, indent=2))

    print(f"Synthetic dataset written to {args.outdir}")
    print("Run:")
    print(
        f"  python scripts/fix_lightcurve_timing.py fix {args.outdir}/data"
        f" --results-dir {args.outdir}/results --ephemerides-dir {args.outdir}/ephemerides"
        f" --nprocs 2"
    )
    print(f"  python scripts/fix_lightcurve_timing.py verify-synthetic {args.outdir}")


def run_verify_synthetic(args: argparse.Namespace) -> None:
    """Check a fixed synthetic dataset against saved expectations; exit 1 on any failure."""
    expectations = json.loads((args.outdir / "expectations.json").read_text())
    hashes = json.loads((args.outdir / "hashes.json").read_text())
    results_dir = args.outdir / "results"
    logs = pd.concat(
        [pd.read_csv(file) for file in results_dir.glob("fix_orbit-*.csv")
         if not file.name.endswith(".errors.csv")],
        ignore_index=True,
    )  # fmt: skip
    logs["tic"] = logs["path"].str.extract(r"(\d+)\.h5$")

    failures = []

    def check(condition: bool, message: str):
        (logger.info if condition else failures.append)(
            f"PASS: {message}" if condition else message
        )

    for tic, expectation in expectations.items():
        if expectation["status"] == "source_missing_group":
            error_files = list(results_dir.glob("fix_orbit-0186_*.errors.csv"))
            check(len(error_files) == 1, f"TIC {tic}: orbit-186 source_missing error record")
            if error_files:
                check(
                    pd.read_csv(error_files[0]).iloc[0]["status"] == "source_missing",
                    f"TIC {tic}: error status is source_missing",
                )
            continue
        matching = logs[logs["tic"] == tic]
        check(len(matching) == 1, f"TIC {tic}: exactly one log row")
        if len(matching) != 1:
            continue
        row = matching.iloc[0]
        check(row["status"] == expectation["status"],
              f"TIC {tic}: status {row['status']} == {expectation['status']}")  # fmt: skip
        if "expected" in expectation:
            expected_tjd = np.load(args.outdir / "expected" / expectation["expected"])
            with h5py.File(row["path"], "r") as file:
                stored = np.asarray(file["LightCurve/BJD"][:], dtype=np.float64)
            if expectation["format"] == "jd":
                stored = stored - audit.TJD_OFFSET
            max_residual = np.abs(stored - expected_tjd).max()
            check(
                max_residual < 1e-9,
                f"TIC {tic}: corrected BJD matches expectation (max residual {max_residual:.2e} d)",
            )
            check(
                abs(row["max_abs_delta_s"] - SYNTHETIC_INJECTED_OFFSET_S) < 0.01,
                f"TIC {tic}: applied delta {row['max_abs_delta_s']:.4f} s ~= "
                f"{SYNTHETIC_INJECTED_OFFSET_S} s",
            )
    for tic, expected_hash in hashes.items():
        matching = logs[logs["tic"] == tic]
        path = Path(matching.iloc[0]["path"]) if len(matching) else next(
            (args.outdir / "data").rglob(f"{tic}.h5")
        )  # fmt: skip
        check(sha256_of(path) == expected_hash, f"TIC {tic}: untouched file bytes unchanged")

    if failures:
        for failure in failures:
            logger.error(f"FAIL: {failure}")
        sys.exit(1)
    logger.info("verify-synthetic: ALL CHECKS PASSED")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    fix_parser = subparsers.add_parser("fix", help="Correct H5 light curve BJDs in place")
    fix_parser.add_argument(
        "inputs", type=Path, nargs="+", help="Directories (recursive *.h5) or manifest text files"
    )
    fix_parser.add_argument("--results-dir", type=Path, required=True)
    fix_parser.add_argument(
        "--ephemerides-dir",
        type=Path,
        default=None,
        help="New-ephemeris cache (default: RESULTS_DIR/ephemerides; point "
        "at the pipeline's cache to avoid any Horizons queries)",
    )
    fix_parser.add_argument(
        "--tglc-data-dir",
        type=Path,
        default=None,
        help="Locate Source pickles via the standard Manifest layout "
        "instead of relative to each H5 file",
    )
    fix_parser.add_argument("--nprocs", type=int, default=1)
    fix_parser.add_argument("--chunk-size", type=int, default=256)
    fix_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log corrections without writing anything",
    )
    fix_parser.add_argument(
        "--replace", action="store_true", help="Redo groups with existing fix logs"
    )
    fix_parser.add_argument(
        "--orbit", type=int, nargs="+", default=None, help="Restrict to these orbits"
    )
    fix_parser.set_defaults(func=run_fix)

    synthetic_parser = subparsers.add_parser(
        "make-synthetic", help="Generate a synthetic verification dataset"
    )
    synthetic_parser.add_argument("outdir", type=Path)
    synthetic_parser.set_defaults(func=run_make_synthetic)

    verify_parser = subparsers.add_parser(
        "verify-synthetic", help="Verify a fixed synthetic dataset"
    )
    verify_parser.add_argument("outdir", type=Path)
    verify_parser.set_defaults(func=run_verify_synthetic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
