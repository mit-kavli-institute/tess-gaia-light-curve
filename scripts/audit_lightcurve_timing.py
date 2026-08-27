"""
Audit existing TGLC H5 light curves for barycentric timing errors caused by the old
ephemeris interpolation.

Background
----------
Light curves produced before commit d55ae79 applied a barycentric correction using spacecraft
positions interpolated from bundled yearly ephemeris CSVs with `np.interp`, which silently
clamps outside a file's JDTDB range and interpolates linearly across any internal gaps. In
addition, the shipped `20260401_tess_ephem.csv` (sectors 102-115) is malformed (12 data fields
under a 9-field header), so positions for those sectors were read from shifted columns.

For each H5 light curve this script computes the timing discrepancy

    dt(t) = ((P_new(t) - P_old(t)) . u_star) / c

where `P_old` reproduces the shipped interpolation literally (including the malformed parse)
and `P_new` interpolates the per-orbit JPL Horizons ephemeris used by the current pipeline.
The corrected BJD stored in the file is used as the evaluation epoch; the difference from the
spacecraft epoch is second-order and negligible for flagging purposes.

Subcommands
-----------
scan            Scan H5 files (directories or manifest lists), write per-orbit results CSVs.
report          Aggregate results CSVs into a PDF report (requires pylatex + a LaTeX toolchain).
make-synthetic  Generate a small synthetic dataset with known injected errors for verification.

Dependencies beyond the tglc environment: `pylatex` (report only; install into the venv with
`pip install pylatex`). Not added to pyproject.toml -- this directory is diagnostics-only.
"""

import argparse
from datetime import UTC, datetime
from functools import partial
import json
import logging
from pathlib import Path
import re
import subprocess
import sys

from astropy.time import Time
import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# Make tglc importable when running this script directly from a repository checkout, without
# requiring the package to be installed in the environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tglc import __version__ as tglc_version  # noqa: E402
from tglc.utils.constants import get_sector_containing_orbit  # noqa: E402
from tglc.utils.mapping import pool_map_if_multiprocessing  # noqa: E402
from tglc.utils.tess_ephemeris import HorizonsError, get_spacecraft_ephemeris  # noqa: E402


logger = logging.getLogger("audit_lightcurve_timing")

SCRIPT_DIRECTORY = Path(__file__).parent
DEFAULT_OLD_EPHEMERIDES_DIRECTORY = SCRIPT_DIRECTORY / "data" / "old_ephemerides"
TJD_OFFSET = 2457000.0
AU_IN_LIGHTSECONDS = (1 * u.au).to_value(u.lightsecond)
ORBIT_PATH_PATTERN = re.compile(r"orbit-(\d+)")
PROBE_FILE_COUNT = 20

# Single-hue figure styling: blue for data marks, dark red reserved for the threshold
# reference line (always labeled, never color-alone), neutral ink for text/grid.
FIGURE_DATA_COLOR = "#3572b0"
FIGURE_THRESHOLD_COLOR = "#8f1f1f"
FIGURE_GRID_COLOR = "#d5d5d5"

RESULT_COLUMNS = [
    "path", "status", "tic_id", "orbit_attr", "sector_attr", "camera", "ccd",
    "orbit_group", "sector_used", "ra", "dec", "n_cadences", "bjd_min", "bjd_max",
    "bjd_format", "max_abs_dt_s", "mean_dt_s", "rms_dt_s", "n_above_threshold",
    "threshold_s", "old_file", "old_clamped_low", "old_clamped_high",
    "old_jdtdb_monotonic", "old_max_gap_days", "error",
]  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Old (pre-d55ae79) ephemeris interpolation, reproduced literally
# ---------------------------------------------------------------------------------------------


def get_old_ephemeris_file_path(sector: int, old_ephemerides_directory: Path) -> Path:
    """
    Sector-to-file mapping as shipped in the old `tglc.utils.tess_ephemeris`, verbatim.

    The `19 <= sector <= 32` branch is unreachable for sector 19 (the previous branch wins);
    this quirk is preserved deliberately to reproduce production behavior.
    """
    if 1 <= sector <= 5:
        return old_ephemerides_directory / "20180720_tess_ephem.csv"
    elif 6 <= sector <= 19:
        return old_ephemerides_directory / "20190101_tess_ephem.csv"
    elif 19 <= sector <= 32:
        return old_ephemerides_directory / "20200101_tess_ephem.csv"
    elif 33 <= sector <= 45:
        return old_ephemerides_directory / "20210101_tess_ephem.csv"
    elif 46 <= sector <= 59:
        return old_ephemerides_directory / "20211215_tess_ephem.csv"
    elif 60 <= sector <= 73:
        return old_ephemerides_directory / "20221201_tess_ephem.csv"
    elif 74 <= sector <= 87:
        return old_ephemerides_directory / "20231201_tess_ephem.csv"
    elif 88 <= sector <= 101:
        return old_ephemerides_directory / "20241201_tess_ephem.csv"
    elif 102 <= sector <= 115:
        return old_ephemerides_directory / "20260401_tess_ephem.csv"
    else:
        raise ValueError(f"No spacecraft ephemeris file assigned for sector {sector}.")


def load_old_ephemeris(sector: int, old_ephemerides_directory: Path) -> dict:
    """
    Load the old ephemeris table for a sector exactly as the old pipeline code parsed it.

    For the malformed 20260401 file this intentionally yields shifted columns (the "JDTDB"
    column contains Y-position values), because that is what production used.
    """
    file_path = get_old_ephemeris_file_path(sector, old_ephemerides_directory)
    ephemeris = pd.read_csv(file_path, comment="#")
    jd = np.asarray(ephemeris["JDTDB"], dtype=np.float64)
    return {
        "file": file_path.stem.removesuffix("_tess_ephem"),
        "jd": jd,
        "x": np.asarray(ephemeris["X"], dtype=np.float64),
        "y": np.asarray(ephemeris["Y"], dtype=np.float64),
        "z": np.asarray(ephemeris["Z"], dtype=np.float64),
        "monotonic": bool(np.all(np.diff(jd) > 0)),
    }


# ---------------------------------------------------------------------------------------------
# Input collection and orbit grouping
# ---------------------------------------------------------------------------------------------


def collect_input_paths(inputs: list[Path]) -> list[Path]:
    """Expand directories (recursive *.h5 glob) and manifest files into a deduplicated list."""
    paths = []
    for input_path in inputs:
        if input_path.is_dir():
            found = sorted(input_path.rglob("*.h5"))
            logger.info(f"Found {len(found)} H5 files under {input_path}")
            paths.extend(found)
        elif input_path.is_file():
            with input_path.open() as manifest:
                lines = [
                    Path(line.strip())
                    for line in manifest
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            logger.info(f"Read {len(lines)} paths from manifest {input_path}")
            paths.extend(lines)
        else:
            raise FileNotFoundError(f"Input {input_path} is neither a directory nor a file")
    return list(dict.fromkeys(paths))


def read_orbit_attribute(path: Path) -> int:
    """Read the Orbit attribute from an H5 file (root attrs, falling back to legacy layout)."""
    with h5py.File(path, "r") as file:
        attrs = file.attrs if "Orbit" in file.attrs else file["LightCurve"].attrs
        return int(attrs["Orbit"])


def group_paths_by_orbit(paths: list[Path]) -> dict[int, list[Path]]:
    """
    Group paths by orbit using the `orbit-NNN` path component; files without the pattern get
    a (slower) H5 attribute read. Unreadable pattern-less files are grouped under orbit -1 so
    they still produce `read_error` rows.
    """
    groups: dict[int, list[Path]] = {}
    patternless = []
    for path in paths:
        match = ORBIT_PATH_PATTERN.search(str(path))
        if match:
            groups.setdefault(int(match.group(1)), []).append(path)
        else:
            patternless.append(path)
    if patternless:
        logger.info(f"Reading Orbit attribute from {len(patternless)} files without orbit paths")
        for path in patternless:
            try:
                orbit = read_orbit_attribute(path)
            except Exception:
                orbit = -1
            groups.setdefault(orbit, []).append(path)
    return groups


# ---------------------------------------------------------------------------------------------
# Per-file scan
# ---------------------------------------------------------------------------------------------


def read_lightcurve_h5(path: Path) -> dict:
    """
    Read attributes and the BJD array from a TGLC H5 light curve.

    Handles both the current layout (attrs on the file root) and the pre-03601b6 layout
    (attrs on the LightCurve group), and normalizes pre-fe6513a files that stored full JD
    instead of TJD in the BJD dataset.
    """
    with h5py.File(path, "r") as file:
        attrs = file.attrs if "TIC ID" in file.attrs else file["LightCurve"].attrs
        bjd = np.asarray(file["LightCurve/BJD"][:], dtype=np.float64)
        data = {
            "tic_id": int(attrs["TIC ID"]),
            "orbit_attr": int(attrs["Orbit"]),
            "sector_attr": int(attrs["Sector"]),
            "camera": int(attrs["Camera"]),
            "ccd": int(attrs["CCD"]),
            "ra": float(attrs["RA"]),
            "dec": float(attrs["Dec"]),
        }
    if len(bjd) > 0 and bjd.min() > 1e6:
        data["bjd_format"] = "jd"
        data["jd"] = bjd
    else:
        data["bjd_format"] = "tjd"
        data["jd"] = bjd + TJD_OFFSET
    return data


def scan_one_file(
    path: Path,
    orbit: int,
    sector: int,
    old_ephemeris: dict,
    new_jd: np.ndarray,
    new_xyz: np.ndarray,
    threshold_s: float,
) -> dict:
    """Compute timing-discrepancy statistics for one light curve. Never raises."""
    row = dict.fromkeys(RESULT_COLUMNS)
    row.update(
        path=str(path),
        orbit_group=orbit,
        sector_used=sector,
        threshold_s=threshold_s,
        old_file=old_ephemeris["file"],
        old_jdtdb_monotonic=old_ephemeris["monotonic"],
    )
    try:
        lightcurve = read_lightcurve_h5(path)
    except Exception as exception:
        row.update(status="read_error", error=repr(exception))
        return row
    jd = lightcurve.pop("jd")
    row.update(lightcurve)

    if len(jd) == 0:
        row.update(status="empty_bjd")
        return row
    row.update(n_cadences=len(jd), bjd_min=jd.min() - TJD_OFFSET, bjd_max=jd.max() - TJD_OFFSET)

    if lightcurve["orbit_attr"] != orbit:
        row.update(status="orbit_mismatch")
        return row
    if jd.min() < new_jd[0] or jd.max() > new_jd[-1]:
        row.update(status="new_ephemeris_gap")
        return row

    ra = np.radians(row["ra"])
    dec = np.radians(row["dec"])
    star_unit_vector = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

    old_jd = old_ephemeris["jd"]
    position_difference_dot_star = np.zeros_like(jd)
    for axis_index, axis in enumerate(["x", "y", "z"]):
        old_position = np.interp(jd, old_jd, old_ephemeris[axis])
        new_position = np.interp(jd, new_jd, new_xyz[axis_index])
        position_difference_dot_star += (new_position - old_position) * star_unit_vector[axis_index]
    dt_seconds = position_difference_dot_star * AU_IN_LIGHTSECONDS

    row.update(
        status="ok",
        max_abs_dt_s=np.abs(dt_seconds).max(),
        mean_dt_s=dt_seconds.mean(),
        rms_dt_s=np.sqrt(np.mean(dt_seconds**2)),
        n_above_threshold=int(np.count_nonzero(np.abs(dt_seconds) > threshold_s)),
        old_clamped_low=bool(jd.min() < old_jd[0]) if old_ephemeris["monotonic"] else False,
        old_clamped_high=bool(jd.max() > old_jd[-1]) if old_ephemeris["monotonic"] else False,
    )
    if old_ephemeris["monotonic"]:
        low, high = np.searchsorted(old_jd, [jd.min(), jd.max()])
        surrounding = old_jd[max(low - 1, 0) : min(high + 1, len(old_jd))]
        if len(surrounding) > 1:
            row.update(old_max_gap_days=np.diff(surrounding).max())
    return row


def scan_chunk(
    paths: list[Path],
    orbit: int,
    sector: int,
    old_ephemeris: dict,
    new_jd: np.ndarray,
    new_xyz: np.ndarray,
    threshold_s: float,
) -> list[dict]:
    """Worker task: scan a chunk of files with shared per-orbit ephemeris arrays."""
    return [
        scan_one_file(path, orbit, sector, old_ephemeris, new_jd, new_xyz, threshold_s)
        for path in paths
    ]


# ---------------------------------------------------------------------------------------------
# Scan driver
# ---------------------------------------------------------------------------------------------


def probe_orbit_jd_span(paths: list[Path]) -> tuple[float, float]:
    """Get the overall JD span from the first few readable light curves in an orbit."""
    jd_min, jd_max = np.inf, -np.inf
    readable = 0
    for path in paths:
        try:
            jd = read_lightcurve_h5(path)["jd"]
        except Exception:
            continue
        if len(jd) > 0:
            jd_min, jd_max = min(jd_min, jd.min()), max(jd_max, jd.max())
            readable += 1
        if readable >= PROBE_FILE_COUNT:
            break
    if not np.isfinite(jd_min):
        raise ValueError("No readable light curves to establish the orbit time span")
    return jd_min, jd_max


def write_orbit_error(results_dir: Path, orbit: int, status: str, message: str) -> None:
    error_file = results_dir / f"orbit-{orbit:04d}.errors.csv"
    pd.DataFrame([{"orbit": orbit, "status": status, "error": message}]).to_csv(
        error_file, index=False
    )
    logger.error(f"Orbit {orbit}: {status}: {message}")


def scan_orbit(orbit: int, paths: list[Path], args: argparse.Namespace) -> None:
    results_file = args.results_dir / f"orbit-{orbit:04d}.csv"
    if results_file.exists() and not args.replace:
        logger.info(f"Orbit {orbit}: results exist, skipping ({len(paths)} files)")
        return

    try:
        sector = get_sector_containing_orbit(orbit)
    except ValueError as exception:
        write_orbit_error(args.results_dir, orbit, "sector_unknown", str(exception))
        return
    try:
        old_ephemeris = load_old_ephemeris(sector, args.old_ephemerides_dir)
    except (ValueError, FileNotFoundError) as exception:
        write_orbit_error(args.results_dir, orbit, "old_ephemeris_unavailable", str(exception))
        return
    try:
        jd_min, jd_max = probe_orbit_jd_span(paths)
        new_ephemeris = get_spacecraft_ephemeris(
            orbit,
            Time(jd_min, format="jd", scale="tdb"),
            Time(jd_max, format="jd", scale="tdb"),
            args.ephemerides_dir,
        )
    except (HorizonsError, ValueError) as exception:
        write_orbit_error(args.results_dir, orbit, "new_ephemeris_unavailable", str(exception))
        return

    task = partial(
        scan_chunk,
        orbit=orbit,
        sector=sector,
        old_ephemeris=old_ephemeris,
        new_jd=np.asarray(new_ephemeris["JDTDB"], dtype=np.float64),
        new_xyz=np.array(
            [new_ephemeris["X"], new_ephemeris["Y"], new_ephemeris["Z"]], dtype=np.float64
        ),
        threshold_s=args.threshold,
    )
    chunks = [paths[i : i + args.chunk_size] for i in range(0, len(paths), args.chunk_size)]
    rows = []
    for chunk_rows in tqdm(
        pool_map_if_multiprocessing(
            task, chunks, nprocs=args.nprocs, pool_map_method="imap_unordered"
        ),
        total=len(chunks),
        desc=f"orbit {orbit}",
        unit="chunk",
    ):
        rows.extend(chunk_rows)

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values("path")
    temporary_file = results_file.with_suffix(".csv.tmp")
    results.to_csv(temporary_file, index=False)
    temporary_file.replace(results_file)
    affected = int((results["max_abs_dt_s"] > args.threshold).sum())
    logger.info(f"Orbit {orbit}: scanned {len(results)} files, {affected} above threshold")


def get_git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIRECTORY,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def run_scan(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.ephemerides_dir is None:
        args.ephemerides_dir = args.results_dir / "ephemerides"

    paths = collect_input_paths(args.inputs)
    logger.info(f"Collected {len(paths)} unique H5 paths")
    groups = group_paths_by_orbit(paths)
    if args.orbit:
        groups = {orbit: group for orbit, group in groups.items() if orbit in args.orbit}
    logger.info(f"Scanning {sum(map(len, groups.values()))} files across {len(groups)} orbits")

    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_revision": get_git_revision(),
        "tglc_version": tglc_version,
        "threshold_s": args.threshold,
        "inputs": [str(path) for path in args.inputs],
        "n_files": len(paths),
        "n_orbits": len(groups),
        "argv": sys.argv,
    }
    (args.results_dir / "scan_meta.json").write_text(json.dumps(metadata, indent=2))

    for orbit in sorted(groups):
        scan_orbit(orbit, groups[orbit], args)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all per-orbit results and orbit-level error records."""
    result_files = sorted(results_dir.glob("orbit-*.csv"))
    error_files = [file for file in result_files if file.name.endswith(".errors.csv")]
    result_files = [file for file in result_files if not file.name.endswith(".errors.csv")]
    if not result_files and not error_files:
        raise FileNotFoundError(f"No orbit-*.csv results found in {results_dir}")
    results = (
        pd.concat([pd.read_csv(file) for file in result_files], ignore_index=True)
        if result_files
        else pd.DataFrame(columns=RESULT_COLUMNS)
    )
    errors = (
        pd.concat([pd.read_csv(file) for file in error_files], ignore_index=True)
        if error_files
        else pd.DataFrame(columns=["orbit", "status", "error"])
    )
    return results, errors


def make_histogram_figure(results: pd.DataFrame, threshold: float, output_file: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_abs_dt = results.loc[results["status"] == "ok", "max_abs_dt_s"].to_numpy()
    clipped = np.clip(max_abs_dt, 1e-7, None)
    upper_decade = max(np.log10(clipped.max()) + 0.5, np.log10(threshold) + 1.5)
    bins = np.logspace(-7, upper_decade, 90)

    figure, axes = plt.subplots(figsize=(7.5, 4.0))
    axes.hist(clipped, bins=bins, color=FIGURE_DATA_COLOR, edgecolor="white", linewidth=0.3)
    axes.axvline(threshold, color=FIGURE_THRESHOLD_COLOR, linestyle="--", linewidth=1.5)
    axes.text(
        threshold,
        0.95,
        f"  threshold = {threshold:g} s",
        color=FIGURE_THRESHOLD_COLOR,
        transform=axes.get_xaxis_transform(),
        va="top",
        fontsize=9,
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel(r"Max $|\Delta t|$ per light curve (s)")
    axes.set_ylabel("Light curves")
    axes.grid(True, which="major", color=FIGURE_GRID_COLOR, linewidth=0.5)
    axes.set_axisbelow(True)
    for spine in ["top", "right"]:
        axes.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_file)
    plt.close(figure)


def compute_dt_series(
    path: Path, orbit: int, old_ephemerides_dir: Path, ephemerides_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the full dt(t) series for one light curve (used for case-study figures)."""
    lightcurve = read_lightcurve_h5(path)
    jd = lightcurve["jd"]
    sector = get_sector_containing_orbit(orbit)
    old_ephemeris = load_old_ephemeris(sector, old_ephemerides_dir)
    new_ephemeris = get_spacecraft_ephemeris(
        orbit,
        Time(jd.min(), format="jd", scale="tdb"),
        Time(jd.max(), format="jd", scale="tdb"),
        ephemerides_dir,
    )
    ra, dec = np.radians(lightcurve["ra"]), np.radians(lightcurve["dec"])
    star_unit_vector = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    dt_seconds = np.zeros_like(jd)
    new_jd = np.asarray(new_ephemeris["JDTDB"], dtype=np.float64)
    for axis_index, axis in enumerate(["x", "y", "z"]):
        old_position = np.interp(jd, old_ephemeris["jd"], old_ephemeris[axis])
        new_position = np.interp(jd, new_jd, np.asarray(new_ephemeris[axis.upper()], float))
        dt_seconds += (new_position - old_position) * star_unit_vector[axis_index]
    return jd - TJD_OFFSET, dt_seconds * AU_IN_LIGHTSECONDS


def make_case_study_figure(row: pd.Series, args: argparse.Namespace, output_file: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        tjd, dt_seconds = compute_dt_series(
            Path(row["path"]), int(row["orbit_group"]), args.old_ephemerides_dir,
            args.ephemerides_dir,
        )  # fmt: skip
    except Exception as exception:
        logger.warning(f"Skipping case study for {row['path']}: {exception!r}")
        return False

    figure, axes = plt.subplots(figsize=(7.5, 3.5))
    axes.plot(tjd, dt_seconds, color=FIGURE_DATA_COLOR, linewidth=1.5)
    axes.axhline(0, color=FIGURE_GRID_COLOR, linewidth=0.8)
    axes.set_xlabel("BJD - 2457000 (TDB)")
    axes.set_ylabel(r"$\Delta t$ (s)")
    axes.set_title(
        f"TIC {int(row['tic_id'])} — orbit {int(row['orbit_group'])}, "
        f"sector {int(row['sector_used'])}",
        fontsize=10,
    )
    axes.grid(True, color=FIGURE_GRID_COLOR, linewidth=0.5)
    axes.set_axisbelow(True)
    for spine in ["top", "right"]:
        axes.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_file)
    plt.close(figure)
    return True


def diagnose_group(group: pd.DataFrame) -> str:
    if group["old_jdtdb_monotonic"].eq(False).any():
        return "malformed"
    if group["old_clamped_low"].eq(True).any() or group["old_clamped_high"].eq(True).any():
        return "clamped"
    return "--"


def run_report(args: argparse.Namespace) -> None:
    from pylatex import (
        Command,
        Document,
        Figure,
        LongTable,
        NoEscape,
        Package,
        Section,
        Subsection,
        Tabular,
    )
    from pylatex.utils import bold, escape_latex

    if args.ephemerides_dir is None:
        args.ephemerides_dir = args.results_dir / "ephemerides"
    results, orbit_errors = load_results(args.results_dir)
    metadata_file = args.results_dir / "scan_meta.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}

    ok = results[results["status"] == "ok"]
    affected = ok[ok["max_abs_dt_s"] > args.threshold]
    status_counts = results["status"].value_counts()

    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    histogram_file = figures_dir / "max_abs_dt_histogram.pdf"
    if len(ok) > 0:
        make_histogram_figure(results, args.threshold, histogram_file)

    document = Document(geometry_options={"margin": "1in"}, document_options=["11pt"], indent=False)
    # The float package's H specifier pins figures at their section position instead of
    # letting them drift across section boundaries
    document.packages.append(Package("float"))
    document.preamble.append(Command("title", "TGLC Light Curve Timing Audit"))
    document.preamble.append(
        Command("author", NoEscape(r"\texttt{scripts/audit\_lightcurve\_timing.py}"))
    )
    document.preamble.append(Command("date", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")))
    document.append(NoEscape(r"\maketitle"))

    with document.create(Section("Summary", numbering=False)):
        with document.create(Tabular("l l")) as table:
            table.add_hline()
            table.add_row("Scan timestamp", metadata.get("timestamp", "unknown"))
            table.add_row("Git revision", (metadata.get("git_revision", "unknown") or "")[:12])
            table.add_row("tglc version", metadata.get("tglc_version", "unknown"))
            table.add_row("Affected threshold", f"{args.threshold:g} s")
            table.add_row("Files scanned", f"{len(results):,}")
            for status, count in status_counts.items():
                table.add_row(f"Status: {status}", f"{count:,}")
            table.add_row(bold("Affected light curves"), bold(f"{len(affected):,}"))
            table.add_row(
                "Affected fraction",
                f"{len(affected) / len(ok):.2%}" if len(ok) else "n/a",
            )
            table.add_row("Orbits scanned", f"{results['orbit_group'].nunique():,}")
            table.add_row("Orbits with affected LCs", f"{affected['orbit_group'].nunique():,}")
            table.add_row("Orbit-level errors", f"{len(orbit_errors):,}")
            if len(affected) > 0:
                worst = affected.loc[affected["max_abs_dt_s"].idxmax()]
                table.add_row(
                    "Worst offender",
                    f"TIC {int(worst['tic_id'])} (orbit {int(worst['orbit_group'])}): "
                    f"{worst['max_abs_dt_s']:.3f} s",
                )
            table.add_hline()

    if histogram_file.exists():
        with document.create(Section("Discrepancy distribution", numbering=False)):
            with document.create(Figure(position="H")) as figure:
                figure.add_image(str(histogram_file.resolve()), width=NoEscape(r"0.85\textwidth"))
                figure.add_caption(
                    NoEscape(
                        r"Distribution of per-light-curve max $|\Delta t|$. Values at the "
                        r"$\sim$1\,ms level reflect the ordinary difference between the "
                        "predicted trajectory in the old files and the definitive JPL "
                        "solution, not a timing defect."
                    )
                )

    with document.create(Section("Per-orbit summary", numbering=False)):
        with document.create(LongTable("r r r r r r r l l")) as table:
            header = [
                "Orbit", "Sector", "LCs", "Affected", "%",
                NoEscape(r"Med $|\Delta t|$ (s)"), NoEscape(r"Max $|\Delta t|$ (s)"),
                "Old file", "Diagnosis",
            ]  # fmt: skip
            table.add_hline()
            table.add_row(header, mapper=bold)
            table.add_hline()
            table.end_table_header()
            table.add_hline()
            table.end_table_last_footer()
            for orbit, group in results.groupby("orbit_group"):
                group_ok = group[group["status"] == "ok"]
                group_affected = (group_ok["max_abs_dt_s"] > args.threshold).sum()
                cells = [
                    f"{orbit}",
                    f"{group['sector_used'].iloc[0]:.0f}",
                    f"{len(group):,}",
                    f"{group_affected:,}",
                    f"{group_affected / len(group_ok) * 100:.2f}" if len(group_ok) else "n/a",
                    f"{group_ok['max_abs_dt_s'].median():.2e}" if len(group_ok) else "n/a",
                    f"{group_ok['max_abs_dt_s'].max():.2e}" if len(group_ok) else "n/a",
                    str(group["old_file"].iloc[0]),
                    diagnose_group(group),
                ]
                table.add_row(cells, mapper=bold if group_affected > 0 else None)
            for _, error_row in orbit_errors.iterrows():
                table.add_row(
                    [
                        f"{error_row['orbit']}",
                        "--",
                        "--",
                        "--",
                        "--",
                        "--",
                        "--",
                        "--",
                        str(error_row["status"]),
                    ],  # fmt: skip
                )

    if len(affected) > 0:
        with document.create(Section(f"Top {args.top_n} worst offenders", numbering=False)):
            worst_rows = affected.nlargest(args.top_n, "max_abs_dt_s")
            with document.create(LongTable("r r r r r r l")) as table:
                table.add_hline()
                table.add_row(
                    [
                        "TIC",
                        "Orbit",
                        NoEscape(r"Max $|\Delta t|$ (s)"),
                        "RMS (s)",
                        NoEscape(r"$N$ over thr."),
                        "Diagnosis",
                        "Path",
                    ],  # fmt: skip
                    mapper=bold,
                )
                table.add_hline()
                table.end_table_header()
                table.add_hline()
                table.end_table_last_footer()
                for _, row in worst_rows.iterrows():
                    if not row["old_jdtdb_monotonic"]:
                        diagnosis = "malformed"
                    elif row["old_clamped_low"] or row["old_clamped_high"]:
                        diagnosis = "clamped"
                    else:
                        diagnosis = "--"
                    path_tail = row["path"][-42:]
                    if len(row["path"]) > 42:
                        path_tail = "..." + path_tail
                    table.add_row([
                        f"{int(row['tic_id'])}",
                        f"{int(row['orbit_group'])}",
                        f"{row['max_abs_dt_s']:.3f}",
                        f"{row['rms_dt_s']:.3f}",
                        f"{int(row['n_above_threshold'])}",
                        diagnosis,
                        NoEscape(r"\scriptsize\texttt{" + escape_latex(path_tail) + "}"),
                    ])  # fmt: skip

        with document.create(Section("Case studies", numbering=False)):
            worst_orbits = (
                affected.groupby("orbit_group")["max_abs_dt_s"]
                .max()
                .nlargest(args.worst_orbits)
                .index
            )
            for orbit in worst_orbits:
                orbit_worst = affected[affected["orbit_group"] == orbit]
                row = orbit_worst.loc[orbit_worst["max_abs_dt_s"].idxmax()]
                case_file = figures_dir / f"case_orbit-{orbit:04d}.pdf"
                if make_case_study_figure(row, args, case_file):
                    with document.create(Figure(position="H")) as figure:
                        figure.add_image(
                            str(case_file.resolve()), width=NoEscape(r"0.85\textwidth")
                        )
                        figure.add_caption(
                            NoEscape(
                                rf"$\Delta t$ vs.\ time for the worst light curve of orbit "
                                rf"{orbit} (max $|\Delta t| = {row['max_abs_dt_s']:.3f}$\,s)."
                            )
                        )

    with document.create(Section("Methodology", numbering=False)):
        document.append(
            NoEscape(
                r"For each light curve, the timing discrepancy is "
                r"$\Delta t(t) = \left(\mathbf{P}_\mathrm{new}(t) - \mathbf{P}_\mathrm{old}(t)"
                r"\right)\cdot\hat{u}_\star / c$, where $\mathbf{P}_\mathrm{old}$ reproduces the "
                r"retired pipeline's interpolation of the bundled yearly ephemeris CSVs "
                r"(including its silent clamping outside file coverage and, for sectors "
                r"102--115, a column-shifted parse of the malformed \texttt{20260401} file), "
                r"and $\mathbf{P}_\mathrm{new}$ interpolates the per-orbit JPL Horizons vector "
                r"table used by the current pipeline. $\Delta t$ is evaluated at the corrected "
                r"BJD stored in each file rather than the (unrecorded) spacecraft time; the "
                r"resulting error is of order $10^{-4}\,|\Delta t|$ and does not affect "
                r"flagging. Light curves are counted as affected when "
                rf"$\max|\Delta t| > {args.threshold:g}$\,s."
            )
        )
        with document.create(Subsection("Old ephemeris file inventory", numbering=False)):
            with document.create(Tabular("l r r r l")) as table:
                table.add_hline()
                table.add_row(["File", "Rows", "JDTDB start", "JDTDB end", "Parse"], mapper=bold)
                table.add_hline()
                for file_path in sorted(args.old_ephemerides_dir.glob("*_tess_ephem.csv")):
                    ephemeris = pd.read_csv(file_path, comment="#")
                    jd = np.asarray(ephemeris["JDTDB"], dtype=np.float64)
                    monotonic = bool(np.all(np.diff(jd) > 0))
                    table.add_row([
                        NoEscape(r"\texttt{" + escape_latex(file_path.stem) + "}"),
                        f"{len(jd):,}",
                        f"{jd[0]:.1f}" if monotonic else "--",
                        f"{jd[-1]:.1f}" if monotonic else "--",
                        "ok" if monotonic else bold("malformed"),
                    ])  # fmt: skip
                table.add_hline()

    output_stem = args.output or (args.results_dir / "timing_audit_report")
    document.generate_pdf(
        str(output_stem), compiler="latexmk", compiler_args=["-pdf"], clean_tex=False
    )
    logger.info(f"Report written to {output_stem}.pdf")


# ---------------------------------------------------------------------------------------------
# Synthetic verification dataset
# ---------------------------------------------------------------------------------------------


def make_synthetic_lightcurve(
    output_file: Path, tic_id: int, orbit: int, sector: int, ra: float, dec: float,
    tjd: np.ndarray, cadence: np.ndarray | None = None, camera: int = 1, ccd: int = 1,
) -> None:  # fmt: skip
    """Write a minimal but schema-complete light curve H5 using the production writer."""
    from astropy.coordinates import SkyCoord

    from tglc.aperture_light_curve import ApertureLightCurve, ApertureLightCurveMetadata

    n = len(tjd)
    if cadence is None:
        cadence = np.arange(n)
    columns = {"cadence": cadence, "quality_flag": np.zeros(n, dtype=int),
               "background_flux": np.ones(n)}  # fmt: skip
    for aperture in ["primary", "small", "large"]:
        columns[f"{aperture}_aperture_magnitude"] = np.full(n, 10.0) + np.random.default_rng(
            tic_id
        ).normal(0, 0.01, n)
        columns[f"{aperture}_aperture_centroid_x"] = np.full(n, 2.0)
        columns[f"{aperture}_aperture_centroid_y"] = np.full(n, 2.0)
    metadata = ApertureLightCurveMetadata(
        tic_id=tic_id, orbit=orbit, sector=sector, camera=camera, ccd=ccd,
        ccd_x=100.0, ccd_y=100.0,
        sky_coord=SkyCoord(ra, dec, unit="deg"), tess_magnitude=10.0,
        exposure_time=200 * u.second,
        primary_aperture_local_background=0 * u.electron,
        small_aperture_local_background=0 * u.electron,
        large_aperture_local_background=0 * u.electron,
    )  # fmt: skip
    lightcurve = ApertureLightCurve(
        time=Time(tjd, format="tjd", scale="tdb"), data=columns, meta=metadata
    )
    lightcurve.write_hdf5(output_file)


def run_make_synthetic(args: argparse.Namespace) -> None:
    """
    Build a synthetic dataset with known timing errors, fully offline.

    The fake "new" per-orbit ephemeris cache copies the old CSV positions with the X
    coordinate offset by delta = 0.5 s / (1 au in light seconds), and extends ~3.5 days past
    the old file's end with linear extrapolation. Consequences, by construction:
      - stars at RA=0, Dec=0 (u_x = 1): injected dt of exactly 0.5 s;
      - stars at RA=90 (u_x = 0): dt = 0;
      - light curves extending past the old file's end additionally see the old clamp error.
    """
    injected_dt = 0.5
    delta_x = injected_dt / AU_IN_LIGHTSECONDS
    orbit, sector = args.orbit, get_sector_containing_orbit(args.orbit)
    old_ephemeris = load_old_ephemeris(sector, DEFAULT_OLD_EPHEMERIDES_DIRECTORY)
    old_jd, old_end = old_ephemeris["jd"], old_ephemeris["jd"][-1]

    # Fake new-ephemeris cache covering the old file's span plus 3.5 days beyond its end
    cache_jd = np.arange(old_end - 430.0, old_end + 3.5, 1 / 24)
    cache = {"JDTDB": cache_jd, "Calendar Date (TDB)": "synthetic"}
    for axis in ["x", "y", "z"]:
        gradient = np.gradient(old_ephemeris[axis], old_jd)
        values = np.interp(cache_jd, old_jd, old_ephemeris[axis])
        beyond = cache_jd > old_end
        values[beyond] += gradient[-1] * (cache_jd[beyond] - old_end)
        cache[axis.upper()] = values + (delta_x if axis == "x" else 0.0)
    cache.update(LT=0.0, RG=1.0, RR=0.0)
    ephemerides_dir = args.outdir / "ephemerides"
    ephemerides_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cache).to_csv(ephemerides_dir / f"tess_ephem_orbit-{orbit:04d}.csv", index=False)

    lc_dir = args.outdir / "data" / f"orbit-{orbit}" / "ffi" / "cam1" / "ccd1" / "LC"
    lc_dir.mkdir(parents=True, exist_ok=True)
    tjd_span = np.linspace(old_end - 6.0, old_end - 4.0, 200) - TJD_OFFSET
    expectations = []
    for i in range(args.n_bad):
        tic_id = 1000 + i
        make_synthetic_lightcurve(
            lc_dir / f"{tic_id}.h5", tic_id, orbit, sector, ra=0.0, dec=0.0, tjd=tjd_span
        )
        expectations.append((tic_id, f"ok, max_abs_dt_s ~= {injected_dt}"))
    for i in range(args.n_good):
        tic_id = 2000 + i
        make_synthetic_lightcurve(
            lc_dir / f"{tic_id}.h5", tic_id, orbit, sector, ra=90.0, dec=30.0, tjd=tjd_span
        )
        expectations.append((tic_id, "ok, max_abs_dt_s ~= 0"))
    # Span extending past the old file's end: old interp clamps, new table keeps moving
    clamp_tjd = np.linspace(old_end - 1.5, old_end + 2.5, 200) - TJD_OFFSET
    make_synthetic_lightcurve(
        lc_dir / "3000.h5", 3000, orbit, sector, ra=0.0, dec=0.0, tjd=clamp_tjd
    )
    expectations.append((3000, "ok, old_clamped_high=True, max_abs_dt_s >> threshold"))
    (lc_dir / "4000.h5").write_bytes(b"this is not an HDF5 file")
    expectations.append((4000, "read_error"))

    # An orbit outside the sector map: produces an orbit-level sector_unknown error record
    unknown_dir = args.outdir / "data" / "orbit-999" / "ffi" / "cam1" / "ccd1" / "LC"
    unknown_dir.mkdir(parents=True, exist_ok=True)
    make_synthetic_lightcurve(
        unknown_dir / "5000.h5", 5000, 999, 120, ra=0.0, dec=0.0, tjd=tjd_span
    )
    expectations.append((5000, "orbit-0999.errors.csv with status sector_unknown"))

    print(f"Synthetic dataset written to {args.outdir}")
    print("Expected scan outcomes:")
    for tic_id, expectation in expectations:
        print(f"  TIC {tic_id}: {expectation}")
    print("\nRun:")
    print(
        f"  python scripts/audit_lightcurve_timing.py scan {args.outdir}/data"
        f" --results-dir {args.outdir}/results --ephemerides-dir {args.outdir}/ephemerides"
        f" --nprocs 2"
    )
    print(
        f"  python scripts/audit_lightcurve_timing.py report"
        f" --results-dir {args.outdir}/results"
        f" --ephemerides-dir {args.outdir}/ephemerides"
    )


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan H5 light curves")
    scan_parser.add_argument(
        "inputs", type=Path, nargs="+", help="Directories (recursive *.h5) or manifest text files"
    )
    scan_parser.add_argument("--results-dir", type=Path, required=True)
    scan_parser.add_argument(
        "--ephemerides-dir",
        type=Path,
        default=None,
        help="New-ephemeris cache directory (default: RESULTS_DIR/ephemerides)",
    )
    scan_parser.add_argument(
        "--old-ephemerides-dir", type=Path, default=DEFAULT_OLD_EPHEMERIDES_DIRECTORY
    )
    scan_parser.add_argument(
        "--threshold", type=float, default=0.05, help="Affected threshold in seconds (default 0.05)"
    )
    scan_parser.add_argument("--nprocs", type=int, default=1)
    scan_parser.add_argument("--chunk-size", type=int, default=256)
    scan_parser.add_argument(
        "--replace", action="store_true", help="Rescan orbits with existing results"
    )
    scan_parser.add_argument(
        "--orbit", type=int, nargs="+", default=None, help="Restrict the scan to these orbits"
    )
    scan_parser.set_defaults(func=run_scan)

    report_parser = subparsers.add_parser("report", help="Build the PDF report")
    report_parser.add_argument("--results-dir", type=Path, required=True)
    report_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output stem (default: RESULTS_DIR/timing_audit_report)",
    )
    report_parser.add_argument("--ephemerides-dir", type=Path, default=None)
    report_parser.add_argument(
        "--old-ephemerides-dir", type=Path, default=DEFAULT_OLD_EPHEMERIDES_DIRECTORY
    )
    report_parser.add_argument("--threshold", type=float, default=0.05)
    report_parser.add_argument("--top-n", type=int, default=20)
    report_parser.add_argument("--worst-orbits", type=int, default=3)
    report_parser.set_defaults(func=run_report)

    synthetic_parser = subparsers.add_parser(
        "make-synthetic", help="Generate a synthetic verification dataset"
    )
    synthetic_parser.add_argument("outdir", type=Path)
    synthetic_parser.add_argument("--orbit", type=int, default=185)
    synthetic_parser.add_argument("--n-good", type=int, default=5)
    synthetic_parser.add_argument("--n-bad", type=int, default=3)
    synthetic_parser.set_defaults(func=run_make_synthetic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
