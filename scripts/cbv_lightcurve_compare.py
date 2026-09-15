"""
Compare two TGLC light-curve trees (e.g. baseline vs. FFI-CBV-corrected) on
per-target precision metrics, with optional wotan detrending and per-magnitude
binning. Designed to scale to ~10^7 targets via process-pool parallelism and
streaming HDF5 writes (one resizable, chunked, gzip-compressed dataset per
column; no Rust build deps).

The TGLC paper defines precision as the MAD of differences between adjacent
flux points; that is the day-one metric implemented here. Other metrics and
stellar covariates are intended to be added by registering into the
``METRICS`` / ``DETRENDERS`` dicts at the top of this file — the driver and
output schema stay the same.

Light-curve schema is read directly from h5py per
``tglc/aperture_light_curve.py:write_hdf5``: file-level attrs include
``TessMag``, and per-aperture ``RawMagnitude`` lives at
``/LightCurve/AperturePhotometry/{Primary,Small,Large}Aperture/RawMagnitude``.
The script does not import any TGLC package code, so it can run anywhere two
``LC/`` directories of ``{tic_id}.h5`` files are available.

Usage:
    python scripts/cbv_lightcurve_compare.py \\
        --baseline-lc-dir /path/to/baseline/LC \\
        --cbv-lc-dir      /path/to/cbv/LC \\
        --out-dir         /path/to/output \\
        [--wotan-window 0.5] [--wotan-method biweight] \\
        [--n-procs $(nproc)] [--flush-every 50000] \\
        [--scatter-max 200000] [--csv] [--limit N]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import faulthandler
from functools import partial
import logging
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback

from astropy.stats import mad_std
import h5py
import matplotlib

from tglc.lc_cbv import LCCBVData, UnsupportedLCCBVFormatError, load_lc_cbvs


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402


# Install the C-level crash handler before we do any heavy lifting. If a
# native call (most likely h5py / libhdf5) segfaults, faulthandler prints a
# Python traceback to stderr — the only signal we get back from a crash.
# ``all_threads=True`` covers any background threads h5py/numba may spawn.
faulthandler.enable(all_threads=True)


logger = logging.getLogger("cbv_lc_compare")

APERTURES = ("Primary", "Small", "Large")
# Channels compared per target: the three aperture magnitudes plus the per-cadence
# background flux. The schema column is named ``channel`` (not ``aperture``) because
# Background isn't an aperture.
BACKGROUND_CHANNEL = "Background"
CHANNELS = (*APERTURES, BACKGROUND_CHANNEL)


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _mad_adjacent_diff_ppm(rel_flux: np.ndarray, good: np.ndarray) -> float:
    """MAD of adjacent-cadence differences in relative flux, reported in ppm.

    This is the TGLC paper's "precision" metric: scatter on the shortest
    timescale, robust to outliers. ``astropy.stats.mad_std`` rescales MAD by
    1/Φ⁻¹(3/4) ≈ 1.4826 so the result is a consistent estimator of σ under
    a Gaussian — the same convention TGLC uses elsewhere
    (``tglc/light_curve.py``).
    """
    y = rel_flux[good]
    if y.size < 2 or not np.isfinite(y).any():
        return float("nan")
    diffs = np.diff(y)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 1:
        return float("nan")
    return 1e6 * float(mad_std(diffs))


METRICS: dict[str, MetricFn] = {
    "mad_adjacent_ppm": _mad_adjacent_diff_ppm,
}


# ---------------------------------------------------------------------------
# Detrender registry
# ---------------------------------------------------------------------------

# Detrenders accept a uniform call shape: positional (time, rel_flux, good) and
# an arbitrary kwargs context. Each detrender consumes only the kwargs it cares
# about; the rest are absorbed into ``**_ctx``. This lets per-target context
# (TIC ID, channel, cadence array) flow into detrenders that need it (lc_cbv)
# without changing the call sites for those that don't (raw, wotan).
DetrenderFn = Callable[..., np.ndarray]


def _raw_detrender(
    _time: np.ndarray, rel_flux: np.ndarray, _good: np.ndarray, **_ctx
) -> np.ndarray:
    return rel_flux


def _wotan_detrender(
    time: np.ndarray,
    rel_flux: np.ndarray,
    good: np.ndarray,
    *,
    method: str,
    window_length: float,
    **_ctx,
) -> np.ndarray:
    """Top-level (picklable) wotan-backed detrender.

    Availability is probed at startup by ``_require_wotan``; the worker
    processes (spawn start method) import wotan when they first call this.
    """
    from wotan import flatten

    masked = rel_flux.copy()
    masked[~good] = np.nan
    if np.count_nonzero(np.isfinite(masked)) < 4:
        return masked
    flat, _ = flatten(time, masked, method=method, window_length=window_length, return_trend=True)
    return flat


def _lc_cbv_detrender(
    _time: np.ndarray,
    rel_flux: np.ndarray,
    _good: np.ndarray,
    *,
    lc_cbv_data: LCCBVData | None,
    tic_id: int,
    cadences: np.ndarray,
    channel: str,
    **_ctx,
) -> np.ndarray:
    """Subtract the LC-CBV-fit systematic trend per target.

    Restricted to the three aperture channels — LC-CBV is fit on aperture
    photometry, so applying it to ``Background`` (a different physical
    quantity) is meaningless. We return all-NaN for ``Background`` and for
    targets absent from the CBV product; the downstream metric then yields
    NaN cleanly and pandas drops those rows in plots.
    """
    if channel == BACKGROUND_CHANNEL or lc_cbv_data is None:
        return np.full_like(rel_flux, np.nan, dtype=np.float64)
    trend = lc_cbv_data.trend_for_target(int(tic_id), cadences)
    if trend is None:
        return np.full_like(rel_flux, np.nan, dtype=np.float64)
    return rel_flux.astype(np.float64, copy=False) - trend


def _build_detrender_map(
    wotan_method: str,
    wotan_window: float,
    lc_cbv_data: LCCBVData | None,
) -> dict[str, DetrenderFn]:
    """Construct the (name → detrender) registry. All entries are picklable
    top-level functions or ``functools.partial`` objects so the dict survives
    the multiprocessing spawn pickle. ``lc_cbv`` is included only when an
    LC-CBV product was loaded.
    """
    detrenders: dict[str, DetrenderFn] = {
        "raw": _raw_detrender,
        f"wotan_{wotan_method}": partial(
            _wotan_detrender, method=wotan_method, window_length=wotan_window
        ),
    }
    if lc_cbv_data is not None:
        detrenders["lc_cbv"] = partial(_lc_cbv_detrender, lc_cbv_data=lc_cbv_data)
    return detrenders


def _require_wotan() -> None:
    """Fail fast at startup if wotan is requested but not installed."""
    try:
        import wotan  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "wotan is not installed but is required for --wotan-method. "
            "Install with: pip install wotan"
        ) from e


# ---------------------------------------------------------------------------
# LC reader
# ---------------------------------------------------------------------------


def _mag_to_rel_flux(mag: np.ndarray) -> np.ndarray:
    """Convert TESS magnitude to relative flux (median-normalised, centred near 1)."""
    med = np.nanmedian(mag)
    if not np.isfinite(med):
        return np.full_like(mag, np.nan, dtype=np.float64)
    return np.power(10.0, -0.4 * (mag - med))


def _bg_to_rel_flux(flux: np.ndarray) -> np.ndarray:
    """Median-normalise a background-flux time series.

    Returns the relative deviation around 0: ``(flux - median) / median``. NaN
    everywhere if the median is non-finite or ≤ 0 (parallels the guard in
    ``_mag_to_rel_flux``). The MAD-of-adjacent-diffs metric is offset-invariant,
    so this choice of centring does not change the reported ppm number relative
    to ``flux / median``; it's chosen for the cleaner zero-centred series in
    the saved series should any future metric ever inspect it directly.
    """
    med = np.nanmedian(flux)
    if not np.isfinite(med) or med <= 0:
        return np.full_like(flux, np.nan, dtype=np.float64)
    return (flux - med) / med


def _peek_lc_attrs(path: Path) -> dict:
    """Read only the file-level HDF5 attrs we need to locate the LC-CBV file.

    Cheaper than the full ``read_tglc_lc`` (no dataset reads) and used exactly
    once in the parent process to derive the per-(orbit, cam, ccd) CBV path.
    """
    with h5py.File(path, "r") as f:
        return {
            "orbit": int(f.attrs["Orbit"]),
            "camera": int(f.attrs["Camera"]),
            "ccd": int(f.attrs["CCD"]),
        }


def read_tglc_lc(path: Path) -> dict:
    """Open a TGLC light-curve HDF5 file and return the data we compare on.

    Returned dict has scalar metadata plus arrays:
      tic_id, tmag, ra, dec, orbit, sector, camera, ccd,
      bjd (n,), cadence (n,), quality (n,),
      rel_flux_by_channel: {"Primary": (n,), "Small": (n,), "Large": (n,),
                            "Background": (n,)}

    The reader owns relative-flux normalisation because the channels don't
    share a formula: apertures use ``10**(-0.4·Δm)`` while the background
    uses ``(flux - median) / median``. Moving that branching out of the
    worker keeps the per-channel loop uniform.

    ``cadence`` is exposed because the LC-CBV detrender aligns its CBVs to
    raw cadence numbers (not BJD); ``orbit`` is exposed so the main process
    can look up the per-(orbit, cam, ccd) LC-CBV FITS file.
    """
    logger.debug("read_tglc_lc: opening %s", path)
    with h5py.File(path, "r") as f:
        attrs = f.attrs
        lc = f["LightCurve"]
        photometry = lc["AperturePhotometry"]
        rel_flux_by_channel: dict[str, np.ndarray] = {}
        for name in APERTURES:
            mag = np.asarray(photometry[f"{name}Aperture"]["RawMagnitude"][:], dtype=np.float64)
            rel_flux_by_channel[name] = _mag_to_rel_flux(mag)
        bg_flux = np.asarray(lc["Background"]["Value"][:], dtype=np.float64)
        rel_flux_by_channel[BACKGROUND_CHANNEL] = _bg_to_rel_flux(bg_flux)
        out = {
            "tic_id": int(attrs["TIC ID"]),
            "tmag": float(attrs["TessMag"]),
            "ra": float(attrs["RA"]),
            "dec": float(attrs["Dec"]),
            "orbit": int(attrs["Orbit"]),
            "sector": int(attrs["Sector"]),
            "camera": int(attrs["Camera"]),
            "ccd": int(attrs["CCD"]),
            "bjd": np.asarray(lc["BJD"][:], dtype=np.float64),
            "cadence": np.asarray(lc["Cadence"][:], dtype=np.int64),
            "quality": np.asarray(lc["QualityFlag"][:], dtype=np.int64),
            "rel_flux_by_channel": rel_flux_by_channel,
        }
    return out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _compute_rows(
    baseline_path: Path,
    cbv_path: Path,
    detrender_map: dict[str, DetrenderFn],
) -> list[dict] | None:
    """Compute one row per (channel, detrender, metric) for a single TIC."""
    base = read_tglc_lc(baseline_path)
    cbv = read_tglc_lc(cbv_path)
    if base["tic_id"] != cbv["tic_id"]:
        logger.warning(
            "TIC mismatch: %s -> %d, %s -> %d",
            baseline_path,
            base["tic_id"],
            cbv_path,
            cbv["tic_id"],
        )
        return None

    base_good = base["quality"] == 0
    cbv_good = cbv["quality"] == 0
    rows: list[dict] = []
    for channel in CHANNELS:
        base_rel = base["rel_flux_by_channel"][channel]
        cbv_rel = cbv["rel_flux_by_channel"][channel]
        # Per-call detrender context. ``raw`` / ``wotan_*`` ignore these (they
        # accept ``**_ctx``); ``lc_cbv`` consumes ``tic_id``, ``cadences``,
        # ``channel`` to look up and subtract the per-target trend.
        base_ctx = dict(tic_id=base["tic_id"], cadences=base["cadence"], channel=channel)
        cbv_ctx = dict(tic_id=cbv["tic_id"], cadences=cbv["cadence"], channel=channel)

        for det_name, detrend in detrender_map.items():
            base_series = detrend(base["bjd"], base_rel, base_good, **base_ctx)
            cbv_series = detrend(cbv["bjd"], cbv_rel, cbv_good, **cbv_ctx)
            base_mask = base_good & np.isfinite(base_series)
            cbv_mask = cbv_good & np.isfinite(cbv_series)

            for metric_name, metric_fn in METRICS.items():
                b = metric_fn(base_series, base_mask)
                c = metric_fn(cbv_series, cbv_mask)
                delta = c - b
                rel_delta = (
                    delta / b if (b is not None and np.isfinite(b) and b != 0) else float("nan")
                )
                rows.append(
                    {
                        "tic_id": base["tic_id"],
                        "tmag": base["tmag"],
                        "ra": base["ra"],
                        "dec": base["dec"],
                        "sector": base["sector"],
                        "camera": base["camera"],
                        "ccd": base["ccd"],
                        "channel": channel,
                        "detrender": det_name,
                        "metric": metric_name,
                        "baseline": b,
                        "cbv": c,
                        "delta": delta,
                        "rel_delta": rel_delta,
                    }
                )
    return rows


def compare_one_target(
    pair: tuple[Path, Path],
    *,
    detrender_map: dict[str, DetrenderFn],
) -> list[dict] | None:
    """Top-level worker. Returns rows on success, ``None`` on any failure.

    We swallow ``Exception`` (not ``BaseException``) so SIGINT still
    propagates and shuts the pool down promptly. A *segfault* inside the
    h5py / numpy / wotan C extensions will NOT be caught here — that crash
    surfaces in the worker's stderr via ``faulthandler``.
    """
    baseline_path, cbv_path = pair
    logger.debug("compare_one_target: start %s", baseline_path.name)
    try:
        rows = _compute_rows(baseline_path, cbv_path, detrender_map)
        logger.debug("compare_one_target: done %s", baseline_path.name)
        return rows
    except Exception as exc:
        logger.warning(
            "Failed on %s: %s: %s",
            baseline_path.name,
            type(exc).__name__,
            exc,
        )
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return None


def _worker_init(log_level: int) -> None:
    """Pool initializer. Spawned children inherit no logging config and no
    faulthandler, so we re-arm both here. Running this once per worker also
    confirms the worker actually started — if you don't see "worker ready"
    for a child, it died during spawn (often a fork/HDF5 init crash).
    """
    import faulthandler as _fh

    _fh.enable(all_threads=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] worker[%(process)d] %(name)s: %(message)s",
        force=True,
    )
    logger.info("worker ready (pid=%d)", os.getpid())


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def _index_lc_dir(lc_dir: Path) -> dict[str, Path]:
    """Map TIC-id string (filename stem) -> Path. Uses iterdir to avoid the
    overhead of fnmatch on giant directories.
    """
    out: dict[str, Path] = {}
    count = 0
    for entry in lc_dir.iterdir():
        if entry.suffix == ".h5":
            out[entry.stem] = entry
            count += 1
            if count % 250_000 == 0:
                logger.info("  …indexed %d files in %s", count, lc_dir)
    logger.info("  indexed %d .h5 files in %s", count, lc_dir)
    return out


def _pair_iterator(
    baseline_dir: Path, cbv_dir: Path, limit: int | None
) -> tuple[list[tuple[Path, Path]], int, int]:
    """Build (matched_pairs, baseline_only, cbv_only) by TIC stem.

    Returns the full list up front because the pool's ``imap_unordered`` needs
    a known total for tqdm and a stable iteration order for reproducibility.
    For 10^7 files the dict + list overhead is ~a few GB — acceptable on the
    machines we run this on. If we outgrow that, swap to a generator and a
    `total=` argument computed via a streaming count first.
    """
    logger.info("Indexing %s", baseline_dir)
    base_idx = _index_lc_dir(baseline_dir)
    logger.info("Indexing %s", cbv_dir)
    cbv_idx = _index_lc_dir(cbv_dir)

    base_only = set(base_idx) - set(cbv_idx)
    cbv_only = set(cbv_idx) - set(base_idx)
    common = sorted(set(base_idx) & set(cbv_idx))
    if limit is not None:
        common = common[:limit]
    pairs = [(base_idx[stem], cbv_idx[stem]) for stem in common]
    return pairs, len(base_only), len(cbv_only)


# ---------------------------------------------------------------------------
# Streaming HDF5 writer
# ---------------------------------------------------------------------------

# Per-column dtype. Strings use h5py's variable-length UTF-8 dtype so the
# column round-trips to Python str rather than fixed-width bytes (which pandas
# would otherwise surface as object/bytes mixes).
_VLEN_STR = h5py.string_dtype(encoding="utf-8")

# Ordered (name, dtype) — drives both writer creation and reader iteration so
# the file layout stays consistent.
_COLUMN_SCHEMA: tuple[tuple[str, object], ...] = (
    ("tic_id", np.int64),
    ("tmag", np.float64),
    ("ra", np.float64),
    ("dec", np.float64),
    ("sector", np.int32),
    ("camera", np.int32),
    ("ccd", np.int32),
    ("channel", _VLEN_STR),
    ("detrender", _VLEN_STR),
    ("metric", _VLEN_STR),
    ("baseline", np.float64),
    ("cbv", np.float64),
    ("delta", np.float64),
    ("rel_delta", np.float64),
)
_COLUMN_NAMES = tuple(name for name, _ in _COLUMN_SCHEMA)


class _H5StreamWriter:
    """Append-mode HDF5 writer: one resizable, chunked, gzip-compressed
    dataset per column. Pure-C dep (h5py / libhdf5); no Rust toolchain
    required at install time.
    """

    def __init__(self, path: Path, chunk_size: int):
        self._path = path
        self._chunk_size = chunk_size
        self._file = h5py.File(path, "w")
        self._n = 0
        for name, dt in _COLUMN_SCHEMA:
            self._file.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=dt,
                compression="gzip",
                compression_opts=4,
            )

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        n = len(rows)
        new_size = self._n + n
        for name, dt in _COLUMN_SCHEMA:
            ds = self._file[name]
            ds.resize((new_size,))
            if dt is _VLEN_STR:
                ds[self._n : new_size] = [r[name] for r in rows]
            else:
                ds[self._n : new_size] = np.asarray([r[name] for r in rows], dtype=dt)
        self._n = new_size

    def close(self) -> None:
        self._file.attrs["n_rows"] = self._n
        self._file.close()

    def __enter__(self) -> _H5StreamWriter:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _read_results(path: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Load (a subset of) result columns from the streaming HDF5 file.

    h5py 3.x surfaces variable-length UTF-8 datasets as ``object`` arrays of
    ``bytes``; we decode those columns to ``str`` so downstream pandas
    ``groupby`` keys and the plot filenames are plain text, not ``b'...'``.
    """
    cols = columns if columns is not None else _COLUMN_NAMES
    data: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for name in cols:
            arr = f[name][:]
            if arr.dtype == object:
                arr = np.array(
                    [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else s for s in arr],
                    dtype=object,
                )
            data[name] = arr
    return pd.DataFrame(data)


def _flush_buffer(buffer: list[dict], writer: _H5StreamWriter) -> None:
    if not buffer:
        return
    writer.write(buffer)
    buffer.clear()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _write_summary(
    results_path: Path,
    summary_path: Path,
    counts: dict[str, int],
    wotan_method: str,
    wotan_window: float,
) -> None:
    """Compute aggregate stats from the streamed HDF5 result file. Reads only
    the columns the summary needs, so peak memory scales with metric columns
    × n_rows, not the full schema.
    """
    df = _read_results(
        results_path,
        columns=("channel", "detrender", "metric", "baseline", "cbv", "delta", "rel_delta"),
    )
    lines: list[str] = []
    lines.append("CBV light-curve comparison summary")
    lines.append("=" * 50)
    lines.append(f"matched pairs:     {counts['matched']}")
    lines.append(f"succeeded:         {counts['succeeded']}")
    lines.append(f"failed (worker):   {counts['failed']}")
    lines.append(f"unmatched (base):  {counts['base_only']}")
    lines.append(f"unmatched (cbv):   {counts['cbv_only']}")
    lines.append(f"wotan: method={wotan_method}, window={wotan_window} d")
    lines.append("")

    grouped = df.groupby(["channel", "detrender", "metric"], sort=True)
    for (ch, det, met), sub in grouped:
        n = len(sub)
        if n == 0:
            continue
        lines.append(f"[{ch} / {det} / {met}]  n={n}")
        for col in ("baseline", "cbv", "delta", "rel_delta"):
            med = float(np.nanmedian(sub[col]))
            lines.append(f"    median {col:9s} = {med:+.6g}")
        beat = float(np.nanmean(sub["cbv"] < sub["baseline"]))
        lines.append(f"    fraction CBV beats baseline = {beat:.3f}")
        lines.append("")

    summary_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", summary_path)


def _plot_precision_vs_tmag(
    results_path: Path,
    out_dir: Path,
    scatter_max: int,
) -> None:
    """One figure per (channel, detrender, metric) combination."""
    df = _read_results(
        results_path,
        columns=("channel", "detrender", "metric", "tmag", "baseline", "cbv"),
    )
    rng = np.random.default_rng(seed=0)

    for (ch, det, met), sub in df.groupby(["channel", "detrender", "metric"]):
        sub = sub.dropna(subset=["tmag", "baseline", "cbv"])
        if sub.empty:
            continue
        if len(sub) > scatter_max:
            idx = rng.choice(len(sub), size=scatter_max, replace=False)
            scatter = sub.iloc[np.sort(idx)]
        else:
            scatter = sub
        bins = np.arange(np.floor(sub["tmag"].min()), np.ceil(sub["tmag"].max()) + 0.5, 0.5)
        sub_cuts = pd.cut(sub["tmag"], bins=bins)
        binned = sub.groupby(sub_cuts, observed=True)[["baseline", "cbv"]].median()
        centres = np.array([interval.mid for interval in binned.index])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(scatter["tmag"], scatter["baseline"], s=2, alpha=0.15, color="C0", label=None)
        ax.scatter(scatter["tmag"], scatter["cbv"], s=2, alpha=0.15, color="C1", label=None)
        ax.plot(centres, binned["baseline"], color="C0", lw=2, label="baseline (binned median)")
        ax.plot(centres, binned["cbv"], color="C1", lw=2, label="CBV (binned median)")
        ax.set_yscale("log")
        ax.set_xlabel("TessMag")
        ax.set_ylabel(f"{met}")
        ax.set_title(f"channel={ch} / detrender={det}")
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / f"precision_vs_tmag_{ch}_{det}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--baseline-lc-dir", type=Path, required=True)
    p.add_argument("--cbv-lc-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--wotan-window", type=float, default=0.5, help="wotan window in days (default 0.5)"
    )
    p.add_argument(
        "--wotan-method", default="biweight", help="wotan detrender method (default biweight)"
    )
    p.add_argument("--n-procs", type=int, default=os.cpu_count() or 1)
    p.add_argument(
        "--flush-every",
        type=int,
        default=50_000,
        help="rows buffered before HDF5 flush; also the per-column chunk size on disk",
    )
    p.add_argument(
        "--scatter-max", type=int, default=200_000, help="max scatter points per plot subgroup"
    )
    p.add_argument(
        "--csv", action="store_true", help="also emit per_target.csv alongside per_target.h5"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="cap total target pairs (for smoke runs)"
    )
    p.add_argument("--chunksize", type=int, default=64, help="Pool.imap_unordered chunk size")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--lc-cbv-dir",
        type=Path,
        default=None,
        help="Directory containing per-(orbit, cam, ccd) LC-CBV FITS files named "
        "cbv_lc_{orbit}_{cam}_{ccd}.fits (qlp-cbv draft-1, PRODUCT='qlp-cbv-lc'). "
        "When set, a third detrender ``lc_cbv`` is registered that subtracts the "
        "per-target trend on aperture channels (Background is skipped).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    # If we crash later, this confirms we made it past argv parsing & logging.
    logger.info(
        "starting; python=%s, pid=%d, n_procs=%d, baseline=%s, cbv=%s, out=%s",
        sys.version.split()[0],
        os.getpid(),
        args.n_procs,
        args.baseline_lc_dir,
        args.cbv_lc_dir,
        args.out_dir,
    )
    logger.info("h5py=%s, hdf5=%s", h5py.__version__, h5py.version.hdf5_version)
    logger.info(
        "HDF5_USE_FILE_LOCKING=%r (libhdf5 file-locking on shared filesystems "
        "is a frequent segfault source — set to 'FALSE' if you see crashes "
        "while opening LC files on NFS)",
        os.environ.get("HDF5_USE_FILE_LOCKING"),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("checking wotan availability")
    _require_wotan()

    logger.info("indexing input directories")
    pairs, base_only, cbv_only = _pair_iterator(args.baseline_lc_dir, args.cbv_lc_dir, args.limit)
    logger.info(
        "Pairing: %d matched, %d baseline-only, %d cbv-only", len(pairs), base_only, cbv_only
    )
    if not pairs:
        logger.error("No matching TIC IDs between the two directories. Exiting.")
        return 1

    # Resolve the LC-CBV product (if any). Loaded once in the parent process so
    # every worker gets the same payload via the partial — avoids 40 concurrent
    # FITS opens on whatever shared filesystem the file lives on, and makes the
    # downstream lc_cbv detrender a pure lookup.
    lc_cbv_data: LCCBVData | None = None
    if args.lc_cbv_dir is not None:
        first_attrs = _peek_lc_attrs(pairs[0][0])
        cbv_path = (
            args.lc_cbv_dir
            / f"cbv_lc_{first_attrs['orbit']}_{first_attrs['camera']}_{first_attrs['ccd']}.fits"
        )
        if not cbv_path.is_file():
            logger.error("LC-CBV file not found at %s — skipping lc_cbv detrender", cbv_path)
        else:
            logger.info("loading LC-CBV product %s", cbv_path)
            try:
                lc_cbv_data = load_lc_cbvs(cbv_path)
                logger.info(
                    "  LC-CBV: orbit=%d cam=%d ccd=%d, NCBV=%d, NCAD=%d, NTGT=%d",
                    lc_cbv_data.orbit,
                    lc_cbv_data.camera,
                    lc_cbv_data.ccd,
                    lc_cbv_data.cbvs.shape[0],
                    lc_cbv_data.cbvs.shape[1],
                    lc_cbv_data.theta.shape[0],
                )
            except UnsupportedLCCBVFormatError as exc:
                logger.error("refused LC-CBV product %s: %s", cbv_path, exc)
                lc_cbv_data = None

    detrender_map = _build_detrender_map(args.wotan_method, args.wotan_window, lc_cbv_data)
    logger.info("active detrenders: %s", sorted(detrender_map.keys()))

    results_path = args.out_dir / "per_target.h5"
    counts = {
        "matched": len(pairs),
        "succeeded": 0,
        "failed": 0,
        "base_only": base_only,
        "cbv_only": cbv_only,
    }
    buffer: list[dict] = []

    worker = partial(compare_one_target, detrender_map=detrender_map)

    # "spawn" is the safer default for HDF5/h5py and matplotlib-loaded processes;
    # the worker function and detrender map are picklable.
    ctx = mp.get_context("spawn")
    logger.info("opening output HDF5 stream at %s", results_path)
    with _H5StreamWriter(results_path, chunk_size=args.flush_every) as writer:
        if args.n_procs > 1:
            logger.info("spawning pool with %d workers", args.n_procs)
            pool = ctx.Pool(
                processes=args.n_procs,
                initializer=_worker_init,
                initargs=(log_level,),
            )
            try:
                logger.info("dispatching %d targets (chunksize=%d)", len(pairs), args.chunksize)
                results = pool.imap_unordered(worker, pairs, chunksize=args.chunksize)
                for rows in tqdm(results, total=len(pairs), unit="target"):
                    if rows is None:
                        counts["failed"] += 1
                        continue
                    counts["succeeded"] += 1
                    buffer.extend(rows)
                    if len(buffer) >= args.flush_every:
                        logger.debug("flushing %d rows to HDF5", len(buffer))
                        _flush_buffer(buffer, writer)
            finally:
                pool.close()
                pool.join()
        else:
            logger.info("running serial loop over %d targets", len(pairs))
            for pair in tqdm(pairs, total=len(pairs), unit="target"):
                rows = worker(pair)
                if rows is None:
                    counts["failed"] += 1
                    continue
                counts["succeeded"] += 1
                buffer.extend(rows)
                if len(buffer) >= args.flush_every:
                    _flush_buffer(buffer, writer)
        _flush_buffer(buffer, writer)

    logger.info(
        "Done: succeeded=%d, failed=%d, written to %s",
        counts["succeeded"],
        counts["failed"],
        results_path,
    )

    if args.csv:
        csv_path = args.out_dir / "per_target.csv"
        _read_results(results_path).to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)

    _write_summary(
        results_path,
        args.out_dir / "summary.txt",
        counts,
        wotan_method=args.wotan_method,
        wotan_window=args.wotan_window,
    )
    _plot_precision_vs_tmag(results_path, args.out_dir, args.scatter_max)
    return 0


if __name__ == "__main__":
    sys.exit(main())
