"""Aggregate edge-compression sweep CSVs into a Han & Brandt 2023 Figure-4-style plot.

Reads the per-cutout CSVs written by `python -m tglc.scripts.edge_compression_sweep`, normalizes
each cutout's MAD-of-residual curve by its own minimum, and plots the per-cutout curves with the
median aggregate (Figure 4 of Han & Brandt 2023 did this for the flux-weighting power; here the
swept parameter is the edge-compression factor in TICA units). Also plots the holdout
cross-validation curve and the wing-mass / flux-fraction / aperture-scatter diagnostics, and
writes a machine-readable recommendation:

- knee_factor: largest factor whose aggregate in-sample MAD is within --knee-tolerance of the
  curve minimum (the in-sample curve is a plateau with a knee, not a U);
- holdout_min_factor: minimum of the aggregate holdout curve;
- recommended_factor: the knee, unless the holdout curve vetoes it (knee's holdout value more
  than the tolerance above the holdout minimum), in which case the largest factor within
  tolerance of the holdout minimum.

Usage:
    python -m tglc.scripts.edge_compression_figure SWEEP_DIR [SWEEP_DIR ...] --outdir DIR
        [--knee-tolerance 0.01] [--metric in_sample|allpix]
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tglc.scripts.edge_compression_common import (
    UNIT_CONVERTED_ANCHOR,
    UPSTREAM_EDGE_COMPRESSION,
    select_largest_within_tolerance,
    select_minimum,
    spoc_equivalent,
)


logger = logging.getLogger(__name__)

INTEGER_FIELDS = {
    "orbit",
    "cutout_x",
    "cutout_y",
    "camera",
    "ccd",
    "n_cadences",
    "n_failed_fits",
    "n_stars",
    "n_isolated_targets",
}


def load_sweeps(sweep_dirs: list[Path]) -> dict:
    """Load per-cutout sweep CSVs keyed by (orbit, cutout_x, cutout_y) -> {factor: row}."""
    data = {}
    for directory in sweep_dirs:
        for path in sorted(Path(directory).glob("sweep_*.csv")):
            with path.open() as f:
                rows = list(csv.DictReader(f))
            if not rows:
                logger.warning(f"{path} is empty (cutout had too few good cadences), skipping")
                continue
            for row in rows:
                for key in row:
                    row[key] = int(row[key]) if key in INTEGER_FIELDS else float(row[key])
            cutout_key = (rows[0]["orbit"], rows[0]["cutout_x"], rows[0]["cutout_y"])
            data[cutout_key] = {row["factor"]: row for row in rows}
    return data


def build_curves(data: dict, metric: str):
    """Min-normalized per-cutout curves and their median aggregate for one MAD metric.

    Returns `(factors, per_cutout_matrix, aggregate, p16, p84)` or `None` when no cutout has a
    finite curve (e.g. the holdout metric of a --no-holdout campaign). Cutouts whose factor grid
    differs from the first cutout's are dropped with a warning: curves are only comparable on a
    shared grid.
    """
    reference_factors = None
    curves = []
    for cutout_key in sorted(data):
        rows = data[cutout_key]
        factors = tuple(sorted(rows))
        if reference_factors is None:
            reference_factors = factors
        if factors != reference_factors:
            logger.warning(f"cutout {cutout_key} has a different factor grid, dropping")
            continue
        values = np.array([rows[factor][metric] for factor in reference_factors])
        if not np.all(np.isfinite(values)) or values.min() <= 0:
            logger.warning(f"cutout {cutout_key} has non-finite/non-positive {metric}, dropping")
            continue
        curves.append(values / values.min())
    if not curves:
        return None
    matrix = np.vstack(curves)
    return (
        np.array(reference_factors),
        matrix,
        np.median(matrix, axis=0),
        np.percentile(matrix, 16, axis=0),
        np.percentile(matrix, 84, axis=0),
    )


def diagnostic_curve(data: dict, factors: np.ndarray, metric: str) -> np.ndarray:
    """Median across cutouts of an (unnormalized) diagnostic column, per factor."""
    reference_factors = tuple(factors)
    values = [
        [rows[factor][metric] for factor in reference_factors]
        for rows in data.values()
        if tuple(sorted(rows)) == reference_factors
    ]
    with np.errstate(invalid="ignore"):
        return np.nanmedian(np.array(values), axis=0)


def make_recommendation(
    factors: np.ndarray,
    aggregate_in_sample: np.ndarray,
    aggregate_holdout: np.ndarray | None,
    tolerance: float,
    exposure: float | None,
) -> dict:
    knee_factor = select_largest_within_tolerance(factors, aggregate_in_sample, tolerance)
    recommendation = {
        "knee_tolerance": tolerance,
        "knee_factor": knee_factor,
        "holdout_min_factor": None,
        "holdout_max_within_tolerance": None,
        "holdout_veto": False,
        "recommended_factor": knee_factor,
    }
    if aggregate_holdout is not None:
        holdout_min = select_minimum(factors, aggregate_holdout)
        holdout_max_within_tolerance = select_largest_within_tolerance(
            factors, aggregate_holdout, tolerance
        )
        recommendation["holdout_min_factor"] = holdout_min
        recommendation["holdout_max_within_tolerance"] = holdout_max_within_tolerance
        knee_holdout_value = aggregate_holdout[np.flatnonzero(factors == knee_factor)[0]]
        if knee_holdout_value > (1 + tolerance) * aggregate_holdout.min():
            recommendation["holdout_veto"] = True
            recommendation["recommended_factor"] = holdout_max_within_tolerance
    if exposure is not None:
        recommendation["exposure"] = exposure
        recommendation["recommended_factor_spoc_equivalent"] = spoc_equivalent(
            recommendation["recommended_factor"], exposure
        )
    return recommendation


def factor_axis_positions(factors: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Log-plottable x positions: factor 0 is drawn a decade below the smallest nonzero factor."""
    nonzero = factors[factors > 0]
    if len(nonzero) == 0 or not np.any(factors == 0):
        return factors, None
    zero_position = nonzero.min() / 10
    return np.where(factors > 0, factors, zero_position), zero_position


def configure_factor_axis(ax, factors: np.ndarray, zero_position: float | None) -> None:
    ax.set_xscale("log")
    nonzero = factors[factors > 0]
    decades = 10.0 ** np.arange(
        np.floor(np.log10(nonzero.min())), np.ceil(np.log10(nonzero.max())) + 1
    )
    ticks = ([zero_position] if zero_position else []) + list(decades)
    labels = (["0"] if zero_position else []) + [f"$10^{{{round(np.log10(d))}}}$" for d in decades]
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("edge-compression factor (TICA units, e- per cadence)")


def add_anchor_lines(ax, recommended_factor: float) -> None:
    ax.axvline(
        UNIT_CONVERTED_ANCHOR, color="tab:green", lw=1, ls=":", label="upstream / exposure$^{1.4}$"
    )
    ax.axvline(UPSTREAM_EDGE_COMPRESSION, color="tab:red", lw=1, ls=":", label="upstream 1e-4")
    ax.axvline(recommended_factor, color="black", lw=1.2, ls="--", label="recommended")


def plot_mad_panel(ax, positions, matrix, aggregate, p16, p84, title, ylabel):
    for curve in matrix:
        ax.plot(positions, curve, color="tab:blue", alpha=0.15, lw=0.7)
    ax.fill_between(positions, p16, p84, color="tab:orange", alpha=0.25)
    ax.plot(positions, aggregate, "o-", color="tab:orange", ms=4, label="median of cutouts")
    ax.set(title=title, ylabel=ylabel)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sweep_dirs", nargs="+", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--knee-tolerance", type=float, default=0.01)
    parser.add_argument("--metric", choices=["in_sample", "allpix"], default="in_sample")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = load_sweeps(args.sweep_dirs)
    if not data:
        raise SystemExit("no sweep CSVs found")
    logger.info(f"{len(data)} cutouts loaded")

    in_sample = build_curves(data, f"mad_{args.metric}")
    if in_sample is None:
        raise SystemExit(f"no cutout has a finite mad_{args.metric} curve")
    factors, in_matrix, in_aggregate, in_p16, in_p84 = in_sample
    holdout = build_curves(data, "mad_holdout")

    exposures = {rows[factor]["exposure"] for rows in data.values() for factor in rows}
    exposure = exposures.pop() if len(exposures) == 1 else None
    recommendation = make_recommendation(
        factors, in_aggregate, holdout[2] if holdout else None, args.knee_tolerance, exposure
    )

    positions, zero_position = factor_axis_positions(factors)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    plot_mad_panel(
        axes[0, 0],
        positions,
        in_matrix,
        in_aggregate,
        in_p16,
        in_p84,
        f"In-sample residual MAD ({len(in_matrix)} cutouts)",
        f"normalized mad_{args.metric}",
    )
    add_anchor_lines(axes[0, 0], recommendation["recommended_factor"])
    axes[0, 0].legend(fontsize=8)
    if exposure is not None:
        scale = exposure**1.4
        secondary = axes[0, 0].secondary_xaxis(
            "top", functions=(lambda f: f * scale, lambda s: s / scale)
        )
        secondary.set_xlabel(f"SPOC-equivalent factor (e-/s units, exposure {exposure:g} s)")

    if holdout is not None:
        _, hold_matrix, hold_aggregate, hold_p16, hold_p84 = holdout
        plot_mad_panel(
            axes[0, 1],
            positions,
            hold_matrix,
            hold_aggregate,
            hold_p16,
            hold_p84,
            f"Holdout residual MAD ({len(hold_matrix)} cutouts)",
            "normalized mad_holdout",
        )
        add_anchor_lines(axes[0, 1], recommendation["recommended_factor"])
        minimum_index = np.flatnonzero(factors == recommendation["holdout_min_factor"])[0]
        axes[0, 1].plot(
            positions[minimum_index],
            hold_aggregate[minimum_index],
            "v",
            color="black",
            ms=9,
            label="holdout minimum",
        )
        axes[0, 1].legend(fontsize=8)
    else:
        axes[0, 1].text(
            0.5, 0.5, "no holdout data", ha="center", va="center", transform=axes[0, 1].transAxes
        )

    wing_mass = diagnostic_curve(data, factors, "wing_mass")
    axes[1, 0].plot(positions, wing_mass, "o-", color="tab:purple", ms=4)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="ePSF wing mass (median of cutouts)", ylabel="flux beyond ±2 px / total")
    add_anchor_lines(axes[1, 0], recommendation["recommended_factor"])

    fraction = diagnostic_curve(data, factors, "flux_fraction_median")
    scatter = diagnostic_curve(data, factors, "aperture_scatter_mmag_median")
    axes[1, 1].plot(positions, fraction, "o-", color="tab:blue", ms=4, label="flux fraction")
    axes[1, 1].axhline(1.0, color="gray", lw=1, ls="--")
    axes[1, 1].set(title="Isolated-target diagnostics (medians)", ylabel="epsf_flux_fraction")
    scatter_axis = axes[1, 1].twinx()
    scatter_axis.plot(positions, scatter, "s-", color="tab:orange", ms=4, label="3x3 scatter")
    scatter_axis.set_ylabel("aperture scatter (mmag)")
    handles1, labels1 = axes[1, 1].get_legend_handles_labels()
    handles2, labels2 = scatter_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles1 + handles2, labels1 + labels2, fontsize=8)

    for ax in axes.flat:
        configure_factor_axis(ax, factors, zero_position)
    fig.tight_layout()
    figure_path = args.outdir / "edge_compression_sweep.pdf"
    fig.savefig(figure_path)
    logger.info(f"wrote {figure_path}")

    summary = {
        "n_cutouts": len(data),
        "n_cutouts_in_sample": len(in_matrix),
        "n_cutouts_holdout": len(holdout[1]) if holdout else 0,
        "metric": f"mad_{args.metric}",
        "factors": list(factors),
        "aggregate_in_sample": list(in_aggregate),
        "aggregate_holdout": list(holdout[2]) if holdout else None,
        "wing_mass": [float(v) for v in wing_mass],
        "flux_fraction_median": [float(v) for v in fraction],
        "aperture_scatter_mmag_median": [float(v) for v in scatter],
        **recommendation,
    }
    summary_path = args.outdir / "edge_compression_recommendation.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
