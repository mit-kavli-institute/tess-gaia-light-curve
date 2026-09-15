"""Sweep edge-compression factors over an orbit's cutouts to calibrate the value in TICA units.

For each (source, epsf) cutout pair: sample good cadences from the stable middle of the orbit,
build the ePSF design matrix once, then refit the ePSF at each candidate edge-compression factor
(rescaling only the regularization rows) and record, per factor:

- MAD of the residual image (Han & Brandt 2023 Figure 4 metric), both in-sample (pixels used by
  the fit) and on a seeded random ~10% pixel holdout excluded from the fit (cross-validation, so
  the curve has a true minimum instead of a plateau-and-knee);
- diagnostics: ePSF wing mass beyond +-2 px, and median flux fraction, model/decontaminated-data
  ratio, and 3x3-aperture scatter over isolated Tmag 9-13 targets.

With --power-sweep the roles swap: the edge-compression factor is held fixed (--edge-factor,
default the full-campaign recommendation) and the flux-uncertainty weighting power is swept
instead (Han & Brandt 2023 Figure 4 proper, which adopted the MAD minimum at l = 1.4), writing
power_{x}_{y}.csv files with the same metrics.

One CSV per cutout (one row per swept value). Resumable: cutouts whose CSV already exists are
skipped. Shardable. The swept grid must be identical across a campaign (see
edge_compression_common). Aggregate the CSVs with `python -m tglc.scripts.edge_compression_figure`.

Usage (run from the repo root with PYTHONPATH=. so the working tree wins over site-packages):
    python -m tglc.scripts.edge_compression_sweep --source-dir DIR --epsf-dir DIR --outdir DIR
        [--factors F ...] [--power-sweep [--powers L ...] [--edge-factor F]] [--cadences 48]
        [--holdout-fraction 0.1] [--no-holdout] [--seed 25] [--shard I --num-shards N] [--limit K]
"""

import argparse
import csv
import gc
import logging
import os
from pathlib import Path
import re
import time

import astropy.units as u
import numpy as np

from tglc.epsf import EPSF, fit_epsf, get_default_epsf_flux_mask, make_tglc_design_matrix
from tglc.ffi import FFICutout
from tglc.io import read_cutout_fits
from tglc.light_curve import (
    evaluate_epsf_model,
    get_cutout_window,
    get_design_matrix_rows_for_window,
    make_field_design_matrix,
    make_target_design_matrix,
)
from tglc.scripts.edge_compression_common import (
    CSV_FIELDS,
    DEFAULT_FACTORS,
    DEFAULT_POWERS,
    FLUX_UNCERTAINTY_POWER,
    POWER_CSV_FIELDS,
    RECOMMENDED_EDGE_COMPRESSION,
    middle_of_orbit,
    spoc_equivalent,
)
from tglc.utils.constants import convert_tess_magnitude_to_tess_flux


logger = logging.getLogger(__name__)

MMAG_PER_RELATIVE_FLUX = 1085.7


def slice_cutout(cutout: FFICutout, cadence_indices: np.ndarray) -> FFICutout:
    """Copy of the cutout restricted to the given cadence indices."""
    sliced = object.__new__(FFICutout)
    for name in (
        "size",
        "orbit",
        "sector",
        "camera",
        "ccd",
        "ccd_x",
        "ccd_y",
        "exposure",
        "cutout_x",
        "cutout_y",
        "wcs",
        "mask",
        "gaia",
        "tic",
    ):
        setattr(sliced, name, getattr(cutout, name))
    sliced.flux = np.ascontiguousarray(cutout.flux[cadence_indices])
    sliced.time = cutout.time[cadence_indices]
    sliced.cadence = cutout.cadence[cadence_indices]
    sliced.quality = cutout.quality[cadence_indices]
    return sliced


def clone_epsf(template: EPSF, array: np.ndarray) -> EPSF:
    return EPSF(
        array,
        psf_size=template.psf_size,
        oversample=template.oversample,
        orbit=template.orbit,
        sector=template.sector,
        camera=template.camera,
        ccd=template.ccd,
        cutout_x=template.cutout_x,
        cutout_y=template.cutout_y,
    )


def choose_cadence_indices(
    cutout: FFICutout, archived: EPSF, n_cadences: int, min_good_cadences: int
) -> np.ndarray:
    """Evenly sample good cadences from the stable middle of the orbit."""
    good = (
        (np.asarray(cutout.quality) == 0)
        & ~archived.failed_cadence_mask
        & middle_of_orbit(cutout.time)
    )
    good_indices = np.flatnonzero(good)
    if len(good_indices) < min_good_cadences:
        return np.array([], dtype=int)
    return np.unique(
        good_indices[np.linspace(0, len(good_indices) - 1, n_cadences).round().astype(int)]
    )


def select_isolated_targets(
    positions: np.ndarray, ratios: np.ndarray, tess_mag: np.ndarray, image_shape: tuple[int, int]
) -> list[int]:
    """Isolated mid-brightness diagnostic targets: Tmag 9-13, away from edges and bright
    neighbors."""
    targets = []
    for i in range(len(positions)):
        if not (9 < tess_mag[i] < 13):
            continue
        x, y = positions[i]
        if not (10 < x < image_shape[1] - 10 and 10 < y < image_shape[0] - 10):
            continue
        distance_squared = np.sum((positions - positions[i]) ** 2, axis=1)
        neighbors = (distance_squared > 0) & (distance_squared < 25) & (ratios > 0.2 * ratios[i])
        if not neighbors.any():
            targets.append(i)
    return targets


def prepare_target_diagnostics(
    target_indices: list[int],
    positions: np.ndarray,
    ratios: np.ndarray,
    tess_mag: np.ndarray,
    template: EPSF,
    base_matrix: np.ndarray,
    image_shape: tuple[int, int],
    exposure: float,
) -> list[dict]:
    """Precompute the factor-independent pieces of the per-target diagnostics.

    The target and field design matrices are built from the data rows of the full design matrix,
    which do not change when the regularization rows are rescaled per candidate factor.
    """
    prepared = []
    for i in target_indices:
        x, y = positions[i]
        window = get_cutout_window(x, y, image_shape, cutout_size=5)
        target_design_matrix = make_target_design_matrix(
            template, window.shape, x - window.left, y - window.bottom, ratios[i]
        )
        field_design_matrix = make_field_design_matrix(
            base_matrix[get_design_matrix_rows_for_window(window, image_shape[1])],
            target_design_matrix,
        )
        center_x, center_y = round(x - window.left), round(y - window.bottom)
        prepared.append(
            {
                "window": window,
                "target_design_matrix": target_design_matrix,
                "field_design_matrix": field_design_matrix,
                "aperture": np.s_[:, center_y - 1 : center_y + 2, center_x - 1 : center_x + 2],
                "expected_flux": convert_tess_magnitude_to_tess_flux(tess_mag[i]).to_value(
                    u.electron / u.s
                )
                * exposure,
            }
        )
    return prepared


def evaluate_diagnostics(epsf: EPSF, prepared_targets: list[dict], flux_cube: np.ndarray) -> dict:
    """Per-factor diagnostics: ePSF wing mass and per-target flux capture / scatter medians."""
    grid = np.nanmedian(epsf.psf_parameters, axis=0).reshape(epsf.oversampled_psf_shape)
    half = epsf.oversampled_psf_shape[0] // 2
    grid_y, grid_x = np.mgrid[0 : 2 * half + 1, 0 : 2 * half + 1]
    # Wing = grid nodes farther than +-2 image pixels from the PSF center
    wing = (np.abs(grid_y - half) > 2 * epsf.oversample) | (
        np.abs(grid_x - half) > 2 * epsf.oversample
    )
    wing_mass = float(grid[wing].sum() / grid.sum())

    n_cadences = flux_cube.shape[0]
    fractions = []
    model_data_ratios = []
    scatters = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for target in prepared_targets:
            window = target["window"]
            target_model = evaluate_epsf_model(
                target["target_design_matrix"], epsf.psf_parameters, window.shape
            )
            decontaminated = flux_cube[
                :, window.bottom : window.top, window.left : window.right
            ] - evaluate_epsf_model(target["field_design_matrix"], epsf.array, window.shape)
            aperture = target["aperture"]
            fractions.append(np.nansum(target_model) / (target["expected_flux"] * n_cadences))
            model_data_ratios.append(
                np.nanmedian(
                    np.nansum(target_model[aperture], axis=(1, 2))
                    / np.nansum(decontaminated[aperture], axis=(1, 2))
                )
            )
            aperture_light_curve = np.nansum(decontaminated[aperture], axis=(1, 2))
            scatters.append(
                MMAG_PER_RELATIVE_FLUX
                * 1.4826
                * np.nanmedian(np.abs(aperture_light_curve - np.nanmedian(aperture_light_curve)))
                / np.nanmedian(aperture_light_curve)
            )
    return {
        "wing_mass": wing_mass,
        "flux_fraction_median": float(np.nanmedian(fractions)) if fractions else np.nan,
        "model_data_ratio_median": (
            float(np.nanmedian(model_data_ratios)) if model_data_ratios else np.nan
        ),
        "aperture_scatter_mmag_median": float(np.nanmedian(scatters)) if scatters else np.nan,
    }


def mad(values: np.ndarray) -> float:
    """Plain median absolute deviation (no Gaussian scaling; downstream use is normalized)."""
    return float(np.nanmedian(np.abs(values - np.nanmedian(values))))


def process_cutout(source_path: Path, epsf_path: Path, out_csv: Path, args) -> list[dict]:
    cutout = read_cutout_fits(source_path)
    archived = EPSF.from_fits(epsf_path)
    if not archived.matches_cutout(cutout):
        raise ValueError(f"{epsf_path.name} does not match {source_path.name}")

    cadence_indices = choose_cadence_indices(
        cutout, archived, args.cadences, args.min_good_cadences
    )
    fieldnames = POWER_CSV_FIELDS if args.power_sweep else CSV_FIELDS
    if len(cadence_indices) == 0:
        logger.warning(f"{source_path.name}: too few good cadences, writing empty CSV")
        write_rows(out_csv, [], fieldnames)
        return []

    sliced = slice_cutout(cutout, cadence_indices)
    template = clone_epsf(archived, archived.array[cadence_indices])
    del cutout, archived
    gc.collect()

    image_shape = sliced.flux.shape[1:]
    positions = sliced.star_positions
    ratios = np.asarray(sliced.gaia["tess_flux_ratio"], dtype=np.float64)
    tess_mag = np.asarray(sliced.gaia["tess_mag"], dtype=np.float64)
    flux_cube = np.asarray(sliced.flux, dtype=np.float64)
    badpix = np.ma.getmaskarray(sliced.mask)
    n_cadences = flux_cube.shape[0]

    base_matrix, regularization_size = make_tglc_design_matrix(
        image_shape,
        (template.psf_size, template.psf_size),
        template.oversample,
        positions,
        ratios,
        np.asarray(sliced.mask.data, dtype=np.float64),
        1.0,  # unit regularization block; rescaled in place per candidate factor
    )
    n_image_rows = image_shape[0] * image_shape[1]
    data_matrix = base_matrix[:n_image_rows]
    regularization_block = base_matrix[n_image_rows:].copy()

    if args.no_holdout:
        holdout_map = np.zeros(image_shape, dtype=bool)
    else:
        rng = np.random.default_rng([args.seed, sliced.orbit, sliced.cutout_x, sliced.cutout_y])
        holdout_map = rng.random(image_shape) < args.holdout_fraction

    # Per-cadence pixel sets. The fit mask matches the pipeline (get_default_epsf_flux_mask) plus
    # the holdout pixels; the metric sets additionally exclude non-finite pixels.
    fit_masks, in_sample_sets, holdout_sets, allpix_sets = [], [], [], []
    for t in range(n_cadences):
        default_mask = get_default_epsf_flux_mask(flux_cube[t], badpix)
        finite = np.isfinite(flux_cube[t])
        fittable = ~default_mask & finite
        fit_masks.append(default_mask | holdout_map)
        in_sample_sets.append(fittable & ~holdout_map)
        holdout_sets.append(fittable & holdout_map)
        allpix_sets.append(~badpix & finite)
    n_fit_pixels_median = float(np.median([m.sum() for m in in_sample_sets]))
    n_holdout_pixels_median = float(np.median([m.sum() for m in holdout_sets]))

    target_indices = select_isolated_targets(positions, ratios, tess_mag, image_shape)
    prepared_targets = prepare_target_diagnostics(
        target_indices,
        positions,
        ratios,
        tess_mag,
        template,
        base_matrix,
        image_shape,
        sliced.exposure,
    )

    if args.power_sweep:
        sweep_points = [(args.edge_factor, power) for power in args.powers]
    else:
        sweep_points = [(factor, FLUX_UNCERTAINTY_POWER) for factor in args.factors]

    rows = []
    for factor, power in sweep_points:
        base_matrix[n_image_rows:] = regularization_block * factor
        parameters = np.full_like(template.array, np.nan)
        for t in range(n_cadences):
            try:
                parameters[t] = fit_epsf(
                    base_matrix,
                    flux_cube[t],
                    badpix,
                    power,
                    regularization_size,
                    flux_mask=fit_masks[t],
                )
            except np.linalg.LinAlgError:
                pass
        succeeded = ~np.isnan(parameters).any(axis=1)

        residuals = flux_cube - evaluate_epsf_model(data_matrix, parameters, image_shape)
        mad_in_sample = [mad(residuals[t][in_sample_sets[t]]) for t in np.flatnonzero(succeeded)]
        mad_holdout = [
            mad(residuals[t][holdout_sets[t]])
            for t in np.flatnonzero(succeeded)
            if holdout_sets[t].any()
        ]
        mad_allpix = [mad(residuals[t][allpix_sets[t]]) for t in np.flatnonzero(succeeded)]

        diagnostics = evaluate_diagnostics(
            clone_epsf(template, parameters), prepared_targets, flux_cube
        )
        rows.append(
            {
                "orbit": sliced.orbit,
                "cutout_x": sliced.cutout_x,
                "cutout_y": sliced.cutout_y,
                "camera": sliced.camera,
                "ccd": sliced.ccd,
                "exposure": sliced.exposure,
                **(
                    {"power": power, "edge_factor": factor}
                    if args.power_sweep
                    else {
                        "factor": factor,
                        "spoc_equivalent_factor": spoc_equivalent(factor, sliced.exposure),
                    }
                ),
                "n_cadences": n_cadences,
                "n_failed_fits": int((~succeeded).sum()),
                "n_stars": len(positions),
                "n_fit_pixels_median": n_fit_pixels_median,
                "n_holdout_pixels_median": n_holdout_pixels_median,
                "mad_in_sample": np.nanmedian(mad_in_sample) if mad_in_sample else np.nan,
                "mad_in_sample_p16": (
                    np.nanpercentile(mad_in_sample, 16) if mad_in_sample else np.nan
                ),
                "mad_in_sample_p84": (
                    np.nanpercentile(mad_in_sample, 84) if mad_in_sample else np.nan
                ),
                "mad_holdout": np.nanmedian(mad_holdout) if mad_holdout else np.nan,
                "mad_holdout_p16": np.nanpercentile(mad_holdout, 16) if mad_holdout else np.nan,
                "mad_holdout_p84": np.nanpercentile(mad_holdout, 84) if mad_holdout else np.nan,
                "mad_allpix": np.nanmedian(mad_allpix) if mad_allpix else np.nan,
                "n_isolated_targets": len(prepared_targets),
                **diagnostics,
            }
        )

    write_rows(out_csv, rows, fieldnames)
    return rows


def write_rows(out_csv: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write the CSV atomically so an interrupted run never leaves a partial file behind."""
    temporary_path = out_csv.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, out_csv)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--epsf-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=DEFAULT_FACTORS,
        help="Candidate edge-compression factors in TICA units (electrons per cadence). "
        "Must be identical for every cutout in a campaign.",
    )
    parser.add_argument(
        "--power-sweep",
        action="store_true",
        help="Sweep the flux-uncertainty weighting power at a fixed edge-compression factor "
        "(Han & Brandt 2023 Figure 4) instead of sweeping factors at the default power.",
    )
    parser.add_argument(
        "--powers",
        type=float,
        nargs="+",
        default=DEFAULT_POWERS,
        help="Candidate weighting powers for --power-sweep.",
    )
    parser.add_argument(
        "--edge-factor",
        type=float,
        default=RECOMMENDED_EDGE_COMPRESSION,
        help="Fixed edge-compression factor (TICA units) used during --power-sweep.",
    )
    parser.add_argument("--cadences", type=int, default=48)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--no-holdout", action="store_true")
    parser.add_argument("--seed", type=int, default=25, help="Base seed (default: issue number)")
    parser.add_argument("--min-good-cadences", type=int, default=10)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.factors = sorted(set(args.factors))
    args.powers = sorted(set(args.powers))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    csv_prefix = "power" if args.power_sweep else "sweep"
    swept = args.powers if args.power_sweep else args.factors
    args.outdir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for epsf_path in sorted(args.epsf_dir.glob("epsf_*_*.fits")):
        x, y = re.match(r"epsf_(\d+)_(\d+)\.fits", epsf_path.name).groups()
        source_path = args.source_dir / f"source_{x}_{y}.fits"
        if source_path.exists():
            pairs.append((source_path, epsf_path, args.outdir / f"{csv_prefix}_{x}_{y}.csv"))
    pairs = pairs[args.shard :: args.num_shards]
    if args.limit:
        pairs = pairs[: args.limit]
    logger.info(
        f"shard {args.shard}/{args.num_shards}: {len(pairs)} cutouts, "
        f"{len(swept)} {csv_prefix} values"
    )

    for source_path, epsf_path, out_csv in pairs:
        if out_csv.exists():
            logger.info(f"{out_csv.name} exists, skipping")
            continue
        start = time.monotonic()
        try:
            rows = process_cutout(source_path, epsf_path, out_csv, args)
        except Exception as error:  # keep the sweep going; missing CSV marks it for retry
            logger.error(f"{source_path.name} FAILED: {error}")
            continue
        finally:
            gc.collect()
        logger.info(f"{out_csv.name}: {len(rows)} values in {time.monotonic() - start:.0f}s")


if __name__ == "__main__":
    main()
