"""Shared constants and helpers for the edge-compression calibration sweep (issue #25).

The ePSF fit weights data rows by ``1/|flux|^flux_uncertainty_power`` but gives the
edge-compression regularization rows unit weight, so the effective regularization strength scales
with the image's flux units to that power. Upstream TGLC calibrated ``edge_compression=1e-4`` on
SPOC e-/s images; this pipeline fits TICA cutouts in electrons per cadence, making the shipped
value ~exposure^1.4 (~1200x at 158.4 s) stronger than calibrated.

The ``edge_compression_sweep`` / ``edge_compression_figure`` modules experimentally re-determine
the factor in TICA units following Han & Brandt 2023 (AJ 165:71) Figure 4: fit many cutouts at
each candidate factor, compare normalized MAD-of-residual-image curves per cutout and in
aggregate, and pick the best value. The candidate factor grid must be identical for every cutout
in a sweep campaign so the per-cutout curves can be aggregated.
"""

import numpy as np


FLUX_UNCERTAINTY_POWER = 1.4
UPSTREAM_EDGE_COMPRESSION = 1e-4
# Upstream's 1e-4 converted from SPOC e-/s to TICA electrons per cadence at the sector 90+
# exposure time: 1e-4 / 158.4^1.4 (158.4^1.4 = 1201.3).
UNIT_CONVERTED_ANCHOR = 8.32e-8

# Half-decade log grid spanning 1e-9 to 1e-3, plus three anchors: 0 (no regularization), the
# unit-converted upstream value, and upstream's raw 1e-4 (the over-strong regime in TICA units).
DEFAULT_FACTORS = [
    0.0,
    1e-9,
    3.16e-9,
    1e-8,
    3.16e-8,
    8.32e-8,
    1e-7,
    3.16e-7,
    1e-6,
    3.16e-6,
    1e-5,
    3.16e-5,
    1e-4,
    3.16e-4,
    1e-3,
]

CSV_FIELDS = [
    "orbit",
    "cutout_x",
    "cutout_y",
    "camera",
    "ccd",
    "exposure",
    "factor",
    "spoc_equivalent_factor",
    "n_cadences",
    "n_failed_fits",
    "n_stars",
    "n_fit_pixels_median",
    "n_holdout_pixels_median",
    "mad_in_sample",
    "mad_in_sample_p16",
    "mad_in_sample_p84",
    "mad_holdout",
    "mad_holdout_p16",
    "mad_holdout_p84",
    "mad_allpix",
    "wing_mass",
    "n_isolated_targets",
    "flux_fraction_median",
    "model_data_ratio_median",
    "aperture_scatter_mmag_median",
]


def spoc_equivalent(factor: float, exposure: float) -> float:
    """Express a TICA-units (electrons per cadence) factor in SPOC-equivalent (e-/s) units."""
    return factor * exposure**FLUX_UNCERTAINTY_POWER


def middle_of_orbit(time: np.ndarray) -> np.ndarray:
    """Mask for the stable middle of the orbit: the first ~20% and last ~10% of the time span
    are dominated by scattered-light systematics."""
    start, end = time.min(), time.max()
    return (time > start + 0.2 * (end - start)) & (time < end - 0.1 * (end - start))


def select_largest_within_tolerance(
    factors: np.ndarray, aggregate: np.ndarray, tolerance: float
) -> float:
    """Largest factor whose aggregate curve value is within (1 + tolerance) of the curve minimum.

    The in-sample residual MAD is expected to be flat at weak regularization and rise at a knee
    (regularization can only increase the in-sample residual), so the calibrated choice is the
    strongest factor still on the plateau.
    """
    eligible = np.asarray(aggregate) <= (1 + tolerance) * np.min(aggregate)
    return float(np.asarray(factors)[eligible].max())


def select_minimum(factors: np.ndarray, aggregate: np.ndarray) -> float:
    """Factor at the aggregate curve minimum (ties broken toward the larger factor)."""
    aggregate = np.asarray(aggregate)
    return float(np.asarray(factors)[np.flatnonzero(aggregate == aggregate.min())[-1]])
