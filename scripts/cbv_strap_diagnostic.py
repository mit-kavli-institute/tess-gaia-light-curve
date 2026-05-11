"""
Compare strap-column residual structure between TGLC source pickles produced
with and without FFI-level CBV correction.

This is a *decision-support* script: TGLC's ePSF design matrix already adds
strap-mask background columns (``tglc/epsf.py:make_tglc_design_matrix``). If
the QLP-CBV product already absorbs strap-related common modes, leaving the
ePSF strap columns on would be a double correction. This script quantifies
how much per-cadence strap-vs-non-strap residual structure remains in each of
the two cutout pickles, so we can decide empirically whether to disable the
ePSF strap-mask path when CBV correction is applied.

Usage:
    python scripts/cbv_strap_diagnostic.py \\
        --no-cbv  path/to/source_X_Y.pkl     (uncorrected run)  \\
        --cbv     path/to/source_X_Y.pkl     (CBV-corrected run) \\
        --out-dir path/to/diagnostic_outputs

Produces:
  - <out>/summary.txt          numerical comparison
  - <out>/strap_residuals.png  per-cadence median strap vs. non-strap flux
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path
import pickle

from astropy.io import fits
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tglc.utils import data as tglc_data  # noqa: E402


def load_strap_mask(camera: int, ccd: int) -> np.ndarray:
    """Load the per-column strap mask TGLC uses for ePSF background fitting.

    The packaged ``median_mask.fits`` is shape (16, 2048) — one row per
    (camera, ccd) pair (camera-major). Strap columns are those where the
    mask is 0 (true background pixels have mask > 0); we invert to get the
    strap-column flag.
    """
    mask_file = resources.files(tglc_data) / "median_mask.fits"
    with fits.open(mask_file) as hdul:
        per_column = hdul[0].data[(camera - 1) * 4 + (ccd - 1), :]
    return per_column == 0  # True where this column is a strap


def cutout_strap_column_flag(source, strap_columns: np.ndarray) -> np.ndarray:
    """Boolean (n_cols,) flag for strap columns inside this cutout."""
    # Source records its (x, y) origin in the FFI as source.x, source.y attributes;
    # the slice is flux = ffi_flux[:, y : y + size, x : x + size].
    x = getattr(source, "x", None)
    size = source.flux.shape[2]
    if x is None:
        # Fall back: assume the cutout's strap pattern was not stored — give up.
        raise RuntimeError(
            "Source pickle is missing the 'x' attribute; cannot map strap columns into cutout."
        )
    return strap_columns[x : x + size]


def per_cadence_median(flux: np.ndarray, col_flag: np.ndarray) -> np.ndarray:
    """Median flux over selected columns, per cadence. flux: (n_cad, h, w)."""
    if not col_flag.any():
        return np.full(flux.shape[0], np.nan)
    return np.nanmedian(flux[:, :, col_flag], axis=(1, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-cbv", type=Path, required=True, help="Uncorrected source pickle.")
    parser.add_argument("--cbv", type=Path, required=True, help="CBV-corrected source pickle.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.no_cbv, "rb") as f:
        src_no_cbv = pickle.load(f)
    with open(args.cbv, "rb") as f:
        src_cbv = pickle.load(f)

    assert src_no_cbv.camera == src_cbv.camera and src_no_cbv.ccd == src_cbv.ccd, (
        "source pickles must be from the same camera/ccd"
    )
    strap_columns = load_strap_mask(src_no_cbv.camera, src_no_cbv.ccd)
    cutout_strap_no = cutout_strap_column_flag(src_no_cbv, strap_columns)
    cutout_strap_yes = cutout_strap_column_flag(src_cbv, strap_columns)
    # The two should agree if cutout origin matches.
    np.testing.assert_array_equal(cutout_strap_no, cutout_strap_yes)
    cutout_strap = cutout_strap_no

    strap_med_no = per_cadence_median(src_no_cbv.flux, cutout_strap)
    other_med_no = per_cadence_median(src_no_cbv.flux, ~cutout_strap)
    strap_med_yes = per_cadence_median(src_cbv.flux, cutout_strap)
    other_med_yes = per_cadence_median(src_cbv.flux, ~cutout_strap)

    def summarise(strap, other, tag):
        ratio = strap / other
        return (
            f"{tag}: median strap/non-strap = {np.nanmedian(ratio):.5f}, "
            f"std(strap-non-strap residual) = "
            f"{np.nanstd(strap - other):.4g}, "
            f"std(strap residual time-series) = {np.nanstd(strap - np.nanmedian(strap)):.4g}"
        )

    summary_lines = [
        f"camera={src_no_cbv.camera}, ccd={src_no_cbv.ccd}",
        f"cutout shape: {src_no_cbv.flux.shape}; strap columns in cutout: {int(cutout_strap.sum())}",
        summarise(strap_med_no, other_med_no, "without CBV"),
        summarise(strap_med_yes, other_med_yes, "with CBV"),
    ]
    summary = "\n".join(summary_lines)
    print(summary)
    (args.out_dir / "summary.txt").write_text(summary + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(strap_med_no, label="strap columns", alpha=0.8)
    axes[0].plot(other_med_no, label="non-strap columns", alpha=0.8)
    axes[0].set_title("without CBV correction")
    axes[0].set_ylabel("median flux")
    axes[0].legend()
    axes[1].plot(strap_med_yes, label="strap columns", alpha=0.8)
    axes[1].plot(other_med_yes, label="non-strap columns", alpha=0.8)
    axes[1].set_title("with CBV correction")
    axes[1].set_xlabel("cadence index")
    axes[1].set_ylabel("median flux")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "strap_residuals.png", dpi=120)
    print(f"wrote {args.out_dir / 'strap_residuals.png'}")


if __name__ == "__main__":
    main()
