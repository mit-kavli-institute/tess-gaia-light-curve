"""
Apply QLP-CBV FFI corrections to TGLC's per-cadence FFI flux array.

The CBV product (separate repo: QLP-CBV ``feature/ffi-cbv-generation``) is a per-CCD
FITS file describing the pixel-by-pixel common-mode systematic trend as
``trend(p, t) = theta_p · C^(S)(t)``. We subtract that trend from the raw FFI
pixels before cutouts are written, so downstream ePSF fitting and light-curve
extraction operate on detrended flux.

Pixel coordinates in the CBV product are imaging-area-local
(``TESS_IMAGING_BBOX = (0, 2048, 44, 2092)``). TGLC's flux array, after
SCIPIXS slicing in ``tglc.ffi._get_ffi_header_data_and_flux``, is indexed in the
same frame, so no coordinate offset is needed.

Only ``FORMAT_V = "draft-1"`` is supported. Unknown versions raise
``UnsupportedCBVFormatError`` — the schema is still in flux.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import warnings

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import numpy as np


logger = logging.getLogger(__name__)

SUPPORTED_FORMAT_VERSION = "draft-1"


class UnsupportedCBVFormatError(ValueError):
    """Raised when a CBV FITS file declares a FORMAT_V we don't understand."""


@dataclass
class CBVSlice:
    """One column-strip slice of the CBV product."""

    label: str
    cbvs: np.ndarray  # (n_cbvs, n_cadences) float64
    singular_values: np.ndarray  # (n_cbvs,) float64
    theta: np.ndarray  # (n_pixels, n_cbvs) float32
    pixel_row: np.ndarray  # (n_pixels,) int32, imaging-local
    pixel_col: np.ndarray  # (n_pixels,) int32, imaging-local
    col_start: int  # inclusive, imaging-local
    col_stop: int  # exclusive, imaging-local
    slice_height: int


def load_ffi_cbvs(
    path: Path,
) -> tuple[dict, np.ndarray, dict[str, CBVSlice]]:
    """Read a draft-1 FFI CBV FITS file.

    Returns
    -------
    meta : dict
        Primary-HDU header (FORMAT_V, CAMERA, CCD, NCAD, ...).
    cadences : ndarray of int64, shape (NCAD,)
        Cadence reference values present in the CBV training set.
    slices : dict[str, CBVSlice]
        Slice payloads keyed by slice label.

    Raises
    ------
    UnsupportedCBVFormatError
        If FORMAT_V is missing or not "draft-1".
    """
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        with fits.open(path, mode="readonly", memmap=False) as hdul:
            meta = dict(hdul[0].header)
            fmt = meta.get("FORMAT_V")
            if fmt != SUPPORTED_FORMAT_VERSION:
                raise UnsupportedCBVFormatError(
                    f"unsupported FORMAT_V in {path}: {fmt!r} "
                    f"(this code supports {SUPPORTED_FORMAT_VERSION!r})"
                )
            cadences = np.asarray(hdul["CADENCES"].data, dtype=np.int64)

            # First pass: collect CBV time series and WEIGHTS tables, keyed by slice label.
            cbv_lists: dict[str, dict[int, tuple[np.ndarray, float]]] = {}
            weights_payloads: dict[str, dict] = {}

            for hdu in hdul[1:]:
                name = hdu.name
                if name == "CADENCES":
                    continue
                if name.startswith("WEIGHTS_"):
                    label = name[len("WEIGHTS_") :]
                    tbl = hdu.data
                    n_cbv = int(hdu.header["NCBV"])
                    theta = np.stack(
                        [np.asarray(tbl[f"THETA_{k + 1:02d}"]) for k in range(n_cbv)],
                        axis=1,
                    )  # (n_pixels, n_cbvs)
                    weights_payloads[label] = {
                        "theta": theta,
                        "pixel_id": np.asarray(tbl["PIXEL_ID"], dtype=np.int64),
                        "pixel_row": np.asarray(tbl["PIXEL_ROW"], dtype=np.int32),
                        "pixel_col": np.asarray(tbl["PIXEL_COL"], dtype=np.int32),
                        "col_start": int(hdu.header["COLSTART"]),
                        "col_stop": int(hdu.header["COLSTOP"]),
                        "slice_height": int(hdu.header["SLICEH"]),
                    }
                elif name.startswith("CBV_"):
                    # CBV_{label}_{kk}; rsplit on the right so labels containing '_' would survive.
                    body, kk = name.rsplit("_", 1)
                    label = body[len("CBV_") :]
                    k = int(kk) - 1
                    cbv_lists.setdefault(label, {})[k] = (
                        np.asarray(hdu.data, dtype=np.float64),
                        float(hdu.header["SVALUE"]),
                    )

    slices: dict[str, CBVSlice] = {}
    for label, weights in weights_payloads.items():
        if label not in cbv_lists:
            raise ValueError(
                f"WEIGHTS_{label} present but no matching CBV_{label}_* HDUs in {path}"
            )
        kk_map = cbv_lists[label]
        ordered = [kk_map[k] for k in sorted(kk_map)]
        cbvs = np.stack([c for c, _ in ordered], axis=0)
        svals = np.asarray([s for _, s in ordered], dtype=np.float64)
        slices[label] = CBVSlice(
            label=label,
            cbvs=cbvs,
            singular_values=svals,
            theta=weights["theta"],
            pixel_row=weights["pixel_row"],
            pixel_col=weights["pixel_col"],
            col_start=weights["col_start"],
            col_stop=weights["col_stop"],
            slice_height=weights["slice_height"],
        )

    return meta, cadences, slices


def apply_cbv_correction(
    flux: np.ndarray,
    cadence: np.ndarray,
    cbv_path: Path,
) -> None:
    """Subtract the CBV systematic trend from ``flux`` in place.

    Parameters
    ----------
    flux : ndarray, shape (n_cadences, 2048, 2048)
        TGLC's time-sorted FFI flux cube, in imaging-area-local pixel coords.
        Modified in place.
    cadence : ndarray of int, shape (n_cadences,)
        Cadence numbers aligned with ``flux``'s leading axis.
    cbv_path : Path
        Path to a draft-1 FFI CBV FITS file.

    Notes
    -----
    Operates slice-by-slice to keep peak memory at one slice's correction
    matrix on top of ``flux`` itself. Cadences not present in the CBV training
    set are passed through unchanged; pixels absent from any ``WEIGHTS_{S}``
    are passed through unchanged.
    """
    meta, cbv_cadences, slices = load_ffi_cbvs(cbv_path)
    logger.info(
        "Loaded CBV product %s (FORMAT_V=%s, NCAD=%d, slices=%s)",
        cbv_path,
        meta.get("FORMAT_V"),
        len(cbv_cadences),
        sorted(slices.keys()),
    )

    cadence = np.asarray(cadence, dtype=np.int64)
    # Map every input cadence to its column in the CBV time-series array (or -1 if absent).
    cbv_sorter = np.argsort(cbv_cadences)
    sorted_cbv_cadences = cbv_cadences[cbv_sorter]
    positions = np.searchsorted(sorted_cbv_cadences, cadence)
    in_bounds = positions < len(sorted_cbv_cadences)
    found = np.zeros_like(cadence, dtype=bool)
    found[in_bounds] = sorted_cbv_cadences[positions[in_bounds]] == cadence[in_bounds]
    n_matched = int(found.sum())
    n_missing = int((~found).sum())
    logger.info(
        "Cadence coverage: %d/%d matched, %d uncorrected (passed through)",
        n_matched,
        len(cadence),
        n_missing,
    )
    if n_matched == 0:
        logger.warning(
            "No input cadences overlap with CBV cadences in %s — flux unchanged.",
            cbv_path,
        )
        return

    # Column-index into each slice's CBV matrix, only for matched input cadences.
    t_idx = cbv_sorter[positions[found]]
    matched_rows = np.nonzero(found)[0]

    for label, slc in sorted(slices.items()):
        # CBVs sampled at the input cadences (matched only): (n_cbvs, n_matched)
        cbvs_at_t = slc.cbvs[:, t_idx]
        # Per-pixel trend at matched cadences: (n_pixels, n_matched), cast to flux dtype
        # so we don't carry a float64 cube into the subtraction.
        trend_pixels = (slc.theta.astype(np.float64) @ cbvs_at_t).astype(flux.dtype, copy=False)

        rows = slc.pixel_row.astype(np.intp)
        cols = slc.pixel_col.astype(np.intp)
        # Loop over matched cadences to keep peak memory at O(n_pixels) extra. With a
        # scalar t and array (rows, cols), ``flux[t, rows, cols] -= x`` is an in-place
        # fancy assignment on ``flux`` itself (pixel coordinates are unique per slice).
        for i, t in enumerate(matched_rows):
            flux[t, rows, cols] -= trend_pixels[:, i]

        logger.debug(
            "Slice %s: subtracted trend for %d pixels across %d cadences",
            label,
            len(rows),
            len(matched_rows),
        )
