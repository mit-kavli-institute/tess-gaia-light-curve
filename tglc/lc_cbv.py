"""
Apply QLP-CBV lightcurve corrections to TGLC per-target aperture series.

The lightcurve CBV product (separate repo: QLP-CBV, branch
``feature/lc-cbv-fits-emission``) is one per-``(orbit, cam, ccd)`` FITS file
that holds a basis set ``CBVS`` (shape ``(NCBV, NCAD)``) plus per-target
coefficients fit against median-normalised aperture lightcurves. The trend
``theta_i @ CBVS`` is the systematic component to subtract from one target's
normalised flux series.

The schema is documented at
https://www.notion.so/360e299747a781fbab38fb0be1a7fd38. Both LC and FFI
products carry ``FORMAT_V = "draft-1"``; they're disambiguated by the
``PRODUCT`` keyword (``"qlp-cbv-lc"`` here, ``"qlp-cbv-ffi"`` for the FFI
sibling — see :mod:`tglc.ffi_cbv`).

Only single-scale layouts (``NBANDS == 0``) are supported. Multiscale
(``NBANDS > 0``) replaces ``CBVS`` / ``SVALS`` / ``WEIGHTS_ROBUST`` with
per-band HDUs and is deferred.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import numpy as np


logger = logging.getLogger(__name__)

SUPPORTED_FORMAT_VERSION = "draft-1"
PRODUCT_LIGHTCURVE_CBV = "qlp-cbv-lc"
SUPPORTED_STELCAT = "TIC82"


class UnsupportedLCCBVFormatError(ValueError):
    """Raised when a FITS file isn't a draft-1 ``qlp-cbv-lc`` we can consume."""


@dataclass
class LCCBVData:
    """Loaded per-(orbit, cam, ccd) lightcurve CBV product.

    ``theta`` is already MAP-blended when the producer wrote ``WEIGHTS_MAP`` +
    ``MAP_MASK``: rows where the mask is True are taken from ``WEIGHTS_MAP``;
    the rest stay at ``WEIGHTS_ROBUST``.
    """

    orbit: int
    camera: int
    ccd: int
    cadences: np.ndarray  # (NCAD,) int64
    cbvs: np.ndarray  # (NCBV, NCAD) float64
    star_ids: np.ndarray  # (NTGT,) int64
    theta: np.ndarray  # (NTGT, NCBV) float32 — robust, MAP-blended in-place
    _id_to_idx: dict[int, int] = field(default_factory=dict, repr=False)

    def trend_for_target(self, tic_id: int, cadences: np.ndarray) -> np.ndarray | None:
        """Reconstruct ``theta_i @ CBVS`` for one target at the input cadences.

        Returns ``None`` if the TIC isn't in this product. Output has the
        same shape as ``cadences``; cadences not in the CBV training set are
        passed through with a 0 contribution (no correction at those points).
        """
        idx = self._id_to_idx.get(int(tic_id))
        if idx is None:
            return None
        theta_i = self.theta[idx].astype(np.float64)  # (NCBV,)
        cadences = np.asarray(cadences, dtype=np.int64)
        out = np.zeros(cadences.shape, dtype=np.float64)
        # CADENCES is not guaranteed monotonic; sort for the searchsorted lookup.
        sorter = np.argsort(self.cadences)
        sorted_cad = self.cadences[sorter]
        positions = np.searchsorted(sorted_cad, cadences)
        in_bounds = positions < len(sorted_cad)
        found = np.zeros_like(cadences, dtype=bool)
        found[in_bounds] = sorted_cad[positions[in_bounds]] == cadences[in_bounds]
        if found.any():
            t_idx = sorter[positions[found]]
            out[found] = theta_i @ self.cbvs[:, t_idx]
        return out


def load_lc_cbvs(path: Path) -> LCCBVData:
    """Read a draft-1 ``qlp-cbv-lc`` FITS file.

    Raises
    ------
    UnsupportedLCCBVFormatError
        If ``FORMAT_V`` / ``PRODUCT`` / ``STELCAT`` are unknown, or
        ``NBANDS > 0`` (multiscale not yet supported), or row/shape
        invariants are violated.
    """
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        with fits.open(path, mode="readonly", memmap=False) as hdul:
            ph = dict(hdul[0].header)
            fmt = ph.get("FORMAT_V")
            product = ph.get("PRODUCT")
            stelcat = ph.get("STELCAT")
            if fmt != SUPPORTED_FORMAT_VERSION or product != PRODUCT_LIGHTCURVE_CBV:
                raise UnsupportedLCCBVFormatError(
                    f"unsupported FORMAT_V/PRODUCT in {path}: "
                    f"FORMAT_V={fmt!r}, PRODUCT={product!r} "
                    f"(supports {SUPPORTED_FORMAT_VERSION!r}/{PRODUCT_LIGHTCURVE_CBV!r})"
                )
            if stelcat != SUPPORTED_STELCAT:
                raise UnsupportedLCCBVFormatError(
                    f"unsupported STELCAT in {path}: {stelcat!r} (expected {SUPPORTED_STELCAT!r})"
                )
            nbands = int(ph.get("NBANDS", 0))
            if nbands > 0:
                raise UnsupportedLCCBVFormatError(
                    f"multiscale lc-CBV (NBANDS={nbands}) not supported in {path}"
                )

            ncad = int(ph["NCAD"])
            ncbv = int(ph["NCBV"])
            ntgt = int(ph["NTGT"])
            has_map = bool(ph["HASMAP"])
            orbit = int(ph["ORBIT"])
            camera = int(ph["CAMERA"])
            ccd = int(ph["CCD"])

            cadences = np.asarray(hdul["CADENCES"].data, dtype=np.int64)
            if cadences.size != ncad:
                raise UnsupportedLCCBVFormatError(
                    f"CADENCES length {cadences.size} != NCAD={ncad} in {path}"
                )

            cbvs = np.asarray(hdul["CBVS"].data, dtype=np.float64)
            if cbvs.shape != (ncbv, ncad):
                raise UnsupportedLCCBVFormatError(
                    f"CBVS shape {cbvs.shape} != (NCBV={ncbv}, NCAD={ncad}) in {path}"
                )

            wt = hdul["WEIGHTS_ROBUST"].data
            star_ids_long = np.asarray(wt["STAR_ID"], dtype=np.int64)
            weights_long = np.asarray(wt["WEIGHT"], dtype=np.float32)
            if star_ids_long.size != ntgt * ncbv:
                raise UnsupportedLCCBVFormatError(
                    f"WEIGHTS_ROBUST has {star_ids_long.size} rows; "
                    f"expected NTGT*NCBV={ntgt * ncbv} in {path}"
                )

            star_ids = star_ids_long.reshape(ntgt, ncbv)[:, 0]
            theta_robust = weights_long.reshape(ntgt, ncbv)

            theta = theta_robust.copy()
            if has_map and "WEIGHTS_MAP" in hdul and "MAP_MASK" in hdul:
                map_table = hdul["WEIGHTS_MAP"].data
                theta_map = np.asarray(map_table["WEIGHT"], dtype=np.float32).reshape(ntgt, ncbv)
                map_mask = np.asarray(hdul["MAP_MASK"].data, dtype=bool)
                theta[map_mask] = theta_map[map_mask]

    id_to_idx = {int(tic): i for i, tic in enumerate(star_ids)}
    return LCCBVData(
        orbit=orbit,
        camera=camera,
        ccd=ccd,
        cadences=cadences,
        cbvs=cbvs,
        star_ids=star_ids,
        theta=theta,
        _id_to_idx=id_to_idx,
    )
