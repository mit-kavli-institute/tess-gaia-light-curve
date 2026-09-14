"""Synthetic data-product builders shared between test modules."""

from astropy.table import MaskedColumn, Table
from astropy.wcs import WCS
import numpy as np

from tglc.epsf import EPSF_BACKGROUND_COLUMNS
from tglc.ffi import FFICutout


def make_synthetic_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [75.0, 75.0]
    wcs.wcs.crval = [120.5, -45.25]
    wcs.wcs.cdelt = [-0.00583, 0.00583]  # ~21" per pixel, comparable to TESS
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def make_synthetic_cutout(
    *,
    n_cadences: int = 5,
    size: int = 12,
    n_stars: int = 4,
    sector: int = 89,
) -> FFICutout:
    """Build an FFICutout by hand, bypassing the heavy __init__ logic.

    Mirrors what `read_cutout_fits` does internally and lets the tests cover
    each persisted attribute without depending on the full ingestion pipeline.
    """
    rng = np.random.default_rng(seed=42)

    cutout = object.__new__(FFICutout)
    cutout.size = size
    cutout.orbit = 185
    cutout.sector = sector
    cutout.camera = 1
    cutout.ccd = 1
    cutout.exposure = 200
    cutout.ccd_x = 44
    cutout.ccd_y = 0
    cutout.cutout_x = 0
    cutout.cutout_y = 0

    cutout.wcs = make_synthetic_wcs()
    cutout.flux = rng.normal(100.0, 5.0, size=(n_cadences, size, size)).astype(np.float32)

    strap_weights = rng.normal(1.0, 0.02, size=(size, size)).astype(np.float32)
    bad_pixels = np.zeros((size, size), dtype=bool)
    bad_pixels[2, 3] = True
    bad_pixels[7, 8] = True
    cutout.mask = np.ma.masked_array(strap_weights, mask=bad_pixels)

    cutout.time = np.linspace(3500.0, 3500.05, n_cadences).astype(np.float64)
    cutout.cadence = np.arange(1000, 1000 + n_cadences, dtype=np.int64)
    cutout.quality = np.zeros(n_cadences, dtype=np.int32)

    designations = [f"Gaia DR3 {1000 + i}" for i in range(n_stars)]
    pmra_values = np.array([1.2, np.nan, 0.5, np.nan], dtype=np.float64)
    pmdec_values = np.array([np.nan, -0.3, 0.0, np.nan], dtype=np.float64)
    gaia = Table(
        {
            "designation": designations,
            "ra": np.array([120.4, 120.5, 120.6, 120.55], dtype=np.float64),
            "dec": np.array([-45.2, -45.25, -45.3, -45.28], dtype=np.float64),
            "phot_g_mean_mag": np.array([10.0, 11.5, 12.3, 13.0], dtype=np.float64),
            "phot_bp_mean_mag": np.array([10.2, 11.7, 12.6, 13.3], dtype=np.float64),
            "phot_rp_mean_mag": np.array([9.7, 11.1, 12.0, 12.7], dtype=np.float64),
            "pmra": MaskedColumn(pmra_values, mask=np.isnan(pmra_values)),
            "pmdec": MaskedColumn(pmdec_values, mask=np.isnan(pmdec_values)),
            "tess_mag": np.array([9.95, 11.45, 12.25, 12.95], dtype=np.float64),
            "tess_flux": np.array([1500.0, 380.0, 180.0, 80.0], dtype=np.float64),
            "tess_flux_ratio": np.array([1.0, 0.25, 0.12, 0.05], dtype=np.float64),
            f"sector_{sector}_x": np.array([2.5, 5.5, 8.5, 10.5], dtype=np.float64),
            f"sector_{sector}_y": np.array([3.5, 6.5, 8.5, 9.5], dtype=np.float64),
        }
    )
    cutout.gaia = gaia

    cutout.tic = Table(
        {
            "TIC": np.array([500001, 500002, 500003, 500004], dtype=np.int64),
            "gaia3": np.array([1000, 1001, 1002, 1003], dtype=np.int64),
        }
    )

    return cutout


def make_synthetic_epsf(n_cadences: int = 3, psf_size: int = 11, oversample: int = 2):
    k = (psf_size * oversample + 1) ** 2 + len(EPSF_BACKGROUND_COLUMNS)
    return np.linspace(0.0, 1.0, n_cadences * k).reshape(n_cadences, k).astype(np.float64)
