"""Unit tests for tglc.lc_cbv: loading and applying QLP-CBV lightcurve products."""

from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from tglc.lc_cbv import (
    PRODUCT_LIGHTCURVE_CBV,
    SUPPORTED_FORMAT_VERSION,
    SUPPORTED_STELCAT,
    UnsupportedLCCBVFormatError,
    load_lc_cbvs,
)


def _write_minimal_lc_cbv_fits(
    path: Path,
    *,
    orbit: int = 42,
    camera: int = 1,
    ccd: int = 3,
    cadences: np.ndarray,
    cbvs: np.ndarray,  # (NCBV, NCAD)
    star_ids: np.ndarray,  # (NTGT,)
    theta_robust: np.ndarray,  # (NTGT, NCBV)
    theta_map: np.ndarray | None = None,  # (NTGT, NCBV)
    map_mask: np.ndarray | None = None,  # (NTGT,) bool
    format_v: str = SUPPORTED_FORMAT_VERSION,
    product: str = PRODUCT_LIGHTCURVE_CBV,
    stelcat: str = SUPPORTED_STELCAT,
    nbands: int = 0,
) -> None:
    """Write a minimal draft-1 qlp-cbv-lc FITS file, single-scale."""
    ncbv, ncad = cbvs.shape
    ntgt = star_ids.size
    has_map = theta_map is not None and map_mask is not None

    primary = fits.PrimaryHDU()
    primary.header["FORMAT_V"] = format_v
    primary.header["PRODUCT"] = product
    primary.header["STELCAT"] = stelcat
    primary.header["ORBIT"] = orbit
    primary.header["CAMERA"] = camera
    primary.header["CCD"] = ccd
    primary.header["NCAD"] = int(ncad)
    primary.header["NCBV"] = int(ncbv)
    primary.header["NTGT"] = int(ntgt)
    primary.header["NBANDS"] = int(nbands)
    primary.header["HASMAP"] = bool(has_map)
    primary.header["DATE"] = "2026-05-14T00:00:00"

    cadences_hdu = fits.ImageHDU(data=cadences.astype(np.int64), name="CADENCES")
    cbvs_hdu = fits.ImageHDU(data=cbvs.astype(np.float64), name="CBVS")
    svals_hdu = fits.ImageHDU(data=np.linspace(1.0, 0.1, ncbv).astype(np.float64), name="SVALS")

    # Long-form WEIGHTS_ROBUST: NTGT * NCBV rows in (target, cbv-index) order.
    star_long = np.repeat(star_ids.astype(np.int64), ncbv)
    cbvn_long = np.tile(np.arange(1, ncbv + 1, dtype=np.int32), ntgt)
    weight_long = theta_robust.astype(np.float32).reshape(-1)
    wt_cols = [
        fits.Column(name="STAR_ID", format="K", array=star_long),
        fits.Column(name="CBV_N", format="J", array=cbvn_long),
        fits.Column(name="WEIGHT", format="E", array=weight_long),
    ]
    weights_robust = fits.BinTableHDU.from_columns(wt_cols, name="WEIGHTS_ROBUST")
    weights_robust.header["NTGT"] = int(ntgt)
    weights_robust.header["NCBV"] = int(ncbv)
    weights_robust.header["NROW"] = int(ntgt * ncbv)

    hdus = [primary, cadences_hdu, cbvs_hdu, svals_hdu, weights_robust]

    if has_map:
        # WEIGHTS_MAP has the same shape (NTGT * NCBV rows).
        map_long = theta_map.astype(np.float32).reshape(-1)  # type: ignore[union-attr]
        map_cols = [
            fits.Column(name="STAR_ID", format="K", array=star_long),
            fits.Column(name="CBV_N", format="J", array=cbvn_long),
            fits.Column(name="WEIGHT", format="E", array=map_long),
        ]
        weights_map = fits.BinTableHDU.from_columns(map_cols, name="WEIGHTS_MAP")
        weights_map.header["NTGT"] = int(ntgt)
        weights_map.header["NCBV"] = int(ncbv)
        weights_map.header["NROW"] = int(ntgt * ncbv)
        map_mask_hdu = fits.ImageHDU(
            data=map_mask.astype(np.uint8),  # type: ignore[union-attr]
            name="MAP_MASK",
        )
        hdus.extend([weights_map, map_mask_hdu])

    fits.HDUList(hdus).writeto(path, overwrite=True)


def _make_spec(n_cad: int, n_targets: int, n_cbvs: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cadences = (1000 + np.arange(n_cad)).astype(np.int64)
    cbvs = rng.standard_normal((n_cbvs, n_cad)).astype(np.float64)
    star_ids = (10000 + np.arange(n_targets)).astype(np.int64)
    theta_robust = rng.standard_normal((n_targets, n_cbvs)).astype(np.float32)
    return {
        "cadences": cadences,
        "cbvs": cbvs,
        "star_ids": star_ids,
        "theta_robust": theta_robust,
    }


def test_load_returns_expected_shapes(tmp_path: Path):
    spec = _make_spec(n_cad=200, n_targets=5, n_cbvs=3)
    path = tmp_path / "cbv_lc_42_1_3.fits"
    _write_minimal_lc_cbv_fits(path, **spec)

    loaded = load_lc_cbvs(path)
    assert loaded.orbit == 42 and loaded.camera == 1 and loaded.ccd == 3
    np.testing.assert_array_equal(loaded.cadences, spec["cadences"])
    assert loaded.cbvs.shape == (3, 200)
    np.testing.assert_array_equal(loaded.star_ids, spec["star_ids"])
    assert loaded.theta.shape == (5, 3)
    # Without MAP, theta equals theta_robust.
    np.testing.assert_allclose(loaded.theta, spec["theta_robust"], rtol=1e-6)


def test_trend_for_target_matches_theta_dot_cbvs(tmp_path: Path):
    spec = _make_spec(n_cad=50, n_targets=4, n_cbvs=2, seed=1)
    path = tmp_path / "cbv_lc_42_1_3.fits"
    _write_minimal_lc_cbv_fits(path, **spec)

    loaded = load_lc_cbvs(path)
    tic = int(spec["star_ids"][2])
    trend = loaded.trend_for_target(tic, spec["cadences"])
    assert trend is not None
    expected = spec["theta_robust"][2].astype(np.float64) @ spec["cbvs"]
    np.testing.assert_allclose(trend, expected, rtol=1e-6)


def test_trend_for_unknown_target_returns_none(tmp_path: Path):
    spec = _make_spec(n_cad=20, n_targets=3, n_cbvs=2)
    path = tmp_path / "cbv_lc_42_1_3.fits"
    _write_minimal_lc_cbv_fits(path, **spec)
    loaded = load_lc_cbvs(path)
    assert loaded.trend_for_target(tic_id=999999999, cadences=spec["cadences"]) is None


def test_cadences_not_in_product_get_zero(tmp_path: Path):
    spec = _make_spec(n_cad=10, n_targets=2, n_cbvs=2, seed=4)
    path = tmp_path / "cbv_lc_42_1_3.fits"
    _write_minimal_lc_cbv_fits(path, **spec)
    loaded = load_lc_cbvs(path)
    tic = int(spec["star_ids"][0])
    # Mix of in-product and out-of-product cadences.
    input_cadences = np.array([spec["cadences"][0], spec["cadences"][3], 99999], dtype=np.int64)
    trend = loaded.trend_for_target(tic, input_cadences)
    assert trend is not None and trend.shape == input_cadences.shape
    # Out-of-product cadence must be zero (pass-through).
    assert trend[-1] == 0.0
    # In-product cadences must equal theta @ cbvs at those columns.
    expected = spec["theta_robust"][0].astype(np.float64) @ spec["cbvs"]
    np.testing.assert_allclose(trend[0], expected[0], rtol=1e-6)
    np.testing.assert_allclose(trend[1], expected[3], rtol=1e-6)


def test_map_blending_overrides_robust_where_mask_true(tmp_path: Path):
    spec = _make_spec(n_cad=30, n_targets=4, n_cbvs=2, seed=2)
    theta_map = spec["theta_robust"] + 1.0  # easily distinguishable
    map_mask = np.array([True, False, True, False], dtype=bool)
    path = tmp_path / "cbv_lc_42_1_3.fits"
    _write_minimal_lc_cbv_fits(
        path,
        **spec,
        theta_map=theta_map.astype(np.float32),
        map_mask=map_mask,
    )
    loaded = load_lc_cbvs(path)
    # mask True rows get MAP, mask False rows keep robust
    expected = spec["theta_robust"].astype(np.float32).copy()
    expected[map_mask] = theta_map[map_mask].astype(np.float32)
    np.testing.assert_array_equal(loaded.theta, expected)


def test_unsupported_format_rejected(tmp_path: Path):
    spec = _make_spec(n_cad=5, n_targets=1, n_cbvs=1)
    path = tmp_path / "cbv_lc_42_1_3.fits"
    _write_minimal_lc_cbv_fits(path, **spec, format_v="future-1")
    with pytest.raises(UnsupportedLCCBVFormatError):
        load_lc_cbvs(path)


def test_unsupported_product_rejected(tmp_path: Path):
    spec = _make_spec(n_cad=5, n_targets=1, n_cbvs=1)
    path = tmp_path / "cbv_lc_42_1_3.fits"
    _write_minimal_lc_cbv_fits(path, **spec, product="qlp-cbv-ffi")
    with pytest.raises(UnsupportedLCCBVFormatError):
        load_lc_cbvs(path)


def test_unsupported_stelcat_rejected(tmp_path: Path):
    spec = _make_spec(n_cad=5, n_targets=1, n_cbvs=1)
    path = tmp_path / "cbv_lc_42_1_3.fits"
    _write_minimal_lc_cbv_fits(path, **spec, stelcat="GAIA3")
    with pytest.raises(UnsupportedLCCBVFormatError):
        load_lc_cbvs(path)


def test_multiscale_rejected(tmp_path: Path):
    spec = _make_spec(n_cad=5, n_targets=1, n_cbvs=1)
    path = tmp_path / "cbv_lc_42_1_3.fits"
    _write_minimal_lc_cbv_fits(path, **spec, nbands=2)
    with pytest.raises(UnsupportedLCCBVFormatError):
        load_lc_cbvs(path)
