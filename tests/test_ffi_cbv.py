"""Unit tests for tglc.ffi_cbv: loading and applying QLP-CBV FFI corrections."""

from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from tglc.ffi_cbv import (
    SUPPORTED_FORMAT_VERSION,
    UnsupportedCBVFormatError,
    apply_cbv_correction,
    load_ffi_cbvs,
)


def _write_minimal_cbv_fits(
    path: Path,
    *,
    cadences: np.ndarray,
    slice_specs: list[dict],
    format_v: str = SUPPORTED_FORMAT_VERSION,
) -> None:
    """Write a minimal draft-1 CBV FITS file.

    Each ``slice_specs`` entry must provide:
        label, cbvs (n_cbvs, n_cad), theta (n_pixels, n_cbvs),
        pixel_row, pixel_col, col_start, col_stop, slice_height.
    """
    primary = fits.PrimaryHDU()
    primary.header["FORMAT_V"] = format_v
    primary.header["CAMERA"] = 1
    primary.header["CCD"] = 1
    primary.header["NCAD"] = len(cadences)
    primary.header["NSLICES"] = len(slice_specs)
    primary.header["DATE"] = "2026-05-11T00:00:00"

    cadence_hdu = fits.ImageHDU(data=cadences.astype(np.int64), name="CADENCES")

    hdus = [primary, cadence_hdu]
    for spec in slice_specs:
        label = spec["label"]
        cbvs = spec["cbvs"]
        for k in range(cbvs.shape[0]):
            cbv_hdu = fits.ImageHDU(
                data=cbvs[k].astype(np.float64), name=f"CBV_{label}_{k + 1:02d}"
            )
            cbv_hdu.header["SLICE"] = label
            cbv_hdu.header["CBVIDX"] = k + 1
            cbv_hdu.header["SVALUE"] = float(cbvs.shape[0] - k)
            hdus.append(cbv_hdu)

        n_pixels, n_cbvs = spec["theta"].shape
        cols = [
            fits.Column(name="PIXEL_ID", format="K", array=np.arange(n_pixels, dtype=np.int64)),
            fits.Column(name="PIXEL_ROW", format="J", array=spec["pixel_row"].astype(np.int32)),
            fits.Column(name="PIXEL_COL", format="J", array=spec["pixel_col"].astype(np.int32)),
        ]
        for k in range(n_cbvs):
            cols.append(
                fits.Column(
                    name=f"THETA_{k + 1:02d}",
                    format="E",
                    array=spec["theta"][:, k].astype(np.float32),
                )
            )
        weights_hdu = fits.BinTableHDU.from_columns(cols, name=f"WEIGHTS_{label}")
        weights_hdu.header["SLICE"] = label
        weights_hdu.header["NPIX"] = n_pixels
        weights_hdu.header["NCBV"] = n_cbvs
        weights_hdu.header["COLSTART"] = spec["col_start"]
        weights_hdu.header["COLSTOP"] = spec["col_stop"]
        weights_hdu.header["SLICEH"] = spec["slice_height"]
        hdus.append(weights_hdu)

    fits.HDUList(hdus).writeto(path, overwrite=True)


def _make_single_slice_spec(
    *, label: str, n_cad: int, n_pixels: int, n_cbvs: int, col_start: int, col_stop: int
) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed=hash(label) & 0xFFFFFFFF)
    cbvs = rng.standard_normal((n_cbvs, n_cad)).astype(np.float64)
    theta = rng.standard_normal((n_pixels, n_cbvs)).astype(np.float32)
    # Pixels live in a regular grid inside this slice's column range.
    width = col_stop - col_start
    assert n_pixels <= 2048 * width, "test pixel count exceeds slice capacity"
    rows = (np.arange(n_pixels) // width).astype(np.int32)
    cols = (col_start + (np.arange(n_pixels) % width)).astype(np.int32)
    return (
        {
            "label": label,
            "cbvs": cbvs,
            "theta": theta,
            "pixel_row": rows,
            "pixel_col": cols,
            "col_start": col_start,
            "col_stop": col_stop,
            "slice_height": 2048,
        },
        cbvs,
    )


def test_load_returns_meta_cadences_and_slices(tmp_path: Path):
    cadences = np.array([100, 101, 102, 103, 104], dtype=np.int64)
    spec, _ = _make_single_slice_spec(
        label="A", n_cad=len(cadences), n_pixels=6, n_cbvs=2, col_start=0, col_stop=3
    )
    path = tmp_path / "cbv.fits"
    _write_minimal_cbv_fits(path, cadences=cadences, slice_specs=[spec])

    meta, loaded_cadences, slices = load_ffi_cbvs(path)

    assert meta["FORMAT_V"] == SUPPORTED_FORMAT_VERSION
    np.testing.assert_array_equal(loaded_cadences, cadences)
    assert set(slices.keys()) == {"A"}
    slc = slices["A"]
    assert slc.cbvs.shape == (2, 5)
    assert slc.theta.shape == (6, 2)
    assert slc.col_start == 0 and slc.col_stop == 3


def test_apply_correction_subtracts_analytic_trend(tmp_path: Path):
    cadences = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    spec, _ = _make_single_slice_spec(
        label="A", n_cad=len(cadences), n_pixels=4, n_cbvs=2, col_start=0, col_stop=2
    )
    path = tmp_path / "cbv.fits"
    _write_minimal_cbv_fits(path, cadences=cadences, slice_specs=[spec])

    # Build the flux cube. We use a small 2048×2048 cube for shape correctness but only
    # care about the 4 pixels listed in the slice spec.
    flux = np.full((len(cadences), 2048, 2048), 1000.0, dtype=np.float32)
    flux_before = flux.copy()
    apply_cbv_correction(flux, cadences, path)

    # Expected trend for the 4 pixels: theta @ cbvs, shape (n_pixels, n_cad).
    expected_trend = (spec["theta"].astype(np.float64) @ spec["cbvs"]).astype(np.float32)
    for i, (r, c) in enumerate(zip(spec["pixel_row"], spec["pixel_col"], strict=False)):
        for t in range(len(cadences)):
            np.testing.assert_allclose(
                flux[t, r, c], flux_before[t, r, c] - expected_trend[i, t], rtol=1e-5
            )


def test_cadences_outside_cbv_set_are_passed_through(tmp_path: Path):
    cadences_in_product = np.array([20, 22, 24], dtype=np.int64)
    spec, _ = _make_single_slice_spec(
        label="A", n_cad=3, n_pixels=2, n_cbvs=1, col_start=0, col_stop=2
    )
    path = tmp_path / "cbv.fits"
    _write_minimal_cbv_fits(path, cadences=cadences_in_product, slice_specs=[spec])

    # Two of these cadences are not in the CBV training set and must be untouched.
    input_cadences = np.array([20, 21, 22, 23, 24], dtype=np.int64)
    flux = np.full((len(input_cadences), 2048, 2048), 50.0, dtype=np.float32)
    flux_before = flux.copy()
    apply_cbv_correction(flux, input_cadences, path)

    # Cadences 21 and 23 are missing — those slices of flux must be unchanged.
    for t in (1, 3):
        np.testing.assert_array_equal(flux[t], flux_before[t])
    # Cadences 20, 22, 24 must differ at the slice's pixels.
    for t in (0, 2, 4):
        diff = flux[t] - flux_before[t]
        assert np.any(diff != 0), f"expected correction at cadence index {t}"


def test_missing_pixel_unchanged(tmp_path: Path):
    cadences = np.array([1, 2, 3], dtype=np.int64)
    spec, _ = _make_single_slice_spec(
        label="A", n_cad=3, n_pixels=2, n_cbvs=1, col_start=0, col_stop=2
    )
    path = tmp_path / "cbv.fits"
    _write_minimal_cbv_fits(path, cadences=cadences, slice_specs=[spec])

    flux = np.full((3, 2048, 2048), 7.0, dtype=np.float32)
    flux_before = flux.copy()
    apply_cbv_correction(flux, cadences, path)

    # A pixel outside the slice's coverage (e.g. row 100, col 100) must be unchanged.
    np.testing.assert_array_equal(flux[:, 100, 100], flux_before[:, 100, 100])


def test_slice_membership_is_column_only(tmp_path: Path):
    """A pixel's slice is decided by its column; the slice boundary is half-open."""
    cadences = np.array([1, 2], dtype=np.int64)
    # Two slices: A covers cols [0, 2), B covers cols [2, 4).
    spec_a, _ = _make_single_slice_spec(
        label="A", n_cad=2, n_pixels=2, n_cbvs=1, col_start=0, col_stop=2
    )
    spec_b, _ = _make_single_slice_spec(
        label="B", n_cad=2, n_pixels=2, n_cbvs=1, col_start=2, col_stop=4
    )
    path = tmp_path / "cbv.fits"
    _write_minimal_cbv_fits(path, cadences=cadences, slice_specs=[spec_a, spec_b])

    _, _, slices = load_ffi_cbvs(path)
    # Pixel at (0, 1) is in slice A; pixel at (0, 2) is in slice B.
    assert any(s.col_start <= 1 < s.col_stop and s.label == "A" for s in slices.values())
    assert any(s.col_start <= 2 < s.col_stop and s.label == "B" for s in slices.values())


def test_unsupported_format_v_is_rejected(tmp_path: Path):
    cadences = np.array([1, 2, 3], dtype=np.int64)
    spec, _ = _make_single_slice_spec(
        label="A", n_cad=3, n_pixels=2, n_cbvs=1, col_start=0, col_stop=2
    )
    path = tmp_path / "cbv.fits"
    _write_minimal_cbv_fits(path, cadences=cadences, slice_specs=[spec], format_v="future-1")

    with pytest.raises(UnsupportedCBVFormatError):
        load_ffi_cbvs(path)


def test_two_slices_apply_independently(tmp_path: Path):
    """Pixels in different column slices use their respective slice's CBVs and theta."""
    cadences = np.array([1, 2, 3, 4], dtype=np.int64)
    spec_a, _ = _make_single_slice_spec(
        label="A", n_cad=4, n_pixels=3, n_cbvs=2, col_start=0, col_stop=3
    )
    spec_b, _ = _make_single_slice_spec(
        label="B", n_cad=4, n_pixels=3, n_cbvs=2, col_start=3, col_stop=6
    )
    path = tmp_path / "cbv.fits"
    _write_minimal_cbv_fits(path, cadences=cadences, slice_specs=[spec_a, spec_b])

    flux = np.full((4, 2048, 2048), 100.0, dtype=np.float32)
    flux_before = flux.copy()
    apply_cbv_correction(flux, cadences, path)

    trend_a = (spec_a["theta"].astype(np.float64) @ spec_a["cbvs"]).astype(np.float32)
    trend_b = (spec_b["theta"].astype(np.float64) @ spec_b["cbvs"]).astype(np.float32)

    for i, (r, c) in enumerate(zip(spec_a["pixel_row"], spec_a["pixel_col"], strict=False)):
        np.testing.assert_allclose(flux[:, r, c], flux_before[:, r, c] - trend_a[i], rtol=1e-5)
    for i, (r, c) in enumerate(zip(spec_b["pixel_row"], spec_b["pixel_col"], strict=False)):
        np.testing.assert_allclose(flux[:, r, c], flux_before[:, r, c] - trend_b[i], rtol=1e-5)
