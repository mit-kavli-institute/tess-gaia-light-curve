"""Tests for the temporary `tglc backfill-epsf-headers` command."""

from pathlib import Path
import sys

from astropy.io import fits
import numpy as np
import pytest

from tglc.__main__ import tglc_main
from tglc.epsf import EPSF, get_reference_flux
from tglc.epsf_throughput import EPSFThroughput
from tglc.io import (
    EPSF_THROUGHPUT_KEYWORDS,
    backfill_epsf_throughput_keywords,
    read_epsf_fits,
    write_cutout_fits,
    write_epsf_fits,
)
from tglc.utils.manifest import Manifest

from .synthetic_data import make_synthetic_cutout, make_synthetic_epsf


def _make_legacy_epsf(n_cadences: int = 5, **overrides) -> EPSF:
    """An ePSF as written before the throughput keywords were recorded."""
    metadata = {
        "psf_size": 11,
        "oversample": 2,
        "orbit": 185,
        "sector": 89,
        "camera": 1,
        "ccd": 1,
        "cutout_x": 0,
        "cutout_y": 0,
    }
    metadata.update(overrides)
    return EPSF(make_synthetic_epsf(n_cadences=n_cadences), **metadata)


@pytest.fixture
def legacy_files(tmp_path: Path):
    cutout = make_synthetic_cutout()
    cutout_path = tmp_path / "source_0_0.fits"
    epsf_path = tmp_path / "epsf_0_0.fits"
    write_cutout_fits(cutout, cutout_path)
    write_epsf_fits(_make_legacy_epsf(), epsf_path)
    return cutout, cutout_path, epsf_path


def test_backfill_records_cutout_scalars(legacy_files):
    cutout, cutout_path, epsf_path = legacy_files
    before_header = fits.getheader(epsf_path)
    before_data = fits.getdata(epsf_path)
    assert not any(keyword in before_header for keyword in EPSF_THROUGHPUT_KEYWORDS)

    assert backfill_epsf_throughput_keywords(epsf_path, cutout_path)

    epsf = read_epsf_fits(epsf_path)
    assert epsf.reference_flux == get_reference_flux(cutout.gaia)
    assert epsf.exposure == cutout.exposure
    assert (epsf.ccd_x, epsf.ccd_y, epsf.cutout_size) == (cutout.ccd_x, cutout.ccd_y, cutout.size)
    # The fit and its provenance are untouched; the backfill is recorded.
    np.testing.assert_array_equal(epsf.array, before_data)
    header = fits.getheader(epsf_path)
    for keyword in ("DATE", "PROCVER", "ORBIT", "CUTOUTX", "BGCOL0"):
        assert header[keyword] == before_header[keyword]
    assert "backfilled from source_0_0.fits" in str(header["HISTORY"])
    assert not epsf_path.with_name(epsf_path.name + ".tmp").exists()


def test_backfilled_epsf_matches_cutout_throughput(legacy_files):
    cutout, cutout_path, epsf_path = legacy_files
    backfill_epsf_throughput_keywords(epsf_path, cutout_path)

    x, y = cutout.star_positions[1]
    np.testing.assert_array_equal(
        EPSFThroughput.from_fits(epsf_path).flux_fraction(x, y),
        EPSFThroughput.from_fits(epsf_path, cutout_path).flux_fraction(x, y),
    )


def test_backfill_recovers_truncated_exposure(tmp_path: Path):
    # Older cutouts stored int(EXPTIME), e.g. 158 for sector 89's 158.4 s.
    cutout = make_synthetic_cutout()
    cutout.exposure = 158
    cutout_path = tmp_path / "source_0_0.fits"
    epsf_path = tmp_path / "epsf_0_0.fits"
    write_cutout_fits(cutout, cutout_path)
    write_epsf_fits(_make_legacy_epsf(), epsf_path)

    backfill_epsf_throughput_keywords(epsf_path, cutout_path)

    assert read_epsf_fits(epsf_path).exposure == 158.4


def test_backfill_skips_complete_files_unless_replace(legacy_files):
    _, cutout_path, epsf_path = legacy_files
    assert backfill_epsf_throughput_keywords(epsf_path, cutout_path)
    complete_bytes = epsf_path.read_bytes()

    assert not backfill_epsf_throughput_keywords(epsf_path, cutout_path)
    assert epsf_path.read_bytes() == complete_bytes

    assert backfill_epsf_throughput_keywords(epsf_path, cutout_path, replace=True)


@pytest.mark.parametrize(
    ("epsf_kwargs", "match"),
    [
        ({"camera": 2}, "has CAMERA 2, but cutout"),
        ({"cutout_y": 3}, "has CUTOUTY 3, but cutout"),
        ({"n_cadences": 4}, "has 4 cadences, but cutout"),
    ],
)
def test_backfill_rejects_mismatched_cutout(tmp_path: Path, epsf_kwargs, match):
    cutout_path = tmp_path / "source_0_0.fits"
    epsf_path = tmp_path / "epsf_0_0.fits"
    write_cutout_fits(make_synthetic_cutout(), cutout_path)
    write_epsf_fits(_make_legacy_epsf(**epsf_kwargs), epsf_path)
    before = epsf_path.read_bytes()

    with pytest.raises(ValueError, match=match):
        backfill_epsf_throughput_keywords(epsf_path, cutout_path)
    assert epsf_path.read_bytes() == before


def test_backfill_accepts_legacy_cutout_index_sentinel(tmp_path: Path):
    cutout_path = tmp_path / "source_0_0.fits"
    epsf_path = tmp_path / "epsf_0_0.fits"
    write_cutout_fits(make_synthetic_cutout(), cutout_path)
    write_epsf_fits(_make_legacy_epsf(cutout_x=-1, cutout_y=-1), epsf_path)

    assert backfill_epsf_throughput_keywords(epsf_path, cutout_path)


def test_backfill_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog):
    manifest = Manifest(tmp_path, orbit=185, camera=1, ccd=1)
    manifest.source_directory.mkdir(parents=True)
    manifest.epsf_directory.mkdir()
    cutout = make_synthetic_cutout()
    for cutout_x in (0, 1):
        cutout.cutout_x = cutout_x
        manifest.cutout_x, manifest.cutout_y = cutout_x, 0
        write_cutout_fits(cutout, manifest.source_file)
        write_epsf_fits(_make_legacy_epsf(cutout_x=cutout_x), manifest.epsf_file)
    # An ePSF whose cutout is missing is reported, not fatal.
    write_epsf_fits(_make_legacy_epsf(cutout_x=2), manifest.epsf_directory / "epsf_2_0.fits")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tglc",
            "backfill-epsf-headers",
            "--tglc-data-dir",
            str(tmp_path),
            "--orbit",
            "185",
            "--ccd",
            "1,1",
        ],
    )
    with caplog.at_level("INFO"):
        tglc_main()

    for cutout_x in (0, 1):
        epsf = read_epsf_fits(manifest.epsf_directory / f"epsf_{cutout_x}_0.fits")
        assert epsf.reference_flux == get_reference_flux(cutout.gaia)
    assert read_epsf_fits(manifest.epsf_directory / "epsf_2_0.fits").reference_flux is None
    assert "Backfilled 2 ePSF files; 0 already had the throughput keywords; 1 failed" in (
        caplog.text
    )
