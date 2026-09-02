"""Tests for the :class:`tglc.ffi.FFICutout` wrapper class."""

import numpy as np

from .synthetic_data import make_synthetic_cutout


def test_ffi_cutout_repr():
    cutout = make_synthetic_cutout()
    assert repr(cutout) == (
        "<FFICutout orbit-185 cam1-ccd1 cutout (0, 0) size=12 cadences=5 gaia=4 tic=4>"
    )


def test_ffi_cutout_repr_legacy_missing_cutout_indices():
    """Legacy pickles predate cutout_x/cutout_y; repr must not raise on them."""
    cutout = make_synthetic_cutout()
    del cutout.cutout_x
    del cutout.cutout_y
    assert "cutout (-1, -1)" in repr(cutout)


def test_ffi_cutout_star_positions():
    cutout = make_synthetic_cutout()

    star_positions = cutout.star_positions

    assert star_positions.shape == (4, 2)
    np.testing.assert_array_equal(
        star_positions,
        np.array(
            [cutout.gaia[f"sector_{cutout.sector}_x"], cutout.gaia[f"sector_{cutout.sector}_y"]]
        ).T,
    )
