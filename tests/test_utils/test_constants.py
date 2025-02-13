"""
Tests for the tglc.util.constants module, which provides common astronomical constants and
conversions, mostly related to TESS.
"""

from astropy.time import Time
import astropy.units as u
import numpy as np
import pytest

from tglc.util.constants import TESS_PIXEL_SCALE, convert_gaia_mags_to_tmag


def test_tess_pixel_scale():
    single_pixel = 1 * u.pix
    assert single_pixel.to(u.arcmin, equivalencies=TESS_PIXEL_SCALE).value == 0.35
    assert np.isclose(single_pixel.to(u.arcsec, equivalencies=TESS_PIXEL_SCALE).value, 21)
    pixel_on_sky = 0.35 * u.arcmin
    assert pixel_on_sky.to(u.pix, equivalencies=TESS_PIXEL_SCALE).value == 1
    assert np.isclose(pixel_on_sky.to(u.arcsec).to(u.pix, equivalencies=TESS_PIXEL_SCALE).value, 1)
    # Each camera has a 24deg x 24deg field of view and has a 2x2 mosaic of CCD detectors, each a
    # 2048x2048 grid, for an effective 4096x4096 grid. There are gaps and the grid does not exactly
    # correspond to the field of view, so we give this check some tolerance.
    detector_size = 2 * 2048 * u.pix
    assert np.isclose(detector_size.to(u.deg, equivalencies=TESS_PIXEL_SCALE).value, 24, 0.15)


def test_convrt_gaia_mags_to_tmag_no_masks():
    # In the big parametrized test, all the objects are masked arrays, some of which just have the
    # masks as all false. This tests that the function works when normal arrays are passed
    G = np.array([10.0, 12.0, 14.0])
    Gbp = np.array([10.5, 12.5, 14.5])
    Grp = np.array([9.5, 11.5, 13.5])
    expected = np.array([9.48243245, 11.48243245, 13.48243245])
    result = convert_gaia_mags_to_tmag(G, Gbp, Grp)
    np.testing.assert_almost_equal(result, expected, decimal=5)
    assert not np.ma.is_masked(result)


# This should test every combination of having/not having masks
# Result values depend on Gbp, Grp masks
# Result mask depends on G mask
@pytest.mark.parametrize(
    "G_mask,Gbp_mask,Grp_mask,expected",
    [
        (None, None, None, [9.48243245, 11.48243245, 13.48243245]),
        (None, [True, True, True], [True, True, True], [9.57, 11.57, 13.57]),
        (None, [False, True, False], [False, True, False], [9.48243245, 11.57, 13.48243245]),
        (None, [False, True, False], [True, False, False], [9.57, 11.57, 13.48243245]),
        (None, [False, False, True], None, [9.48243245, 11.48243245, 13.57]),
        (None, None, [False, False, True], [9.48243245, 11.48243245, 13.57]),
        ([True, False, False], None, None, [9.48243245, 11.48243245, 13.48243245]),
        ([True, False, False], [False, True, False], None, [9.48243245, 11.57, 13.48243245]),
        ([True, False, False], None, [False, True, False], [9.48243245, 11.57, 13.48243245]),
        (
            [False, True, False],
            [True, False, False],
            [True, False, False],
            [9.57, 11.48243245, 13.48243245],
        ),
    ],
)
def test_convert_gaia_mags_to_tmag(G_mask, Gbp_mask, Grp_mask, expected):
    G = np.ma.masked_array([10.0, 12.0, 14.0], mask=G_mask)
    Gbp = np.ma.masked_array([10.5, 12.5, 14.5], mask=Gbp_mask)
    Grp = np.ma.masked_array([9.5, 11.5, 13.5], mask=Grp_mask)
    # Mask of result should match mask of G, since all calculations involve G, and calculations can
    # be done using only G if other values are missing.
    expected = np.ma.masked_array(expected, mask=G_mask)
    result = convert_gaia_mags_to_tmag(G, Gbp, Grp)
    np.testing.assert_almost_equal(result, expected, decimal=6)
    # When the mask is all False (i.e., no values masked), the mask produced by the function is
    # sometimes a scalar False instead of an array full of False. Using np.array_equiv instead of
    # np.array_equal allows broadcasting
    assert np.array_equiv(result.mask, expected.mask)


def test_tessjd_format():
    tjd = Time(2457000.0, format="jd", scale="tdb")
    assert tjd.tjd == 0.0
    tjd2 = Time(0.0, format="tjd", scale="tdb")
    assert tjd2.jd == 2457000.0
