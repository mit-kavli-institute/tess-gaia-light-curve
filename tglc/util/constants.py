"""
Astronomical constants and conversions used by TGLC, mostly related to TESS.
"""

from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.time.formats import TimeFromEpoch
import astropy.units as u
import numpy as np
import numpy.typing as npt


TESS_PIXEL_SCALE = u.pixel_scale(0.35 * u.arcmin / u.pixel)
"""
Astropy units equivalency for TESS pixels taken from Ricker et al, 2014, S4.1, table 1.

See <https://doi.org/10.1117/1.JATIS.1.1.014003>.
"""


def convert_gaia_mags_to_tmag(
    G: npt.ArrayLike, Gbp: npt.ArrayLike, Grp: npt.ArrayLike
) -> np.ma.MaskedArray:
    """
    Convert Gaia magnitudes to Tmag based on the conversion in Stassun et al, 2019, S2.3.1, eq 1.

    See <https://doi.org/10.3847/1538-3881/ab3467>.

    When G_bp and G_rp are available (as indicated by masked arrays), the formula used is

    $$
        T = G - 0.00522555(G_bp - G_rp)^3
            + 0.0891337(G_bp - G_rp)^2
            - 0.633923(G_bp - G_rp)
            + 0.0324473
    $$

    When G_bp or G_rp is not available, the formula used is

    $$
        T = G - 0.430
    $$

    Parameters
    ----------
    G : ArrayLike
        Gaia G passband magnitudes. Masked arrays are supported.
    Gbp : ArrayLike
        Gaia G_bp passband magnitudes. Masked arrays are supported.
    Grp : ArrayLike
        Gaia G_rp passband magnitudes. Masked arrays are supported.

    Returns
    -------
    T : MaskedArray
        Converted Tmag values. Masked where `G` input is masked.
    """
    br_difference = Gbp - Grp
    nominal_conversion = (
        G
        - 0.00522555 * (br_difference**3)
        + 0.0891337 * (br_difference**2)
        - 0.633923 * br_difference
        + 0.0324473
    )

    no_br_conversion = G - 0.430
    br_available = (
        ~br_difference.mask if np.ma.isMaskedArray(br_difference) else np.isfinite(br_difference)
    )
    return np.ma.where(br_available, nominal_conversion, no_br_conversion)


class TESSJD(TimeFromEpoch):
    """
    Astropy time format for TESS Julian Date, TJD = JD - 2457000, reported in TDB.

    Importing this class registers the `"tjd"` format with `astropy.time`.
    """

    name = "tjd"
    unit = 1
    epoch_val = 2457000 * u.day
    epoch_val2 = None
    epoch_scale = "tdb"
    epoch_format = "jd"


def apply_barycentric_correction(
    tjd: npt.ArrayLike, coord: SkyCoord, spacecraft_position: u.km.physical_type
) -> Time:
    """
    Apply barycentric time correction to TESS spacecraft timestamps.

    Parameters
    ----------
    tjd : Time
        Timestamps as recorded on TESS spacecraft.
    coord : SkyCoord
        Sky coordinates of target star(s) for which correction is being applied.
    spacecraft_position : Quantity["length"]
        (x, y, z) position of spacecraft relative to solar system barycenter at each time stamp.
        Length should match `tjd`.

    Returns
    -------
    btjd : Time
        Barycentric JD, TDB timestamps.
        If `coord` is a scalar, an array matching the shape of `tjd` is returned.
        If `coord` is an array, a 2D array with one row per object is returned.
    """
    star_vector = coord.cartesian.xyz

    star_projection = np.dot(spacecraft_position, star_vector)
    light_time = TimeDelta(
        star_projection.to(u.lightsecond).value / (60 * 60 * 24),
        format="jd",
        scale="tdb",
    )
    if len(coord.shape) == 0:
        # Scalar coordinate
        return tjd + light_time
    return tjd[:, np.newaxis] + light_time
