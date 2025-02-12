"""
Provides sky coordinates of the TESS camera pointings for each sector. Data is based on the CSV
linked on <https://tess.mit.edu/observations/>, which at the time of writing is
| <https://tess.mit.edu/wp-content/uploads/Year1-7_pointings_long.csv>.
The CSV data should be updated as more sectors become available.
"""

from importlib import resources

from astropy.coordinates import SkyCoord
from astropy.table import Column, QTable
import astropy.units as u

from . import data


tglc_data = resources.files(data)
TESS_SECTOR_POINTINGS: QTable = QTable.read(tglc_data / "Year1-7_pointings_long.csv", format="csv")


# Spacecraft and camera pointings are stored as strings formatted as "{ra}, {dec}, {roll}", so we
# need to parse those into actual useful values
def _parse_ra_dec_roll_column(column: Column) -> tuple[Column, Column, Column]:
    """Parse a column which has string values with the format "{ra}, {dec}, {roll}"."""
    ra = []
    dec = []
    roll = []
    for entry in column:
        ra_val, dec_val, roll_val = map(float, entry.split(", "))
        ra.append(ra_val)
        dec.append(dec_val)
        roll.append(roll_val)
    return (
        Column(ra * u.deg, name=f"{column.name} RA"),
        Column(dec * u.deg, name=f"{column.name} Dec"),
        Column(roll * u.deg, name=f"{column.name} Roll"),
    )


sc_ra, sc_dec, sc_roll = _parse_ra_dec_roll_column(TESS_SECTOR_POINTINGS["Spacecraft"])
TESS_SECTOR_POINTINGS.add_columns([sc_ra, sc_dec, sc_roll])
del TESS_SECTOR_POINTINGS["Spacecraft"]
for camera in ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]:
    ra_col, dec_col, roll_col = _parse_ra_dec_roll_column(TESS_SECTOR_POINTINGS[camera])
    TESS_SECTOR_POINTINGS.add_columns([ra_col, dec_col, roll_col])
    del TESS_SECTOR_POINTINGS[camera]


def get_sector_camera_pointing(sector: int, camera: int) -> SkyCoord:
    """
    Get the sky coordinate of the camera pointing in a given sector.

    Parameters
    ----------
    sector : int
        Sector of interest
    caemra : int
        Camera of interest

    Returns
    -------
    camera_pointing : SkyCoord
        Sky coordinate of camera center pointing during the given sector.

    Raises
    ------
    ValueError
        If the given sector is not contained in the known data downloaded from <tess.mit.edu>.
    """
    if sector not in range(1, 97):
        raise ValueError(f"No data available for sector {sector}.")
    if camera not in range(1, 5):
        raise ValueError(f"Camera {camera} is invalid, must be in range [1-4].")
    sector_data = TESS_SECTOR_POINTINGS[TESS_SECTOR_POINTINGS["Sector"] == sector][0]
    ra = sector_data[f"Camera {camera} RA"]
    dec = sector_data[f"Camera {camera} Dec"]
    return SkyCoord(ra, dec)
