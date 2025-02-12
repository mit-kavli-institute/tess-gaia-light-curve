"""
Script that creates cached versions of the TIC and Gaia databases with the entries relevant for the
current sector.
"""

import argparse
from logging import getLogger
from pathlib import Path

from astropy.table import QTable
import astropy.units as u

from tglc.util._optional_deps import HAS_PYTICDB
from tglc.util.cli import base_parser
from tglc.util.logging import setup_logging
from tglc.util.tess_pointings import get_sector_camera_pointing


logger = getLogger(__name__)

TIC_CATALOG_FIELDS = ["ID", "ra", "dec", "Tmag", "pmRA", "pmDEC", "Jmag", "Kmag", "Vmag"]

GAIA_CATALOG_FIELDS = [
    "DESIGNATION",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "ra",
    "dec",
    "pmra",
    "pmdec",
]


def get_tic_catalog_data(
    sector: int,
    camera: int,
    magnitude_cutoff: float = 13.5,
    mdwarf_magnitude_cutoff: float | None = None,
) -> QTable:
    """
    Query the TESS Input Catalog for stars in a cone around the camera during the sector.

    Automatically selects between Pyticdb and astroquery as the database engine based on the
    installed packages available.

    Parameters
    ----------
    sector : int
        The sector to use for pointing data
    camera : int
        The camera we want stars for
    magnitude_cutoff : float
        Stars brighter than the magnitude cutoff will be included in the query. Default = 13.5
    mdwarf_magnitude_cutoff : float
        Separate magnitude cutoff for M dwarf stars. If excluded, the main magnitude cutoff will be
        used.

    Returns
    -------
    tic_results : QTable
        Table containing the TIC catalog fields with appropriate units
    """
    if mdwarf_magnitude_cutoff is None:
        mdwarf_magnitude_cutoff = magnitude_cutoff

    camera_pointing = get_sector_camera_pointing(sector, camera)

    if HAS_PYTICDB:
        from pyticdb import TICEntry
        import sqlalchemy as sa

        from tglc.databases import TIC

        tic = TIC("tic82")

        base_query = tic.select("ticentries", *(field.lower() for field in TIC_CATALOG_FIELDS))
        magnitude_filter = TICEntry.c.tmag < magnitude_cutoff
        # M dwarfs: magnitude < 15, T_eff < 4,000K, radius < 0.8 solar radii
        mdwarf_filter = sa.and_(
            TICEntry.c.tmag.between(magnitude_cutoff, mdwarf_magnitude_cutoff),
            TICEntry.c.teff < 4_000,
            TICEntry.c.rad < 0.8,
        )

        tic_cone_query = base_query.where(
            tic.in_cone("ticentries", camera_pointing.ra.deg, camera_pointing.dec.deg, width=18.0)
        ).where(sa.or_(magnitude_filter, mdwarf_filter))
        logger.debug(f"Querying TIC via Pyticdb for 18.0 deg cone around {camera_pointing}")
        tic_results = QTable.from_pandas(tic.to_df(tic_cone_query))

    else:
        from astroquery.mast import Catalogs

        logger.debug(
            f"Querying TIC at MAST via astroquery for 18.0deg cone around {camera_pointing}"
        )
        tic_data = Catalogs.query_region(
            camera_pointing,
            catalog="TIC",
            radius=18.0 * u.deg,
            objType="STAR",
        )
        # Apply magnitude and M dwarf filters after the query because I don't know how to include
        # them in the region query
        magnitude_filter = tic_data["Tmag"] < magnitude_cutoff
        # M dwarfs: magnitude < 15, T_eff < 4,000K, radius < 0.8 solar radii
        mdwarf_filter = (
            (tic_data["Tmag"] >= magnitude_cutoff)
            & (tic_data["Tmag"] < mdwarf_magnitude_cutoff)
            & (tic_data["Teff"] < 4_000)
            & (tic_data["rad"] < 0.8)
        )

        tic_results = QTable(tic_data[magnitude_filter | mdwarf_filter][TIC_CATALOG_FIELDS])

    tic_results["ra"].unit = u.deg
    tic_results["dec"].unit = u.deg
    tic_results["pmra"].unit = u.mas / u.yr
    tic_results["pmdec"].unit = u.mas / u.yr
    logger.debug(
        f"Found {len(tic_results)} stars around {camera_pointing} after applying magnitude "
        f"(<{magnitude_cutoff} Tmag) and M dwarf (<{mdwarf_magnitude_cutoff} Tmag, <4,000K T_eff, "
        "<0.8 solar rad) filters"
    )

    return tic_results


def get_gaia_catalog_data(sector: int, camera: int) -> QTable:
    """
    Query Gaia for stars in a cone around the camera during the sector.

    Automatically selects between Pyticdb and astroquery as the database engine based on the
    installed packages available.

    Parameters
    ----------
    sector : int
        The sector to use for pointing data
    camera : int
        The camera we want stars for

    Returns
    -------
    gaia_results : QTable
        Table containing the Gaia catalog fields with appropriate units
    """
    camera_pointing = get_sector_camera_pointing(sector, camera)

    if HAS_PYTICDB:
        from tglc.databases import Gaia

        gaia = Gaia("gaia3")
        gaia_cone_query = gaia.query_by_loc(
            "gaia_source",
            camera_pointing.ra.deg,
            camera_pointing.dec.deg,
            18.0,
            *(field.lower() for field in GAIA_CATALOG_FIELDS),
            as_query=True,
        )
        logger.debug(f"Querying Gaia via Pyticdb for 18.0deg cone around {camera_pointing}")
        gaia_results = QTable.from_pandas(gaia.to_df(gaia_cone_query))
        gaia_results["ra"].unit = u.deg
        gaia_results["dec"].unit = u.deg
        gaia_results["pmra"].unit = u.mas / u.yr
        gaia_results["pmdec"].unit = u.mas / u.yr
        logger.debug(f"Found {len(gaia_results)} stars around {camera_pointing}")
        return gaia_results

    else:
        from astroquery.gaia import Gaia

        logger.debug(f"Querying Gaia via astroquery for 18.0deg cone around {camera_pointing}")
        gaia_results = Gaia.cone_search(
            camera_pointing, radius=18.0 * u.deg, columns=GAIA_CATALOG_FIELDS
        ).get_results()
        # Convert names to lower case
        gaia_results = QTable(gaia_results[[col.lower() for col in gaia_results.columns]])
        logger.debug(f"Found {len(gaia_results)} stars around {camera_pointing}")
        return gaia_results


def make_catalog_main():
    parser = argparse.ArgumentParser(
        description="Create cached versions of the TIC and Gaia databses with relevant entries for a given sector.",
        parents=[base_parser],
    )
    parser.add_argument("-s", "--sector", type=int, required=True, help="Sector to query for.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory for cached catalog files.",
    )
    parser.add_argument("--maglim", type=float, default=13.5, help="Magnitude limit for TIC query")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.tglc_data_dir / f"sector{args.sector:04d}" / "catalogs"
    args.output_dir.mkdir(exist_ok=True)

    setup_logging(args.debug, args.logfile)

    for camera in range(1, 5):
        logger.info(f"Creating catalogs for camera {camera} in {args.output_dir}")

        tic_results = get_tic_catalog_data(args.sector, camera)
        tic_catalog_file = args.output_dir / f"TIC_camera{camera}.ecsv"
        tic_results.write(tic_catalog_file)

        gaia_results = get_gaia_catalog_data(args.sector, camera)
        gaia_catalog_file = args.output_dir / f"Gaia_camera{camera}.ecsv"
        gaia_results.write(gaia_catalog_file)


if __name__ == "__main__":
    make_catalog_main()
