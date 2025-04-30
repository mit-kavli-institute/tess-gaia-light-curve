"""
Script that creates cached versions of the TIC and Gaia databases with the entries relevant for the
current sector.
"""

import argparse
from logging import getLogger
from pathlib import Path

from astropy.table import QTable
import astropy.units as u
import numpy as np
import pandas as pd
import sqlalchemy as sa
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tglc.databases import TIC, Gaia
from tglc.util.cli import base_parser
from tglc.util.logging import setup_logging
from tglc.util.multiprocessing import pool_map_if_multiprocessing
from tglc.util.tess_pointings import get_sector_camera_pointing


logger = getLogger(__name__)

TIC_CATALOG_FIELDS = ["ID", "GAIA", "ra", "dec", "Tmag", "pmRA", "pmDEC", "Jmag", "Kmag", "Vmag"]

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


def _get_camera_query_grid_coordinates(ra: float, dec: float) -> np.ndarray:
    """Get centers of 5deg-radius cones that will cover a 24x24 deg field centered at (ra, dec)."""
    grid_points = np.arange(-9, 12, 3)
    ra_grid, dec_grid = np.meshgrid(ra + grid_points, dec + grid_points)
    return np.array([ra + ra_grid, dec + dec_grid]).T.reshape(-1, 2)


def _run_tic_cone_query(
    radec: tuple[float, float],
    radius: float = 5.0,
    magnitude_cutoff: float = 13.5,
    mdwarf_magnitude_cutoff: float | None = None,
) -> pd.DataFrame:
    """
    Get results of TIC cone query centered at (ra, dec). All arguments have degree units.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks `ra` and `dec` from the first argument.
    """
    ra, dec = radec
    if mdwarf_magnitude_cutoff is None:
        mdwarf_magnitude_cutoff = magnitude_cutoff

    tic = TIC("tic_82")
    TICEntry = tic.table("ticentries")

    base_query = tic.select("ticentries", *(field.lower() for field in TIC_CATALOG_FIELDS))
    magnitude_filter = TICEntry.c.tmag < magnitude_cutoff
    # M dwarfs: magnitude < 15, T_eff < 4,000K, radius < 0.8 solar radii
    mdwarf_filter = sa.and_(
        TICEntry.c.tmag.between(magnitude_cutoff, mdwarf_magnitude_cutoff),
        TICEntry.c.teff < 4_000,
        TICEntry.c.rad < 0.8,
    )

    tic_cone_query = base_query.where(tic.in_cone("ticentries", ra, dec, width=radius)).where(
        sa.or_(magnitude_filter, mdwarf_filter)
    )
    logger.debug(f"Querying TIC via Pyticdb for {radius:.2f}deg cone around {ra=:.2f}, {dec=:.2f}")
    return tic.to_df(tic_cone_query)


def get_tic_catalog_data(
    sector: int,
    camera: int,
    magnitude_cutoff: float = 13.5,
    mdwarf_magnitude_cutoff: float | None = None,
    nprocs: int = 1,
) -> QTable:
    """
    Query the TESS Input Catalog for stars in a grid of cones covering the camera during the sector.

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
    nprocs : int
        Number of processes to use to distribute queries

    Returns
    -------
    tic_data : QTable
        Table containing the TIC catalog fields with appropriate units
    """
    camera_pointing = get_sector_camera_pointing(sector, camera)
    # Pyticdb can't handle np.float types, which is what camera_pointing.xx.deg are by default
    ra = float(camera_pointing.ra.deg)
    dec = float(camera_pointing.dec.deg)
    query_results = list(
        pool_map_if_multiprocessing(
            _run_tic_cone_query, _get_camera_query_grid_coordinates(ra, dec), nprocs=nprocs
        )
    )
    tic_data = QTable.from_pandas(pd.concat(query_results).drop_duplicates("id"))
    tic_data["ra"].unit = u.deg
    tic_data["dec"].unit = u.deg
    tic_data["pmra"].unit = u.mas / u.yr
    tic_data["pmdec"].unit = u.mas / u.yr
    logger.debug(
        f"Found {len(tic_data)} TIC stars for camera {camera} after applying magnitude "
        f"(<{magnitude_cutoff} Tmag) and M dwarf (<{mdwarf_magnitude_cutoff} Tmag, <4,000K T_eff, "
        "<0.8 solar rad) filters"
    )

    return tic_data


def _run_gaia_cone_query(radec: tuple[float, float], radius: float = 5.0) -> pd.DataFrame:
    """
    Get results of Gaia cone query centered at (ra, dec). All arguments have degree units.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks `ra` and `dec` from the first argument.
    """
    ra, dec = radec
    gaia = Gaia("gaia3")
    gaia_cone_query = gaia.query_by_loc(
        "gaia_source",
        ra,
        dec,
        radius,
        *(field.lower() for field in GAIA_CATALOG_FIELDS),
        as_query=True,
    )
    logger.debug(f"Querying Gaia via Pyticdb for {radius:.2f}deg cone around {ra=:.2f}, {dec=:.2f}")
    return gaia.to_df(gaia_cone_query)


def get_gaia_catalog_data(sector: int, camera: int, nprocs: int = 1) -> QTable:
    """
    Query Gaia for stars in a grid of cones covering the camera during the sector.

    Parameters
    ----------
    sector : int
        The sector to use for pointing data
    camera : int
        The camera we want stars for
    nprocss : int
        Number of processes to use to distribute queries

    Returns
    -------
    gaia_data : QTable
        Table containing the Gaia catalog fields with appropriate units
    """
    camera_pointing = get_sector_camera_pointing(sector, camera)
    # Pyticdb can't handle np.float types, which is what camera_pointing.xx.deg are by default
    ra = float(camera_pointing.ra.deg)
    dec = float(camera_pointing.dec.deg)
    query_results = list(
        pool_map_if_multiprocessing(
            _run_gaia_cone_query, _get_camera_query_grid_coordinates(ra, dec), nprocs=nprocs
        )
    )
    gaia_data = QTable.from_pandas(pd.concat(query_results).drop_duplicates("designation"))
    gaia_data["ra"].unit = u.deg
    gaia_data["dec"].unit = u.deg
    gaia_data["pmra"].unit = u.mas / u.yr
    gaia_data["pmdec"].unit = u.mas / u.yr
    logger.debug(f"Found {len(gaia_data)} Gaia stars for camera {camera}")
    return gaia_data


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

    with logging_redirect_tqdm():
        for camera in tqdm(range(1, 5), desc="Creating catalogs for cameras 1-4", unit="camera"):
            tic_catalog_file: Path = args.output_dir / f"TIC_camera{camera}.ecsv"
            if args.replace or not tic_catalog_file.is_file():
                tic_results = get_tic_catalog_data(args.sector, camera, nprocs=args.nprocs)
                tic_results.write(tic_catalog_file, overwrite=args.replace)
            else:
                logger.info(
                    f"TIC catalog at {tic_catalog_file} already exists and will not be overwritten"
                )

            gaia_catalog_file: Path = args.output_dir / f"Gaia_camera{camera}.ecsv"
            if args.replace or not gaia_catalog_file.is_file():
                gaia_results = get_gaia_catalog_data(args.sector, camera, nprocs=args.nprocs)
                gaia_results.write(gaia_catalog_file, overwrite=args.replace)
            else:
                logger.info(
                    f"Gaia catalog at {gaia_catalog_file} already exists and will not be overwritten"
                )


if __name__ == "__main__":
    make_catalog_main()
