"""
Script that creates cached versions of the TIC and Gaia databases with the entries relevant for the
current sector.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import product
from logging import getLogger
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import numpy as np
import pandas as pd
import sqlalchemy as sa
import tesswcs
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tglc.databases import TIC, Gaia
from tglc.util.cli import base_parser
from tglc.util.constants import TESS_CCD_SHAPE
from tglc.util.logging import setup_logging
from tglc.util.multiprocessing import pool_map_if_multiprocessing


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


def _get_camera_query_grid_centers(sector: int, camera: int, ccd: int) -> SkyCoord:
    """Get centers of 5deg-radius cones that will cover a CCD FOV in a given sector."""
    ra, dec, roll = tesswcs.pointings[tesswcs.pointings["Sector"] == sector][0]["RA", "Dec", "Roll"]
    wcs = tesswcs.WCS.predict(ra, dec, roll, camera, ccd, warp=False)
    ccd_rows, ccd_columns = TESS_CCD_SHAPE
    query_center_ccd_x, query_center_ccd_y = np.meshgrid(
        np.arange(ccd_columns / 4, ccd_columns, ccd_columns / 4, dtype=float),
        np.arange(ccd_rows / 4, ccd_rows, ccd_rows / 4, dtype=float),
    )
    return wcs.pixel_to_world(query_center_ccd_x.ravel(), query_center_ccd_y.ravel())


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
    ccd: int,
    magnitude_cutoff: float = 13.5,
    mdwarf_magnitude_cutoff: float | None = None,
    nprocs: int = 1,
) -> Table:
    """
    Query the TESS Input Catalog for stars in a grid of cones covering the camera during the sector.

    Parameters
    ----------
    sector, camera, ccd : int
        TESS sector, camera, and CCD identifying the field of view to create a catalog for.
    magnitude_cutoff : float
        Stars brighter than the magnitude cutoff will be included in the query. Default = 13.5
    mdwarf_magnitude_cutoff : float
        Separate magnitude cutoff for M dwarf stars. If excluded, the main magnitude cutoff will be
        used.
    nprocs : int
        Number of processes to use to distribute queries

    Returns
    -------
    tic_data : Table
        Table containing the TIC catalog fields with appropriate units
    """
    query_grid_centers = _get_camera_query_grid_centers(sector, camera, ccd)
    run_tic_cone_query_with_mag_cutoffs = partial(
        _run_tic_cone_query,
        magnitude_cutoff=magnitude_cutoff,
        mdwarf_magnitude_cutoff=mdwarf_magnitude_cutoff,
    )
    with ThreadPoolExecutor(max_workers=nprocs) as executor:
        query_results = executor.map(
            run_tic_cone_query_with_mag_cutoffs,
            zip(query_grid_centers.ra.deg, query_grid_centers.dec.deg),
        )
    tic_data = Table.from_pandas(pd.concat(query_results).drop_duplicates("id"))
    tic_data["ra"].unit = u.deg
    tic_data["dec"].unit = u.deg
    tic_data["pmra"].unit = u.mas / u.yr
    tic_data["pmdec"].unit = u.mas / u.yr
    logger.debug(
        f"Found {len(tic_data)} TIC stars for camera {camera}, CCD {ccd} after applying magnitude "
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


def get_gaia_catalog_data(sector: int, camera: int, ccd: int, nprocs: int = 1) -> Table:
    """
    Query Gaia for stars in a grid of cones covering the camera during the sector.

    Parameters
    ----------
    sector, camera, ccd : int
        TESS sector, camera, and CCD identifying the field of view to create a catalog for.
    nprocss : int
        Number of processes to use to distribute queries

    Returns
    -------
    gaia_data : Table
        Table containing the Gaia catalog fields with appropriate units
    """
    query_grid_centers = _get_camera_query_grid_centers(sector, camera, ccd)
    with ThreadPoolExecutor(max_workers=nprocs) as executor:
        query_results = executor.map(
            _run_gaia_cone_query,
            zip(query_grid_centers.ra.deg, query_grid_centers.dec.deg),
        )
    gaia_data = Table.from_pandas(pd.concat(query_results).drop_duplicates("designation"))
    gaia_data["ra"].unit = u.deg
    gaia_data["dec"].unit = u.deg
    gaia_data["pmra"].unit = u.mas / u.yr
    gaia_data["pmdec"].unit = u.mas / u.yr
    logger.debug(f"Found {len(gaia_data)} Gaia stars for camera {camera}, CCD {ccd}")
    return gaia_data


def _make_tic_and_gaia_catalogs(
    camera_ccd: tuple[int, int],
    sector: int,
    tic_magnitude_limit: float,
    output_directory: Path,
    nprocs: int = 1,
    replace: bool = False,
):
    """
    Make TIC and Gaia catalog files for a camera/CCD in a sector.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks `camera` and `ccd` from the first argument.
    """
    camera, ccd = camera_ccd
    tic_catalog_file: Path = output_directory / f"TIC_camera{camera}_ccd{ccd}.ecsv"
    if replace or not tic_catalog_file.is_file():
        tic_results = get_tic_catalog_data(
            sector, camera, ccd, magnitude_cutoff=tic_magnitude_limit, nprocs=nprocs
        )
        # Astropy's fast ascii writer doesn't work with ecsv by default, but we can write the
        # header and then write the data to get an equivalent file.
        tic_results[:0].write(tic_catalog_file, overwrite=replace)
        with open(tic_catalog_file, "a") as tic_output:
            tic_results.write(tic_output, format="ascii.fast_no_header", delimiter=" ", strip_whitespace=False)
    else:
        logger.info(f"TIC catalog at {tic_catalog_file} already exists and will not be overwritten")

    gaia_catalog_file: Path = output_directory / f"Gaia_camera{camera}_ccd{ccd}.ecsv"
    if replace or not gaia_catalog_file.is_file():
        gaia_results = get_gaia_catalog_data(sector, camera, ccd, nprocs=nprocs)
        # Astropy's fast ascii writer doesn't work with ecsv by default, but we can write the
        # header and then write the data to get an equivalent file.
        gaia_results[:0].write(gaia_catalog_file, overwrite=replace)
        with open(gaia_catalog_file, "a") as gaia_output:
            gaia_results.write(gaia_output, format="ascii.fast_no_header", delimiter=" ", strip_whitespace=False)
    else:
        logger.info(
            f"Gaia catalog at {gaia_catalog_file} already exists and will not be overwritten"
        )


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

    make_tic_and_gaia_catalogs_for_camera_and_ccd = partial(
        _make_tic_and_gaia_catalogs,
        sector=args.sector,
        tic_magnitude_limit=args.maglim,
        output_directory=args.output_dir,
        nprocs=max(args.nprocs // 16, 1),  # Controls how many threads to use for queries
        replace=args.replace,
    )
    make_catalogs_iterator = tqdm(
        pool_map_if_multiprocessing(
            make_tic_and_gaia_catalogs_for_camera_and_ccd,
            product(range(1, 5), repeat=2),
            nprocs=min(args.nprocs, 16),
            pool_map_method="imap_unordered",
        ),
        desc="Creating catalogs for cameras 1-4, CCDs 1-4",
        unit="ccd",
        total=16,
    )
    with logging_redirect_tqdm():
        # Consume catalog iterator to execute _make_tic_and_gaia_catalogs
        for _ in make_catalogs_iterator:
            pass


if __name__ == "__main__":
    make_catalog_main()
