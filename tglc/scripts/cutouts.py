"""
Create cutout objects from TESS full frame images that include relevant catalog data.

Assumes `tglc catalogs` has already been run.
"""

import argparse
import logging
from pathlib import Path

from tglc.ffi import ffi
from tglc.utils.constants import get_sector_containing_orbit


logger = logging.getLogger(__name__)


def make_cutouts_main(args: argparse.Namespace):
    """
    Create cutout objects from TESS full frame images that include relevant catalog data.

    Assumes `tglc catalogs` has already been run.
    """
    orbit_directory: Path = args.tglc_data_dir / f"orbit{args.orbit:04d}"
    (orbit_directory / "source").mkdir(exist_ok=True)

    # Check that prerequisite directories
    ffi_directory = orbit_directory / "ffi"
    if not ffi_directory.is_dir():
        logger.error(f"FFI directory at {ffi_directory.resolve()} not found, exiting")
        return
    catalog_directory = orbit_directory / "catalogs"
    if not catalog_directory.is_dir():
        logger.error(f"Catalog directory at {catalog_directory.resolve()} not found, exiting")
        return
    source_directory = orbit_directory / "source"
    source_directory.mkdir()

    for camera, ccd in args.ccd:
        ffi(
            camera,
            ccd,
            args.orbit,
            get_sector_containing_orbit(args.orbit),
            base_directory=orbit_directory,
            cutout_size=args.cutout_size,
            produce_mask=False,
            nprocs=args.nprocs,
            replace=args.replace,
        )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )
