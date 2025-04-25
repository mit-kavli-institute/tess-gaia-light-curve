"""
Script that creates cutout objects from TESS full frame images that include catalog data.

Assumes `make_catalogs.py` has already been run.
"""

import argparse
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path

from tglc.ffi import ffi
from tglc.util.cli import base_parser, limit_math_multithreading
from tglc.util.constants import get_sector_containing_orbit
from tglc.util.logging import setup_logging


def make_cutouts_main():
    parser = argparse.ArgumentParser(
        description="Create cutouts with catalog data from FFIs", parents=[base_parser]
    )
    parser.add_argument(
        "-o", "--orbit", type=int, required=True, help="Orbit containing full frame images"
    )
    parser.add_argument("-s", "--cutout-size", type=int, default=150, help="Cutout side length")
    args = parser.parse_args()

    limit_math_multithreading(1)
    setup_logging(args.debug, args.logfile)
    orbit_directory: Path = args.tglc_data_dir / f"orbit{args.orbit:04d}"
    (orbit_directory / "source").mkdir(exist_ok=True)

    with Pool(args.nprocs) as pool:
        ffi_for_camera_and_ccd = partial(
            ffi,
            orbit=args.orbit,
            sector=get_sector_containing_orbit(args.orbit),
            size=args.cutout_size,
            local_directory=str(orbit_directory) + "/",
            producing_mask=False,
        )
        pool.starmap(ffi_for_camera_and_ccd, product(range(1, 5), repeat=2))


if __name__ == "__main__":
    make_cutouts_main()
