"""
Script that creates cutout objects from TESS full frame images that include catalog data.

Assumes `make_catalogs.py` has already been run.
"""

import argparse
from multiprocessing import set_start_method
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
    parser.add_argument(
        "-s", "--cutout-size", type=int, default=150, help="Cutout side length. Default=150."
    )
    args = parser.parse_args()

    limit_math_multithreading(1)
    setup_logging(args.debug, args.logfile)
    orbit_directory: Path = args.tglc_data_dir / f"orbit{args.orbit:04d}"
    (orbit_directory / "source").mkdir(exist_ok=True)

    set_start_method("fork")
    for camera in range(1, 5):
        for ccd in range(1, 5):
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
    make_cutouts_main()
