"""
Script that creates cutout objects from TESS full frame images that include catalog data.

Assumes `make_catalogs.py` has already been run.
"""

import argparse
from pathlib import Path

from tglc.ffi import ffi
from tglc.util.constants import get_sector_containing_orbit


def make_cutouts_main(args: argparse.Namespace):
    orbit_directory: Path = args.tglc_data_dir / f"orbit{args.orbit:04d}"
    (orbit_directory / "source").mkdir(exist_ok=True)

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
