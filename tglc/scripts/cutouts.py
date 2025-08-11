"""
Create cutout objects from TESS full frame images that include relevant catalog data.

Assumes `tglc catalogs` has already been run.
"""

import argparse
import logging

from tglc.ffi import ffi
from tglc.utils.manifest import Manifest


logger = logging.getLogger(__name__)


def make_cutouts_main(args: argparse.Namespace):
    """
    Create cutout objects from TESS full frame images that include relevant catalog data.

    Assumes `tglc catalogs` has already been run.
    """
    manifest = Manifest(args.tglc_data_dir)

    for camera, ccd in args.ccd:
        ffi(
            args.orbit,
            camera,
            ccd,
            args.cutout,
            manifest,
            cutout_size=args.cutout_size,
            cutout_overlap=args.overlap,
            produce_mask=False,
            nprocs=args.nprocs,
            replace=args.replace,
        )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )
