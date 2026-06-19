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
    ffi_cbv_dir = getattr(args, "ffi_cbv_dir", None)

    for camera, ccd in args.ccd:
        ffi_cbv_file = None
        if ffi_cbv_dir is not None:
            manifest.orbit = args.orbit
            manifest.camera = camera
            manifest.ccd = ccd
            ffi_cbv_file = manifest.ffi_cbv_file(ffi_cbv_dir)
            if ffi_cbv_file is None:
                logger.warning(
                    "No CBV file found under %s for orbit %d cam %d ccd %d — "
                    "FFI pixels will not be CBV-corrected for this CCD.",
                    ffi_cbv_dir,
                    args.orbit,
                    camera,
                    ccd,
                )

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
            ffi_cbv_file=ffi_cbv_file,
        )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )
