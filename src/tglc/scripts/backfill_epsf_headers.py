"""
TEMPORARY: add the ePSF throughput keywords to ePSF files fit before they were recorded.

The keywords (``REFFLUX``, ``EXPOSURE``, ``CCDX``, ``CCDY``, ``CUTSIZE``) let
`tglc.epsf_throughput.EPSFThroughput` correct light curves from the ePSF file alone, without the
cutout. Their values come from each ePSF's cutout, so no re-fit is needed. Remove this script, the
`backfill-epsf-headers` command, and `tglc.io.backfill_epsf_throughput_keywords` once existing
ePSF products have been backfilled.

Assumes `tglc cutouts` and `tglc epsfs` have already been run.
"""

import argparse
from functools import partial
import logging
from pathlib import Path

from tglc.io import backfill_epsf_throughput_keywords
from tglc.utils.manifest import Manifest
from tglc.utils.mapping import iterate_with_progress_bar, pool_map_if_multiprocessing


logger = logging.getLogger(__name__)


def backfill_epsf_file(epsf_and_source_files: tuple[Path, Path], replace: bool) -> str:
    """
    Backfill one ePSF file, returning ``"updated"``, ``"skipped"``, or ``"failed"``.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks the file paths from the first argument. Errors are logged rather than raised so one
    bad file doesn't stop the campaign.
    """
    epsf_file, source_file = epsf_and_source_files
    if not source_file.is_file():
        logger.error(f"No cutout {source_file.resolve()} for ePSF {epsf_file.resolve()}, skipping")
        return "failed"
    try:
        updated = backfill_epsf_throughput_keywords(epsf_file, source_file, replace=replace)
    except (OSError, ValueError, KeyError) as e:
        # OSError: unreadable file; ValueError: mismatched cutout; KeyError: missing keyword/HDU
        logger.error(f"Could not backfill ePSF {epsf_file.resolve()}: {e}")
        return "failed"
    if not updated:
        logger.debug(f"ePSF {epsf_file.resolve()} already has the throughput keywords")
    return "updated" if updated else "skipped"


def backfill_epsf_headers_main(args: argparse.Namespace):
    """Backfill the ePSF throughput keywords for an orbit's ePSF files."""
    manifest = Manifest(args.tglc_data_dir, orbit=args.orbit)
    counts = {"updated": 0, "skipped": 0, "failed": 0}

    for camera, ccd in args.ccd:
        manifest.camera = camera
        manifest.ccd = ccd
        if args.cutout is not None:
            epsf_files = []
            for cutout_x, cutout_y in args.cutout:
                manifest.cutout_x = cutout_x
                manifest.cutout_y = cutout_y
                if manifest.epsf_file.is_file():
                    epsf_files.append(manifest.epsf_file)
        else:
            epsf_files = sorted(manifest.epsf_directory.glob("epsf_*.fits"))
        if len(epsf_files) == 0:
            logger.warning(f"No ePSF files found for camera {camera} CCD {ccd}, skipping")
            continue
        source_files = [
            manifest.source_directory / f"source{epsf_file.stem.removeprefix('epsf')}.fits"
            for epsf_file in epsf_files
        ]

        for status in iterate_with_progress_bar(
            pool_map_if_multiprocessing(
                partial(backfill_epsf_file, replace=args.replace),
                zip(epsf_files, source_files, strict=True),
                nprocs=args.nprocs,
                pool_map_method="imap_unordered",
            ),
            desc=f"Backfilling ePSF headers for {camera}-{ccd}",
            unit="ePSF",
            total=len(epsf_files),
        ):
            counts[status] += 1

    logger.info(
        f"Backfilled {counts['updated']} ePSF files; {counts['skipped']} already had the "
        f"throughput keywords; {counts['failed']} failed"
    )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )
