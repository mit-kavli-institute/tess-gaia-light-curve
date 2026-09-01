"""
TEMPORARY script migrating legacy data products to the FITS format (issue #1).

Converts source cutout pickles (``source_{x}_{y}.pkl``) and ePSF numpy files
(``epsf_{x}_{y}.npy``) to the FITS formats written by `tglc.io`. Delete this
script (and its CLI wiring) once the retroactive reprocessing campaign is done.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from pathlib import Path
import re

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tglc.io import migrate_cutout_pickle, migrate_epsf_npy
from tglc.utils.constants import get_sector_containing_orbit
from tglc.utils.manifest import Manifest


logger = logging.getLogger(__name__)

_LEGACY_FILE_STEM = re.compile(r"(?:source|epsf)_(\d+)_(\d+)")


@dataclass(frozen=True)
class _WorkItem:
    kind: str
    """Type of legacy file: "source" or "epsf"."""
    legacy_path: Path
    camera: int
    ccd: int
    cutout_x: int
    cutout_y: int


def _discover_work(args: argparse.Namespace) -> list[_WorkItem]:
    """Find legacy files to migrate, honoring --ccd, --cutout, and --replace."""
    work = []
    skipped_existing = 0
    manifest = Manifest(args.tglc_data_dir, orbit=args.orbit)
    for camera, ccd in args.ccd:
        manifest.camera = camera
        manifest.ccd = ccd
        for kind, directory, pattern in [
            ("source", manifest.source_directory, "source_*.pkl"),
            ("epsf", manifest.epsf_directory, "epsf_*.npy"),
        ]:
            if not directory.is_dir():
                logger.warning(f"Directory {directory.resolve()} not found, skipping")
                continue
            for legacy_path in sorted(directory.glob(pattern)):
                stem_match = _LEGACY_FILE_STEM.fullmatch(legacy_path.stem)
                if stem_match is None:
                    logger.warning(f"Unrecognized file name {legacy_path.resolve()}, skipping")
                    continue
                cutout_x, cutout_y = int(stem_match[1]), int(stem_match[2])
                if args.cutout is not None and (cutout_x, cutout_y) not in args.cutout:
                    continue
                if not args.replace and legacy_path.with_suffix(".fits").is_file():
                    skipped_existing += 1
                    continue
                work.append(_WorkItem(kind, legacy_path, camera, ccd, cutout_x, cutout_y))
    if skipped_existing:
        logger.info(
            f"Skipping {skipped_existing} legacy files that already have FITS files "
            "(use --replace to overwrite)"
        )
    return work


def migrate_main(args: argparse.Namespace):
    """Migrate legacy source pickles and ePSF numpy files to FITS."""
    sector = get_sector_containing_orbit(args.orbit)

    def migrate_item(item: _WorkItem) -> str:
        try:
            if item.kind == "source":
                # Legacy pickles predate the cutout_x/cutout_y attributes, so supply
                # them from the file name.
                migrate_cutout_pickle(
                    item.legacy_path,
                    cutout_x=item.cutout_x,
                    cutout_y=item.cutout_y,
                    delete_original=args.delete_original,
                )
            else:
                # migrate_epsf_npy validates the array shape against psf_size/oversample and
                # raises ValueError on mismatch, which is logged and counted below.
                migrate_epsf_npy(
                    item.legacy_path,
                    psf_size=args.psf_size,
                    oversample=args.oversample,
                    orbit=args.orbit,
                    sector=sector,
                    camera=item.camera,
                    ccd=item.ccd,
                    cutout_x=item.cutout_x,
                    cutout_y=item.cutout_y,
                    delete_original=args.delete_original,
                )
        except Exception:
            logger.warning(f"Failed to migrate {item.legacy_path.resolve()}", exc_info=True)
            return "failed"
        return "migrated"

    work = _discover_work(args)
    with ThreadPoolExecutor(max_workers=args.nprocs) as executor, logging_redirect_tqdm():
        results = Counter(
            tqdm(
                executor.map(migrate_item, work),
                desc=f"Migrating legacy files for orbit {args.orbit}",
                unit="file",
                total=len(work),
            )
        )
    logger.info(f"Migration complete: {results['migrated']} migrated, {results['failed']} failed")


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )
