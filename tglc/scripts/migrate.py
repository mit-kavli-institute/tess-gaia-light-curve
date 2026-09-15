"""
TEMPORARY script migrating legacy data products to the FITS format (issue #1).

Converts source cutout pickles (``source_{x}_{y}.pkl``) and ePSF numpy files
(``epsf_{x}_{y}.npy``) to the FITS formats written by `tglc.io`. Cutout
migration re-derives the Gaia/TIC catalog tables from the per-CCD ECSV
catalogs (proper-motion propagation, the ``*_ref`` columns, and the
``PMEPOCH``/``PMREFEP`` epochs), so those catalog files must be on disk;
regenerate them with ``tglc catalogs`` if needed (database queries only, no
FFI reads). Cutout FITS files produced by earlier versions of this script
carry stale catalogs — they are detected by their missing ``PMEPOCH`` keyword,
or a ``FILTMARG`` keyword absent or different from the requested
``--filter-margin``, and re-migrated automatically without requiring
``--replace``. Delete this script (and its CLI wiring) once the retroactive
reprocessing campaign is done.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
import re

from astropy.io import fits
from astropy.table import QTable
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


def _existing_fits_is_current(fits_path: Path, kind: str, filter_margin: float) -> bool:
    """Whether an existing FITS sibling is current, i.e. skippable without --replace.

    A cutout FITS file is current only if it carries propagated catalogs (``PMEPOCH``
    present — files from the old naive migration lack it) built with the requested
    star-selection margin (``FILTMARG`` present and equal; the float round-trips
    through the header exactly). ePSF files have no such markers, so existence is
    enough. An unreadable FITS file counts as not current.
    """
    if kind != "source":
        return True
    try:
        header = fits.getheader(fits_path)
    except Exception:
        return False
    return header.get("PMEPOCH") is not None and header.get("FILTMARG") == float(filter_margin)


def _discover_work(args: argparse.Namespace) -> dict[tuple[int, int], list[_WorkItem]]:
    """Find legacy files to migrate per (camera, ccd), honoring --ccd/--cutout/--replace."""
    work: dict[tuple[int, int], list[_WorkItem]] = {}
    skipped_existing = 0
    stale_refreshed = 0
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
                fits_sibling = legacy_path.with_suffix(".fits")
                if not args.replace and fits_sibling.is_file():
                    if _existing_fits_is_current(fits_sibling, kind, args.filter_margin):
                        skipped_existing += 1
                        continue
                    stale_refreshed += 1
                work.setdefault((camera, ccd), []).append(
                    _WorkItem(kind, legacy_path, camera, ccd, cutout_x, cutout_y)
                )
    if skipped_existing:
        logger.info(
            f"Skipping {skipped_existing} legacy files that already have current FITS files "
            "(use --replace to overwrite)"
        )
    if stale_refreshed:
        logger.info(
            f"Re-migrating {stale_refreshed} cutouts whose FITS files predate proper-motion "
            "propagation or were built with a different filter margin"
        )
    return work


def _load_catalogs(manifest: Manifest, camera: int, ccd: int) -> tuple[QTable, QTable] | None:
    """Read the full-CCD Gaia/TIC ECSV catalogs, or warn and return None if missing."""
    manifest.camera = camera
    manifest.ccd = ccd
    if not (manifest.gaia_catalog_file.is_file() and manifest.tic_catalog_file.is_file()):
        logger.warning(
            f"Catalog files for camera {camera} CCD {ccd} not found in "
            f"{manifest.catalog_directory.resolve()}; skipping cutout migration for this CCD. "
            f"Generate them with 'tglc catalogs --orbit {manifest.orbit} --ccd {camera},{ccd}' "
            "(database queries only, no FFI reads)."
        )
        return None
    return QTable.read(manifest.gaia_catalog_file), QTable.read(manifest.tic_catalog_file)


def migrate_main(args: argparse.Namespace):
    """Migrate legacy source pickles and ePSF numpy files to FITS."""
    sector = get_sector_containing_orbit(args.orbit)
    manifest = Manifest(args.tglc_data_dir, orbit=args.orbit)

    def migrate_item(item: _WorkItem, gaia_catalog=None, tic_catalog=None) -> str:
        try:
            if item.kind == "source":
                # Legacy pickles predate the cutout_x/cutout_y attributes, so supply
                # them from the file name.
                migrate_cutout_pickle(
                    item.legacy_path,
                    gaia_catalog=gaia_catalog,
                    tic_catalog=tic_catalog,
                    cutout_x=item.cutout_x,
                    cutout_y=item.cutout_y,
                    filter_margin=args.filter_margin,
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

    work_by_ccd = _discover_work(args)
    total = sum(len(items) for items in work_by_ccd.values())
    results = Counter()
    with (
        ThreadPoolExecutor(max_workers=args.nprocs) as executor,
        logging_redirect_tqdm(),
        tqdm(
            desc=f"Migrating legacy files for orbit {args.orbit}", unit="file", total=total
        ) as progress,
    ):
        for (camera, ccd), items in work_by_ccd.items():
            # Full-CCD catalogs can be large, so they are loaded once per CCD (only
            # when the CCD has cutout pickles) and released before the next CCD.
            # Threads share them read-only: derive_catalogs never mutates its inputs.
            catalogs = None
            if any(item.kind == "source" for item in items):
                catalogs = _load_catalogs(manifest, camera, ccd)
                if catalogs is None:
                    skipped = sum(item.kind == "source" for item in items)
                    results["skipped"] += skipped
                    progress.update(skipped)
                    items = [item for item in items if item.kind != "source"]
            gaia_catalog, tic_catalog = catalogs if catalogs is not None else (None, None)
            migrate = partial(migrate_item, gaia_catalog=gaia_catalog, tic_catalog=tic_catalog)
            for outcome in executor.map(migrate, items):
                results[outcome] += 1
                progress.update(1)
    logger.info(
        f"Migration complete: {results['migrated']} migrated, {results['failed']} failed, "
        f"{results['skipped']} skipped (missing catalogs)"
    )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )
