"""
TEMPORARY script migrating legacy data products to the FITS format (issue #1).

Converts source cutout pickles (``source_{x}_{y}.pkl``) and ePSF numpy files
(``epsf_{x}_{y}.npy``) to the FITS formats written by `tglc.io`. Cutout
migration re-derives the Gaia/TIC catalog tables from the per-CCD ECSV
catalogs (the ``*_ref`` columns and the ``PMEPOCH``/``PMREFEP`` epochs), so
those catalog files must be on disk; regenerate them with ``tglc catalogs``
if needed (database queries only, no FFI reads). Old-format Gaia catalog
files without proper-motion-propagated positions are upgraded (and rewritten
on disk) once per CCD when loaded. Cutout FITS files produced by earlier
versions of this script
carry stale catalogs — they are detected by their missing ``PMEPOCH`` keyword,
or a ``FILTMARG`` keyword absent or different from the requested
``--filter-margin``, and re-migrated automatically without requiring
``--replace``. Delete this script (and its CLI wiring) once the retroactive
reprocessing campaign is done.
"""

import argparse
from collections import Counter
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
from tglc.proper_motion import load_propagated_gaia_catalog
from tglc.utils.constants import get_sector_containing_orbit
from tglc.utils.manifest import Manifest
from tglc.utils.mapping import pool_map_if_multiprocessing


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


# Full-CCD catalog tables keyed by their (gaia, tic) file paths. Holds at most one CCD's
# catalogs. migrate_main warms it in the parent process before creating each per-CCD pool,
# so under the "fork" start method (the tglc default, see tglc.__main__) workers inherit
# the tables copy-on-write instead of each task pickling them or each worker re-reading
# the ECSV files. Under "spawn" the cache starts empty in every worker, which then falls
# back to reading the files once.
_catalog_cache: dict[tuple[Path, Path], tuple[QTable, QTable]] = {}


def _get_catalogs(
    gaia_catalog_file: Path, tic_catalog_file: Path, orbit: int
) -> tuple[QTable, QTable]:
    """Read the full-CCD Gaia/TIC ECSV catalogs, reusing (and refilling) `_catalog_cache`.

    Old-format Gaia catalog files are upgraded to proper-motion-propagated form (and
    rewritten on disk). The parent-process cache warm does this once per CCD before the
    worker pool exists; spawned workers then re-read the already-rewritten file.
    """
    key = (gaia_catalog_file, tic_catalog_file)
    if key not in _catalog_cache:
        _catalog_cache.clear()
        _catalog_cache[key] = (
            load_propagated_gaia_catalog(gaia_catalog_file, orbit),
            QTable.read(tic_catalog_file),
        )
    return _catalog_cache[key]


def _load_catalogs(manifest: Manifest, camera: int, ccd: int) -> tuple[Path, Path] | None:
    """Locate and pre-load the full-CCD catalogs, returning their paths, or None if missing."""
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
    catalog_files = (manifest.gaia_catalog_file, manifest.tic_catalog_file)
    _get_catalogs(*catalog_files, manifest.orbit)
    return catalog_files


def _migrate_item(
    item: _WorkItem,
    *,
    catalog_files: tuple[Path, Path] | None,
    filter_margin: float,
    psf_size: int,
    oversample: int,
    orbit: int,
    sector: int,
    delete_original: bool,
) -> str:
    """Migrate one legacy file, returning "migrated" or "failed".

    Runs in worker processes: everything it takes is cheap to pickle (the catalogs
    travel as file paths and are resolved through `_get_catalogs`).
    """
    try:
        if item.kind == "source":
            gaia_catalog, tic_catalog = _get_catalogs(*catalog_files, orbit)
            # Legacy pickles predate the cutout_x/cutout_y attributes, so supply
            # them from the file name.
            migrate_cutout_pickle(
                item.legacy_path,
                gaia_catalog=gaia_catalog,
                tic_catalog=tic_catalog,
                cutout_x=item.cutout_x,
                cutout_y=item.cutout_y,
                filter_margin=filter_margin,
                delete_original=delete_original,
            )
        else:
            # migrate_epsf_npy validates the array shape against psf_size/oversample and
            # raises ValueError on mismatch, which is logged and counted below.
            migrate_epsf_npy(
                item.legacy_path,
                psf_size=psf_size,
                oversample=oversample,
                orbit=orbit,
                sector=sector,
                camera=item.camera,
                ccd=item.ccd,
                cutout_x=item.cutout_x,
                cutout_y=item.cutout_y,
                delete_original=delete_original,
            )
    except Exception:
        logger.warning(f"Failed to migrate {item.legacy_path.resolve()}", exc_info=True)
        return "failed"
    return "migrated"


def migrate_main(args: argparse.Namespace):
    """Migrate legacy source pickles and ePSF numpy files to FITS."""
    sector = get_sector_containing_orbit(args.orbit)
    manifest = Manifest(args.tglc_data_dir, orbit=args.orbit)

    work_by_ccd = _discover_work(args)
    total = sum(len(items) for items in work_by_ccd.values())
    results = Counter()
    with (
        logging_redirect_tqdm(),
        tqdm(
            desc=f"Migrating legacy files for orbit {args.orbit}", unit="file", total=total
        ) as progress,
    ):
        for (camera, ccd), items in work_by_ccd.items():
            # Full-CCD catalogs can be large, so they are loaded once per CCD (only when
            # the CCD has cutout pickles) and released before the next CCD. Loading them
            # here, before the per-CCD pool is created, lets forked workers inherit the
            # tables copy-on-write (see _catalog_cache) and performs any old-format
            # catalog upgrade exactly once per CCD; derive_catalogs never mutates its
            # inputs.
            catalog_files = None
            if any(item.kind == "source" for item in items):
                catalog_files = _load_catalogs(manifest, camera, ccd)
                if catalog_files is None:
                    skipped = sum(item.kind == "source" for item in items)
                    results["skipped"] += skipped
                    progress.update(skipped)
                    items = [item for item in items if item.kind != "source"]
            if not items:
                continue
            # Worker processes rather than threads: derive_catalogs' full-catalog
            # pixel-coordinate computation is CPU-bound and holds the GIL.
            migrate = partial(
                _migrate_item,
                catalog_files=catalog_files,
                filter_margin=args.filter_margin,
                psf_size=args.psf_size,
                oversample=args.oversample,
                orbit=args.orbit,
                sector=sector,
                delete_original=args.delete_original,
            )
            for outcome in pool_map_if_multiprocessing(
                migrate, items, nprocs=args.nprocs, pool_map_method="imap_unordered"
            ):
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
