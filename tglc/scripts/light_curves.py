"""
Extract light curves from FFI cutouts using best-fit ePSFs.

Assumes `tglc cutouts` and `tglc epsfs` have already been run.
"""

import argparse
from functools import partial
import logging
import os
from pathlib import Path
import time

from tglc.ffi import FFICutout
from tglc.io import read_cutout_fits, read_epsf_fits
from tglc.light_curve import generate_light_curves
from tglc.utils.benchmark import format_benchmark_record, get_peak_rss_bytes
from tglc.utils.manifest import Manifest
from tglc.utils.mapping import consume_iterator_with_progress_bar, pool_map_if_multiprocessing


logger = logging.getLogger()


def read_source_and_epsf_and_save_light_curves(
    source_and_epsf_files: tuple[Path, Path],
    manifest: Manifest,
    replace: bool,
    psf_size: int,
    oversample_factor: int,
    tic_ids: list[int] | None = None,
    apertures: list[str] | None = None,
):
    """
    Read an :class:`FFICutout` FITS file and its matching ePSF FITS file, and extract and save
    light curves.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks I/O file paths from first argument.
    """
    start = time.perf_counter()
    source_file, epsf_file = source_and_epsf_files
    source: FFICutout = read_cutout_fits(source_file)
    epsf, _epsf_metadata = read_epsf_fits(epsf_file)
    read_elapsed = time.perf_counter() - start
    star_count = 0
    for light_curve in generate_light_curves(
        source, epsf, psf_size, oversample_factor, tic_ids, apertures
    ):
        manifest.tic_id = light_curve.meta["tic_id"]
        if replace or not manifest.light_curve_file.is_file():
            light_curve.write_hdf5(manifest.light_curve_file)
        else:
            logger.debug(
                f"Light curve file {manifest.light_curve_file.resolve()} exists and will not be"
                " overwritten"
            )
        star_count += 1
    elapsed = time.perf_counter() - start
    logger.info(
        format_benchmark_record(
            "cutout_light_curves",
            cutout=source_file.stem,
            pid=os.getpid(),
            stars=star_count,
            read_s=read_elapsed,
            elapsed_s=elapsed,
            stars_per_s=star_count / elapsed if elapsed > 0 else 0.0,
            peak_rss_mb=get_peak_rss_bytes() / 2**20,
        )
    )


def make_light_curves_main(args: argparse.Namespace):
    """
    Extract light curves from FFI cutouts using best-fit ePSFs.

    Assumes `tglc cutouts` and `tglc epsfs` have already been run.
    """
    manifest = Manifest(args.tglc_data_dir, orbit=args.orbit)

    for camera, ccd in args.ccd:
        manifest.camera = camera
        manifest.ccd = ccd
        ccd_source_files = sorted(manifest.source_directory.glob("source_*.fits"))
        if len(ccd_source_files) == 0:
            logger.warning(f"No cutout source files found for camera {camera} CCD {ccd}, skipping")
            continue

        ccd_source_and_epsf_files = []
        for source_file in ccd_source_files:
            epsf_file = (
                manifest.epsf_directory / f"epsf{source_file.stem.removeprefix('source')}.fits"
            )
            if epsf_file.is_file():
                ccd_source_and_epsf_files.append((source_file, epsf_file))
            else:
                logger.warning(f"ePSF for source file {source_file.resolve()} not found, skipping")
        if len(ccd_source_and_epsf_files) == 0:
            logger.warning(f"No ePSF files found for camera {camera} CCD {ccd}, skipping")
            continue

        manifest.light_curve_directory.mkdir(exist_ok=True)

        if args.tic is not None:
            logger.info(
                "Light curves for the ONLY the following TIC IDs will be produced: "
                + ", ".join(map(str, args.tic))
            )

        save_light_curves_with_argparse_args = partial(
            read_source_and_epsf_and_save_light_curves,
            manifest=manifest,
            replace=args.replace,
            psf_size=args.psf_size,
            oversample_factor=args.oversample,
            tic_ids=args.tic,
            apertures=args.apertures,
        )
        consume_iterator_with_progress_bar(
            pool_map_if_multiprocessing(
                save_light_curves_with_argparse_args,
                ccd_source_and_epsf_files,
                nprocs=args.nprocs,
                pool_map_method="imap_unordered",
            ),
            desc=f"Extracting light curves for {camera}-{ccd}",
            unit="cutout",
            total=len(ccd_source_and_epsf_files),
        )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )
