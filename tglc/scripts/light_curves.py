"""
Extract light curves from FFI cutouts using best-fit ePSFs.

Assumes `tglc cutouts` and `tglc epsfs` have already been run.
"""

import argparse
from functools import partial
import logging
from pathlib import Path
import pickle

import numpy as np

from tglc.ffi import Source
from tglc.light_curve import generate_light_curves
from tglc.util.mapping import consume_iterator_with_progress_bar, pool_map_if_multiprocessing


logger = logging.getLogger()


def read_source_and_epsf_and_save_light_curves(
    source_and_epsf_files: tuple[Path, Path],
    light_curve_directory: Path,
    replace: bool,
    psf_size: int,
    oversample_factor: int,
    max_magnitude: float,
):
    """
    Read a pickled `Source` object and a numpy-saved ePSF, and extract and save light curves.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks I/O file paths from first argument.
    """
    source_file, epsf_file = source_and_epsf_files
    with source_file.open("rb") as source_pickle:
        source: Source = pickle.load(source_pickle)
    epsf = np.load(epsf_file)
    for light_curve in generate_light_curves(
        source, epsf, psf_size, oversample_factor, max_magnitude
    ):
        light_curve_file = light_curve_directory / f"{light_curve.meta['tic_id']}.h5"
        if light_curve_file.is_file() and not replace:
            logger.debug(
                f"Light curve file {light_curve_file.resolve()} exists and will not be overwritten"
            )
        else:
            light_curve.write_hdf5(light_curve_file)


def make_light_curves_main(args: argparse.Namespace):
    """
    Extract light curves from FFI cutouts using best-fit ePSFs.

    Assumes `tglc cutouts` and `tglc epsfs` have already been run.
    """
    orbit_directory: Path = args.tglc_data_dir / f"orbit{args.orbit:04d}"

    source_directory = orbit_directory / "source"
    if not source_directory.is_dir():
        logger.error("Source directory not found, exiting")
        return
    epsf_directoory = orbit_directory / "epsf"
    if not epsf_directoory.is_dir():
        logger.error("ePSF directory not found, exiting")
        return
    light_curve_directory = orbit_directory / "lc"
    light_curve_directory.mkdir(exist_ok=True)

    for camera, ccd in args.ccd:
        ccd_source_directory = source_directory / f"{camera}-{ccd}"
        ccd_source_files = list(ccd_source_directory.glob("source_*_*.pkl"))
        if len(ccd_source_files) == 0:
            logger.warning(f"No cutout source files found for camera {camera} CCD {ccd}, skipping")
            continue
        ccd_epsf_directory = epsf_directoory / f"{camera}-{ccd}"
        ccd_source_and_epsf_files = []
        for source_file in ccd_source_files:
            epsf_file = ccd_epsf_directory / f"epsf{source_file.stem.removeprefix('source')}.npy"
            if epsf_file.is_file():
                ccd_source_and_epsf_files.append((source_file, epsf_file))
            else:
                logger.warning(f"ePSF for source file {source_file.resolve()} not found, skipping")
        if len(ccd_source_and_epsf_files) == 0:
            logger.warning(f"No ePSF files found for camera {camera} CCD {ccd}, skipping")
            continue

        ccd_light_curve_directory = light_curve_directory / f"{camera}-{ccd}"
        ccd_light_curve_directory.mkdir(exist_ok=True)

        save_light_curves_with_argparse_args = partial(
            read_source_and_epsf_and_save_light_curves,
            light_curve_directory=ccd_light_curve_directory,
            replace=args.replace,
            psf_size=args.psf_size,
            oversample_factor=args.oversample,
            max_magnitude=args.max_magnitude,
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
