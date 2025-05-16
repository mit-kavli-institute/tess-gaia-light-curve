"""
Script that creates light curves from FFI cutout objects.

Assumes `make_cutouts.py` and `make_epsfs.py` have already been run.
"""

import argparse
import logging
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tglc.ffi import Source
from tglc.light_curve import generate_light_curves


logger = logging.getLogger()


def read_source_and_epsf_and_save_light_curve(
    source_epsf_files_and_lightcurve_directory: tuple[Path, Path, Path],
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
    source_file, epsf_file, light_curve_directory = source_epsf_files_and_lightcurve_directory
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
    orbit_directory: Path = args.tglc_data_dir / f"orbit{args.orbit:04d}"

    source_directory = orbit_directory / "source"
    epsf_directoory = orbit_directory / "epsf"
    light_curve_directory = orbit_directory / "lc"
    light_curve_directory.mkdir(exist_ok=True)

    for camera, ccd in args.ccd:
        ccd_source_directory = source_directory / f"{camera}-{ccd}"
        if not ccd_source_directory.is_dir():
            logger.warning(f"Source directory for CCD {camera}-{ccd} not found, skipping")
            continue
        ccd_epsf_directory = epsf_directoory / f"{camera}-{ccd}"
        if not ccd_epsf_directory.is_dir():
            logger.warning(f"ePSF directory for CCD {camera}-{ccd} not found, skipping")
            continue
        ccd_light_curve_directory = light_curve_directory / f"{camera}-{ccd}"
        ccd_light_curve_directory.mkdir(exist_ok=True)

        ccd_source_files = list(ccd_source_directory.glob("source_*_*.pkl"))
        with logging_redirect_tqdm():
            for source_file in tqdm(
                ccd_source_files,
                desc=f"Extracting light curves for cutouts in {camera}-{ccd}",
                unit="cutout",
            ):
                epsf_file = (
                    ccd_epsf_directory / f"epsf{source_file.stem.removeprefix('source')}.npy"
                )
                if not epsf_file.is_file():
                    logger.warning(
                        f"ePSF for source file {source_file.resolve()} not found, skipping"
                    )
                    continue
                read_source_and_epsf_and_save_light_curve(
                    (source_file, epsf_file, ccd_light_curve_directory),
                    args.replace,
                    args.psf_size,
                    args.oversample,
                    args.max_magnitude,
                )


if __name__ == "__main__":
    make_light_curves_main()
