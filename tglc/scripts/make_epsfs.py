"""Script that fits and saves epsfs for FFI cutout `Source` objects."""

import argparse
from functools import partial
import logging
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tglc.effective_psf import fit_psf, get_psf
from tglc.ffi import Source
from tglc.util.multiprocessing import pool_map_if_multiprocessing


logger = logging.getLogger(__name__)


def fit_epsf(
    source: Source,
    psf_size: int,
    oversample_factor: float,
    edge_compression_factor: float,
    brightness_power: float,
) -> np.ndarray:
    """Fit an epsf for the FFI cutout `Source`."""
    logger.debug(
        f"Fitting ePSF for source in {source.camera}-{source.ccd} at {source.ccd_x}, {source.ccd_y}"
    )
    A, _, oversampled_psf_size, _, _ = get_psf(
        source,
        psf_size=psf_size,
        factor=oversample_factor,
        edge_compression=edge_compression_factor,
    )
    background_degrees_of_freedom = 6
    e_psf = np.zeros((len(source.time), oversampled_psf_size**2 + background_degrees_of_freedom))
    for i in range(len(source.time)):
        e_psf[i] = fit_psf(A, source, oversampled_psf_size, power=brightness_power, time=i)
    return e_psf


def read_source_and_fit_and_save_epsf(
    source_and_epsf_files: tuple[Path, Path],
    replace: bool,
    psf_size: int,
    oversample_factor: float,
    edge_compression_factor: float,
    brightness_power: float,
):
    """Designed for multiprocessing pool with a functools partial, so unpacks file paths from first
    argument.
    """
    source_file, epsf_output_file = source_and_epsf_files
    if not replace and epsf_output_file.is_file():
        logger.debug(f"ePSF file {epsf_output_file.resolve()} exists and will not be overwritten")
        return
    with source_file.open("rb") as source_pickle:
        source: Source = pickle.load(source_pickle)
    e_psf = fit_epsf(source, psf_size, oversample_factor, edge_compression_factor, brightness_power)
    np.save(epsf_output_file, e_psf)


def make_epsfs_main(args: argparse.Namespace):
    orbit_directory: Path = args.tglc_data_dir / f"orbit{args.orbit:04d}"
    source_directory = orbit_directory / "source"
    if not source_directory.is_dir():
        logger.error("Source directory not found, exiting")
        exit()
    epsf_directory = orbit_directory / "epsf"
    epsf_directory.mkdir(exist_ok=True)

    for camera in range(1, 5):
        for ccd in range(1, 5):
            ccd_source_directory = source_directory / f"{camera}-{ccd}"
            ccd_source_files = list(ccd_source_directory.glob("source_*_*.pkl"))
            ccd_epsf_directory = epsf_directory / f"{camera}-{ccd}"
            ccd_epsf_files = [
                ccd_epsf_directory
                / (
                    f"epsf{source_file.stem.removeprefix('source')}"
                    f"_orbit_{args.orbit}_{camera}-{ccd}.npy"
                )
                for source_file in ccd_source_files
            ]
            if len(ccd_source_files) == 0:
                logger.warning(f"No source files found for {camera}-{ccd}, skipping")
            else:
                fit_and_save_epsf_with_argparse_args = partial(
                    read_source_and_fit_and_save_epsf,
                    replace=args.replace,
                    psf_size=args.psf_size,
                    oversample_factor=args.oversample,
                    edge_compression_factor=args.edge_compression_factor,
                    brightness_power=args.brightness_power,
                )
                fit_and_save_epsf_iterator = pool_map_if_multiprocessing(
                    fit_and_save_epsf_with_argparse_args,
                    zip(ccd_source_files, ccd_epsf_files),
                    nprocs=args.nprocs,
                    pool_map_method="imap_unordered",
                )
                with logging_redirect_tqdm():
                    for _ in tqdm(
                        fit_and_save_epsf_iterator,
                        desc=f"Fitting ePSFs for {camera}-{ccd}",
                        unit="cutout",
                        total=len(ccd_source_files),
                    ):
                        pass


if __name__ == "__main__":
    make_epsfs_main()
