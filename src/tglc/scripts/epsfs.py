"""
Fit and save ePSFs for FFI cutouts.

Assumes `tglc cutouts` has already been run.
"""

import argparse
from functools import partial
import logging
import multiprocessing
from pathlib import Path
import re

from tglc.epsf import EPSF
from tglc.ffi import FFICutout
from tglc.io import read_cutout_fits
from tglc.utils._optional_deps import HAS_CUPY
from tglc.utils.manifest import Manifest
from tglc.utils.mapping import consume_iterator_with_progress_bar, pool_map_if_multiprocessing


logger = logging.getLogger(__name__)


def read_source_and_fit_and_save_epsf(
    source_and_epsf_files: tuple[Path, Path],
    replace: bool,
    psf_size: int,
    oversample_factor: int,
    edge_compression_factor: float,
    flux_uncertainty_power: float,
    use_gpu: bool = True,
):
    """
    Read an :class:`FFICutout` FITS file, fit an ePSF for each of its cadences, and save the
    results.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks I/O file paths from first argument.

    Most arguments are passed to `fit_epsf_for_source`.
    """
    source_file, epsf_output_file = source_and_epsf_files
    if not replace and epsf_output_file.is_file():
        logger.debug(f"ePSF file {epsf_output_file.resolve()} exists and will not be overwritten")
        return
    source: FFICutout = read_cutout_fits(source_file)

    process_name = multiprocessing.current_process().name
    pool_worker_name_match = re.match(r".*PoolWorker-(\d+)", process_name)
    if pool_worker_name_match:
        pool_worker_id = int(pool_worker_name_match[1])
    else:
        pool_worker_id = -1

    if use_gpu and HAS_CUPY:
        # Figure out which GPU to use, making sure they're evenly disributed
        import cupy

        if pool_worker_id > 0:
            cuda_device = (pool_worker_id - 1) % cupy.cuda.runtime.getDeviceCount()
            logger.debug(f"Pool worker {pool_worker_id} using GPU {cuda_device}")
        else:
            cuda_device = 0
            logger.debug(f"Non-pool process {process_name} using GPU 0")
        cuda_device_context = cupy.cuda.Device(cuda_device)
    else:
        from contextlib import nullcontext

        cuda_device = None
        cuda_device_context = nullcontext()

        if pool_worker_id > 0:
            logger.debug(f"Pool worker {pool_worker_id} using CPU")
        else:
            logger.debug(f"Non-pool process {process_name} using CPU")

    with cuda_device_context:
        epsf = EPSF.from_cutout_fit(
            source,
            psf_size=psf_size,
            oversample=oversample_factor,
            edge_compression_factor=edge_compression_factor,
            flux_uncertainty_power=flux_uncertainty_power,
            use_gpu=use_gpu,
        )
    epsf.to_fits(epsf_output_file)


def make_epsfs_main(args: argparse.Namespace):
    """
    Fit and save ePSFs for FFI cutouts.

    Assumes `tglc cutouts` has already been run.
    """
    manifest = Manifest(args.tglc_data_dir, orbit=args.orbit)

    for camera, ccd in args.ccd:
        manifest.camera = camera
        manifest.ccd = ccd
        ccd_source_files = sorted(manifest.source_directory.glob("source_*.fits"))
        if args.cutout is not None:
            # Filter `ccd_source_files` by cutouts specified by user
            args_cutout_source_files = []
            # The `Manifest` class doesn't support temporary parameters, so there's no good way to
            # make this a list comprehension, which it should be.
            for cutout_x, cutout_y in args.cutout:
                manifest.cutout_x = cutout_x
                manifest.cutout_y = cutout_y
                args_cutout_source_files.append(manifest.source_file.resolve())
            ccd_source_files = [
                file for file in ccd_source_files if file.resolve() in args_cutout_source_files
            ]
        if len(ccd_source_files) == 0:
            logger.warning(f"No cutout source files found for camera {camera} CCD {ccd}, skipping")
            continue

        manifest.epsf_directory.mkdir(exist_ok=True)
        ccd_epsf_files = [
            manifest.epsf_directory / f"epsf{source_file.stem.removeprefix('source')}.fits"
            for source_file in ccd_source_files
        ]

        fit_and_save_epsf_with_argparse_args = partial(
            read_source_and_fit_and_save_epsf,
            replace=args.replace,
            psf_size=args.psf_size,
            oversample_factor=args.oversample,
            edge_compression_factor=args.edge_compression_factor,
            flux_uncertainty_power=args.uncertainty_power,
            use_gpu=not args.no_gpu,
        )
        # For GPU multiprocessing, the "spawn" start method is necessary
        # TODO logging from workers is ignored with the "spawn" method
        mp_start_method = "spawn" if not args.no_gpu else None
        consume_iterator_with_progress_bar(
            pool_map_if_multiprocessing(
                fit_and_save_epsf_with_argparse_args,
                zip(ccd_source_files, ccd_epsf_files, strict=True),
                nprocs=args.nprocs,
                pool_map_method="imap_unordered",
                mp_start_method=mp_start_method,
            ),
            desc=f"Fitting ePSFs for {camera}-{ccd}",
            unit="cutout",
            total=len(ccd_source_files),
        )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )
