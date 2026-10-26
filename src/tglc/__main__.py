"""Main entrypoint for TGLC scripts."""

import logging
from multiprocessing import set_start_method
import os

from tglc import __version__ as tglc_version
from tglc.cli import parse_tglc_args
from tglc.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def tglc_main():
    args = parse_tglc_args()
    setup_logging(args.debug, args.logfile, args.enable_runtime_warnings)
    logger.info(f"TGLC version {tglc_version}")
    printable_args = "\n".join(
        f"{a}: {getattr(args, a)}" for a in dir(args) if not a.startswith("_")
    )
    logger.debug(f"Parsed command line arguments:\n{printable_args}")

    # Use the "fork" start method to make logging from subprocesses work.
    # Note that the "spawn" method is needed for GPU multiprocessing, but should be explicitly
    # required using `multiprocessing.get_context` by parts of the package that need it. This is
    # done through `tglc.utils.mapping.pool_map_if_multiprocessing` by the ePSFs script, for
    # example.
    # Update: this function is used in end-to-end tests and pytests seems to set the start method,
    # so we try/catch the resulting RuntimeError and just continue if this fails, since it basically
    # only affects logging.
    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    if args.nprocs > 1:
        # Stop numpy & other math libraries from multithreading
        # on top of our multiprocessing
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

    if args.tglc_command == "catalogs":
        from tglc.scripts.catalogs import make_catalog_main

        make_catalog_main(args)
    elif args.tglc_command == "cutouts":
        from tglc.scripts.cutouts import make_cutouts_main

        make_cutouts_main(args)
    elif args.tglc_command == "epsfs":
        from tglc.scripts.epsfs import make_epsfs_main

        make_epsfs_main(args)
    elif args.tglc_command == "lightcurves":
        from tglc.scripts.light_curves import make_light_curves_main

        make_light_curves_main(args)
    elif args.tglc_command == "all":

        def log_heading(msg: str):
            """
            Prints a header message with a border like this:
            ******************
            * Header Message *
            ******************
            """
            logger.info(f"\n{'*' * (len(msg) + 4)}\n* {msg} *\n{'*' * (len(msg) + 4)}\n")

        log_heading("Running all TGLC scripts")

        from tglc.scripts.catalogs import make_catalog_main

        log_heading("Creating catalogs")
        make_catalog_main(args)

        from tglc.scripts.cutouts import make_cutouts_main

        log_heading("Making FFI cutouts")
        make_cutouts_main(args)

        from tglc.scripts.epsfs import make_epsfs_main

        log_heading("Fitting ePSFs")
        # Don't allow more GPU workers than CUDA devices
        old_nprocs = args.nprocs
        from tglc.utils._optional_deps import HAS_CUPY

        if HAS_CUPY:
            import cupy

            num_cuda_devices = cupy.cuda.runtime.getDeviceCount()
            args.nprocs = min(args.nprocs, num_cuda_devices)

        make_epsfs_main(args)

        args.nprocs = old_nprocs

        from tglc.scripts.light_curves import make_light_curves_main

        log_heading("Extracting light curves")
        make_light_curves_main(args)

    else:
        raise ValueError(f"Unrecognized TGLC command: {args.tglc_command}")


if __name__ == "__main__":
    tglc_main()
