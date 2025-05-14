"""Main entrypoint for TGLC scripts."""

import argparse
import logging
from multiprocessing import set_start_method
import os
from typing import Callable

from tglc import __version__ as tglc_version
from tglc.cli import parse_tglc_args
from tglc.util.logging import setup_logging


logger = logging.getLogger(__name__)


def tglc_main():
    args = parse_tglc_args()
    setup_logging(args.debug, args.logfile)
    logger.info(f"TGLC version {tglc_version}")

    if args.tglc_command == "epsfs" and not args.no_gpu:
        # For GPU multiprocessing, the "spawn" method is necessary.
        # TODO logging from workers gets ignored currently; figure out how to fix this.
        set_start_method("spawn")
    else:
        # Otherwise, the "fork" method is best for proper logging.
        set_start_method("fork")

    if args.nprocs > 1:
        # Stop numpy & other math libraries from multithreading
        # on top of our multiprocessing
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

    main: Callable[[argparse.Namespace], None]
    if args.tglc_command == "catalogs":
        from tglc.scripts.make_catalogs import make_catalog_main as main
    elif args.tglc_command == "cutouts":
        from tglc.scripts.make_cutouts import make_cutouts_main as main
    elif args.tglc_command == "epsfs":
        from tglc.scripts.make_epsfs import make_epsfs_main as main
    elif args.tglc_command == "lightcurves":
        from tglc.scripts.make_light_curves import make_light_curves_main as main
    else:
        raise ValueError(f"Unrecognized TGLC command: {args.tglc_command}")

    main(args)


if __name__ == "__main__":
    tglc_main()
