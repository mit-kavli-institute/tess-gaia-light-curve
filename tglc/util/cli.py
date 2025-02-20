"""Command line interface utilities for TGLC scripts."""

import argparse
from pathlib import Path


# Default value for --tglc-data-dir command line argument
# I would prefer to do this with a custom argparse.Action subclass, but argparse doesn't apply
# actions to the default, so we need to have a default value prepared ahead of time to give the
# parser.
# We look in the directory hierarchy in above the CWD for any directory called "tglc-data", and if
# we don't find one, fall back to the CWD.
def get_parent_tglc_data_dir(path: Path):
    """Find a directory named "tglc-data" in directory hierarchy containing the given path."""
    current_path = path.parent
    while current_path != current_path.parent:
        if current_path.name == "tglc-data":
            return current_path
        current_path = current_path.parent
    return path


TGLC_DATA_DIR_DEFAULT = get_parent_tglc_data_dir(Path.cwd().expanduser()).resolve()


base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument("-n", "--nprocs", type=int, default=1, help="Number of processes to use")
base_parser.add_argument(
    "-r", "--replace", action="store_true", help="Whether to overwrite existing data products"
)
base_parser.add_argument("--debug", action="store_true", help="Whether to output debug-level logs")
base_parser.add_argument("-l", "--logfile", type=Path, help="File to write logs")
base_parser.add_argument(
    "--tglc-data-dir",
    type=Path,
    default=TGLC_DATA_DIR_DEFAULT,
    help="Base directory for TGLC data. Default is first directory containing the working "
    'directory called "tglc-data", or the working directory if no "tglc-data" directory is found.',
)
