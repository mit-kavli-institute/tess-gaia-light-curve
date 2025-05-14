"""Command line interface definintion for TGLC."""

import argparse
from pathlib import Path

from tglc import __version__ as tglc_version


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


command_base_parser = argparse.ArgumentParser(add_help=False)
command_base_parser.add_argument(
    "-n", "--nprocs", type=int, default=1, help="Number of processes to use"
)
command_base_parser.add_argument(
    "-r", "--replace", action="store_true", help="Whether to overwrite existing data products"
)
command_base_parser.add_argument(
    "--debug", action="store_true", help="Whether to output debug-level logs"
)
command_base_parser.add_argument("-l", "--logfile", type=Path, help="File to write logs")
command_base_parser.add_argument(
    "--tglc-data-dir",
    type=Path,
    default=TGLC_DATA_DIR_DEFAULT,
    help="Base directory for TGLC data. Default is first directory containing the working "
    'directory called "tglc-data", or the working directory if no "tglc-data" directory is found.',
)


def parse_tglc_args() -> argparse.Namespace:
    tglc_parser = argparse.ArgumentParser(
        description="TESS-Gaia Light Curve",
    )
    tglc_parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {tglc_version}"
    )
    tglc_commands = tglc_parser.add_subparsers(
        dest="tglc_command", required=True, help="TGLC script to run"
    )

    catalogs_parser = tglc_commands.add_parser(
        "catalogs",
        description="Create cached TIC and Gaia catalogs with data for an orbit.",
        help="Create cached TIC and Gaia catalogs with data for an orbit.",
        parents=[command_base_parser],
    )
    catalogs_parser.add_argument(
        "-o", "--orbit", type=int, required=True, help="TESS orbit of observations to query"
    )
    catalogs_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for cached catalog files",
    )
    catalogs_parser.add_argument(
        "--maglim", type=float, default=13.5, help="Magnitude limit for TIC query"
    )

    cutouts_parser = tglc_commands.add_parser(
        "cutouts",
        description="Create FFI cutouts using catalog data (requires tglc catalogs to be run)",
        help="Create FFI cutouts using catalog data (requires tglc catalogs to be run)",
        parents=[command_base_parser],
    )
    cutouts_parser.add_argument(
        "-o", "--orbit", type=int, required=True, help="TESS orbit of observations."
    )
    cutouts_parser.add_argument(
        "-s", "--cutout-size", type=int, default=150, help="Cutout side length. Default=150."
    )

    epsfs_parser = tglc_commands.add_parser(
        "epsfs",
        description="Fit and save ePSFs for FFI cutouts (requires tglc cutouts to be run)",
        help="Fit and save ePSFs for FFI cutouts (requires tglc cutouts to be run)",
        parents=[command_base_parser],
    )
    epsfs_parser.add_argument(
        "-o", "--orbit", type=int, required=True, help="TESS orbit of observations"
    )
    epsfs_parser.add_argument(
        "--psf-size", type=int, default=11, help="Side length in pixels of the ePSF. Default=11."
    )
    epsfs_parser.add_argument(
        "--oversample",
        type=int,
        default=2,
        help="Factor by which to oversample the ePSF compared to image pixels. Default=2",
    )
    epsfs_parser.add_argument(
        "--uncertainty-power",
        type=float,
        default=1.4,
        help="Power of pixel value used as observational uncertainty in ePSF fit. <1 emphasizes "
        "contributions from dimmer stars, 1 means all contributions are equal. Default=1.4 "
        "determined empirically.",
    )
    epsfs_parser.add_argument(
        "--edge-compression-factor",
        type=float,
        default=1e-4,
        help="Scale factor used when forcing edges of ePSF to 0. Default=1e-4.",
    )
    epsfs_parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Do not use scipy sparse linear algebra methods to fit ePSFs",
    )
    epsfs_parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Do not use GPUs to fit ePSFs (ignored if cupy is not installed)",
    )

    lightcurves_parser = tglc_commands.add_parser(
        "lightcurves",
        description="Create light curves using fitted ePSFs (requires tglc epsfs to be run)",
        help="Create light curves using fitted ePSFs (requires tglc epsfs to be run)",
        parents=[command_base_parser],
    )
    lightcurves_parser.add_argument(
        "-o", "--orbit", type=int, required=True, help="Orbit of light curves"
    )

    args = tglc_parser.parse_args()

    # Custom post-parsing logic
    if args.tglc_command == "catalogs":
        if args.output_dir is None:
            args.output_dir = args.tglc_data_dir / f"orbit{args.orbit:04d}" / "catalogs"

    return args
