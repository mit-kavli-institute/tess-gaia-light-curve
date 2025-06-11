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


def ccd(arg: str) -> tuple[int, int]:
    """Parse "cam,ccd" as passed to --ccd. Used as a type for argparse."""
    try:
        cam, ccd = arg.split(",")
        cam, ccd = int(cam), int(ccd)
    except Exception as e:
        raise ValueError(f"Invalid CCD specifier: '{arg}'. Should be 'cam,ccd'.") from e
    if cam not in range(1, 5):
        raise ValueError(f"Invalid camera: {cam}. Must be in [1, 2, 3, 4].")
    if ccd not in range(1, 5):
        raise ValueError(f"Invalid CCD: {ccd}. Must be in [1, 2, 3, 4].")
    return cam, ccd


command_base_parser = argparse.ArgumentParser(add_help=False)
command_base_parser.add_argument("-o", "--orbit", type=int, required=True, help="TESS orbit to run")
command_base_parser.add_argument(
    "--ccd",
    type=ccd,
    nargs="+",
    help="cam,ccd pairs to run. For example, --ccd 2,4 specifies camera 2, CCD 4. "
    "Multiple arguments are allowed separated by spaces, like --ccd 2,4 3,2.",
)

_general_options = command_base_parser.add_argument_group("General Options")
_general_options.add_argument(
    "-n", "--nprocs", type=int, default=1, help="Number of processes to use"
)
_general_options.add_argument(
    "-r", "--replace", action="store_true", help="Whether to overwrite existing data products"
)
_general_options.add_argument(
    "--tglc-data-dir",
    type=Path,
    default=TGLC_DATA_DIR_DEFAULT,
    help="Base directory for TGLC data. Default is first directory containing the working "
    'directory called "tglc-data", or the working directory if no "tglc-data" directory is found.',
)

_logging_options = command_base_parser.add_argument_group("Logging Options")
_logging_options.add_argument(
    "--debug", action="store_true", help="Output debug-level logs (default=info-level logs)"
)
_logging_options.add_argument("-l", "--logfile", type=Path, help="File to write logs")
_logging_options.add_argument(
    "--enable-runtime-warnings",
    action="store_true",
    help="Allow numpy runtime warnings (silenced by default)",
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

    all_parser = tglc_commands.add_parser(
        "all",
        description="Run all TGLC steps for an orbit.",
        help="Run all TGLC steps for an orbit.",
        parents=[command_base_parser],
    )
    all_parser.add_argument(
        "--max-magnitude",
        type=float,
        default=13.5,
        help="Magnitude limit for TIC queries and light curve production",
    )
    all_parser.add_argument(
        "-s", "--cutout-size", type=int, default=150, help="Cutout side length. Default=150."
    )
    all_parser.add_argument(
        "--psf-size", type=int, default=11, help="Side length in pixels of square PSF. Default=11."
    )
    all_parser.add_argument(
        "--oversample",
        type=int,
        default=2,
        help="Factor used to oversample the PSF compared to image pixels. Default=2.",
    )
    all_parser.add_argument(
        "--uncertainty-power",
        type=float,
        default=1.4,
        help="Power of pixel value used as observational uncertainty in ePSF fit. <1 emphasizes "
        "contributions from dimmer stars, 1 means all contributions are equal. Default=1.4 "
        "determined empirically.",
    )
    all_parser.add_argument(
        "--edge-compression-factor",
        type=float,
        default=1e-4,
        help="Scale factor used when forcing edges of ePSF to 0. Default=1e-4.",
    )
    all_parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Do not use GPUs to fit ePSFs (ignored if cupy not installed or GPUs not available)",
    )

    catalogs_parser = tglc_commands.add_parser(
        "catalogs",
        description="Create cached TIC and Gaia catalogs with data for an orbit.",
        help="Create cached TIC and Gaia catalogs with data for an orbit.",
        parents=[command_base_parser],
    )
    catalogs_parser.add_argument(
        "--max-magnitude", type=float, default=13.5, help="Magnitude limit for TIC query"
    )

    cutouts_parser = tglc_commands.add_parser(
        "cutouts",
        description="Create FFI cutouts using catalog data (requires tglc catalogs to be run)",
        help="Create FFI cutouts using catalog data (requires tglc catalogs to be run)",
        parents=[command_base_parser],
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
        "--psf-size", type=int, default=11, help="Side length in pixels of square PSF. Default=11."
    )
    epsfs_parser.add_argument(
        "--oversample",
        type=int,
        default=2,
        help="Factor used to oversample the PSF compared to image pixels. Default=2.",
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
        "--no-gpu",
        action="store_true",
        help="Do not use GPUs to fit ePSFs (ignored if cupy not installed or GPUs not available)",
    )

    lightcurves_parser = tglc_commands.add_parser(
        "lightcurves",
        description="Create light curves using fitted ePSFs (requires tglc epsfs to be run)",
        help="Create light curves using fitted ePSFs (requires tglc epsfs to be run)",
        parents=[command_base_parser],
    )
    lightcurves_parser.add_argument(
        "-t", "--tic", type=int, nargs="+", help="Produce light curves only for listed TIC IDs."
    )
    lightcurves_parser.add_argument(
        "--psf-size", type=int, default=11, help="Side length in pixels of square PSF. Default=11."
    )
    lightcurves_parser.add_argument(
        "--oversample",
        type=int,
        default=2,
        help="Factor used to oversample the PSF compared to image pixels. Default=2.",
    )
    lightcurves_parser.add_argument(
        "--max-magnitude",
        type=float,
        default=13.5,
        help="Maximum magnitude for which light curves should be extracted",
    )

    args = tglc_parser.parse_args()

    # Custom post-parsing logic
    if args.ccd is None:
        args.ccd = [(camera, ccd) for camera in range(1, 5) for ccd in range(1, 5)]

    return args
