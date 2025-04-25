"""
Script that creates light curves from FFI cutout objects.

Assumes `make_cutouts.py` has already been run.
"""

import argparse
from functools import partial
from glob import glob
from itertools import product
from multiprocessing import Pool
from pathlib import Path
import pickle

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from tglc.target_lightcurve import epsf
from tglc.util.cli import base_parser, limit_math_multithreading
from tglc.util.logging import setup_logging


def plot_epsf(sector=1, camccd="", local_directory=""):
    fig = plt.figure(constrained_layout=False, figsize=(20, 9))
    gs = fig.add_gridspec(14, 30)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(196):
        cut_x = i // 14
        cut_y = i % 14
        psf = np.load(
            f"{local_directory}epsf/{camccd}/epsf_{cut_x:02d}_{cut_y:02d}_sector_{sector}_{camccd}.npy"
        )
        cmap = "bone"
        if np.isnan(psf).any():
            cmap = "inferno"
        ax = fig.add_subplot(gs[13 - cut_y, cut_x])
        ax.imshow(psf[0, : 23**2].reshape(23, 23), cmap=cmap, origin="lower")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False)
        ax.tick_params(axis="y", left=False)
    input_files = glob(f"{local_directory}ffi/*{camccd}-????-?_ffic.fits")
    with fits.open(input_files[0], mode="denywrite") as hdul:
        flux = hdul[1].data[0:2048, 44:2092]
        ax_1 = fig.add_subplot(gs[:, 16:])
        ax_1.imshow(np.log10(flux), origin="lower")
    fig.text(0.25, 0.08, "CUT X (0-13)", ha="center")
    fig.text(0.09, 0.5, "CUT Y (0-13)", va="center", rotation="vertical")
    fig.suptitle(f"ePSF for sector:{sector} camera-ccd:{camccd}", x=0.5, y=0.92, size=20)
    plt.savefig(
        f"{local_directory}log/epsf_sector_{sector}_{camccd}.png", bbox_inches="tight", dpi=300
    )


def make_light_curves_for_cutout(camera: int, ccd: int, x: int, y: int, local_directory: Path):
    with open(
        local_directory / "source" / f"{camera}-{ccd}" / f"source_{x:02d}_{y:02d}.pkl", "rb"
    ) as cutout_source:
        source = pickle.load(cutout_source)
    (local_directory / "epsf" / f"{camera}-{ccd}").mkdir(exist_ok=True)
    epsf(
        source,
        psf_size=11,
        factor=2,
        cut_x=x,
        cut_y=y,
        sector=source.sector,
        power=1.4,
        local_directory=str(local_directory) + "/",
        limit_mag=13.5,
        save_aper=False,
        no_progress_bar=True,
    )


def make_light_curves_main():
    parser = argparse.ArgumentParser(
        description="Create light curves from FFI cutouts", parents=[base_parser]
    )
    parser.add_argument("-o", "--orbit", type=int, required=True, help="Orbit of light curves")
    args = parser.parse_args()

    limit_math_multithreading(1)
    setup_logging(args.debug, args.logfile)
    orbit_directory: Path = args.tglc_data_dir / f"orbit{args.orbit:04d}"
    (orbit_directory / "epsf").mkdir(exist_ok=True)
    (orbit_directory / "lc").mkdir(exist_ok=True)

    with Pool(args.nprocs) as pool:
        make_light_curves_for_ccd = partial(
            make_light_curves_for_cutout, local_directory=orbit_directory
        )
        pool.starmap(
            make_light_curves_for_ccd, product(range(1, 5), range(1, 5), range(14), range(14))
        )


if __name__ == "__main__":
    make_light_curves_main()
