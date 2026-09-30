"""
Extract light curves from FFI cutouts using best-fit ePSFs.

Assumes `tglc cutouts` and `tglc epsfs` have already been run.
"""

import argparse
from functools import partial
import logging
from pathlib import Path
import re

from tglc.ffi import FFICutout
from tglc.io import read_cutout_fits, read_epsf_fits
from tglc.light_curve import generate_light_curves
from tglc.utils.manifest import Manifest
from tglc.utils.mapping import iterate_with_progress_bar, pool_map_if_multiprocessing


logger = logging.getLogger()


_TIC_ID_FILE_COMMENT = re.compile(r"#[^\n]*")
"""A `#` comment character and the rest of its line in a TIC ID file."""

_TIC_ID_FILE_SEPARATOR = re.compile(r"[,\s]+")
"""Separator between TIC IDs in a TIC ID file: any mix of whitespace and commas."""


def read_tic_id_file(tic_id_file: Path) -> list[int]:
    """
    Read a list of TIC IDs from a text file.

    IDs may be separated by any mix of whitespace and commas, so both one ID per line and a single
    comma-separated line are accepted. Blank lines and anything following a `#` comment character
    are ignored, and duplicate IDs are collapsed.

    Parameters
    ----------
    tic_id_file : Path
        File to read TIC IDs from.

    Returns
    -------
    tic_ids : list[int]
        TIC IDs in the order they first appear in the file.

    Raises
    ------
    ValueError
        If the file contains an entry that isn't an integer.
    """
    contents = _TIC_ID_FILE_COMMENT.sub("", tic_id_file.read_text())
    tic_ids = []
    for entry in _TIC_ID_FILE_SEPARATOR.split(contents):
        if not entry:
            continue
        try:
            tic_ids.append(int(entry))
        except ValueError as e:
            raise ValueError(
                f"Invalid TIC ID {entry!r} in TIC ID file {tic_id_file.resolve()}"
            ) from e
    # dict preserves insertion order, so this deduplicates without sorting the IDs
    return list(dict.fromkeys(tic_ids))


def get_requested_tic_ids(args: argparse.Namespace) -> list[int] | None:
    """
    Combine the TIC IDs requested on the command line with those listed in a TIC ID file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments, using the `tic` and `tic_file` attributes.

    Returns
    -------
    tic_ids : list[int] | None
        Requested TIC IDs, or `None` if neither `--tic` nor `--tic-file` was given.
    """
    if args.tic is None and args.tic_file is None:
        return None
    tic_ids = list(args.tic) if args.tic is not None else []
    if args.tic_file is not None:
        tic_ids += read_tic_id_file(args.tic_file)
    tic_ids = list(dict.fromkeys(tic_ids))
    if len(tic_ids) == 0:
        logger.warning(f"No TIC IDs found in TIC ID file {args.tic_file.resolve()}")
    return tic_ids


def read_source_and_epsf_and_save_light_curves(
    source_and_epsf_files: tuple[Path, Path],
    manifest: Manifest,
    replace: bool,
    tic_ids: list[int] | None = None,
    max_magnitude: float | None = None,
) -> list[int]:
    """
    Read an :class:`FFICutout` FITS file and its matching ePSF FITS file, and extract and save
    light curves.

    The ePSF configuration (PSF size, oversampling) comes from the ePSF FITS header.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks I/O file paths from first argument.

    Returns the subset of `tic_ids` that had a light curve produced from this cutout, so the caller
    can report requested targets that never produced one. A requested target in this cutout's TIC
    catalog is excluded if it has no matching Gaia row or sits too close to a cutout edge. Empty
    when no TIC IDs were requested.
    """
    source_file, epsf_file = source_and_epsf_files
    source: FFICutout = read_cutout_fits(source_file)
    epsf = read_epsf_fits(epsf_file)
    requested_tic_ids = set(tic_ids) if tic_ids is not None else set()
    produced_tic_ids = []
    for light_curve in generate_light_curves(
        source, epsf, manifest.ephemerides_directory, tic_ids, max_magnitude
    ):
        tic_id = light_curve.meta["tic_id"]
        if tic_id in requested_tic_ids:
            produced_tic_ids.append(tic_id)
        manifest.tic_id = tic_id
        if replace or not manifest.light_curve_file.is_file():
            light_curve.write_hdf5(manifest.light_curve_file)
        else:
            logger.debug(
                f"Light curve file {manifest.light_curve_file.resolve()} exists and will not be"
                " overwritten"
            )
    return produced_tic_ids


def make_light_curves_main(args: argparse.Namespace):
    """
    Extract light curves from FFI cutouts using best-fit ePSFs.

    Assumes `tglc cutouts` and `tglc epsfs` have already been run.
    """
    manifest = Manifest(args.tglc_data_dir, orbit=args.orbit)
    manifest.ephemerides_directory.mkdir(exist_ok=True)

    requested_tic_ids = get_requested_tic_ids(args)
    produced_tic_ids: set[int] = set()

    # The target selection is the same for every CCD, so describe it once up front.
    selections = []
    if args.light_curve_max_magnitude is not None:
        selections.append(f"targets brighter than TESS magnitude {args.light_curve_max_magnitude}")
    if requested_tic_ids is not None:
        selections.append(f"the {len(requested_tic_ids)} requested TIC IDs")
    if len(selections) > 0:
        logger.info("Light curves will ONLY be produced for " + ", plus ".join(selections))

    for camera, ccd in args.ccd:
        manifest.camera = camera
        manifest.ccd = ccd
        ccd_source_files = sorted(manifest.source_directory.glob("source_*.fits"))
        if len(ccd_source_files) == 0:
            logger.warning(f"No cutout source files found for camera {camera} CCD {ccd}, skipping")
            continue

        ccd_source_and_epsf_files = []
        for source_file in ccd_source_files:
            epsf_file = (
                manifest.epsf_directory / f"epsf{source_file.stem.removeprefix('source')}.fits"
            )
            if epsf_file.is_file():
                ccd_source_and_epsf_files.append((source_file, epsf_file))
            else:
                logger.warning(f"ePSF for source file {source_file.resolve()} not found, skipping")
        if len(ccd_source_and_epsf_files) == 0:
            logger.warning(f"No ePSF files found for camera {camera} CCD {ccd}, skipping")
            continue

        manifest.light_curve_directory.mkdir(exist_ok=True)

        save_light_curves_with_argparse_args = partial(
            read_source_and_epsf_and_save_light_curves,
            manifest=manifest,
            replace=args.replace,
            tic_ids=requested_tic_ids,
            max_magnitude=args.light_curve_max_magnitude,
        )
        for cutout_produced_tic_ids in iterate_with_progress_bar(
            pool_map_if_multiprocessing(
                save_light_curves_with_argparse_args,
                ccd_source_and_epsf_files,
                nprocs=args.nprocs,
                pool_map_method="imap_unordered",
            ),
            desc=f"Extracting light curves for {camera}-{ccd}",
            unit="cutout",
            total=len(ccd_source_and_epsf_files),
        ):
            produced_tic_ids.update(cutout_produced_tic_ids)

    if requested_tic_ids is not None:
        missing_tic_ids = [tic_id for tic_id in requested_tic_ids if tic_id not in produced_tic_ids]
        if len(missing_tic_ids) > 0:
            logger.warning(
                f"{len(missing_tic_ids)} of {len(requested_tic_ids)} requested TIC IDs were not "
                "found in any processed cutout, so no light curves were produced for them "
                "(targets outside the processed cameras/CCDs, missing from the cutout catalogs, "
                "or too close to a cutout edge)"
            )
            logger.debug(
                "Requested TIC IDs with no light curve: " + ", ".join(map(str, missing_tic_ids))
            )


if __name__ == "__main__":
    raise RuntimeError(
        "TGLC scripts can't be run directly: use the 'tglc' command or run 'python -m tglc'!"
    )
