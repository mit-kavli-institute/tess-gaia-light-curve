from tglc.utils.manifest import Manifest
import multiprocessing as mp
from itertools import product
import sys
from qlp.io.backends import default_io_backend as io
import pickle
import numpy as np
import pathlib

from tglc.utils.constants import (
    TESS_PIXEL_SATURATION_LEVEL,
    convert_tess_flux_to_tess_magnitude,
    convert_tess_magnitude_to_tess_flux,
)
from tglc.epsf import make_tglc_design_matrix
from tglc.light_curve import get_cutout_for_light_curve
import astropy.units as u
from astropy.stats import mad_std
from scipy.ndimage import center_of_mass
from qlp.util.databases import TIC
from rich import print
import pandas as pd
from tqdm import tqdm

psf_size = 11
psf_oversample_factor = 2

tic_db = TIC("tic_82")

def find_largest_group(arr: np.ndarray, value: int) -> tuple[int, int] | None:
    """
    Find the largest contiguous group of `value` in `arr` and return
    the (start_index, center_index) in terms of the original array.

    Returns None if `value` is not present.
    """
    # Boolean mask where the value matches
    mask = arr == value

    # Pad with False on both ends so that diff catches edge-starting/ending groups
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.view(np.int8))

    # Rising edges (+1) mark group starts, falling edges (-1) mark group ends
    starts = np.where(diff == 1)[0]   # indices into original arr
    ends   = np.where(diff == -1)[0]  # exclusive ends

    if len(starts) == 0:
        return None

    lengths = ends - starts
    best    = np.argmax(lengths)

    start  = starts[best]
    end    = ends[best]          # exclusive
    center = (start + end - 1) // 2   # inclusive midpoint

    return start, center


def find_safe_cadence_index(orbit, camera, ccd):
    """
    Using operator assigned quality flags, predict a "safe" cadence
    that has minimal scattered light and instrumental noise.

    This is performed by finding the longest continuous group of "good"
    quality flags and then we pick the cadence at the center of that
    group. This should be a "safe" FFI, trusting the operator.
    """
    from lightcurvedb import db, models as m
    from qlp.contrib.lcdb.models import CCDWideQualityFlags, TESSObservation

    import sqlalchemy as sa
    from sqlalchemy import orm

    camera_inst = orm.aliased(m.Instrument, name="camera")
    ccd_inst = orm.aliased(m.Instrument, name="ccd")

    q = (
        sa.select(CCDWideQualityFlags)
        .join(TESSObservation, CCDWideQualityFlags.observation_id == TESSObservation.id)
        .join(ccd_inst, TESSObservation.instrument_id == ccd_inst.id)
        .join(camera_inst, ccd_inst.parent_id == camera_inst.id)
        .where(
            TESSObservation.orbit_id == orbit,
            camera_inst.name == f"Camera {camera}",
            ccd_inst.name == f"CCD {ccd}"
        )
    )
    with db:
        qflags_obj = db.scalar(q)
        if qflags_obj is None:
            print(f"No quality flags for {orbit} Cam: {camera} CCD: {ccd}")
            sys.exit(1)

        result = find_largest_group(qflags_obj.quality_flags, 0)
        if result is None:
            print(f"No good-quality cadences for {orbit} Cam: {camera} CCD: {ccd}")
            sys.exit(1)
        _, idx = result
        cadence = qflags_obj.observation.cadence_reference[idx]
    return cadence, idx


def get_aperture_limits(
    aperture_size: int, x: int, y: int, top_limit: int, right_limit: int
) -> tuple[int, int, int, int]:
    """Get (bottom, top, left, right) limits for aperture within 5x5 pixel grid."""
    bottom = max(0, y - aperture_size // 2)
    top = min(top_limit, y + aperture_size // 2 + 1)
    left = max(0, x - aperture_size // 2)
    right = min(right_limit, x + aperture_size // 2 + 1)
    return bottom, top, left, right


def predict_epsf_file_from_source(source_path):
    epsf_file = f"epsf{source_path.stem.removeprefix('source')}.npy"
    return epsf_file


def get_unnormalized_aperture_photometry(
    images: np.ndarray,
    quality_flags: np.ndarray,
    aperture_size: int,
    x: int,
    y: int,
    tmag: float,
    exposure_time: u.Quantity,
    flux_portion: np.ndarray,
):
    bottom, top, left, right = get_aperture_limits(
        aperture_size, x, y, images.shape[1], images.shape[2]
    )
    flux = np.nansum(images[:, bottom:top, left:right], axis=(1, 2)) * u.electron
    centroids = (
        np.array([center_of_mass(image[bottom:top, left:right]) for image in images]) * u.pixel
    )
    centroids[:, 0] += bottom * u.pixel
    centroids[:, 1] += left * u.pixel

    is_saturated = flux > TESS_PIXEL_SATURATION_LEVEL * (aperture_size**2) * exposure_time / (
        2.0 * u.second
    )
    flux[is_saturated] = np.nan
    centroids[is_saturated, :] = np.nan

    expected_total_flux_per_cadence = convert_tess_magnitude_to_tess_flux(tmag) * exposure_time
    flux_portion_in_aperture = np.nansum(flux_portion[bottom:top, left:right])
    expected_aperture_flux = expected_total_flux_per_cadence * flux_portion_in_aperture

    # Matches tglc/aperture_photometry.py: median of good-quality flux minus expected level.
    local_background = np.nanmedian(flux[quality_flags == 0]) - expected_aperture_flux

    return (
        flux / exposure_time,
        flux_portion_in_aperture,
        expected_aperture_flux / exposure_time,
        local_background,
    )


def analyze(source_path, epsf_file, safe_idx):
    with source_path.open("rb") as source_pickle:
        source = pickle.load(source_pickle)
    epsf = np.load(epsf_file)

    tic_match_table = source.tic
    star_positions = np.array(
        [source.gaia[f"sector_{source.sector}_x"], source.gaia[f"sector_{source.sector}_y"]]
    ).T

    design_matrix, _ = make_tglc_design_matrix(
        source.flux.shape[1:],
        (psf_size, psf_size),
        psf_oversample_factor,
        star_positions,
        source.gaia["tess_flux_ratio"].data,
        source.mask.data
    )
    flat_background = epsf[:, -6]
    high_background_points = np.abs(flat_background - np.nanmedian(flat_background)) >= mad_std(
        flat_background, ignore_nan=True
    )
    quality_mask = np.array(source.quality) | high_background_points
    nearest_pixel_x = np.round(source.gaia[f"sector_{source.sector}_x"]).astype(int)
    nearest_pixel_y = np.round(source.gaia[f"sector_{source.sector}_y"]).astype(int)

    # Targets outside these bounds have too little data to make light curves
    pixel_left_bound = 1.5
    pixel_right_bound = source.size - 2.5
    pixel_bottom_bound = 1.5
    pixel_top_bound = source.size - 2.5

    # Batch TIC tmag lookup for all targets in this cutout (one query instead of N).
    import sqlalchemy as sa

    tic_ids_in_source = [int(row[0]) for row in tic_match_table]
    tic_table = tic_db.table("ticentries")
    tmag_query = sa.select(tic_table.c.id, tic_table.c.tmag, tic_table.c.e_tmag).where(
        tic_table.c.id.in_(tic_ids_in_source)
    )
    tmag_lookup = {row.id: (row.tmag, row.e_tmag) for row in tic_db.execute(tmag_query)}

    payload = []
    for tic_id, gaia3_id in tic_match_table:
        try:
            i = np.nonzero(source.gaia["designation"] == f"Gaia DR3 {gaia3_id}")[0][0]
        except IndexError:
            continue
        if not (
                (pixel_left_bound <= nearest_pixel_x[i] <= pixel_right_bound)
                and (pixel_bottom_bound <= nearest_pixel_y[i] <= pixel_top_bound)
        ):
            continue

        light_curve_cutout, star_x, star_y, psf_portions = get_cutout_for_light_curve(
            source.flux,
            epsf,
            design_matrix,
            star_positions[i][0],
            star_positions[i][1],
            source.gaia["tess_flux_ratio"].data[i],
            (psf_size, psf_size),
            psf_oversample_factor,
            cutout_size=5,
        )
        aperture_size = 3
        target_ccd_x = star_positions[i][0] + source.ccd_x
        target_ccd_y = star_positions[i][1] + source.ccd_y

        flux, flux_ratio, expected_flux, local_background = get_unnormalized_aperture_photometry(
            light_curve_cutout,
            quality_mask,
            aperture_size,
            round(star_x),
            round(star_y),
            source.gaia["tess_mag"][i],
            source.exposure * u.second,
            psf_portions,
        )
        if tic_id not in tmag_lookup:
            continue
        tmag, e_tmag = tmag_lookup[tic_id]
        payload.append(dict(
            tic_id=tic_id,
            ccd_x=target_ccd_x,
            ccd_y=target_ccd_y,
            observed_flux=flux[safe_idx].value,
            expected_flux=expected_flux.value,
            local_background=local_background.value,
            tmag=tmag,
            e_tmag=e_tmag
        ))
    return source_path, payload


def wrapper(args):
    return analyze(*args)


SECTOR = 100

CAMERAS = CCDS = [1, 2, 3, 4]
destination_dir = pathlib.Path("~/raw-phot-chunks/").expanduser().resolve()
sector_info = io.read_sector_info(SECTOR)

for orbit in sector_info.orbit_info:
    for camera, ccd in product(CAMERAS, CCDS):
        cadence, safe_cadence_idx = find_safe_cadence_index(orbit.orbit_number, camera, ccd)
        manifest = Manifest(pathlib.Path("/pdo/qlp-data"), orbit=orbit.orbit_number, camera=camera, ccd=ccd)
        jobs = [
            (source_path,
            manifest.epsf_directory / predict_epsf_file_from_source(source_path),
            safe_cadence_idx)
            for source_path in manifest.source_directory.iterdir()
        ]
        with mp.Pool() as pool:
            _iter = tqdm(pool.imap_unordered(wrapper, jobs), total=len(jobs))
            for used_source_path, result in _iter:
                df = pd.DataFrame(result)
                df["cadence"] = cadence
                df["camera"] = camera
                df["ccd"] = ccd
                df["orbit"] = orbit.orbit_number
                stem = used_source_path.stem
                dest = destination_dir / f"o{orbit.orbit_number}_{camera}_{ccd}_{stem}_extr.csv"
                df.to_csv(dest, index=False)
