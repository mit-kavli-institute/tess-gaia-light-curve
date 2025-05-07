from functools import partial
from importlib import resources
from itertools import product
import logging
from multiprocessing import get_context
from pathlib import Path
import pickle
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, MaskedColumn, QTable, Table, hstack
from astropy.time import Time
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
import bottleneck as bn
from erfa.core import ErfaWarning
import numba
from numba import jit, float32, prange
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tglc.util import data
from tglc.util.constants import convert_gaia_mags_to_tmag
from tglc.util.multiprocessing import pool_map_if_multiprocessing


logger = logging.getLogger(__name__)


def crossmatch_tic_to_gaia(
    tic: QTable,
    gaia: QTable,
    match_tmag_tolerance: float = 0.1,
    match_angular_distance_tolerance: u.deg.physical_type = 2.1 * u.arcsec,
):
    """
    Crossmatch between TIC and Gaia sources.

    See `SkyCoord.match_to_catalog_sky` for more information about how matches are identified.

    Parameters
    ----------
    tic : QTable
        TIC data for sources to be identifid in Gaia data. Should include at least the following
        columns:
            - `"ra"`
            - `"dec"`
            - `"pmra"`
            - `"pmdec"`
            - `"tmag"`
    gaia : QTable
        Gaia data to be searched against for TIC sources. Should include at least the following
        columns:
            - `"ra"`
            - `"dec"`
            - `"phot_g_mean_mag"`
            - `"phot_bp_mean_mag"`
            - `"phot_rp_mean_mag"`
            - `"designation"`
    tmag_tolerance : float
        Tolerance for difference between Tmag from the TIC and converted Tmag based on Gaia data.
        See `convert_gaia_mags_to_tmag` for more information about the conversion.
        Default = 0.1
    pix_dist_tolerance : u.Quantity
        Tolerance for angular distance between matched sources.
        Default = 2.1 arcsec = 0.1 TESS pixels

    Returns
    -------
    gaia_designations : MaskedColumn
        Column containing Gaia designations for sources in TIC data provided. Rows where no match
        could successfully be made with the provided tolerances are masked.
    """
    if len(tic) == 0 or len(gaia) == 0:
        return MaskedColumn([], name="gaia_designation", mask=[])

    tic_coords = SkyCoord(
        ra=tic["ra"],
        dec=tic["dec"],
        pm_ra_cosdec=tic["pmra"],
        pm_dec=tic["pmdec"],
        frame="icrs",
        obstime=Time("J2000"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        pm_adjusted_tic_coords = tic_coords.apply_space_motion(Time("J2016"))
        gaia_coords = SkyCoord(gaia["ra"], gaia["dec"])
        match_idx, match_dist_angle, _match_dist_3d = pm_adjusted_tic_coords.match_to_catalog_sky(
            gaia_coords
        )

    close_matches = match_dist_angle <= match_angular_distance_tolerance

    gaia_tmag = convert_gaia_mags_to_tmag(
        gaia["phot_g_mean_mag"][match_idx],
        gaia["phot_bp_mean_mag"][match_idx],
        gaia["phot_rp_mean_mag"][match_idx],
    )
    tmag_difference = np.abs(gaia_tmag - tic["tmag"])
    close_tmags = tmag_difference <= match_tmag_tolerance

    successful_matches = close_matches & close_tmags

    return MaskedColumn(
        gaia["designation"][match_idx], name="gaia_designation", mask=~successful_matches
    )


# from Tim
def background_mask(im=None):
    imfilt = im * 1.0
    for i in range(im.shape[1]):
        imfilt[:, i] = ndimage.percentile_filter(im[:, i], 50, size=51)

    ok = im < imfilt
    # Don't use saturated pixels!
    satfactor = 0.4
    ok *= im < satfactor * np.amax(im)
    running_factor = 1
    cal_factor = np.zeros(im.shape[1])
    cal_factor[0] = 1

    di = 1
    i = 0
    while i < im.shape[1] - 1 and i + di < im.shape[1]:
        _ok = ok[:, i] * ok[:, i + di]
        coef = np.median(im[:, i + di][_ok] / im[:, i][_ok])
        if 0.5 < coef < 2:
            running_factor *= coef
            cal_factor[i + di] = running_factor
            i += di
            di = 1  # Reset the stepsize to one.
        else:
            # Label the column as bad, then skip it.
            cal_factor[i + di] = 0
            di += 1

    # cal_factor[im > 0.4 * np.amax(im)] = 0
    return cal_factor


class Source:
    def __init__(
        self,
        x=0,
        y=0,
        flux=None,
        time=None,
        wcs=None,
        quality=None,
        mask=None,
        exposure=1800,
        orbit=0,
        sector=0,
        size=150,
        camera=1,
        ccd=1,
        cadence=None,
        gaia_catalog=None,
        tic_catalog=None,
    ):
        """
        Source object that includes all data from TESS and Gaia DR2
        :param x: int, required
        starting horizontal pixel coordinate
        :param y: int, required
        starting vertical pixel coordinate
        :param flux: np.ndarray, required
        3d data cube, the time series of a all FFI from a CCD
        :param time: np.ndarray, required
        1d array of time
        :param wcs: astropy.wcs.wcs.WCS, required
        WCS Keywords of the TESS FFI
        :param orbit: int, required
        TESS orbit number
        :param sector: int, required
        TESS sector number
        :param size: int, optional
        the side length in pixel  of TESScut image
        :param camera: int, optional
        camera number
        :param ccd: int, optional
        CCD number
        :param cadence: list, required
        list of cadences of TESS FFI
        :param gaia_catalog: QTable, required
        Gaia catalog data
        :param tic_catalog: QTable, required
        TIC catalog data
        """
        if cadence is None:
            cadence = []
        if quality is None:
            quality = []
        if wcs is None:
            wcs = []
        if time is None:
            time = []
        if flux is None:
            flux = []

        self.size = size
        self.orbit = orbit
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.cadence = cadence
        self.quality = quality
        self.exposure = exposure
        self.wcs = wcs
        self.ccd_x = x + 44
        self.ccd_y = y

        # Load catalog files and find relevant stars
        gaia_sky_coordinates = SkyCoord(gaia_catalog["ra"], gaia_catalog["dec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            gaia_x, gaia_y = wcs.world_to_pixel(gaia_sky_coordinates)
        gaia_x_in_source = (self.ccd_x <= gaia_x) & (gaia_x <= self.ccd_x + size)
        gaia_y_in_source = (self.ccd_y <= gaia_y) & (gaia_y <= self.ccd_y + size)
        gaia_in_source = gaia_x_in_source & gaia_y_in_source
        catalogdata = gaia_catalog[gaia_in_source]

        tic_sky_coordinates = SkyCoord(tic_catalog["ra"], tic_catalog["dec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            tic_x, tic_y = wcs.world_to_pixel(tic_sky_coordinates)
        tic_x_in_source = (self.ccd_x <= tic_x) & (tic_x <= self.ccd_x + size)
        tic_y_in_source = (self.ccd_y <= tic_y) & (tic_y <= self.ccd_y + size)
        tic_in_source = tic_x_in_source & tic_y_in_source
        catalogdata_tic = tic_catalog[tic_in_source]

        # Cross match TIC and Gaia
        tic_match_table = Table()
        tic_match_table.add_column(catalogdata_tic["id"], name="TIC")
        tic_match_table.add_column(
            crossmatch_tic_to_gaia(catalogdata_tic, catalogdata), name="gaia_designation"
        )
        self.tic = tic_match_table

        # TODO remove this at some point, but right now units aren't expected downstream
        for name, col in catalogdata.columns.items():
            if np.ma.is_masked(col):
                catalogdata[name] = MaskedColumn(col.data, mask=col.mask)
            else:
                catalogdata[name] = Column(col.data)

        self.flux = flux[:, y : y + size, x : x + size]
        self.mask = mask[y : y + size, x : x + size]
        self.time = np.array(time)
        median_time = np.median(self.time)
        interval = (median_time - 388.5) / 365.25
        # Julian Day Number:	2457000.0 (TBJD=0)
        # Calendar Date/Time:	2014-12-08 12:00:00 388.5 days before J2016

        num_gaia = len(catalogdata)
        tic_id = np.zeros(num_gaia)
        x_gaia = np.zeros(num_gaia)
        y_gaia = np.zeros(num_gaia)
        tess_mag = np.zeros(num_gaia)
        in_frame = [True] * num_gaia
        for i, designation in enumerate(catalogdata["designation"]):
            ra = catalogdata["ra"][i]
            dec = catalogdata["dec"][i]
            if not np.isnan(catalogdata["pmra"].mask[i]):  # masked?
                ra += catalogdata["pmra"][i] * np.cos(np.deg2rad(dec)) * interval / 1000 / 3600
            if not np.isnan(catalogdata["pmdec"].mask[i]):
                dec += catalogdata["pmdec"][i] * interval / 1000 / 3600
            pixel = self.wcs.all_world2pix(
                np.array([catalogdata["ra"][i], catalogdata["dec"][i]]).reshape((1, 2)),
                0,
                quiet=True,
            )
            x_gaia[i] = pixel[0][0] - self.ccd_x
            y_gaia[i] = pixel[0][1] - self.ccd_y
            try:
                tic_id[i] = catalogdata_tic["ID"][
                    np.where(catalogdata_tic["GAIA"] == designation.split()[2])[0][0]
                ]
            except:
                tic_id[i] = np.nan
            if np.isnan(catalogdata["phot_g_mean_mag"][i]):
                in_frame[i] = False
            elif catalogdata["phot_g_mean_mag"][i] >= 25:
                in_frame[i] = False
            elif -4 < x_gaia[i] < self.size + 3 and -4 < y_gaia[i] < self.size + 3:
                dif = catalogdata["phot_bp_mean_mag"][i] - catalogdata["phot_rp_mean_mag"][i]
                with warnings.catch_warnings():
                    # Warnings for for masked value conversion to nan
                    warnings.simplefilter("ignore", UserWarning)
                    tess_mag[i] = (
                        catalogdata["phot_g_mean_mag"][i]
                        - 0.00522555 * dif**3
                        + 0.0891337 * dif**2
                        - 0.633923 * dif
                        + 0.0324473
                    )
                    if np.isnan(tess_mag[i]):
                        tess_mag[i] = catalogdata["phot_g_mean_mag"][i] - 0.430
                    if np.isnan(tess_mag[i]):
                        in_frame[i] = False
            else:
                in_frame[i] = False

        tess_flux = 10 ** (-tess_mag / 2.5)
        t = Table()
        t["tess_mag"] = tess_mag[in_frame]
        t["tess_flux"] = tess_flux[in_frame]
        t["tess_flux_ratio"] = tess_flux[in_frame] / (
            np.nanmax(tess_flux[in_frame]) if len(tess_flux[in_frame]) > 0 else 1
        )
        t[f"sector_{self.sector}_x"] = x_gaia[in_frame]
        t[f"sector_{self.sector}_y"] = y_gaia[in_frame]
        catalogdata = hstack([catalogdata[in_frame], t])
        catalogdata.sort("tess_mag")
        self.gaia = catalogdata


def _get_science_pixel_limits(scipixs_string: str) -> tuple[int, int, int, int]:
    """
    Parse string of the form `"[min_x:max_x,min_y:max_y]"`.

    TICA FFI FITS headers have a "SCIPIXS" keyword indicating which pixels are science pixels
    (rather than buffer rows/columns) of the form indicated above.

    Note
    ----
    The string in TICA headers is 1-indexed, but the pixel coordinates are converted to 0-indexed
    values for use with data arrays in this function. So, the science data from the detector is
    obtained by
    ```python
    science_data = data[min_y:max_y + 1, min_x:max_x + 1]
    ```
    where `data` is the primary extension data from the TICA FFI FITS file.

    Returns
    -------
    (min_x, max_x, min_y, max_y) : tuple[int, int, int, int]
        Tuple containing science pixel limits (inclusive)
    """
    x_range, y_range = scipixs_string.strip("[]").split(",")
    min_x, max_x = map(int, x_range.split(":"))
    min_y, max_y = map(int, y_range.split(":"))
    return (min_x - 1, max_x - 1, min_y - 1, max_y - 1)


def _get_ffi_header_data_and_flux(
    ffi_file: Path, camera: int
) -> tuple[int, int, float, np.ndarray]:
    """
    Harvest important header values and pixel flux values from a TICA FFI file.

    Parameters
    ----------
    ffi_file : Path
        Path to FFI FITS file to read.
    camera : int
        Camera of FFI. Needed to read stray light header value.

    Returns
    -------
    (quality, cadence, time, flux) : tuple[int, int, float, array_like]
        Tuple containing quality flag, cadence, and time values pulled from header and flux array
        taken from science pixels.
    """
    try:
        with warnings.catch_warnings():
            # TICA FITS headers get wrangled by Astropy, but it creates no problems
            warnings.simplefilter("ignore", AstropyWarning)
            with fits.open(ffi_file, mode="readonly", memmap=False) as hdulist:
                primary_header = hdulist[0].header
                # Map quality indicators to bits that align with FFI quality flags.
                # See https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014-Rev-F.pdf?page=56
                quality = (
                    (primary_header["COARSE"] << 2)
                    & (primary_header["RW_DESAT"] << 5)
                    & (primary_header[f"STRAYLT{camera}"] << 11)
                )
                cadence = primary_header["CADENCE"]
                time = primary_header["MIDTJD"]
                min_x, max_x, min_y, max_y = _get_science_pixel_limits(primary_header["SCIPIXS"])
                flux = hdulist[0].data[min_y : max_y + 1, min_x : max_x + 1]
        return (quality, cadence, time, flux)
    except Exception as e:
        logger.warning(f"Invalid FFI file {ffi_file.resolve()}: got error {e}")
        return (0, 0, np.nan, np.full((2048, 2048), np.nan))


def _make_source_and_write_pickle(
    xy: tuple[int, int], output_directory: Path, replace: bool, **kwargs
):
    """
    Construct source object and write pickle file.

    Designed for use with `multiprocessing.Pool.imap_unordered` and a `functools.partial`, so
    unpacks `x` and `y` from the first argument.
    """
    x, y = xy
    output_file = output_directory / f"source_{x:02d}_{y:02d}.pkl"
    if not replace and (output_file.is_file() and output_file.stat().st_size > 0):
        logger.debug(
            f"Source file for camera {kwargs['camera']}, CCD {kwargs['ccd']}, {x}_{y} already exists, skipping"
        )
        return
    kwargs["x"] = x * (kwargs["size"] - 4)
    kwargs["y"] = y * (kwargs["size"] - 4)
    source = Source(**kwargs)
    with open(output_file, "wb") as output:
        pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)


@jit(float32[:, :](float32[:, :, :]), nogil=True, parallel=True)
def _fast_nanmedian_axis0(array):
    """
    Fast JIT-compiled, multithreaded version of np.nanmedian(array, axis=0)
    
    Computing a nanmedian image from all the FFI data is necessary to detect bad pixels, but on
    arrays with roughly the shape (6000, 2048, 2048), this is incredibly. We use numba here to
    distribute the work to many cores.
    """
    result = np.empty(array.shape[1:], dtype=np.float32)
    for i in prange(result.shape[0]):
        for j in prange(result.shape[1]):
            result[i, j] = np.nanmedian(array[:, i, j])
    return result


def ffi(
    camera: int,
    ccd: int,
    orbit: int,
    sector: int,
    base_directory: Path,
    cutout_size: int = 150,
    produce_mask: bool = False,
    nprocs: int = 1,
    replace: bool = False,
):
    """
    Produce `Source` object pickle file from calibrated FFI files.

    Within `base_directory`, the `source/{camera}-{ccd}/` directory is created and populated. If
    `produce_mask` is `True`, the `mask/` directory is created and populated instead.

    Parameters
    ----------
    camera, ccd : int
        TESS camera and CCD of FFIs that should be used.
    orbit, sector : int
        TESS orbit and sector containing the orbit of the FFI observations.
    base_directory : Path
        Base path to contain data products. Should contain populated `catalogs/` and `ffi/`
        subdirectories.
    cutout_size : int
        Side length of cutouts. Large numbers recommended for better quality. Default = 150.
    produce_mask : bool
        Produce CCD mask instead of making cutout `Source` objects.
    nprocs : int
        Processes to use for in multiprocessing pool. Default = 1.
    replace : bool
        Replace existing files with new data. Default = False.
    """
    base_directory = Path(base_directory)
    ffi_directory = base_directory / "ffi"
    ffi_files = list(ffi_directory.glob(f"*cam{camera}-ccd{ccd}*_img.fits"))

    if len(ffi_files) == 0:
        logger.warning(f"No FFI files found for camera {camera} CCD {ccd}, skipping")
        return
    logger.info(f"Found {len(ffi_files)} FFI files for camera {camera} CCD {ccd}")

    time = np.full_like(ffi_files, np.nan, dtype=np.float64)
    cadence = np.zeros_like(ffi_files, dtype=np.int64)
    quality = np.zeros_like(ffi_files, dtype=np.int32)
    flux = np.full((len(ffi_files), 2048, 2048), np.nan, dtype=np.float32)
    get_ffi_header_data_and_flux_for_camera = partial(_get_ffi_header_data_and_flux, camera=camera)
    ffi_data_iterator = tqdm(
        pool_map_if_multiprocessing(
            get_ffi_header_data_and_flux_for_camera,
            ffi_files,
            nprocs=nprocs,
            pool_map_method="imap_unordered",
        ),
        desc=f"Reading FFI files for {camera}-{ccd}",
        unit="file",
        total=len(ffi_files),
    )
    with logging_redirect_tqdm():
        for i, (ffi_quality, ffi_cadence, ffi_time, ffi_flux) in enumerate(ffi_data_iterator):
            quality[i] = ffi_quality
            cadence[i] = ffi_cadence
            time[i] = ffi_time
            flux[i] = ffi_flux
    logger.info("Sorting FFI data by timestamp")
    time_order = np.argsort(time)
    time = time[time_order]
    cadence = cadence[time_order]
    quality = quality[time_order]
    flux = flux[time_order, :, :]

    if np.min(np.diff(cadence)) != 1:
        logger.warning(f"{(np.diff(cadence) != 1).sum()} cadence gaps != 1 detected.")

    # Load or save mask
    numba.set_num_threads(nrpocs)
    if produce_mask:
        logger.info("Saving background mask")
        median_flux = _fast_nanmedian_axis0(flux)
        mask = background_mask(im=median_flux)
        mask /= ndimage.median_filter(mask, size=51)
        np.save(base_directory / f"mask/mask_sector{sector:04d}_cam{camera}_ccd{ccd}.npy", mask)
        return
    logger.info("Loading background mask")
    mask_file = resources.files(data) / "median_mask.fits"
    with fits.open(mask_file) as hdulist:
        mask = hdulist[0].data[(camera - 1) * 4 + (ccd - 1), :]
    mask = np.repeat(mask.reshape(1, 2048), repeats=2048, axis=0)

    logger.info("Detecting bad pixels")
    bad_pixels = np.zeros(flux.shape[1:], dtype=bool)
    median_flux = _fast_nanmedian_axis0(flux)
    bad_pixels[median_flux > 0.8 * bn.nanmax(median_flux)] = 1
    bad_pixels[median_flux < 0.2 * bn.nanmedian(median_flux)] = 1
    bad_pixels[np.isnan(median_flux)] = 1

    # Mark neighbors of bad pixels as also bad
    bad_y, bad_x = np.nonzero(bad_pixels)
    for x, y in zip(bad_x, bad_y):
        bad_pixels[min(y + 1, 2047), x] = 1
        bad_pixels[max(y - 1, 0), x] = 1
        bad_pixels[y, min(x + 1, 2047)] = 1
        bad_pixels[y, max(x - 1, 0)] = 1

    mask = np.ma.masked_array(mask, mask=bad_pixels | (mask == 0))

    logger.info("Loading WCS pixel-to-world solution")
    # Get WCS object from good-quality FFI
    first_good_quality_ffi = ffi_files[np.nonzero(quality == 0)[0][0]]
    with warnings.catch_warnings():
        # TICA FITS headers get wrangled by Astropy, but it creates no problems
        warnings.simplefilter("ignore", AstropyWarning)
        with fits.open(first_good_quality_ffi) as hdulist:
            wcs = WCS(hdulist[0].header)
            exposure = int(hdulist[0].header["EXPTIME"])

    catalogs_directory = base_directory / "catalogs"
    logger.info(
        f"Reading catalogs for camera {camera} CCD {ccd} from {catalogs_directory.resolve()}"
    )
    gaia_catalog = QTable.read(catalogs_directory / f"Gaia_camera{camera}_ccd{ccd}.ecsv")
    tic_catalog = QTable.read(catalogs_directory / f"TIC_camera{camera}_ccd{ccd}.ecsv")

    source_directory = base_directory / f"source/{camera}-{ccd}"
    source_directory.mkdir(exist_ok=True)
    write_source_pickle_from_x_y = partial(
        _make_source_and_write_pickle,
        output_directory=source_directory,
        replace=replace,
        flux=flux,
        mask=mask,
        orbit=orbit,
        sector=sector,
        time=time,
        size=cutout_size,
        quality=quality,
        wcs=wcs,
        camera=camera,
        ccd=ccd,
        exposure=exposure,
        cadence=cadence,
        gaia_catalog=gaia_catalog,
        tic_catalog=tic_catalog,
    )
    # TODO remove this comment when the issue is resolved
    # Currently we don't actually do multiprocessing here because the catalogs can't be pickled.
    # There is a problem pickling astropy `MaskedQuantity` objects. There was some progress in
    # v7.0.1, and there appears to be an issue tracking the remaining problem:
    # https://github.com/astropy/astropy/issues/16352
    source_writer_iterator = tqdm(
        pool_map_if_multiprocessing(
            write_source_pickle_from_x_y,
            product(range(14), repeat=2),
            nprocs=1,  # TODO change to `nprocs=procs` when issue above is resolved
            pool_map_method="imap_unordered",
        ),
        desc=f"Writing source pickle files for {camera}-{ccd}",
        unit="source",
        total=14 * 14,
    )
    # Lazy iterator needs to be consumed
    with logging_redirect_tqdm():
        for _ in source_writer_iterator:
            pass
