import json
import os
import pickle
import sys
import astropy.units as u
import numpy as np
import pkg_resources
import requests
import time

from glob import glob
from os.path import exists
from urllib.parse import quote as urlencode
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, QTable, hstack, Column, MaskedColumn
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from scipy import ndimage
from tqdm import tqdm, trange

from tglc.util.constants import convert_gaia_mags_to_tmag

Gaia.ROW_LIMIT = -1
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # TODO: dr3 MJD = 2457388.5, TBJD = 388.5


# The next three functions are adopted from astroquery MAST API https://mast.stsci.edu/api/v0/pyex.html#incPy
def mast_query(request):
    """Perform a MAST query.

        Parameters
        ----------
        request (dictionary): The MAST request json object

        Returns head,content where head is the response HTTP headers, and content is the returned data"""

    # Base API url
    request_url = 'https://mast.stsci.edu/api/v0/invoke'
    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))
    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent": "python-requests/" + version}
    # Encoding the request as a json string
    req_string = json.dumps(request)
    req_string = urlencode(req_string)
    # Perform the HTTP request
    resp = requests.post(request_url, data="request=" + req_string, headers=headers)
    # Pull out the headers and response content
    head = resp.headers
    content = resp.content.decode('utf-8')
    return head, content


def mast_json2table(json_obj):
    data_table = Table()
    for col, atype in [(x['name'], x['type']) for x in json_obj['fields']]:
        if atype == "string":
            atype = "str"
        if atype == "boolean":
            atype = "bool"
        data_table[col] = np.array([x.get(col, None) for x in json_obj['data']], dtype=atype)
    return data_table



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
    tic_coords = SkyCoord(
        ra=tic["ra"],
        dec=tic["dec"],
        pm_ra_cosdec=tic["pmra"],
        pm_dec=tic["pmdec"],
        frame="icrs",
        obstime=Time("J2000"),
    )
    pm_adjusted_tic_coords = tic_coords.apply_space_motion(Time("J2016"))
    gaia_coords = SkyCoord(gaia["ra"], gaia["dec"])
    match_idx, match_dist_angle, _match_dist_3d = pm_adjusted_tic_coords.match_to_catalog_sky(gaia_coords)

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
    imfilt = im * 1.
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
        self, x=0, y=0,
        flux=None, time=None, wcs=None, quality=None, mask=None, exposure=1800,
        sector=0, size=150, camera=1, ccd=1, cadence=None, catalogs_directory=None
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
        :param catalogs_directory: str, required
        path to directory containing Gaia and TIC catalog files
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
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.cadence = cadence
        self.quality = quality
        self.exposure = exposure
        self.wcs = wcs

        # Load catalog files and find relevant stars
        gaia_catalog_file = catalogs_directory + f"Gaia_camera{camera}.ecsv"
        gaia_catalog = QTable.read(gaia_catalog_file)
        gaia_sky_coordinates = SkyCoord(gaia_catalog["ra"], gaia_catalog["dec"])
        gaia_x, gaia_y = wcs.world_to_pixel(gaia_sky_coordinates)
        gaia_x_in_source = (x <= gaia_x) & (gaia_x <= x + size)
        gaia_y_in_source = (y <= gaia_y) & (gaia_y <= y + size)
        gaia_in_source = gaia_x_in_source & gaia_y_in_source
        catalogdata = gaia_catalog[gaia_in_source]

        tic_catalog_file = catalogs_directory + f"TIC_camera{camera}.ecsv"
        tic_catalog = QTable.read(tic_catalog_file)
        tic_sky_coordinates = SkyCoord(tic_catalog["ra"], tic_catalog["dec"])
        tic_x, tic_y = wcs.world_to_pixel(tic_sky_coordinates)
        tic_x_in_source = (x <= tic_x) & (tic_x <= x + size)
        tic_y_in_source = (y <= tic_y) & (tic_y <= y + size)
        tic_in_source = tic_x_in_source & tic_y_in_source
        catalogdata_tic = tic_catalog[tic_in_source]

        # Cross match TIC and Gaia
        tic_match_table = Table()
        tic_match_table.add_column(catalogdata_tic["id"], name="TIC")
        tic_match_table.add_column(crossmatch_tic_to_gaia(catalogdata_tic, catalogdata), name="gaia_designation")
        self.tic = tic_match_table

        # TODO remove this at some point, but right now units aren't expected downstream
        for name, col in catalogdata.columns.items():
            if np.ma.is_masked(col):
                catalogdata[name] = MaskedColumn(col.data, mask=col.mask)
            else:
                catalogdata[name] = Column(col.data)

        self.flux = flux[:, y:y + size, x:x + size]
        self.mask = mask[y:y + size, x:x + size]
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
        for i, designation in enumerate(catalogdata['designation']):
            ra = catalogdata['ra'][i]
            dec = catalogdata['dec'][i]
            if not np.isnan(catalogdata['pmra'].mask[i]):  # masked?
                ra += catalogdata['pmra'][i] * np.cos(np.deg2rad(dec)) * interval / 1000 / 3600
            if not np.isnan(catalogdata['pmdec'].mask[i]):
                dec += catalogdata['pmdec'][i] * interval / 1000 / 3600
            pixel = self.wcs.all_world2pix(
                np.array([catalogdata['ra'][i], catalogdata['dec'][i]]).reshape((1, 2)), 0, quiet=True)
            x_gaia[i] = pixel[0][0] - x - 44
            y_gaia[i] = pixel[0][1] - y
            try:
                tic_id[i] = catalogdata_tic['ID'][np.where(catalogdata_tic['GAIA'] == designation.split()[2])[0][0]]
            except:
                tic_id[i] = np.nan
            if np.isnan(catalogdata['phot_g_mean_mag'][i]):
                in_frame[i] = False
            elif catalogdata['phot_g_mean_mag'][i] >= 25:
                in_frame[i] = False
            elif -4 < x_gaia[i] < self.size + 3 and -4 < y_gaia[i] < self.size + 3:
                dif = catalogdata['phot_bp_mean_mag'][i] - catalogdata['phot_rp_mean_mag'][i]
                tess_mag[i] = catalogdata['phot_g_mean_mag'][
                                  i] - 0.00522555 * dif ** 3 + 0.0891337 * dif ** 2 - 0.633923 * dif + 0.0324473
                if np.isnan(tess_mag[i]):
                    tess_mag[i] = catalogdata['phot_g_mean_mag'][i] - 0.430
                if np.isnan(tess_mag[i]):
                    in_frame[i] = False
            else:
                in_frame[i] = False

        tess_flux = 10 ** (- tess_mag / 2.5)
        t = Table()
        t[f'tess_mag'] = tess_mag[in_frame]
        t[f'tess_flux'] = tess_flux[in_frame]
        t[f'tess_flux_ratio'] = tess_flux[in_frame] / np.nanmax(tess_flux[in_frame])
        t[f'sector_{self.sector}_x'] = x_gaia[in_frame]
        t[f'sector_{self.sector}_y'] = y_gaia[in_frame]
        catalogdata = hstack([catalogdata[in_frame], t])
        catalogdata.sort('tess_mag')
        self.gaia = catalogdata

    def search_gaia(self, x, y, co1, co2):
        coord = self.wcs.pixel_to_world([x + co1 + 44], [y + co2])[0].to_string()
        radius = u.Quantity((self.size / 2 + 4) * 21 * 0.707 / 3600, u.deg)
        attempt = 0
        while attempt < 5:
            try:
                catalogdata = Gaia.cone_search_async(coord, radius=radius,
                                             columns=['DESIGNATION', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                                                      'phot_rp_mean_mag', 'ra', 'dec', 'pmra', 'pmdec']).get_results()
                return catalogdata
            except:
                attempt += 1
                time.sleep(10)
                print(f'Trying Gaia search again. Coord = {coord}, radius = {radius}')

def ffi(ccd=1, camera=1, sector=1, size=150, local_directory='', producing_mask=False):
    """
    Generate Source object from the calibrated FFI downloaded directly from MAST
    :param sector: int, required
    TESS sector number
    :param camera: int, required
    camera number
    :param ccd: int, required
    ccd number
    :param size: int, optional
    size of the FFI cut, default size is 150. Recommend large number for better quality.
    :param local_directory: string, required
    path to the FFI folder
    :return:
    """
    # input_files = glob(f'/pdo/spoc-data/sector-{sector:03d}/ffi*/**/*{camera}-{ccd}-????-?_ffic.fits*')
    input_files = glob(f'{local_directory}ffi/*cam{camera}-ccd{ccd}*_img.fits')
    print('camera: ' + str(camera) + '  ccd: ' + str(ccd) + '  num of files: ' + str(len(input_files)))
    time = []
    quality = []
    cadence = []
    flux = np.empty((len(input_files), 2048, 2048), dtype=np.float32)
    for i, file in enumerate(tqdm(input_files)):
        try:
            with fits.open(file, mode='denywrite', memmap=False) as hdul:
                quality_flag = (
                    (hdul[0].header['COARSE'] << 2)
                    & (hdul[0].header['RW_DESAT'] << 5)
                    & (hdul[0].header[f'STRAYLT{camera}'])
                )
                quality.append(quality_flag)
                cadence.append(hdul[0].header['CADENCE'])
                flux[i] = hdul[0].data[0:2048, 44:2092]
                time.append(hdul[0].header['MIDTJD'])

        except:
            print(f'Corrupted file {file}, download again ...')
            response = requests.get(
                f'https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/{os.path.basename(file)}')
            open(file, 'wb').write(response.content)
            with fits.open(file, mode='denywrite', memmap=False) as hdul:
                quality_flag = (
                    (hdul[0].header['COARSE'] << 2)
                    & (hdul[0].header['RW_DESAT'] << 5)
                    & (hdul[0].header[f'STRAYLT{camera}'])
                )
                quality.append(quality_flag)
                cadence.append(hdul[0].header['CADENCE'])
                flux[i] = hdul[0].data[0:2048, 44:2092]
                time.append(hdul[0].header['MIDTJD'])
    time_order = np.argsort(np.array(time))
    time = np.array(time)[time_order]
    flux = flux[time_order, :, :]
    quality = np.array(quality)[time_order]
    cadence = np.array(cadence)[time_order]
    # mask = np.array([True] * 2048 ** 2).reshape(2048, 2048)
    # for i in range(len(time)):
    #     mask[np.where(flux[i] > np.percentile(flux[i], 99.95))] = False
    #     mask[np.where(flux[i] < np.median(flux[i]) / 2)] = False
    if np.min(np.diff(cadence)) != 1:
        np.save(f'{local_directory}/Wrong_Cadence_sector{sector:04d}_cam{camera}_ccd{ccd}.npy', np.min(np.diff(cadence)))
    if producing_mask:
        median_flux = np.median(flux, axis=0)
        mask = background_mask(im=median_flux)
        mask /= ndimage.median_filter(mask, size=51)
        np.save(f'{local_directory}mask/mask_sector{sector:04d}_cam{camera}_ccd{ccd}.npy', mask)
        return
    # load mask
    mask = pkg_resources.resource_stream(__name__, f'background_mask/median_mask.fits')
    with fits.open(mask) as mask_hdul:
        mask = mask_hdul[0].data[(camera - 1) * 4 + (ccd - 1), :]
    mask = np.repeat(mask.reshape(1, 2048), repeats=2048, axis=0)
    bad_pixels = np.zeros(np.shape(flux[0]))
    med_flux = np.median(flux, axis=0)
    bad_pixels[med_flux > 0.8 * np.nanmax(med_flux)] = 1
    bad_pixels[med_flux < 0.2 * np.nanmedian(med_flux)] = 1
    bad_pixels[np.isnan(med_flux)] = 1

    x_b, y_b = np.where(bad_pixels)
    for i in range(len(x_b)):
        if x_b[i] < 2047:
            bad_pixels[x_b[i] + 1, y_b[i]] = 1
        if x_b[i] > 0:
            bad_pixels[x_b[i] - 1, y_b[i]] = 1
        if y_b[i] < 2047:
            bad_pixels[x_b[i], y_b[i] + 1] = 1
        if y_b[i] > 0:
            bad_pixels[x_b[i], y_b[i] - 1] = 1

    mask = np.ma.masked_array(mask, mask=bad_pixels)
    mask = np.ma.masked_equal(mask, 0)

    for i in range(10):
        with fits.open(input_files[np.nonzero(np.array(quality) == 0)[0][i]]) as hdul:
            wcs = WCS(hdul[0].header)
        if wcs.axis_type_names == ['RA', 'DEC']:
            exposure = int(hdul[0].header['EXPTIME'])
            break


    # 95*95 cuts with 2 pixel redundant, (22*22 cuts)
    # try 77*77 with 4 redundant, (28*28 cuts)
    os.makedirs(f'{local_directory}source/{camera}-{ccd}/', exist_ok=True)
    for i in trange(14):  # 22
        for j in range(14):  # 22
            source_path = f'{local_directory}source/{camera}-{ccd}/source_{i:02d}_{j:02d}.pkl'
            source_exists = exists(source_path)
            if source_exists and os.path.getsize(source_path) > 0:
                # print(f'{source_path} exists. ')
                pass
            else:
                with open(source_path, 'wb') as output:
                    source = Source(x=i * (size - 4), y=j * (size - 4), flux=flux, mask=mask, sector=sector,
                                    time=time, size=size, quality=quality, wcs=wcs, camera=camera, ccd=ccd,
                                    exposure=exposure, cadence=cadence, catalogs_directory=f"{local_directory}catalogs/")
                    pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)
