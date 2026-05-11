import pathlib
from loguru import logger
from astropy.coordinates import SkyCoord
from qlp.find import project_sky_to_pixel
from astropy.io import fits
from qlp.io.backends import default_io_backend as io
import pandas as pd
import numpy as np

# tess2020281075104-00129824-3-crm-ffi_dehoc_likelihood.fits
BEAM_DIR = pathlib.Path("/pdo/users/djtufto/BEAM/model_outputs/sector30/likelihood/")

def extract_beam_cadence(path) -> int:
    tokens = path.stem.split("-")
    cadence = int(tokens[1])
    return cadence

def load_beam_data(beam_dir):
    files = list(beam_dir.glob("*_likelihood.fits"))
    return files

logger.info("Loading BEAM FITS")
beam_files = [(path, extract_beam_cadence(path)) for path in load_beam_data(BEAM_DIR)]
beam_files = sorted(beam_files, key=lambda r: r[1])
logger.info(f"Found {len(beam_files)} BEAM FITS files")

SECTOR = 30
CAMERA = 3

logger.info(f"Reacting s{SECTOR:04d} catalog for camera {CAMERA}")
catalog = pd.read_csv("/pdo/qlp-data/orbit-67/ffi/run/catalog_67_3_1_full.txt", sep="\s+", names=["tic_id", "ra", "dec", "tmag", "_", "__", "___", "____", "_____"])
coordinates = SkyCoord(catalog["ra"], catalog["dec"], unit="degree", frame="icrs")

logger.info("Projecting Catalog for DEHOC camera coordinates")
sector_info = io.read_sector_info(SECTOR)
projection = project_sky_to_pixel(coordinates, orbit=sector_info.orbit_info[0].orbit_number, frame="camera")

logger.info("Reducing catalog to only camera {camera} targets")
catalog["camera"] = np.nan
catalog["camera_x"] = np.nan
catalog["camera_y"] = np.nan
catalog.loc[projection.target_index, "camera"] = projection.camera
catalog.loc[projection.target_index, "camera_x"] = projection.pixel_x
catalog.loc[projection.target_index, "camera_y"] = projection.pixel_y

camera3_catalog = catalog[catalog["camera"] == 3]
# Build apertures...



def perform_photometry(beam_file):
    with fits.open(beam_file) as fin:
        
