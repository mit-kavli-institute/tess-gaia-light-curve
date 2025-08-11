"""
Script/functions for downloading sample data products.

Uses [pooch](https://www.fatiando.org/pooch/latest/index.html) to handle actual downloads.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pooch


def download_ffis():
    """
    Download the TICA calibrated FFIs for end-to-end tests.

    Files are stored in the `ffi/` subdirectory of this directory (`tests/sample_data`).

    See the sample data readme for information on updating the data used here.
    """
    base_url = "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/tica/s0089/cam1-ccd1/"
    file_fmt = "hlsp_tica_tess_ffi_s0089-o1-{cadence:08d}-cam1-ccd1_tess_v01_img.fits"
    cadences = range(1078000, 1078005)
    hashes = [
        "md5:e980f06c83563151b7c00ba610e31b53",
        "md5:c70c89e319d77d8fe129e5c124abc340",
        "md5:7c5c9e005fd787721bf27cb6fa8085b0",
        "md5:fcfa0f782ed05e18867f6f0cfad28bc4",
        "md5:b2b9f3f05a63ff248e41966cc7c7e814",
    ]
    ffi_download_manager = pooch.create(
        Path(__file__).parent / "ffi",
        base_url,
        registry={
            file_fmt.format(cadence=cadence): f"md5:{file_hash}"
            for (cadence, file_hash) in zip(cadences, hashes, strict=False)
        },
    )

    def fetch_ffi(cadence):
        ffi_download_manager.fetch(file_fmt.format(cadence=cadence))

    # Download each FFI in a separate thread
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(fetch_ffi, cadences)


if __name__ == "__main__":
    download_ffis()
