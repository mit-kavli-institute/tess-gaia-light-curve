# TGLC Testing Sample Data

Testing TGLC, especially end-to-end tests that run high-level commands, requires some sample data to operate on. The prerequisite data requirements for TGLC are TESS FFIs and TIC & Gaia databases to generate catalogs from. The sample databases can be small, including only the data for a single FFI cutout. The FFIs, however, are quite large and would be impractical to include in the repository. Therefore, we have download mechanisms to fetch the data when it is needed for testing.

Orbit 185 (sector 89), camera 1, CCD 1, cutout (0, 0) was chosen for sample data. There is no particular reason for this aside from the fact that sector 89 was used for initial testing of the pipeline, so there were data products to copy. The information in this document and the sample databases readme should be enough to set up new sample data if it ever becomes necessary or desirable.

## FFI Download Setup

Once downloaded, FFIs are stored in the [ffi/](./ffi/) subdirectory of this sample data directory. The files need to be fetched and preprocessed ahead of time because [pooch](https://www.fatiando.org/pooch/latest/index.html) needs file hashes to check. The process is outlined here:

1. Decide on a sector/camera/CCD and list of cadences to use. Encode that information into the `file_fmt` and `cadences` variables in the `download_ffis` function in [download.py](./download.py).
2. Download the files from MAST directly - see for example the bulk download scripts:
   > https://archive.stsci.edu/hlsp/tica#h3-CK-72718332-3aaa-4bfb-8dd3-57d7457fe30e
3. Run `pooch.file_hash(file, alg="md5")` on each file to determine the hashes. Collect them into the `hashes` variable in `download_ffis()`.
4. Update the test of the `tmp_orbit_directory` fixture in [/tests/end_to-send/test_full_pipeline.py](/tests/end_to_end/test_full_pipeline.py) to check for the correct number of FFIs in the correct locations (camera/CCD).

At the time of writing, MAST indicates that the URL for a TICA FFI is

```
https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/tica/<sector>/<cam>-<ccd>
```

and the file names within that URL are

```
hlsp_tica_tess_ffi_<sector>-<orbit>-<cadence>-<cam>-<ccd>_tess_v01_<extension>
```

where:

- `<sector>` = The Sector number as a four-digit, zero-padded number preceded by an 's', e.g.,
  's0027' for Sector 27.
- `<orbit>` = The relative orbit number within this Sector, either "o1" or "o2". Note that
  this does not correlate to the absolute Orbit Number that is tracked and reported in the
  Data Release Notes and in mission-produced products.
- `<cadence>` = The cadence number, as an eight-digit, zero-padded integer, equal to the
  FFIINDEX header keyword for mission-produced FFIs.
- `<cam>` = The camera number (1-4), e.g., "cam2" for Camera #2.
- `<ccd>` = The CCD number (1-4) for that camera, e.g., "ccd1" for CCD #1.
- `<extension>` = "img.fits" for the calibrated full frame images.
