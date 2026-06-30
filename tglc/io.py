"""
FITS I/O for TGLC intermediate data products.

This module replaces the older pickle (`.pkl`) and NumPy (`.npy`) formats used
for FFI cutouts (:class:`tglc.ffi.FFICutout`) and fitted ePSFs. See GitHub
issue #1 for motivation. The on-disk layouts are documented next to each
writer function.

Translators are provided to convert legacy products to the new FITS format
before archival on the MIT TSO servers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import pickle
from typing import TYPE_CHECKING
import warnings

from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Column, MaskedColumn, Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS, FITSFixedWarning
import numpy as np


if TYPE_CHECKING:
    from tglc.ffi import FFICutout


logger = logging.getLogger(__name__)


EPSF_BACKGROUND_COLUMNS = (
    "y_strap",
    "x_strap",
    "flat_strap",
    "x_gradient",
    "y_gradient",
    "flat",
)
"""Names of the 6 background columns appended after the ePSF parameters.

The order matches the construction in :func:`tglc.epsf.make_tglc_design_matrix`.
"""


def _atomic_write(hdul: fits.HDUList, path: Path) -> None:
    """Write a FITS HDUList atomically via a temporary file and ``os.replace``.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        HDUList to serialize.
    path : pathlib.Path
        Final destination path. A sibling ``<name>.tmp`` file is written first
        and renamed into place once the write completes successfully.
    """
    tmp_path = path.parent / (path.name + ".tmp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        hdul.writeto(tmp_path, overwrite=True)
    os.replace(tmp_path, path)


def _convert_table_to_native_byteorder(table: Table) -> None:
    """Rewrite numeric columns of ``table`` in native byte order, in place.

    FITS stores numeric data as big-endian; on a little-endian host the
    columns come back with dtypes like ``'>f8'``. Numba (used in
    :func:`tglc.epsf.make_tglc_design_matrix`) refuses non-native-byteorder
    arrays, so this normalizes at the I/O boundary.

    Parameters
    ----------
    table : astropy.table.Table
        Table whose numeric columns will be rewritten. String, byte, and
        object columns are left untouched. Modified in place.
    """
    for name in list(table.colnames):
        col = table[name]
        dtype = col.dtype
        if dtype.byteorder not in ("=", "|") and dtype.kind not in ("U", "S", "O"):
            native = dtype.newbyteorder("=")
            if isinstance(col, MaskedColumn):
                table[name] = MaskedColumn(
                    np.asarray(col.data, dtype=native),
                    mask=col.mask,
                    name=name,
                    unit=col.unit,
                )
            else:
                table[name] = Column(
                    np.asarray(col, dtype=native),
                    name=name,
                    unit=col.unit,
                )


# ---------------------------------------------------------------------
# FFI cutout I/O
# ---------------------------------------------------------------------


def write_cutout_fits(cutout: FFICutout, path: Path) -> None:
    """Write an FFI cutout to a multi-extension FITS file.

    HDU layout:

    * PRIMARY -- empty data; scalar metadata in header
    * FLUX -- (t, size, size) float32 image cube; WCS keys in header
    * MASK -- (size, size) float32 strap weights (``cutout.mask.data``)
    * BADPIX -- (size, size) uint8 bad-pixel mask (``cutout.mask.mask``)
    * CADENCES -- BINTABLE of co-indexed ``time``, ``cadence``, ``quality``
    * GAIA -- BINTABLE of the gaia catalog
    * TIC -- BINTABLE of the TIC <-> Gaia DR3 crossmatch

    Parameters
    ----------
    cutout : tglc.ffi.FFICutout
        Cutout object to serialize. All persisted attributes
        (``orbit``/``sector``/``camera``/``ccd``/``size``/``ccd_x``/``ccd_y``/
        ``exposure``/``cutout_x``/``cutout_y``/``wcs``/``flux``/``mask``/
        ``time``/``cadence``/``quality``/``gaia``/``tic``) are read from it.
    path : pathlib.Path
        Output FITS file path. The file is written atomically via
        :func:`_atomic_write`.
    """
    path = Path(path)

    primary_header = fits.Header()
    primary_header["ORBIT"] = int(cutout.orbit)
    primary_header["SECTOR"] = int(cutout.sector)
    primary_header["CAMERA"] = int(cutout.camera)
    primary_header["CCD"] = int(cutout.ccd)
    primary_header["CUTSIZE"] = int(cutout.size)
    primary_header["CCDX"] = int(cutout.ccd_x)
    primary_header["CCDY"] = int(cutout.ccd_y)
    primary_header["EXPOSURE"] = int(cutout.exposure)
    primary_header["CUTOUTX"] = int(getattr(cutout, "cutout_x", -1))
    primary_header["CUTOUTY"] = int(getattr(cutout, "cutout_y", -1))

    primary_hdu = fits.PrimaryHDU(header=primary_header)

    flux_header = fits.Header()
    if cutout.wcs is not None and isinstance(cutout.wcs, WCS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            warnings.simplefilter("ignore", VerifyWarning)
            warnings.simplefilter("ignore", AstropyWarning)
            flux_header.extend(cutout.wcs.to_header(relax=True))
    flux_hdu = fits.ImageHDU(
        data=np.ascontiguousarray(cutout.flux, dtype=np.float32),
        header=flux_header,
        name="FLUX",
    )

    mask_array = cutout.mask
    if hasattr(mask_array, "mask"):
        mask_data = np.ascontiguousarray(np.ma.getdata(mask_array), dtype=np.float32)
        badpix_data = np.ascontiguousarray(np.ma.getmaskarray(mask_array).astype(np.uint8))
    else:
        mask_data = np.ascontiguousarray(mask_array, dtype=np.float32)
        badpix_data = np.zeros_like(mask_data, dtype=np.uint8)
    mask_hdu = fits.ImageHDU(data=mask_data, name="MASK")
    badpix_hdu = fits.ImageHDU(data=badpix_data, name="BADPIX")

    cadences_table = Table(
        {
            "time": np.asarray(cutout.time, dtype=np.float64),
            "cadence": np.asarray(cutout.cadence, dtype=np.int64),
            "quality": np.asarray(cutout.quality, dtype=np.int32),
        }
    )
    cadences_hdu = fits.BinTableHDU(cadences_table, name="CADENCES")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        gaia_hdu = fits.BinTableHDU(cutout.gaia, name="GAIA")
        tic_hdu = fits.BinTableHDU(cutout.tic, name="TIC")

    hdul = fits.HDUList(
        [primary_hdu, flux_hdu, mask_hdu, badpix_hdu, cadences_hdu, gaia_hdu, tic_hdu]
    )

    _atomic_write(hdul, path)


def read_cutout_fits(path: Path) -> FFICutout:
    """Read an :class:`FFICutout` previously written by :func:`write_cutout_fits`.

    Reconstructs the object without invoking ``FFICutout.__init__``, which
    performs catalog cross-matching and would discard the persisted state.

    Parameters
    ----------
    path : pathlib.Path
        Path to a FITS file produced by :func:`write_cutout_fits`.

    Returns
    -------
    cutout : tglc.ffi.FFICutout
        Reconstructed cutout with all persisted attributes populated. Table
        columns are converted to native byte order; ``gaia['designation']``
        is decoded to ``str``; ``gaia['pmra']`` and ``gaia['pmdec']`` are
        promoted back to :class:`astropy.table.MaskedColumn` if astropy
        returned them as plain :class:`Column`.

    Raises
    ------
    Exception
        Propagates any error raised by :class:`astropy.wcs.WCS` if the
        persisted WCS header cannot be parsed. Per the issue #1 design,
        WCS parse failures are not silently swallowed.
    """
    from tglc.ffi import FFICutout  # local import: breaks the ffi <-> io cycle

    path = Path(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        warnings.simplefilter("ignore", VerifyWarning)
        warnings.simplefilter("ignore", AstropyWarning)
        with fits.open(path) as hdul:
            primary_header = hdul[0].header
            flux_hdu = hdul["FLUX"]
            # naxis=2 keeps WCS construction 2D in spite of NAXIS3 on the cube.
            # Without it, real TICA WCS headers (which carry SIP distortion)
            # error out: SIP only supports 2-axis WCS.
            wcs = WCS(flux_hdu.header, naxis=2, relax=True)
            flux = np.array(flux_hdu.data, dtype=np.float32)
            mask_data = np.array(hdul["MASK"].data, dtype=np.float32)
            badpix_data = np.array(hdul["BADPIX"].data, dtype=bool)
            cadences_table = Table.read(hdul["CADENCES"])
            gaia_table = Table.read(hdul["GAIA"])
            tic_table = Table.read(hdul["TIC"])

    cutout = object.__new__(FFICutout)
    cutout.size = int(primary_header["CUTSIZE"])
    cutout.orbit = int(primary_header["ORBIT"])
    cutout.sector = int(primary_header["SECTOR"])
    cutout.camera = int(primary_header["CAMERA"])
    cutout.ccd = int(primary_header["CCD"])
    cutout.ccd_x = int(primary_header["CCDX"])
    cutout.ccd_y = int(primary_header["CCDY"])
    cutout.exposure = int(primary_header["EXPOSURE"])
    cutout.cutout_x = int(primary_header.get("CUTOUTX", -1))
    cutout.cutout_y = int(primary_header.get("CUTOUTY", -1))
    cutout.wcs = wcs
    cutout.flux = flux
    cutout.mask = np.ma.masked_array(mask_data, mask=badpix_data)
    cutout.time = np.array(cadences_table["time"], dtype=np.float64)
    cutout.cadence = np.array(cadences_table["cadence"], dtype=np.int64)
    cutout.quality = np.array(cadences_table["quality"], dtype=np.int32)

    if "designation" in gaia_table.colnames:
        col = gaia_table["designation"]
        if col.dtype.kind in ("S", "O"):
            gaia_table["designation"] = Column(
                [d.decode() if isinstance(d, bytes) else str(d) for d in col],
                name="designation",
            )
    for col_name in ("pmra", "pmdec"):
        if col_name in gaia_table.colnames and not isinstance(gaia_table[col_name], MaskedColumn):
            col_data = np.asarray(gaia_table[col_name])
            gaia_table[col_name] = MaskedColumn(col_data, mask=np.isnan(col_data), name=col_name)
    _convert_table_to_native_byteorder(gaia_table)
    _convert_table_to_native_byteorder(tic_table)
    cutout.gaia = gaia_table
    cutout.tic = tic_table

    return cutout


# ---------------------------------------------------------------------
# ePSF I/O
# ---------------------------------------------------------------------


def write_epsf_fits(
    path: Path,
    epsf: np.ndarray,
    *,
    psf_size: int,
    oversample: int,
    orbit: int,
    sector: int,
    camera: int,
    ccd: int,
    cutout_x: int,
    cutout_y: int,
    n_background: int = len(EPSF_BACKGROUND_COLUMNS),
) -> None:
    """Write a fitted ePSF parameter array to a single-HDU FITS file.

    The primary HDU stores the ``(t, k)`` float64 array directly. The
    background columns at ``epsf[:, -n_background:]`` follow the fixed order
    in :data:`EPSF_BACKGROUND_COLUMNS`, also recorded in ``BGCOL*`` header
    keywords for self-description.

    Parameters
    ----------
    path : pathlib.Path
        Output FITS file path. The file is written atomically via
        :func:`_atomic_write`.
    epsf : np.ndarray
        2D ``(t, k)`` array of best-fit ePSF + background parameters, as
        returned by :func:`tglc.epsf.fit_epsf`. Coerced to ``float64`` on
        write.
    psf_size : int
        Side length in pixels of the square PSF model.
    oversample : int
        Factor by which the PSF was oversampled relative to image pixels.
    orbit, sector, camera, ccd : int
        TESS identifiers for the cutout this ePSF was fit for. Used to
        match the file back to its originating :class:`FFICutout`.
    cutout_x, cutout_y : int
        Cutout grid indices matching :attr:`FFICutout.cutout_x` /
        :attr:`FFICutout.cutout_y` on the originating cutout.
    n_background : int, optional
        Number of background-parameter columns appended after the ePSF
        parameters. Defaults to ``len(EPSF_BACKGROUND_COLUMNS)`` (i.e. 6).
    """
    path = Path(path)

    header = fits.Header()
    header["PSF_SIZE"] = int(psf_size)
    header["OVRSAMPL"] = int(oversample)
    header["N_BG"] = int(n_background)
    header["ORBIT"] = int(orbit)
    header["SECTOR"] = int(sector)
    header["CAMERA"] = int(camera)
    header["CCD"] = int(ccd)
    header["CUTOUTX"] = int(cutout_x)
    header["CUTOUTY"] = int(cutout_y)
    for i, name in enumerate(EPSF_BACKGROUND_COLUMNS[:n_background]):
        header[f"BGCOL{i}"] = name

    hdu = fits.PrimaryHDU(data=np.asarray(epsf, dtype=np.float64), header=header)
    _atomic_write(fits.HDUList([hdu]), path)


def read_epsf_fits(path: Path) -> tuple[np.ndarray, dict]:
    """Read an ePSF parameter array and its metadata from a FITS file.

    Parameters
    ----------
    path : pathlib.Path
        Path to a FITS file produced by :func:`write_epsf_fits`.

    Returns
    -------
    epsf : np.ndarray
        2D ``(t, k)`` float64 array of best-fit ePSF and background
        parameters.
    metadata : dict
        Header values keyed by ``psf_size``, ``oversample``,
        ``n_background``, ``orbit``, ``sector``, ``camera``, ``ccd``,
        ``cutout_x``, ``cutout_y``, and ``background_columns`` (a tuple of
        background-column names of length ``n_background``).
    """
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        warnings.simplefilter("ignore", VerifyWarning)
        with fits.open(path) as hdul:
            header = hdul[0].header
            data = np.array(hdul[0].data, dtype=np.float64)
    n_background = int(header["N_BG"])
    metadata = {
        "psf_size": int(header["PSF_SIZE"]),
        "oversample": int(header["OVRSAMPL"]),
        "n_background": n_background,
        "orbit": int(header["ORBIT"]),
        "sector": int(header["SECTOR"]),
        "camera": int(header["CAMERA"]),
        "ccd": int(header["CCD"]),
        "cutout_x": int(header["CUTOUTX"]),
        "cutout_y": int(header["CUTOUTY"]),
        "background_columns": tuple(str(header.get(f"BGCOL{i}", "")) for i in range(n_background)),
    }
    return data, metadata


# ---------------------------------------------------------------------
# Legacy-format migration
# ---------------------------------------------------------------------


def _default_fits_path(src: Path) -> Path:
    return src.with_suffix(".fits")


def migrate_cutout_pickle(
    pkl_path: Path,
    fits_path: Path | None = None,
    *,
    delete_original: bool = False,
) -> Path:
    """Convert a legacy cutout pickle into a FITS file.

    Reads the pickled cutout, writes the FITS file (atomically), verifies it
    is readable, and optionally removes the original pickle. The original is
    only deleted after a successful round-trip read of the new FITS file.

    Parameters
    ----------
    pkl_path : pathlib.Path
        Path to the legacy ``.pkl`` file produced by older versions of TGLC.
        Pickles referencing ``tglc.ffi.Source`` still load thanks to the
        backwards-compat alias kept at the end of :mod:`tglc.ffi`.
    fits_path : pathlib.Path, optional
        Output FITS file path. Defaults to ``pkl_path`` with the ``.fits``
        suffix.
    delete_original : bool, optional
        If ``True``, remove ``pkl_path`` after the new FITS file has been
        verified readable. Defaults to ``False`` so the legacy file is
        retained unless the caller explicitly opts in.

    Returns
    -------
    fits_path : pathlib.Path
        Path to the written FITS file.
    """
    pkl_path = Path(pkl_path)
    fits_path = Path(fits_path) if fits_path is not None else _default_fits_path(pkl_path)

    with pkl_path.open("rb") as pickle_file:
        cutout = pickle.load(pickle_file)

    write_cutout_fits(cutout, fits_path)
    read_cutout_fits(fits_path)

    if delete_original:
        pkl_path.unlink()

    return fits_path


def migrate_epsf_npy(
    npy_path: Path,
    fits_path: Path | None = None,
    *,
    psf_size: int,
    oversample: int,
    orbit: int,
    sector: int,
    camera: int,
    ccd: int,
    cutout_x: int,
    cutout_y: int,
    delete_original: bool = False,
) -> Path:
    """Convert a legacy ePSF ``.npy`` file into a FITS file.

    Legacy ``.npy`` files carry no metadata, so all FITS header keywords
    must be supplied by the caller from the surrounding pipeline state
    (e.g., orbit/camera/ccd from the directory layout, ``psf_size`` and
    ``oversample`` from the originating ePSF fit configuration).

    Parameters
    ----------
    npy_path : pathlib.Path
        Path to the legacy ``.npy`` file produced by older versions of TGLC.
    fits_path : pathlib.Path, optional
        Output FITS file path. Defaults to ``npy_path`` with the ``.fits``
        suffix.
    psf_size : int
        Side length in pixels of the square PSF model the ePSF was fit
        against. Used to populate the FITS header.
    oversample : int
        Factor by which the PSF was oversampled relative to image pixels.
    orbit, sector, camera, ccd : int
        TESS identifiers for the originating cutout.
    cutout_x, cutout_y : int
        Cutout grid indices matching the originating
        :class:`FFICutout`'s persisted ``cutout_x`` / ``cutout_y``.
    delete_original : bool, optional
        If ``True``, remove ``npy_path`` after the new FITS file has been
        verified readable. Defaults to ``False``.

    Returns
    -------
    fits_path : pathlib.Path
        Path to the written FITS file.
    """
    npy_path = Path(npy_path)
    fits_path = Path(fits_path) if fits_path is not None else _default_fits_path(npy_path)

    epsf = np.load(npy_path)
    write_epsf_fits(
        fits_path,
        epsf,
        psf_size=psf_size,
        oversample=oversample,
        orbit=orbit,
        sector=sector,
        camera=camera,
        ccd=ccd,
        cutout_x=cutout_x,
        cutout_y=cutout_y,
    )
    read_epsf_fits(fits_path)

    if delete_original:
        npy_path.unlink()

    return fits_path
