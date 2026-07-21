"""
TGLC organizes files in a particular way, and the `Manifest` class helps find those files.
"""

from dataclasses import dataclass, fields
from pathlib import Path

from tglc.utils.constants import get_orbits_in_sector, get_sector_containing_orbit


def get_tic_id_shard_path(tic_id: int) -> Path:
    """
    Relative shard directory for a light curve file, derived from the TIC ID.

    The TIC ID is zero-padded to 16 digits and the first 13 digits are split into
    1+3+3+3+3-digit directory components, e.g. ``12345678901`` ->
    ``Path("0/000/012/345/678")``. The last 3 digits vary within a leaf directory, keeping
    every directory level at <=1000 entries.

    Parameters
    ----------
    tic_id : int
        TESS Input Catalog ID. Must be a non-negative integer below 10^16.

    Returns
    -------
    shard_path : Path
        Relative directory path with 5 components.
    """
    if not 0 <= tic_id < 10**16:
        raise ValueError(f"TIC ID must be a non-negative integer below 10^16, got {tic_id}")
    padded = f"{tic_id:016d}"
    return Path(padded[0], padded[1:4], padded[4:7], padded[7:10], padded[10:13])


@dataclass
class Manifest:
    """This class manages/specifies the file structure that TGLC expects."""

    tglc_data_dir: Path
    """Base directory for TGLC data products."""

    orbit: int | None = None
    """TESS orbit"""

    camera: int | None = None
    """TESS camera"""

    ccd: int | None = None
    """TESS CCD"""

    cadence: int | None = None
    """TESS cadence number"""

    tic_id: int | None = None
    """TESS Input Catalog ID"""

    cutout_x: int | None = None
    """X coordinate of cutout (among all cutouts) in FFIs"""

    cutout_y: int | None = None
    """Y coordinate of cutout (among all cutouts) in FFIs"""

    def __getattribute__(self, name):
        """Raise a custom RuntimeError if an attribute that has not been specified is accessed."""
        value = super().__getattribute__(name)
        if value is None:
            raise RuntimeError(
                f"{type(self).__name__} does not have the required attribute {name} set to a non-null value!"
            )
        return value

    def __repr__(self) -> str:
        """Create a repr that excludes null attributes."""
        non_null_fields = []
        for field in fields(self):
            try:
                non_null_fields.append((field.name, getattr(self, field.name)))
            except RuntimeError:
                pass
        fields_string = ", ".join(f"{name}={repr(value)}" for name, value in non_null_fields)
        return f"{type(self).__name__}({fields_string})"

    @property
    def orbit_directory(self) -> Path:
        """Directory containing data products related to the TESS orbit."""
        return self.tglc_data_dir / f"orbit-{self.orbit}" / "ffi"

    @property
    def catalog_directory(self) -> Path:
        """Directory containing catalogs for the orbit."""
        return self.orbit_directory / "catalogs"

    @property
    def gaia_catalog_file(self) -> Path:
        """Gaia catalog eCSV file."""
        return self.catalog_directory / f"Gaia_cam{self.camera}_ccd{self.ccd}.ecsv"

    @property
    def tic_catalog_file(self) -> Path:
        """TIC catalog eCSV file."""
        return self.catalog_directory / f"TIC_cam{self.camera}_ccd{self.ccd}.ecsv"

    @property
    def camera_directory(self) -> Path:
        """Directory containing data products related to the TESS camera."""
        return self.orbit_directory / f"cam{self.camera}"

    @property
    def ccd_directory(self) -> Path:
        """Directory containing data products related to the TESS CCD."""
        return self.camera_directory / f"ccd{self.ccd}"

    @property
    def ffi_directory(self) -> Path:
        """Directory containing full frame images."""
        return self.ccd_directory / "ffi"

    @property
    def tica_ffi_file_pattern(self) -> Path:
        """
        Glob pattern for TICA FFIs from any cadence, but known orbit, camera, and CCD.

        Meant to be used like `manifest.ffi_directory.glob(manifest.tica_ffi_file_pattern)`.
        """
        sector = get_sector_containing_orbit(self.orbit)
        # Assumes orbits in a sector are consecutive
        orbit_within_sector = self.orbit - min(get_orbits_in_sector(sector)) + 1
        return f"hlsp_tica_tess_ffi_s{sector:04d}-o{orbit_within_sector}-*-cam{self.camera}-ccd{self.ccd}_tess_v01_img.fits"

    @property
    def tica_ffi_file(self) -> Path:
        """TICA calibrated full frame image file."""
        sector = get_sector_containing_orbit(self.orbit)
        # Assumes orbits in a sector are consecutive
        orbit_within_sector = self.orbit - min(get_orbits_in_sector(sector)) + 1
        return (
            self.ffi_directory
            / f"hlsp_tica_tess_ffi_s{sector:04d}-o{orbit_within_sector}-{self.cadence:08d}-cam{self.camera}-ccd{self.ccd}_tess_v01_img.fits"
        )

    @property
    def source_directory(self) -> Path:
        """Directory containing source cutout files."""
        return self.ccd_directory / "source"

    @property
    def source_file(self) -> Path:
        """FFI cutout FITS file."""
        return self.source_directory / f"source_{self.cutout_x}_{self.cutout_y}.fits"

    @property
    def source_file_legacy_pkl(self) -> Path:
        """Legacy pickle path used before the move to FITS (issue #1).

        Retained so migration helpers can locate old files.
        """
        return self.source_directory / f"source_{self.cutout_x}_{self.cutout_y}.pkl"

    @property
    def epsf_directory(self) -> Path:
        """Directory containing ePSF files."""
        return self.ccd_directory / "epsf"

    @property
    def epsf_file(self) -> Path:
        """ePSF FITS file."""
        return self.epsf_directory / f"epsf_{self.cutout_x}_{self.cutout_y}.fits"

    @property
    def epsf_file_legacy_npy(self) -> Path:
        """Legacy NumPy ePSF path used before the move to FITS (issue #1)."""
        return self.epsf_directory / f"epsf_{self.cutout_x}_{self.cutout_y}.npy"

    @property
    def light_curve_directory(self) -> Path:
        """Directory containing light curves."""
        return self.ccd_directory / "LC"

    @property
    def light_curve_file(self) -> Path:
        """Light curve in HDF5 format, sharded by zero-padded TIC ID."""
        return self.light_curve_directory / get_tic_id_shard_path(self.tic_id) / f"{self.tic_id}.h5"
