"""Aperture light curve class."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
from astropy.timeseries import TimeSeries
import h5py
import numpy as np

from tglc.apertures import APERTURE_NAMES, APERTURE_SIZES
from tglc.utils.constants import TESSJD


@dataclass
class ApertureLightCurveMetadata:
    """Metadata for an aperture light curve."""

    tic_id: int
    """TIC ID of light curve target star."""

    orbit: int
    """Orbit containing light curve data."""

    sector: int
    """Sector containing light curve data."""

    camera: int
    """Camera containing target star."""

    ccd: int
    """CCD containing target star."""

    ccd_x: float
    """X coordinate on CCD of target star (projected from Gaia coordinates)."""

    ccd_y: float
    """Y coordinate on CCD of target star (projected from Gaia coordinates)."""

    sky_coord: SkyCoord
    """Sky coordinates of target star from the Gaia."""

    tess_magnitude: float
    """Brightness of target star in TESS magnitude."""

    exposure_time: u.Quantity["time"]  # noqa: F821
    """"Exposure time of light curve data points."""

    primary_aperture_local_background: u.Quantity[u.electron] | None = None
    """Local background level in primary aperture, subtracted to bring flux median to expect level.
    """

    small_aperture_local_background: u.Quantity[u.electron] | None = None
    """Local background level in small aperture, subtracted to bring flux median to expect level."""

    large_aperture_local_background: u.Quantity[u.electron] | None = None
    """Local background level in large aperture, subtracted to bring flux median to expect level."""


class ApertureLightCurve(TimeSeries):
    """Aperture light curve containing photometry data for one or more apertures."""

    _base_required_columns = ["cadence", "quality_flag", "background_flux"]
    _aperture_column_suffixes = ["magnitude", "centroid_x", "centroid_y"]
    _required_metadata = [
        field.name
        for field in fields(ApertureLightCurveMetadata)
        if not field.name.endswith("_aperture_local_background")
    ]

    def __init__(
        self,
        *args,
        meta: ApertureLightCurveMetadata | Any = None,
        apertures: Sequence[str] | None = None,
        **kwargs,
    ):
        if isinstance(meta, ApertureLightCurveMetadata):
            meta = asdict(meta)
        super().__init__(*args, meta=meta, **kwargs)

        if apertures is None:
            # Copy/slice operations propagate `meta`, including any previous aperture selection
            apertures = self.meta.get("apertures")
        if apertures is None:
            # Fall back to whichever apertures have columns present
            apertures = [
                name for name in APERTURE_NAMES if f"{name}_aperture_magnitude" in self.colnames
            ]
        invalid_apertures = [name for name in apertures if name not in APERTURE_SIZES]
        if invalid_apertures:
            raise ValueError(
                f"Unrecognized apertures for light curve: {', '.join(invalid_apertures)}"
            )
        if not apertures:
            raise ValueError("Aperture light curve requires at least one aperture")
        self.meta["apertures"] = [name for name in APERTURE_NAMES if name in set(apertures)]

        required_columns = self._base_required_columns + [
            f"{aperture_name}_aperture_{data_name}"
            for aperture_name in self.meta["apertures"]
            for data_name in self._aperture_column_suffixes
        ]
        missing_columns = [name for name in required_columns if name not in self.colnames]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for aperture light curve: {', '.join(missing_columns)}"
            )

        missing_metadata = [key for key in self._required_metadata if key not in self.meta] + [
            f"{aperture_name}_aperture_local_background"
            for aperture_name in self.meta["apertures"]
            if self.meta.get(f"{aperture_name}_aperture_local_background") is None
        ]
        if missing_metadata:
            raise ValueError(
                f"Missing required metadata for aperture light curve: {', '.join(missing_metadata)}"
            )

    def write_hdf5(self, output_file: Path):
        with h5py.File(output_file, "w") as file:
            file.attrs["TIC ID"] = self.meta["tic_id"]
            file.attrs["Orbit"] = self.meta["orbit"]
            file.attrs["Sector"] = self.meta["sector"]
            file.attrs["Camera"] = self.meta["camera"]
            file.attrs["CCD"] = self.meta["ccd"]
            file.attrs["RA"] = self.meta["sky_coord"].ra.deg
            file.attrs["Dec"] = self.meta["sky_coord"].dec.deg
            file.attrs["BJDoffset"] = TESSJD.epoch_val.to(u.day)
            file.attrs["TessMag"] = self.meta["tess_magnitude"]

            lc_group = file.create_group("LightCurve")
            lc_group.create_dataset("BJD", data=self.time.tjd, dtype=np.float64)
            lc_group.create_dataset("Cadence", data=self["cadence"], dtype=np.int64)
            lc_group.create_dataset(
                "X",
                data=np.full_like(self.time, self.meta["ccd_x"], dtype=np.float64),
                dtype=np.float64,
            )
            lc_group.create_dataset(
                "Y",
                data=np.full_like(self.time, self.meta["ccd_y"], dtype=np.float64),
                dtype=np.float64,
            )
            lc_group.create_dataset("QualityFlag", data=self["quality_flag"], dtype=np.int64)

            background_group = lc_group.create_group("Background")
            background_group.create_dataset("Value", data=self["background_flux"], dtype=np.float64)
            background_group.create_dataset(
                "Error",
                data=np.full_like(self["background_flux"], mad_std(self["background_flux"])),
                dtype=np.float64,
            )

            photometry_group = lc_group.create_group("AperturePhotometry")
            for aperture_name in self.meta["apertures"]:
                aperture_size = APERTURE_SIZES[aperture_name]
                aperture_group = photometry_group.create_group(
                    f"{aperture_name.capitalize()}Aperture"
                )
                aperture_group.attrs["name"] = f"TGLCAperture{aperture_name.capitalize()}"
                aperture_group.attrs["description"] = f"{aperture_size}x{aperture_size} square"
                aperture_group.attrs["localbackground"] = self.meta[
                    f"{aperture_name}_aperture_local_background"
                ]

                aperture_data = self[f"{aperture_name}_aperture_magnitude"]
                aperture_group.create_dataset("RawMagnitude", data=aperture_data, dtype=np.float64)
                aperture_group.create_dataset(
                    "RawMagnitudeError",
                    data=np.full_like(aperture_data, mad_std(aperture_data)),
                    dtype="f",
                )
                aperture_group.create_dataset(
                    "X", data=self[f"{aperture_name}_aperture_centroid_x"], dtype="f"
                )
                aperture_group.create_dataset(
                    "Y", data=self[f"{aperture_name}_aperture_centroid_y"], dtype="f"
                )
