"""Multiplicative ePSF throughput correction for TGLC light curves.

The ePSF fit absorbs TESS's optical throughput loss: toward the camera edge, the fitted PSF's
integrated response per unit catalog flux is systematically lower. Light curves are normalized
additively (their good-cadence median is pinned to the catalog-expected flux, see
`tglc.aperture_photometry.get_normalized_aperture_photometry`), so the median throughput deficit
is folded into the recorded `local_background` and the multiplicative factor is invisible in the
product.

The factor is a time series, so it is not stored in every light curve file. :class:`EPSFThroughput`
recomputes it on demand from the ePSF data product alone (which records the reference flux,
exposure, and cutout geometry; older ePSF files also need their cutout). The correction first
undoes the additive normalization, then divides the factor out of the un-normalized aperture flux:
dividing it out of the normalized flux instead would count the median deficit twice.

Because each star's design-matrix contribution is scaled by its catalog flux ratio to the
brightest star in the cutout, the modeled target flux divided by the target's catalog flux does not
depend on the target's magnitude: the fraction is set by the cutout's ePSF, its brightest star, and
(weakly) the target's sub-pixel position.
"""

import logging
from pathlib import Path

from astropy.table import QTable
from astropy.time import Time
import astropy.units as u
import h5py
import numpy as np

from tglc.aperture_light_curve import LIGHT_CURVE_APERTURES, ApertureLightCurve
from tglc.epsf import EPSF, get_reference_flux
from tglc.ffi import FFICutout
from tglc.io import read_cutout_fits, read_epsf_fits
from tglc.utils.constants import (
    TESSJD,  # noqa: F401 imported to register the tjd time format
    convert_tess_flux_to_tess_magnitude,
)


logger = logging.getLogger(__name__)


def _shift_into_footprint_frame(coordinate: float, half_size: int) -> float:
    """
    Shift a coordinate by an even integer so its nearest pixel is `half_size + 1` or `+ 2`.

    The shifted star's full PSF footprint then fits in an image of side `2 * half_size + 4`. The
    shift is even so that the rounding of exact half-pixel positions (banker's rounding in
    `tglc.epsf.make_tglc_design_matrix`) picks the same footprint as at the original position.
    """
    shift = round(coordinate) - (half_size + 1)
    shift -= shift % 2
    return coordinate - shift


def _undo_normalization(normalized_flux: np.ndarray, local_background: float) -> np.ndarray:
    """Add back the local background; a NaN background means none was subtracted."""
    if np.isnan(local_background):
        return normalized_flux
    return normalized_flux + local_background


class EPSFThroughput:
    """
    Per-cadence fraction of a star's catalog-expected flux captured by a fitted ePSF.

    The fraction ``f(t)`` is the target-only ePSF model summed over the full ``psf_size`` square
    footprint (never clamped by an image edge), divided by the target's catalog-expected flux per
    cadence. Dividing the un-normalized aperture flux by ``f(t)`` removes the throughput loss the
    fit absorbed, putting the flux in the catalog frame.

    This differs from the ``flux_fraction_median`` diagnostic of
    `tglc.scripts.edge_compression_sweep`, which sums only over the 5x5 light curve cutout window.
    """

    def __init__(
        self,
        epsf: EPSF,
        *,
        reference_flux: u.Quantity,
        exposure_time: u.Quantity,
        ccd_x: int,
        ccd_y: int,
        cutout_size: int,
    ):
        """
        Parameters
        ----------
        epsf : EPSF
            Fitted ePSF for one cutout.
        reference_flux : u.Quantity (e-/s)
            Catalog flux of the star with flux ratio 1 (the brightest catalog star) in the cutout
            the ePSF was fit for; see `tglc.epsf.get_reference_flux`.
        exposure_time : u.Quantity (time)
            Exposure time per cadence of the cutout.
        ccd_x, ccd_y : int
            CCD coordinates of the cutout origin, used to convert CCD positions to the cutout
            frame.
        cutout_size : int
            Side length in pixels of the cutout, used to reject targets outside it.
        """
        self.epsf = epsf
        self.reference_flux = u.Quantity(reference_flux, u.electron / u.second)
        self.exposure_time = u.Quantity(exposure_time, u.second)
        self.ccd_x = int(ccd_x)
        self.ccd_y = int(ccd_y)
        self.cutout_size = int(cutout_size)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.epsf!r} reference_flux={self.reference_flux} "
            f"exposure_time={self.exposure_time}>"
        )

    @classmethod
    def from_epsf(cls, epsf: EPSF) -> "EPSFThroughput":
        """
        Make the throughput wrapper from an ePSF alone, using the scalars recorded with it.

        Raises
        ------
        ValueError
            If the ePSF predates the recorded reference flux, exposure, and cutout geometry; use
            :meth:`from_cutout` for those.
        """
        reference_flux, exposure, ccd_x, ccd_y, cutout_size = (
            epsf.reference_flux,
            epsf.exposure,
            epsf.ccd_x,
            epsf.ccd_y,
            epsf.cutout_size,
        )
        if (
            reference_flux is None
            or exposure is None
            or ccd_x is None
            or ccd_y is None
            or cutout_size is None
        ):
            missing = [
                name
                for name, value in [
                    ("reference_flux", reference_flux),
                    ("exposure", exposure),
                    ("ccd_x", ccd_x),
                    ("ccd_y", ccd_y),
                    ("cutout_size", cutout_size),
                ]
                if value is None
            ]
            raise ValueError(
                f"ePSF {epsf!r} does not record {', '.join(missing)}; "
                "pass the cutout it was fit for (EPSFThroughput.from_cutout)"
            )
        return cls(
            epsf,
            reference_flux=reference_flux * u.electron / u.second,
            exposure_time=exposure * u.second,
            ccd_x=ccd_x,
            ccd_y=ccd_y,
            cutout_size=cutout_size,
        )

    @classmethod
    def from_cutout(cls, epsf: EPSF, cutout: FFICutout) -> "EPSFThroughput":
        """
        Make the throughput wrapper for an ePSF and the cutout it was fit for.

        Needed only for ePSFs that predate the recorded scalars (see :meth:`from_epsf`).

        Raises
        ------
        ValueError
            If the ePSF was not fit for `cutout`.
        """
        if not epsf.matches_cutout(cutout):
            raise ValueError(f"ePSF {epsf!r} does not match cutout {cutout!r}")
        return cls(
            epsf,
            reference_flux=get_reference_flux(cutout.gaia) * u.electron / u.second,
            exposure_time=cutout.exposure * u.second,
            ccd_x=cutout.ccd_x,
            ccd_y=cutout.ccd_y,
            cutout_size=cutout.size,
        )

    @classmethod
    def from_fits(cls, epsf_path: Path, cutout_path: Path | None = None) -> "EPSFThroughput":
        """
        Make the throughput wrapper from an ePSF FITS file.

        The cutout FITS file is needed only for ePSF files that predate the recorded scalars.
        """
        epsf = read_epsf_fits(epsf_path)
        if cutout_path is None:
            return cls.from_epsf(epsf)
        return cls.from_cutout(epsf, read_cutout_fits(cutout_path))

    @property
    def reference_flux_per_cadence(self) -> float:
        """Catalog flux in electrons per cadence of a star with flux ratio 1."""
        return (self.reference_flux * self.exposure_time).to_value(u.electron)

    def flux_fraction(self, x: float, y: float) -> np.ndarray:
        """
        Fraction of a star's catalog-expected flux captured by the ePSF model in each cadence.

        Parameters
        ----------
        x, y : float
            Position of the star in the cutout frame (as in `FFICutout.star_positions`).

        Returns
        -------
        flux_fraction : array
            1D array with one entry per cadence. NaN for cadences whose ePSF fit failed.
        """
        half_size = (self.epsf.psf_size - 1) // 2
        footprint_side = 2 * half_size + 4
        design_matrix, _ = self.epsf.make_design_matrix(
            (footprint_side, footprint_side),
            np.array(
                [
                    [
                        _shift_into_footprint_frame(x, half_size),
                        _shift_into_footprint_frame(y, half_size),
                    ]
                ]
            ),
            np.array([1.0]),
        )
        # Summing the design matrix over pixels first gives each PSF node's total weight, so the
        # per-cadence model total is a single matrix-vector product.
        return (
            self.epsf.psf_parameters @ design_matrix.sum(axis=0) / self.reference_flux_per_cadence
        )

    def flux_fraction_at_ccd(self, ccd_x: float, ccd_y: float) -> np.ndarray:
        """:meth:`flux_fraction` for a star at a CCD position (as in light curve metadata)."""
        return self.flux_fraction(ccd_x - self.ccd_x, ccd_y - self.ccd_y)

    @staticmethod
    def correct_flux(raw_flux, flux_fraction: np.ndarray):
        """
        Divide the ePSF flux fraction out of un-normalized aperture flux.

        Non-positive flux fractions, which only a pathological fit produces, are treated like
        failed cadences: the corrected flux is NaN and a warning is logged.
        """
        flux_fraction = np.asarray(flux_fraction, dtype=np.float64)
        non_positive = flux_fraction <= 0
        if non_positive.any():
            logger.warning(
                f"{non_positive.sum()} cadences have a non-positive ePSF flux fraction; their "
                "corrected flux is NaN"
            )
            flux_fraction = np.where(non_positive, np.nan, flux_fraction)
        return raw_flux / flux_fraction

    def _correct_aperture(
        self,
        raw_flux: u.Quantity,
        flux_fraction: np.ndarray,
        flux_portion: float,
    ) -> tuple[u.Quantity, np.ndarray]:
        """Corrected flux (electrons) and the TESS magnitude it implies for the whole star."""
        flux = self.correct_flux(raw_flux, flux_fraction)
        # Non-positive flux has no magnitude; NaN matches the normalized product's convention.
        positive_flux = np.where(flux > 0, flux, np.nan * u.electron)
        magnitude = convert_tess_flux_to_tess_magnitude(
            positive_flux / flux_portion / self.exposure_time
        )
        return flux, np.asarray(magnitude)

    def _check_target(
        self,
        description: str,
        *,
        orbit: int,
        camera: int,
        ccd: int,
        cutout_x: int,
        cutout_y: int,
        ccd_x: float,
        ccd_y: float,
        n_cadences: int,
    ) -> None:
        """Raise `ValueError` unless a light curve was extracted from this ePSF's cutout."""
        epsf = self.epsf
        if (orbit, camera, ccd) != (epsf.orbit, epsf.camera, epsf.ccd):
            raise ValueError(
                f"{description} is for orbit {orbit} {camera}-{ccd}, but the ePSF is for orbit "
                f"{epsf.orbit} {epsf.camera}-{epsf.ccd}"
            )
        # -1 is the ePSF's legacy "not set" cutout index sentinel.
        for own_index, index in [(epsf.cutout_x, cutout_x), (epsf.cutout_y, cutout_y)]:
            if own_index != -1 and own_index != index:
                raise ValueError(
                    f"{description} is for cutout ({cutout_x}, {cutout_y}), but the ePSF is for "
                    f"cutout ({epsf.cutout_x}, {epsf.cutout_y})"
                )
        if n_cadences != epsf.n_cadences:
            raise ValueError(
                f"{description} has {n_cadences} cadences, but the ePSF has {epsf.n_cadences}"
            )
        x, y = ccd_x - self.ccd_x, ccd_y - self.ccd_y
        if not (0 <= x <= self.cutout_size and 0 <= y <= self.cutout_size):
            raise ValueError(
                f"{description} target at CCD ({ccd_x}, {ccd_y}) lies outside the ePSF's "
                f"{self.cutout_size} px cutout at CCD ({self.ccd_x}, {self.ccd_y})"
            )

    def correct_light_curve(self, light_curve: ApertureLightCurve) -> ApertureLightCurve:
        """
        Divide the ePSF flux fraction out of each aperture's un-normalized flux.

        Returns a copy of `light_curve` with an added `epsf_flux_fraction` column. Each
        `*_aperture_flux` column is replaced by the un-normalized aperture flux (the normalized
        flux plus the local background) divided by the flux fraction, and each
        `*_aperture_magnitude` column by the TESS magnitude that flux implies for the whole star.
        Cadences clipped as non-positive during normalization stay NaN.

        Raises
        ------
        ValueError
            If the light curve was not extracted from this ePSF's cutout, or its cadence count
            differs from the ePSF's.
        """
        meta = light_curve.meta
        self._check_target(
            "Light curve",
            orbit=meta["orbit"],
            camera=meta["camera"],
            ccd=meta["ccd"],
            cutout_x=meta["cutout_x"],
            cutout_y=meta["cutout_y"],
            ccd_x=meta["ccd_x"],
            ccd_y=meta["ccd_y"],
            n_cadences=len(light_curve),
        )
        flux_fraction = self.flux_fraction_at_ccd(meta["ccd_x"], meta["ccd_y"])
        corrected = light_curve.copy()
        corrected["epsf_flux_fraction"] = flux_fraction
        for aperture_name, _ in LIGHT_CURVE_APERTURES:
            raw_flux = _undo_normalization(
                u.Quantity(corrected[f"{aperture_name}_aperture_flux"], u.electron),
                u.Quantity(meta[f"{aperture_name}_aperture_local_background"], u.electron),
            )
            flux, magnitude = self._correct_aperture(
                raw_flux, flux_fraction, meta[f"{aperture_name}_aperture_flux_portion"]
            )
            corrected[f"{aperture_name}_aperture_flux"] = flux
            corrected[f"{aperture_name}_aperture_magnitude"] = magnitude
        return corrected

    def correct_hdf5(self, path: Path) -> QTable:
        """
        Read an HDF5 light curve and divide the ePSF flux fraction out of it.

        Same correction as :meth:`correct_light_curve`, using the scalars recorded by
        `ApertureLightCurve.write_hdf5`. The file is not modified.

        Returns
        -------
        light_curve : QTable
            Table with `time` (TDB, TJD format), `cadence`, `quality_flag`, `background_flux`,
            `epsf_flux_fraction`, and per-aperture corrected `{aperture}_aperture_flux` (electrons)
            and `{aperture}_aperture_magnitude` columns.

        Raises
        ------
        ValueError
            If the file predates the recorded scalars, was not extracted from this ePSF's
            cutout, or its cadence count differs from the ePSF's.
        """
        with h5py.File(path, "r") as file:
            missing = sorted(
                {"Orbit", "Camera", "CCD", "CutoutX", "CutoutY"} - set(file.attrs.keys())
            )
            if missing:
                raise ValueError(
                    f"Light curve file {path} does not record {', '.join(missing)}; it predates "
                    "the ePSF throughput correction"
                )
            light_curve_group = file["LightCurve"]
            bjd = light_curve_group["BJD"][()]
            x = light_curve_group["X"][()]
            y = light_curve_group["Y"][()]
            self._check_target(
                f"Light curve file {path}",
                orbit=int(file.attrs["Orbit"]),
                camera=int(file.attrs["Camera"]),
                ccd=int(file.attrs["CCD"]),
                cutout_x=int(file.attrs["CutoutX"]),
                cutout_y=int(file.attrs["CutoutY"]),
                # The position is constant; an empty file fails the cadence count check first.
                ccd_x=float(x[0]) if len(x) else np.nan,
                ccd_y=float(y[0]) if len(y) else np.nan,
                n_cadences=len(bjd),
            )
            flux_fraction = self.flux_fraction_at_ccd(float(x[0]), float(y[0]))
            table = QTable(
                {
                    "time": Time(bjd, format="tjd", scale="tdb"),
                    "cadence": light_curve_group["Cadence"][()],
                    "quality_flag": light_curve_group["QualityFlag"][()],
                    "background_flux": light_curve_group["Background/Value"][()],
                    "epsf_flux_fraction": flux_fraction,
                }
            )
            for aperture_name, _ in LIGHT_CURVE_APERTURES:
                aperture_group = light_curve_group["AperturePhotometry"][
                    f"{aperture_name.capitalize()}Aperture"
                ]
                flux, magnitude = self._correct_aperture(
                    read_hdf5_raw_flux(file, aperture_name),
                    flux_fraction,
                    float(aperture_group.attrs["fluxportion"]),
                )
                table[f"{aperture_name}_aperture_flux"] = flux
                table[f"{aperture_name}_aperture_magnitude"] = magnitude
        return table


def read_hdf5_raw_flux(file: h5py.File, aperture_name: str) -> u.Quantity:
    """
    Recover an aperture's un-normalized flux from an HDF5 light curve's magnitudes.

    Inverts the additive normalization with the scalars recorded by
    `ApertureLightCurve.write_hdf5`. Cadences clipped as non-positive during normalization are
    stored as NaN and stay NaN.

    Parameters
    ----------
    file : h5py.File
        Open HDF5 light curve file.
    aperture_name : str
        Aperture name from `tglc.aperture_light_curve.LIGHT_CURVE_APERTURES`, e.g. ``"primary"``.

    Returns
    -------
    raw_flux : u.Quantity
        Per-cadence aperture flux in electrons.

    Raises
    ------
    ValueError
        If the file predates the `ExposureTime`, `ZeroPointFlux`, and `fluxportion` attributes.
    """
    aperture_group = file["LightCurve"]["AperturePhotometry"][
        f"{aperture_name.capitalize()}Aperture"
    ]
    missing = sorted({"ExposureTime", "ZeroPointFlux"} - set(file.attrs.keys())) + sorted(
        {"fluxportion", "localbackground"} - set(aperture_group.attrs.keys())
    )
    if missing:
        raise ValueError(
            f"Light curve file {file.filename} does not record {', '.join(missing)}; it predates "
            "the absolute photometry scalars"
        )
    normalized_flux = (
        aperture_group.attrs["fluxportion"]
        * file.attrs["ExposureTime"]
        * file.attrs["ZeroPointFlux"]
        * 10 ** ((aperture_group["RawMagnitude"][()] - 10) / -2.5)
    )
    return (
        _undo_normalization(normalized_flux, float(aperture_group.attrs["localbackground"]))
        * u.electron
    )
