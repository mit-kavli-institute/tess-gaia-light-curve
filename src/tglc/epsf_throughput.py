"""Multiplicative ePSF throughput correction for TGLC light curves.

The ePSF fit absorbs TESS's optical throughput loss: toward the camera edge, the fitted PSF's
integrated response per unit catalog flux is systematically lower. Light curves are normalized
additively (their good-cadence median is pinned to the catalog-expected flux, see
`tglc.aperture_photometry.get_normalized_aperture_photometry`), so the multiplicative factor is
invisible in the product.

The factor is a time series, so it is not stored in every light curve file. :class:`EPSFThroughput`
recomputes it on demand from the ePSF data product alone (which records the reference flux,
exposure, and cutout origin; older ePSF files also need their cutout) and divides it out of the
background-offset flux and magnitude.

Because each star's design-matrix contribution is scaled by its catalog flux ratio to the
brightest star in the cutout, the modeled target flux divided by the target's catalog flux does not
depend on the target's magnitude: the fraction is set by the cutout's ePSF, its brightest star, and
(weakly) the target's sub-pixel position.
"""

from pathlib import Path

from astropy.table import QTable
import astropy.units as u
import h5py
import numpy as np

from tglc.aperture_light_curve import ApertureLightCurve
from tglc.epsf import EPSF, get_reference_flux
from tglc.ffi import FFICutout
from tglc.io import read_cutout_fits, read_epsf_fits
from tglc.light_curve import LIGHT_CURVE_APERTURES


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


class EPSFThroughput:
    """
    Per-cadence fraction of a star's catalog-expected flux captured by a fitted ePSF.

    The fraction ``f(t)`` is the target-only ePSF model summed over the full ``psf_size`` square
    footprint (never clamped by an image edge), divided by the target's catalog-expected flux per
    cadence. Dividing the background-offset light curve by ``f(t)`` removes the throughput loss the
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
        """
        self.epsf = epsf
        self.reference_flux = u.Quantity(reference_flux, u.electron / u.second)
        self.exposure_time = u.Quantity(exposure_time, u.second)
        self.ccd_x = int(ccd_x)
        self.ccd_y = int(ccd_y)

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
            If the ePSF predates the recorded reference flux, exposure, and cutout origin; use
            :meth:`from_cutout` for those.
        """
        reference_flux, exposure, ccd_x, ccd_y = (
            epsf.reference_flux,
            epsf.exposure,
            epsf.ccd_x,
            epsf.ccd_y,
        )
        if reference_flux is None or exposure is None or ccd_x is None or ccd_y is None:
            missing = [
                name
                for name, value in [
                    ("reference_flux", reference_flux),
                    ("exposure", exposure),
                    ("ccd_x", ccd_x),
                    ("ccd_y", ccd_y),
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
            reference_flux=get_reference_flux(cutout) * u.electron / u.second,
            exposure_time=cutout.exposure * u.second,
            ccd_x=cutout.ccd_x,
            ccd_y=cutout.ccd_y,
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
    def correct_flux(flux, flux_fraction: np.ndarray):
        """Divide the ePSF flux fraction out of background-offset flux."""
        return flux / flux_fraction

    @staticmethod
    def correct_magnitude(magnitude, flux_fraction: np.ndarray):
        """Divide the ePSF flux fraction out of background-offset magnitude."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return magnitude + 2.5 * np.log10(flux_fraction)

    def _check_identifiers(self, orbit: int, camera: int, ccd: int, description: str) -> None:
        if (orbit, camera, ccd) != (self.epsf.orbit, self.epsf.camera, self.epsf.ccd):
            raise ValueError(
                f"{description} is for orbit {orbit} {camera}-{ccd}, but the ePSF is for orbit "
                f"{self.epsf.orbit} {self.epsf.camera}-{self.epsf.ccd}"
            )

    def correct_light_curve(self, light_curve: ApertureLightCurve) -> ApertureLightCurve:
        """
        Divide the ePSF flux fraction out of each aperture's flux and magnitude.

        Returns a copy of `light_curve` with an added `epsf_flux_fraction` column and corrected
        `*_aperture_flux` and `*_aperture_magnitude` columns.

        Raises
        ------
        ValueError
            If the light curve's orbit, camera, or CCD doesn't match the ePSF.
        """
        meta = light_curve.meta
        self._check_identifiers(meta["orbit"], meta["camera"], meta["ccd"], "Light curve")
        flux_fraction = self.flux_fraction_at_ccd(meta["ccd_x"], meta["ccd_y"])
        corrected = light_curve.copy()
        corrected["epsf_flux_fraction"] = flux_fraction
        for aperture_name, _ in LIGHT_CURVE_APERTURES:
            for column, correct in [
                (f"{aperture_name}_aperture_flux", self.correct_flux),
                (f"{aperture_name}_aperture_magnitude", self.correct_magnitude),
            ]:
                if column in corrected.colnames:
                    corrected[column] = correct(corrected[column], flux_fraction)
        return corrected

    def correct_hdf5(self, path: Path) -> QTable:
        """
        Read an HDF5 light curve and divide the ePSF flux fraction out of it.

        The file is not modified. Per-aperture flux columns are included only for files that
        record `ExposureTime`, `ZeroPointFlux`, and `fluxportion` (see `ApertureLightCurve.write_hdf5`).

        Returns
        -------
        light_curve : QTable
            Table with `bjd`, `cadence`, `epsf_flux_fraction`, and corrected
            `{aperture}_aperture_magnitude` (and `{aperture}_aperture_flux`, in electrons)
            columns.

        Raises
        ------
        ValueError
            If the file's orbit, camera, or CCD doesn't match the ePSF.
        """
        with h5py.File(path, "r") as file:
            self._check_identifiers(
                int(file.attrs["Orbit"]),
                int(file.attrs["Camera"]),
                int(file.attrs["CCD"]),
                f"Light curve file {path}",
            )
            light_curve_group = file["LightCurve"]
            flux_fraction = self.flux_fraction_at_ccd(
                float(light_curve_group["X"][0]), float(light_curve_group["Y"][0])
            )
            table = QTable(
                {
                    "bjd": light_curve_group["BJD"][()],
                    "cadence": light_curve_group["Cadence"][()],
                    "epsf_flux_fraction": flux_fraction,
                }
            )
            for aperture_name, _ in LIGHT_CURVE_APERTURES:
                aperture_group = light_curve_group["AperturePhotometry"][
                    f"{aperture_name.capitalize()}Aperture"
                ]
                magnitude = aperture_group["RawMagnitude"][()]
                table[f"{aperture_name}_aperture_magnitude"] = self.correct_magnitude(
                    magnitude, flux_fraction
                )
                offset_flux = read_hdf5_offset_flux(file, aperture_name)
                if offset_flux is not None:
                    table[f"{aperture_name}_aperture_flux"] = self.correct_flux(
                        offset_flux, flux_fraction
                    )
        return table


def read_hdf5_offset_flux(file: h5py.File, aperture_name: str) -> u.Quantity | None:
    """
    Recover an aperture's background-offset flux from an HDF5 light curve's magnitudes.

    Adding the aperture's `localbackground` attribute to the result gives the un-normalized
    aperture flux; see `ApertureLightCurve.write_hdf5`.

    Parameters
    ----------
    file : h5py.File
        Open HDF5 light curve file.
    aperture_name : str
        Aperture name from `tglc.light_curve.LIGHT_CURVE_APERTURES`, e.g. ``"primary"``.

    Returns
    -------
    offset_flux : u.Quantity | None
        Per-cadence flux in electrons, or `None` if the file predates the `ExposureTime`,
        `ZeroPointFlux`, and `fluxportion` attributes.
    """
    aperture_group = file["LightCurve"]["AperturePhotometry"][
        f"{aperture_name.capitalize()}Aperture"
    ]
    if not (
        {"ExposureTime", "ZeroPointFlux"} <= file.attrs.keys()
        and "fluxportion" in aperture_group.attrs
    ):
        return None
    magnitude = aperture_group["RawMagnitude"][()]
    return (
        aperture_group.attrs["fluxportion"]
        * file.attrs["ExposureTime"]
        * file.attrs["ZeroPointFlux"]
        * 10 ** ((magnitude - 10) / -2.5)
        * u.electron
    )
