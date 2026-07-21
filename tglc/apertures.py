"""Canonical definitions of the photometric apertures used for TGLC light curves.

Kept free of heavy dependencies so the CLI module can import it without pulling in the
scientific stack.
"""

APERTURE_SIZES: dict[str, int] = {"primary": 3, "small": 1, "large": 5}
"""Mapping from aperture name to side length (pixels) of the square aperture."""

APERTURE_NAMES: tuple[str, ...] = tuple(APERTURE_SIZES)
"""Canonical aperture order, matching the historical HDF5 output order."""
