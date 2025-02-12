"""
This module indicates whether the various optional dependencies of TGLC are installed.

This idea comes from `astropy.util.compat.optional_deps`.
"""

from importlib.util import find_spec


_optional_dependencies = ["pyticdb"]
_deps = {name.upper(): name for name in _optional_dependencies}

__all__ = [f"HAS_{pkg}" for pkg in _deps]


def __getattr__(name):
    if name in __all__:
        return find_spec(_deps[name.removeprefix("HAS_")]) is not None

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
