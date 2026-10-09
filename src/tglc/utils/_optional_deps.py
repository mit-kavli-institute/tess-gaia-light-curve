"""
This module indicates whether the various optional dependencies of TGLC are installed.

This idea comes from `astropy.util.compat.optional_deps`.
"""

from importlib.util import find_spec


_optional_dependencies = ["pyticdb", "cupy"]
_deps = {name.upper(): name for name in _optional_dependencies}

__all__ = [f"HAS_{pkg}" for pkg in _deps]


def __getattr__(name):
    if name == "HAS_CUPY":
        # For cupy, we not only check that cupy is installed but also that CUDA exists and there is
        # at least one CUDA device available.
        cupy_installed = find_spec("cupy") is not None
        if not cupy_installed:
            return False
        import cupy
        from cupy_backends.cuda.api.runtime import CUDARuntimeError

        try:
            num_cuda_devices = cupy.cuda.runtime.getDeviceCount()
            return num_cuda_devices > 0
        except CUDARuntimeError:
            return False
    if name in __all__:
        return find_spec(_deps[name.removeprefix("HAS_")]) is not None

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
