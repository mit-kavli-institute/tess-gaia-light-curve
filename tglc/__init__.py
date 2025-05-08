from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("tglc")
except PackageNotFoundError:
    # package is not installed
    pass

__author__ = "Te Han, Timothy Brandt, Jack Haviland"
__credits__ = "University of California, Santa Barbara; MIT"
