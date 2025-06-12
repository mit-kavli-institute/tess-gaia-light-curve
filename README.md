![TESS-Gaia Light Curve Logo](/logo/TGLC_Title.png)
[![TGLC DOI Badge](https://zenodo.org/badge/420868490.svg)](https://zenodo.org/badge/latestdoi/420868490)
[![TGLC Citation Badge](https://img.shields.io/badge/Cite-TGLC-blue)](https://www.tomwagg.com/software-citation-station/?auto-select=tglc)

## Introduction

This is the version of TESS-Gaia Light Curve adapted for the TESS Quick-Look Pipeline at MIT. It uses TGLC's methods for ePSF fitting and aperture photometry with two additional apertures (small 1x1 and large 5x5) to produce light curves suitable for QLP's systematics correction, detrending, and planet search process.

Refer to [Han & Brandt (2023)](https://iopscience.iop.org/article/10.3847/1538-3881/acaaa7) and the [original TGLC repository](https://github.com/TeHanHunter/TESS_Gaia_Light_Curve) for more information on TGLC's methods.

## Usage

Install this version of TGLC via pip:

```shell
pip install git+ssh://git@github.com:mit-kavli-institute/tess-gaia-light-curve.git
```

This will create a `tglc` executable command in your environment. It has four subcommands, which can be listed with `tglc -h`. They correspond to the four steps that TGLC must do to create light curves: download catalogs, create FFI cutouts, fit ePSFs, and extract photometry. TGLC does not download FFI data; you are repsonsible for ensuring that data is available in the right location on your system.

```
$ tglc -h
usage: tglc [-h] [-V] {catalogs,cutouts,epsfs,lightcurves} ...

TESS-Gaia Light Curve

positional arguments:
  {catalogs,cutouts,epsfs,lightcurves}
                        TGLC script to run
    catalogs            Create cached TIC and Gaia catalogs with data for an orbit.
    cutouts             Create FFI cutouts using catalog data (requires tglc catalogs to be run)
    epsfs               Fit and save ePSFs for FFI cutouts (requires tglc cutouts to be run)
    lightcurves         Create light curves using fitted ePSFs (requires tglc epsfs to be run)

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
```

Each of the subcommands has additional information available via a similar help message, for example with `tglc cutouts -h`.

## Development

If you want to work directly with this code base, clone the repository, create a virtual environment, and install the project in editable mode.

```shell
git clone git@github.com:mit-kavli-institute/tess-gaia-light-curve.git
python3 -m venv .venv  # or use conda or uv
source .venv/bin/activate  # if you used venv as above
pip install -e ".[dev]"
```

You now have the `tglc` package and all its dependencies available to use in scripts and notebooks. If you edit the codebase, you can run the tools that are set up for checking the code.

```shell
ruff format .  # formatter
ruff check .  # linter
pytest  # test suite
```
