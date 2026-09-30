![TESS-Gaia Light Curve Logo](/logo/TGLC_Title.png)
[![TGLC DOI Badge](https://zenodo.org/badge/420868490.svg)](https://zenodo.org/badge/latestdoi/420868490)
[![TGLC Citation Badge](https://img.shields.io/badge/Cite-TGLC-blue)](https://www.tomwagg.com/software-citation-station/?auto-select=tglc)

## Introduction

This is the version of TESS-Gaia Light Curve adapted for the TESS Quick-Look Pipeline at MIT. It uses TGLC's methods for ePSF fitting and aperture photometry with two additional apertures (small 1x1 and large 5x5) to produce light curves suitable for QLP's systematics correction, detrending, and planet search process.

Refer to [Han & Brandt (2023)](https://iopscience.iop.org/article/10.3847/1538-3881/acaaa7) and the [original TGLC repository](https://github.com/TeHanHunter/TESS_Gaia_Light_Curve) for more information on TGLC's methods.

## Usage

Install this version of TGLC via pip:

```shell
pip install git+https://github.com/mit-kavli-institute/tess-gaia-light-curve.git
```

This will create a `tglc` executable command in your environment. Its subcommands can be listed with `tglc -h`. Four of them correspond to the four steps that TGLC must do to create light curves: download catalogs, create FFI cutouts, fit ePSFs, and extract photometry. `all` runs those four steps in sequence for an orbit, and `migrate` is a temporary command for converting legacy data products (see below). TGLC does not download FFI data; you are responsible for ensuring that data is available in the right location on your system.

```
$ tglc -h
usage: tglc [-h] [-V] {all,catalogs,cutouts,epsfs,lightcurves,migrate} ...

TESS-Gaia Light Curve

positional arguments:
  {all,catalogs,cutouts,epsfs,lightcurves,migrate}
                        TGLC script to run
    all                 Run all TGLC steps for an orbit.
    catalogs            Create cached TIC and Gaia catalogs with data for an orbit.
    cutouts             Create FFI cutouts using catalog data (requires tglc catalogs to be run)
    epsfs               Fit and save ePSFs for FFI cutouts (requires tglc cutouts to be run)
    lightcurves         Create light curves using fitted ePSFs (requires tglc epsfs to be run)
    migrate             Migrate legacy .pkl/.npy data products to FITS (temporary)

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
```

Each of the subcommands has additional information available via a similar help message, for example with `tglc cutouts -h`.

`migrate` converts legacy pickle/`.npy` cutout and ePSF files to the FITS formats this version reads. It exists only for the retroactive reprocessing campaign and will be removed once that is complete, so new runs should not need it.

### Selecting which targets get light curves

By default, `tglc lightcurves` produces a light curve for every target in each cutout's TIC catalog — that is, everything admitted by the magnitude limits given to `tglc catalogs`. Three options narrow that down:

- `--max-magnitude` keeps only targets brighter than a TESS magnitude, using the same strictly-brighter-than convention as the TIC query in `tglc catalogs`. The limit is applied to the Gaia-derived magnitude recorded in each light curve, not the TIC `Tmag` the catalog query filters on, because the cutout's TIC table carries only the TIC ↔ Gaia crossmatch.
- `-t`/`--tic` takes TIC IDs directly on the command line.
- `--tic-file` reads TIC IDs from a file, for target lists too long to pass with `--tic`. IDs are separated by any mix of whitespace and commas, so one ID per line and a single comma-separated line both work; blank lines and `#` comments are ignored.

`--tic` and `--tic-file` are combined into one list of requested IDs. Requested IDs that never turn up in a processed cutout are reported in a warning at the end of the run, so the same target list can be passed for every orbit and CCD.

The magnitude limit and the requested IDs are **additive**: with both given, a target gets a light curve if it is bright enough *or* it is explicitly listed. That extracts a magnitude-limited sample alongside a list of fainter targets of interest. Either one alone acts as the sole selection.

```shell
# Everything brighter than Tmag 10, plus a list of fainter targets of interest
tglc lightcurves -o 223 --max-magnitude 10 --tic-file targets.txt
```

Note that the `all` command's `--max-magnitude` applies to the TIC query only. It is deliberately not reapplied to the light curve step, which would otherwise drop the M dwarfs admitted by `--mdwarf-magnitude`.

## Development

If you want to work directly with this code base, clone the repository, create a virtual environment, and install the project in editable mode. The `[dev]` extra pulls in `pyticdb`, which lives on the MIT-Kavli PyPI index, so the index needs to be passed to `pip` with `--extra-index-url`.

```shell
git clone git@github.com:mit-kavli-institute/tess-gaia-light-curve.git
python3 -m venv .venv  # or use conda or uv
source .venv/bin/activate  # if you used venv as above
pip install -e ".[dev]" \
  --extra-index-url https://mit-kavli-institute.github.io/MIT-Kavli-PyPi/
```

You now have the `tglc` package and all its dependencies available to use in scripts and notebooks. If you edit the codebase, you can run the tools that are set up for checking the code.

```shell
ruff format .  # formatter
ruff check .  # linter
pytest  # test suite
```

### End-to-end tests

The end-to-end suite in `tests/end_to_end/` exercises the full pipeline (catalogs → cutouts → ePSFs → light curves) against fake TIC and Gaia databases that are brought up as PostgreSQL containers via `docker compose`, plus a handful of TICA FFIs downloaded from MAST via [pooch](https://www.fatiando.org/pooch/). To run it you need:

- The Docker daemon running locally (Docker Desktop, Colima, or equivalent).
- `psycopg`'s binary wheel, so libpq does not need to be installed system-wide:

  ```shell
  pip install "psycopg[binary]"
  ```

- An internet connection on first run, for `pooch` to fetch the sample FFIs (cached afterward in `tests/sample_data/ffi/`).

With those in place, the e2e tests run as part of the standard `pytest` invocation. They can also be exercised in isolation:

```shell
pytest tests/end_to_end/
```

The unit-test suite (`tests/test_io.py`, `tests/test_utils/`, etc.) does not require Docker or `psycopg`, so a plain `pytest tests/test_io.py` will succeed without those prerequisites.

## Edge-compression calibration note

The default `--edge-compression-factor` of `3.16e-7` was **determined experimentally for 200 s
FFIs** (TICA cutouts fit in electrons per cadence, 158.4 s effective exposure), using the sweep in
`tglc/scripts/edge_compression_sweep.py` / `edge_compression_figure.py` over all 392 cutouts of
sector 106 (orbits 223–224). Three independent metrics agree on the value: the knee of the
residual-image MAD curve, the minimum of a 10%-pixel holdout cross-validation, and the minimum of
the small-aperture light-curve scatter (see issue #25). It matches upstream TGLC's `1e-4` — which
was calibrated on SPOC images in e-/s — converted to these units
(`1e-4 / 158.4^1.4 ≈ 8.3e-8`) to within one half-decade grid step.

Because the ePSF fit weights data rows by `1/flux^1.4` while the regularization rows have unit
weight, the appropriate factor scales with the image's flux units. For FFIs at other cadences,
rescale by `(effective exposure / 158.4)^1.4`, or re-derive the value with the sweep scripts.
