# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This is the MIT Quick-Look Pipeline fork of TESS-Gaia Light Curve (TGLC). It uses TGLC's ePSF fitting and aperture photometry methods (Han & Brandt, 2023) to produce light curves from TESS FFIs for downstream QLP systematics correction, detrending, and planet search. TGLC does **not** download FFIs — callers are responsible for putting FFI files in the expected location before running.

## Development commands

```shell
pip install -e ".[dev]"   # editable install with dev tools (ruff, pytest, pyticdb)
ruff format .             # formatter
ruff check .              # linter (extends select: B, I, UP, NPY, PD; line-length 100)
pytest                    # full test suite (some tests need Docker, see below)
pytest tests/test_epsf.py::test_name  # single test
```

`pytest` is configured with `filterwarnings = ["error"]` and `xfail_strict = true` — warnings fail tests.

End-to-end tests in `tests/end_to_end/` spin up Postgres containers via `pytest-docker` (compose file at `tests/sample_data/databases/docker-compose.yml`) and download sample FFIs via `pooch`. They require Docker to be running and the `pyticdb` extra installed. See `tests/sample_data/README.md` and `tests/sample_data/databases/README.md` for the sample-data provenance and how to regenerate it.

## CLI architecture

The package exposes a single `tglc` console script (entry: `tglc.__main__:tglc_main`) with five subcommands that form a strict pipeline; each step assumes the previous step's outputs exist on disk:

1. `tglc catalogs` — query TIC + Gaia (via `pyticdb`) for the orbit's FOV and cache as eCSV catalogs.
2. `tglc cutouts` — slice FFIs into overlapping cutouts (default 150 px, 2 px overlap) and pickle a `Source` object per cutout containing flux cube + WCS + matched catalog rows.
3. `tglc epsfs` — fit per-cadence ePSF parameters for each cutout via least squares, saved as `.npy`. Uses CuPy on GPU when available (auto-detected by `tglc.utils._optional_deps.HAS_CUPY`).
4. `tglc lightcurves` — combine source pickles with fitted ePSFs to extract decontaminated aperture photometry per star, written as one HDF5 file per TIC ID.
5. `tglc all` — runs steps 1–4 in sequence for an orbit.

All subcommands share a common parser (`tglc/cli.py:command_base_parser`) and accept `--orbit`, `--ccd cam,ccd` (defaults to all 16), `--cutout x,y`, `-n/--nprocs`, `-r/--replace`, `--tglc-data-dir`, `--debug`, `-l/--logfile`. The `--tglc-data-dir` default walks up the cwd looking for a directory named `tglc-data`.

## Data layout (the `Manifest` class)

All file paths are centralized in `tglc/utils/manifest.py`. `Manifest` is a dataclass with `orbit/camera/ccd/cadence/cutout_x/cutout_y/tic_id` fields and properties that derive paths from them. Crucial behavior: accessing a property whose required attributes are unset raises `RuntimeError` — set the fields you need before reading the path.

Directory tree under `tglc-data-dir`:

```
orbit-<orbit>/ffi/
  catalogs/{TIC,Gaia}_cam<C>_ccd<D>.ecsv
  cam<C>/ccd<D>/
    ffi/    <— place TICA FFI .fits files here before running (not downloaded by TGLC)
    source/source_<x>_<y>.fits      (FFICutout MEF: PRIMARY/FLUX/MASK/BADPIX/CADENCES/GAIA/TIC)
    epsf/epsf_<x>_<y>.fits          (fitted ePSF + metadata in header)
    LC/<tic_id>.h5                  (light curves)
```

The cutout and ePSF intermediate products migrated from pickle/`.npy` to FITS
in issue #1. `tglc/io.py` exposes `write_cutout_fits`/`read_cutout_fits` and
`write_epsf_fits`/`read_epsf_fits` (free functions, not methods, to avoid an
`ffi.py` ↔ `io.py` import cycle), plus `migrate_cutout_pickle`/`migrate_epsf_npy`
for translating legacy on-disk products. The class is `tglc.ffi.FFICutout`;
`tglc.ffi.Source` is kept as a backwards-compat alias so legacy pickles still
unpickle. `Manifest` exposes both the current `.fits` paths and `_legacy_*` paths
for the old extensions.

## TESS sector/orbit mapping

`tglc/utils/constants.py:get_sector_containing_orbit` and `get_orbits_in_sector` encode the (non-monotonic) mapping between TESS orbits and sectors. This includes special-cased ranges (sectors 97 and 98 have four orbits each, breaking the otherwise simple `sector*2+7,8` rule). Update these together when extending into new mission years.

## Multiprocessing model

- `tglc.utils.mapping.pool_map_if_multiprocessing` is the standard fan-out: `map()` when `nprocs=1`, otherwise a `multiprocessing.Pool` with optional `mp_start_method` override.
- `tglc/__main__.py` calls `set_start_method("fork")` so logs from workers propagate. The ePSF script overrides to `"spawn"` when GPUs are used (required by CUDA), at the cost of losing worker logs.
- When `--nprocs > 1`, numpy/BLAS thread env vars (`OPENBLAS_NUM_THREADS`, etc.) are pinned to 1 in `__main__.py` to prevent oversubscription.
- For GPU runs, `tglc all` caps workers at the number of CUDA devices for the ePSF step only.
- Pyticdb queries within `tglc catalogs` use a `ThreadPoolExecutor` (sized `nprocs // 16`) inside each worker process — the outer `Pool` distributes CCDs, the inner threads distribute cone queries within a CCD.

## Optional dependencies

`tglc/utils/_optional_deps.py` exposes `HAS_<DEP>` flags. `pyticdb` (in the `[pyticdb]` extra, installed by `[dev]`) is required for `tglc catalogs` and is sourced from MIT-Kavli's PyPI index — see `[tool.uv]` in `pyproject.toml`. `cupy` (in the `[cupy]` extra) is checked for both install and runtime CUDA device availability, not just `find_spec`.

## Logging

Always create loggers via `logging.getLogger(__name__)`. `tglc/utils/logging.py:setup_logging` is called once from `__main__.py` based on CLI flags; it silences `RuntimeWarning` by default (override with `--enable-runtime-warnings`) and downgrades numba's debug spam.

## Scripts entry-point convention

Files under `tglc/scripts/` are imported lazily from `__main__.py` and each defines a `make_*_main(args)` function. They explicitly `raise RuntimeError` if run directly — always invoke via `tglc <subcommand>` or `python -m tglc`.
