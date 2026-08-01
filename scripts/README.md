# TGLC diagnostics scripts

Standalone diagnostics and data-exploration tools. These are **not** part of the `tglc` package:
their extra dependencies are intentionally not declared in `pyproject.toml`, and they add the
repository root to `sys.path` at startup so they run from any checkout without installing tglc.

## `audit_lightcurve_timing.py`

Audits existing H5 light curves for barycentric timing errors introduced by the retired
ephemeris interpolation (see the module docstring for the method). For each light curve it
computes the discrepancy between the old correction (bundled yearly CSVs + clamped `np.interp`,
reproduced literally from the pre-d55ae79 code) and the current correction (per-orbit JPL
Horizons ephemeris), then aggregates everything into a PDF report.

Known systematics it detects and classifies:

- **`clamped`** — the light curve's time span fell partly outside its yearly CSV's coverage, so
  the old code silently held the spacecraft position at the file edge (errors grow with distance
  past the edge; seconds to minutes).
- **`malformed`** — sectors 102–115 used `20260401_tess_ephem.csv`, whose data rows have 12
  fields under a 9-field header; the old parser read shifted columns, so every correction from
  that file was computed from meaningless values.
- Light curves with max |Δt| at the ~1 ms level are **fine**: that is the ordinary difference
  between the predicted trajectory in the old files and the definitive JPL solution.

Both current and legacy H5 layouts are handled (root-level vs. `LightCurve`-group attributes,
TJD vs. full-JD `BJD` datasets).

### Prerequisites

- The repo's Python environment (`.venv`) — the scan needs only tglc's own dependencies.
- For `report`: `pip install pylatex` plus a LaTeX toolchain (`latexmk`/`pdflatex` on `PATH`).
- Network access to `ssd.jpl.nasa.gov` for the first scan of each orbit (one query per orbit,
  cached in `--ephemerides-dir`). For offline hosts, pre-populate the cache directory — e.g.
  copy `ephemerides/` from a pipeline data directory that has already processed those orbits.

### Usage

```sh
# Scan: inputs are directories (searched recursively for *.h5) and/or manifest text files
# (one H5 path per line). Results land in per-orbit CSVs under --results-dir.
python scripts/audit_lightcurve_timing.py scan /data/tglc-data /data/extra_lightcurves.txt \
    --results-dir /data/timing_audit --nprocs 32

# Report: aggregates all per-orbit CSVs into a PDF (plus figures/ and the .tex source).
python scripts/audit_lightcurve_timing.py report --results-dir /data/timing_audit
```

Useful options: `--threshold` (seconds; default 0.05) sets the "affected" cutoff — the report
recomputes affected counts from stored per-file statistics, so you can re-run `report` with a
different threshold without rescanning. `--orbit N ...` restricts a scan; `--replace` rescans
orbits that already have results (by default finished orbits are skipped, making scans
resumable after interruption).

For millions of files, prefer a pre-built manifest (`find ... -name '*.h5' > manifest.txt`)
over directory inputs: recursive globbing of network filesystems can take longer than the scan
itself. The scan is I/O bound (~1–2 ms/file locally, 5–20 ms/file on network storage).

### Verification

A fully offline end-to-end check with known injected errors:

```sh
python scripts/audit_lightcurve_timing.py make-synthetic /tmp/timing_synthetic
# then run the scan + report commands it prints
```

The synthetic dataset contains light curves with an injected 0.5 s discrepancy (stars with
û_x = 1 against an X-shifted fake ephemeris cache), control stars at û_x = 0 (expect ~0),
a span extending past the old file's coverage (expect the `clamped` diagnosis), a corrupt file
(expect `read_error`), and an orbit outside the sector map (expect an orbit-level error record).
Expected outcomes are printed when the dataset is generated.

### `data/old_ephemerides/`

Byte-identical copies of the yearly ephemeris CSVs that shipped with tglc before commit
d55ae79 (extracted from git history), including the malformed `20260401_tess_ephem.csv` —
deliberately left broken so the audit reproduces exactly what production computed. Do not
"fix" these files.

## `fix_lightcurve_timing.py`

Corrects the `LightCurve/BJD` dataset of existing H5 light curves **in place**. The true
spacecraft times are recovered from the cutout `Source` pickles (`Source.cadence` →
`Source.time`, matched per (orbit, camera, ccd) because FFI timestamps differ slightly between
CCDs), and the barycentric correction is recomputed with the production code path
(`get_tess_spacecraft_position` + `apply_barycentric_correction`), so repaired files are
bit-identical to what the current pipeline would write. Only the BJD values change — no other
datasets, no attributes, no backups (per project decision; MAST and other filesystems hold
copies). The old BJD is never an input, so re-running is idempotent and even garbage BJDs
(e.g. the sector ≥ 102 `malformed` cases) repair fully. Legacy files are handled: pre-2025
attribute layouts, and full-JD `BJD` datasets are corrected in their own convention.

Groups whose Source pickles are missing are **skipped and reported** (`source_missing`) —
regenerate them with `tglc cutouts` for those orbits and re-run.

```sh
# Always dry-run first: computes and logs every correction, provably writes nothing
# (files are opened read-only in this mode)
python scripts/fix_lightcurve_timing.py fix /data/tglc-data \
    --results-dir /data/timing_fix --ephemerides-dir /data/tglc-data/ephemerides \
    --nprocs 32 --dry-run

# Then the real thing (dry-run logs use a .dryrun.csv suffix and don't block this)
python scripts/fix_lightcurve_timing.py fix /data/tglc-data \
    --results-dir /data/timing_fix --ephemerides-dir /data/tglc-data/ephemerides --nprocs 32
```

Inputs are directories and/or manifest files, as for the audit. Point `--ephemerides-dir` at
the pipeline's existing cache to avoid any Horizons queries. Per-group fix logs
(`fix_orbit-0185_cam1_ccd1.csv`) make interrupted runs resumable (finished groups are skipped;
`--replace` redoes them); an interrupted group is safe to redo because the correction is
recomputed from scratch. `--tglc-data-dir` locates Source pickles via the standard Manifest
layout when H5 paths are nonstandard. On network filesystems with broken POSIX locking set
`HDF5_USE_FILE_LOCKING=FALSE`. Do not run two fixer invocations over overlapping inputs.

Offline end-to-end verification (injected 0.5 s errors, subset cadences, legacy formats,
must-not-touch cases):

```sh
python scripts/fix_lightcurve_timing.py make-synthetic /tmp/fix_synthetic
# run the printed fix command, then:
python scripts/fix_lightcurve_timing.py verify-synthetic /tmp/fix_synthetic
```
