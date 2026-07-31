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
