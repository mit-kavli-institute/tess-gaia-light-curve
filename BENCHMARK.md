# Benchmarking TGLC: TMag ≤ 18, primary aperture only

This branch benchmarks TGLC under a production-candidate configuration:

- Light curves extracted for stars down to **TMag 18** (default catalog cutoff is 13.5)
- Only the **primary (3×3) aperture** written to output files (default is primary + small + large)

Metrics captured: per-step wall-clock time, per-cutout star counts and throughput in the
lightcurves step, and peak memory (RSS).

## Prerequisites

- Real orbit FFIs placed in the expected layout (TGLC does **not** download FFIs):
  `<tglc-data-dir>/orbit-<N>/ffi/cam<C>/ccd<D>/ffi/*.fits`
- `pyticdb` installed and configured with access to the TIC and Gaia databases
- For GPU ePSF fitting, the `cupy` extra and CUDA devices

## Running the benchmark

One-shot:

```shell
tglc all --orbit <N> --max-magnitude 18 --apertures primary -n <P> --replace \
    --logfile bench-orbit<N>-tmag18-primary.log
```

Or per step (e.g. to run the ePSF step on a GPU node):

```shell
tglc catalogs    --orbit <N> --max-magnitude 18 -n <P> --replace -l bench.log
tglc cutouts     --orbit <N> -n <P> --replace -l bench.log
tglc epsfs       --orbit <N> -n <P> --replace -l bench.log
tglc lightcurves --orbit <N> --apertures primary -n <P> --replace -l bench.log
```

Using `--logfile` is recommended: tqdm progress bars go to stderr while benchmark records land
cleanly in the file.

`--mdwarf-magnitude` can be left at its default: with `--max-magnitude 18` the M-dwarf extension
clause (TMag between the main and M-dwarf cutoffs) is empty and subsumed by the main cutoff.

### Changing the magnitude limit requires regenerating catalogs and cutouts

The TIC target table is queried by `tglc catalogs` and baked into each cutout FITS file by
`tglc cutouts`. Changing `--max-magnitude` therefore has no effect on existing cutouts — rerun
`catalogs` and `cutouts` with `--replace`, or (recommended for clean benchmarks) use a fresh
`--tglc-data-dir`. The Gaia catalog used for ePSF fitting is not magnitude-limited, so ePSFs are
in principle reusable across magnitude configurations, but full regeneration is the safe default.

### Baseline comparison

Run the identical command with the defaults for comparison:

```shell
tglc all --orbit <N> --max-magnitude 13.5 --apertures primary small large -n <P> --replace \
    --logfile bench-orbit<N>-baseline.log
```

## Reading the results

Benchmark records are INFO-level log lines with a stable `key=value` format:

```
TGLC-BENCH event=step step=lightcurves elapsed_s=123.456 peak_rss_mb=1024.000 children_peak_rss_mb=2048.000
TGLC-BENCH event=cutout_light_curves cutout=source_0_0 pid=12345 stars=1500 read_s=2.345 elapsed_s=98.765 stars_per_s=15.188 peak_rss_mb=2048.000
```

One `event=step` record is logged per pipeline step (plus an `event=all` total for `tglc all`),
and one `event=cutout_light_curves` record per cutout processed by the lightcurves step.

Useful one-liners:

```shell
# Per-step wall times
grep 'event=step' bench.log

# Total stars and aggregate throughput across all cutouts
grep 'event=cutout_light_curves' bench.log \
    | sed -E 's/.* stars=([0-9]+) .* elapsed_s=([0-9.]+) .*/\1 \2/' \
    | awk '{stars += $1; s += $2} END {print stars " stars, " s "s total cutout time, " stars/s " stars/s"}'

# Peak worker memory
grep 'event=cutout_light_curves' bench.log \
    | sed -E 's/.* peak_rss_mb=([0-9.]+).*/\1/' | sort -n | tail -1
```

### Metric semantics

- **Peak RSS is a lifetime high-water mark**, not a per-step delta: per-step `peak_rss_mb` values
  for the parent process are non-decreasing across a `tglc all` run, so use the maximum (or final)
  value as the run's parent memory footprint.
- **`children_peak_rss_mb` is a maximum over reaped worker processes, not a sum.** Total memory
  demand with `-n P` workers is roughly `P ×` the per-worker peak.
- **Forked workers inherit the parent's memory baseline**, so per-worker `peak_rss_mb` includes
  memory the parent had already touched at fork time.
- `ru_maxrss` units differ by platform (bytes on macOS, kilobytes on Linux); values are normalized
  to bytes/MiB in the code.
