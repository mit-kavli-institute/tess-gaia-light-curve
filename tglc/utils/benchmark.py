"""Lightweight benchmarking helpers: wall-clock timing and peak-memory (RSS) reporting.

Benchmark records are INFO-level log lines with a stable, grep-able key=value format:

    TGLC-BENCH event=<event> key1=value1 key2=value2 ...

Unix-only: uses the `resource` module, which is unavailable on Windows.
"""

from contextlib import contextmanager
import logging
import resource
import sys
import time


logger = logging.getLogger(__name__)


_RU_MAXRSS_BYTES_PER_UNIT = 1 if sys.platform == "darwin" else 1024
"""`resource.getrusage` reports `ru_maxrss` in bytes on macOS but kilobytes on Linux."""


def get_peak_rss_bytes(children: bool = False) -> int:
    """
    Get the peak resident set size (high-water memory usage) in bytes.

    Parameters
    ----------
    children : bool
        If true, report the maximum peak RSS among reaped child processes instead of the
        current process. Note that this is a maximum over individual children, not a sum,
        and only includes children that have already been waited on.

    Returns
    -------
    peak_rss : int
        Peak resident set size in bytes.
    """
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    return resource.getrusage(who).ru_maxrss * _RU_MAXRSS_BYTES_PER_UNIT


def format_benchmark_record(event: str, **fields) -> str:
    """
    Format a `TGLC-BENCH` benchmark record log line.

    Parameters
    ----------
    event : str
        Name identifying the kind of benchmark record, e.g. "step".
    **fields
        Values to include in the record. Floats are rendered with 3 decimal places.

    Returns
    -------
    record : str
        Log line like `TGLC-BENCH event=<event> key1=value1 key2=value2 ...`.
    """
    formatted_fields = " ".join(
        f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in fields.items()
    )
    return f"TGLC-BENCH event={event} {formatted_fields}"


@contextmanager
def benchmark_step(step: str):
    """
    Context manager logging wall-clock time and peak RSS for a pipeline step.

    Emits an INFO-level `TGLC-BENCH event=step` record on exit with the elapsed time in
    seconds and the peak RSS of this process and its reaped children in MiB. Peak RSS is a
    lifetime high-water mark, so per-step values are non-decreasing over a multi-step run.

    Parameters
    ----------
    step : str
        Name of the pipeline step being benchmarked.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            format_benchmark_record(
                "step",
                step=step,
                elapsed_s=time.perf_counter() - start,
                peak_rss_mb=get_peak_rss_bytes() / 2**20,
                children_peak_rss_mb=get_peak_rss_bytes(children=True) / 2**20,
            )
        )
