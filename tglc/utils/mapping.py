"""Utilities for mapping functions in TGLC."""

from collections.abc import Callable, Iterable
from multiprocessing import Pool

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def pool_map_if_multiprocessing(
    func: Callable,
    iterable: Iterable,
    nprocs: int = 1,
    pool_map_method: str = "map",
):
    """Map a function over an iterable, conditionally using a multiprocessing pool if `nprocs > 1`."""
    if nprocs > 1:
        with Pool(nprocs) as pool:
            yield from getattr(pool, pool_map_method)(func, iterable)
    else:
        yield from map(func, iterable)


def consume_iterator_with_progress_bar(iterator: Iterable, *args, **kwargs):
    """
    Consume an iterator with a progress bar. Logging is redirected.

    Additional positional and keyword arguments are passed to `tqdm.tqdm`.
    """
    with logging_redirect_tqdm():
        for _ in tqdm(iterator, *args, **kwargs):
            pass
