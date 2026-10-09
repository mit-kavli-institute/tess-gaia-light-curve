"""Utilities for mapping functions in TGLC."""

from collections.abc import Callable, Iterable
from multiprocessing import get_context

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def pool_map_if_multiprocessing(
    func: Callable,
    iterable: Iterable,
    nprocs: int = 1,
    pool_map_method: str = "map",
    mp_start_method: str | None = None,
):
    """Map a function over an iterable, conditionally using a multiprocessing pool if `nprocs > 1`."""
    mp_context = get_context(mp_start_method)
    if nprocs > 1:
        with mp_context.Pool(nprocs) as pool:
            yield from getattr(pool, pool_map_method)(func, iterable)
    else:
        yield from map(func, iterable)


def iterate_with_progress_bar(iterator: Iterable, *args, **kwargs):
    """
    Yield items from an iterator with a progress bar. Logging is redirected.

    Additional positional and keyword arguments are passed to `tqdm.tqdm`.
    """
    with logging_redirect_tqdm():
        yield from tqdm(iterator, *args, **kwargs)


def consume_iterator_with_progress_bar(iterator: Iterable, *args, **kwargs):
    """
    Consume an iterator with a progress bar, discarding results. Logging is redirected.

    Additional positional and keyword arguments are passed to `tqdm.tqdm`.
    """
    for _ in iterate_with_progress_bar(iterator, *args, **kwargs):
        pass
