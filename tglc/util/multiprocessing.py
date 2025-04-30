"""Multiprocessing utilities for TGLC."""

from collections.abc import Iterable
from multiprocessing import Pool
from typing import Callable


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
