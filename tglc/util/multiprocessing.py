"""Multiprocessing utilities for TGLC."""

from collections.abc import Iterable
from multiprocessing import Pool
from typing import Callable

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def pool_map_if_multiprocessing_with_tqdm(
    func: Callable,
    iterable: Iterable,
    nprocs: int = 1,
    pool_map_method: str = "map",
    **kwargs,
):
    """
    Map a function over an iterable with a progress bar that updates as the result is consumed.

    Conditionally uses a multiprocessing pool if `nprocs > 1`.

    `kwargs` are forwarded to `tqdm.tqdm`.
    """
    with logging_redirect_tqdm():
        if nprocs > 1:
            with Pool(nprocs) as pool:
                yield from tqdm(
                    getattr(pool, pool_map_method)(func, iterable),
                    **kwargs,
                )
        else:
            yield from tqdm(map(func, iterable), **kwargs)
