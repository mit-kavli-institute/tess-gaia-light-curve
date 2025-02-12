"""
Logging utilities mainly meant for use in scripts. All logging throughout TGLC should be done
through a logger retrieved with `logging.getLogger(__name__)` to ensure defaults are propagated
correctly.
"""

import logging
from pathlib import Path


def setup_logging(debug: bool = False, logfile: Path | None = None):
    """Set up logging with a reasonable set of defaults."""
    log_level = logging.DEBUG if debug else logging.INFO

    log_fmt: str
    if debug:
        log_fmt = "%(asctime)s %(levelname)s %(funcName)s: %(message)s"
    else:
        log_fmt = "%(asctime)s %(message)s"

    handler: logging.Handler
    if logfile is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(logfile)
    date_fmt = "%Y.%m.%d %H:%M:%S"
    handler.setFormatter(logging.Formatter(log_fmt, date_fmt))

    # TODO: Eventually this should be logging.getLogger("tglc") once we have a real CLI set up
    base_tglc_logger = logging.getLogger()
    base_tglc_logger.setLevel(log_level)
    base_tglc_logger.addHandler(handler)
