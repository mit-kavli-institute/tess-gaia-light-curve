"""
Tests for the tglc.util.logging module, which provides helper functions for setting up logging for
TGLC scripts.
"""

import logging
from pathlib import Path

import pytest

from tglc.utils.logging import setup_logging


TEST_LOG_LEVELS = [
    (level_name, logging.getLevelNamesMapping()[level_name])
    for level_name in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
]


@pytest.mark.parametrize("logger_name", ["tglc", "tglc.submodule"])
@pytest.mark.parametrize("level_name,level", TEST_LOG_LEVELS)
def test_setup_logging_defaults(caplog, logger_name: str, level_name: str, level: int):
    setup_logging()
    logger = logging.getLogger(logger_name)
    logger.log(level, f"{level_name.lower()} message")

    if level_name == "DEBUG":
        assert len(caplog.records) == 0
    else:
        assert len(caplog.records) == 1
        logged_record = caplog.records[0]
        assert logged_record.name == logger_name
        assert logged_record.levelname == level_name
        assert logged_record.message == f"{level_name.lower()} message"


@pytest.mark.parametrize("logger_name", ["tglc", "tglc.submodule"])
@pytest.mark.parametrize("level_name,level", TEST_LOG_LEVELS)
def test_setup_logging_debug(caplog, logger_name: str, level_name: str, level: int):
    setup_logging(debug=True)
    logger = logging.getLogger(logger_name)
    logger.log(level, f"{level_name.lower()} message")

    assert len(caplog.records) == 1
    logged_record = caplog.records[0]
    assert logged_record.name == logger_name
    assert logged_record.levelname == level_name
    assert logged_record.message == f"{level_name.lower()} message"


@pytest.mark.parametrize("logger_name", ["tglc", "tglc.submodule"])
@pytest.mark.parametrize("level_name,level", TEST_LOG_LEVELS)
def test_setup_logging_to_file(
    caplog, tmp_path: Path, logger_name: str, level_name: str, level: int
):
    log_file = tmp_path / "logfile.txt"
    setup_logging(logfile=log_file)
    logger = logging.getLogger(logger_name)
    logger.log(level, f"{level_name.lower()} message")

    if level_name == "DEBUG":
        assert len(caplog.records) == 0
    else:
        assert len(caplog.records) == 1
        logged_record = caplog.records[0]
        assert logged_record.name == logger_name
        assert logged_record.levelname == level_name
        assert logged_record.message == f"{level_name.lower()} message"
        with log_file.open() as logs:
            lines = logs.readlines()
            assert len(lines) == 1
            assert lines[0].endswith(f"{level_name.lower()} message\n")


@pytest.mark.parametrize("logger_name", ["tglc", "tglc.submodule"])
@pytest.mark.parametrize("level_name,level", TEST_LOG_LEVELS)
def test_setup_logging_debug_to_file(
    caplog, tmp_path: Path, logger_name: str, level_name: str, level: int
):
    log_file = tmp_path / "logfile.txt"
    setup_logging(debug=True, logfile=log_file)
    logger = logging.getLogger(logger_name)
    logger.log(level, f"{level_name.lower()} message")

    assert len(caplog.records) == 1
    logged_record = caplog.records[0]
    assert logged_record.name == logger_name
    assert logged_record.levelname == level_name
    assert logged_record.message == f"{level_name.lower()} message"

    with log_file.open() as logs:
        lines = logs.readlines()
        assert len(lines) == 1
        assert level_name in lines[0]
        assert "test_setup_logging_debug_to_file" in lines[0]
        assert lines[0].endswith(f"{level_name.lower()} message\n")
