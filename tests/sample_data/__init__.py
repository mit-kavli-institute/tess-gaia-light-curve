"""Pytest fixtures for accessing sample data."""

from pathlib import Path

import pytest

from .download import download_ffis


SAMPLE_DATA_DIRECTORY = Path(__file__).parent


@pytest.fixture(scope="session")
def sample_ffis():
    """Pytest fixture that ensures FFIs are downloaded for tests that require them."""
    download_ffis()
    return SAMPLE_DATA_DIRECTORY / "ffi"
