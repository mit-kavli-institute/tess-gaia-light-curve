"""Test configuration for the TGLC package."""

from collections.abc import Generator
import importlib
from pathlib import Path

import pytest
import pyticdb
import sqlalchemy as sa

from .sample_data import sample_ffis  # noqa: F401


#######################################################################
#### Database Fixtures ################################################
#######################################################################


TEST_PYTICDB_CONFIG = """[tic_82]
username=tglctester
password=password
database=
port=5432

[gaia3]
username=tglctester
password=password
database=
port=5433
"""


@pytest.fixture
def tmp_pyticdb_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    """Pytest fixture that creates a pyticdb configuration file and monkeypatches pyticdb to look
    in the proper directory for the configuration.

    Returns a `Path` object for the configuration directory.
    """
    config_dir = tmp_path / ".config" / "tic"
    config_dir.mkdir(parents=True)
    with open(config_dir / "db.conf", "w") as db_conf_file:
        db_conf_file.write(TEST_PYTICDB_CONFIG)

    # Monkeypatch the configuration directory to look at the sample configuration.
    # Pyticdb bakes the config location into a lot of things at import time, so the appropriate
    # modules need to be reloaded to force the new config location to take effect. Then, after
    # undoing the monkeypatch, we need to reload again to reset the config location.
    try:
        with monkeypatch.context() as m:
            m.setattr(
                pyticdb.conn, "Databases", pyticdb.conn.TableReflectionCache(config_dir / "db.conf")
            )
            importlib.reload(pyticdb)
            importlib.reload(pyticdb.query)
            yield config_dir
    finally:
        importlib.reload(pyticdb)
        importlib.reload(pyticdb.query)


@pytest.fixture(scope="session")
def docker_compose_file():
    return Path(__file__).parent / "sample_data" / "databases" / "docker-compose.yml"


def is_pyticdb_running(db_name: str):
    try:
        with pyticdb.Databases[db_name][1]().connection():
            return True
    except sa.exc.OperationalError:
        return False


@pytest.fixture(scope="session")
def pyticdb_database_service(docker_services):
    docker_services.wait_until_responsive(
        timeout=20.0, pause=0.25, check=lambda: is_pyticdb_running("tic_82")
    )
    docker_services.wait_until_responsive(
        timeout=20.0, pause=0.25, check=lambda: is_pyticdb_running("gaia3")
    )


@pytest.fixture
def pyticdb_databases(tmp_pyticdb_config, pyticdb_database_service):
    """Pytest fixture that makes all pyticdb databases, plus configuration, available."""
    pass
