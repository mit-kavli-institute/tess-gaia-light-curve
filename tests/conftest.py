"""Test configuration for the TGLC package."""

from collections.abc import Generator
import importlib
from pathlib import Path

import pytest

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

    Skips the test if `pyticdb` is not installed -- the package lives on the MIT-Kavli PyPI index
    and is not always available locally.

    Returns a `Path` object for the configuration directory.
    """
    pyticdb = pytest.importorskip("pyticdb")
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


@pytest.fixture(scope="session")
def pyticdb_database_service(docker_services):
    """Wait until both TIC and Gaia postgres containers are responsive on their published ports.

    Probes the containers directly via psycopg rather than through pyticdb. The pyticdb config that
    points at these ports is set up by the function-scoped :func:`tmp_pyticdb_config` fixture,
    which runs after session-scoped fixtures, so it isn't available here.
    """
    pytest.importorskip("pyticdb")
    psycopg = pytest.importorskip("psycopg")

    def is_postgres_ready(port: int) -> bool:
        try:
            with psycopg.connect(
                host="127.0.0.1",
                port=port,
                user="tglctester",
                password="password",
                dbname="postgres",
                connect_timeout=2,
            ):
                return True
        except psycopg.OperationalError:
            return False

    docker_services.wait_until_responsive(
        timeout=60.0, pause=0.5, check=lambda: is_postgres_ready(5432)
    )
    docker_services.wait_until_responsive(
        timeout=60.0, pause=0.5, check=lambda: is_postgres_ready(5433)
    )


@pytest.fixture
def pyticdb_databases(tmp_pyticdb_config, pyticdb_database_service):
    """Pytest fixture that makes all pyticdb databases, plus configuration, available."""
    pass
