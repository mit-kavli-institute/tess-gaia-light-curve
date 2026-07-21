"""
This file tests a full execution of the TGLC pipeline, designed to mimic the way it would be run in
practice. This end-to-end/integration tests verifies that the data products produced by each step
are compatible with subsequent steps.
"""

from pathlib import Path
import re
import shutil
import sys

from astropy.table import QTable
import h5py
import pytest

from tglc.__main__ import tglc_main
from tglc.io import read_cutout_fits, read_epsf_fits
from tglc.utils.manifest import get_tic_id_shard_path


TEST_ORBIT = 185
"""The orbit used for testing purposes. Chosen arbitrarily because it was one of the orbits used
for actual production testing."""


@pytest.fixture
def tmp_orbit_directory(tmp_path: Path, sample_ffis: Path) -> Path:
    """Set up an orbit directory with whatever sample FFIs are available"""
    orbit_directory = tmp_path / f"orbit-{TEST_ORBIT}" / "ffi"
    (orbit_directory / "run").mkdir(parents=True)
    (orbit_directory / "catalogs").mkdir()
    for camera in range(1, 5):
        for ccd in range(1, 5):
            ccd_directory = orbit_directory / f"cam{camera}" / f"ccd{ccd}"
            ccd_directory.mkdir(parents=True)
            for subdir in ["ffi", "source", "epsf", "LC"]:
                (ccd_directory / subdir).mkdir()
    for file in sample_ffis.glob("*.fits"):
        cam_ccd_match = re.search(r"cam([1-4])-ccd([1-4])", file.stem)
        if cam_ccd_match:
            camera = cam_ccd_match.group(1)
            ccd = cam_ccd_match.group(2)
            shutil.copy(file, orbit_directory / f"cam{camera}" / f"ccd{ccd}" / "ffi")
    return orbit_directory


def test_orbit_directory_setup(tmp_orbit_directory: Path):
    assert (tmp_orbit_directory / "run").is_dir()
    assert (tmp_orbit_directory / "catalogs").is_dir()
    for camera in range(1, 5):
        for ccd in range(1, 5):
            for subdirectory in ["ffi", "source", "epsf", "LC"]:
                assert (tmp_orbit_directory / f"cam{camera}" / f"ccd{ccd}" / subdirectory).is_dir()
    assert len(list((tmp_orbit_directory / "cam1/ccd1/ffi").iterdir())) == 5


def test_catalogs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_orbit_directory: Path,
    pyticdb_databases,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tglc",
            "catalogs",
            "--tglc-data-dir",
            str(tmp_path.resolve()),
            "--orbit",
            str(TEST_ORBIT),
            "--ccd",
            "1,1",
        ],
    )
    tglc_main()

    catalogs_directory = tmp_orbit_directory / "catalogs"
    catalog_files = list(catalogs_directory.iterdir())
    assert len(catalog_files) == 2
    for file in catalog_files:
        catalog = QTable.read(file)
        assert len(catalog) > 0


def test_full_pipeline_with_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_orbit_directory: Path,
    pyticdb_databases,
):
    ccd_directory = tmp_orbit_directory / "cam1" / "ccd1"

    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "tglc",
                "catalogs",
                "--tglc-data-dir",
                str(tmp_path.resolve()),
                "--orbit",
                str(TEST_ORBIT),
                "--ccd",
                "1,1",
            ],
        )
        tglc_main()

    catalogs_directory = tmp_orbit_directory / "catalogs"
    catalog_files = list(catalogs_directory.iterdir())
    assert len(catalog_files) == 2
    for file in catalog_files:
        catalog = QTable.read(file)
        assert len(catalog) > 0

    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "tglc",
                "cutouts",
                "--tglc-data-dir",
                str(tmp_path.resolve()),
                "--orbit",
                str(TEST_ORBIT),
                "--ccd",
                "1,1",
                "--cutout",
                "0,0",
            ],
        )
        tglc_main()

    source_directory = ccd_directory / "source"
    source_files = list(source_directory.iterdir())
    assert len(source_files) == 1
    for source_file in source_files:
        assert source_file.suffix == ".fits"
        read_cutout_fits(source_file)

    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "tglc",
                "epsfs",
                "--tglc-data-dir",
                str(tmp_path.resolve()),
                "--orbit",
                str(TEST_ORBIT),
                "--ccd",
                "1,1",
                "--cutout",
                "0,0",
            ],
        )
        tglc_main()

    epsf_directory = ccd_directory / "epsf"
    epsf_files = list(epsf_directory.iterdir())
    assert len(epsf_files) == 1
    for epsf_file in epsf_files:
        assert epsf_file.suffix == ".fits"
        read_epsf_fits(epsf_file)

    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "tglc",
                "lightcurves",
                "--tglc-data-dir",
                str(tmp_path.resolve()),
                "--orbit",
                str(TEST_ORBIT),
                "--ccd",
                "1,1",
                "--cutout",
                "0,0",
            ],
        )
        tglc_main()

    lc_directory = ccd_directory / "LC"
    lc_files = sorted(lc_directory.rglob("*.h5"))
    assert len(lc_files) > 0
    # Light curves should be sharded by zero-padded TIC ID, not flat in the LC root
    assert not any(path.suffix == ".h5" for path in lc_directory.iterdir())
    for lc_file in lc_files:
        assert lc_file.parent == lc_directory / get_tic_id_shard_path(int(lc_file.stem))
        with h5py.File(lc_file) as lc_data:
            assert "LightCurve" in lc_data.keys()
            assert sorted(lc_data["LightCurve/AperturePhotometry"].keys()) == [
                "LargeAperture",
                "PrimaryAperture",
                "SmallAperture",
            ]

    # Re-running with a restricted aperture selection should only produce the selected apertures
    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "tglc",
                "lightcurves",
                "--tglc-data-dir",
                str(tmp_path.resolve()),
                "--orbit",
                str(TEST_ORBIT),
                "--ccd",
                "1,1",
                "--cutout",
                "0,0",
                "--apertures",
                "primary",
                "--replace",
            ],
        )
        tglc_main()

    lc_files = sorted(lc_directory.rglob("*.h5"))
    assert len(lc_files) > 0
    for lc_file in lc_files:
        with h5py.File(lc_file) as lc_data:
            assert list(lc_data["LightCurve/AperturePhotometry"].keys()) == ["PrimaryAperture"]


def test_full_pipeline_with_all(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_orbit_directory: Path,
    pyticdb_databases,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tglc",
            "all",
            "--tglc-data-dir",
            str(tmp_path.resolve()),
            "--orbit",
            str(TEST_ORBIT),
            "--ccd",
            "1,1",
            "--cutout",
            "0,0",
        ],
    )
    tglc_main()

    catalogs_directory = tmp_orbit_directory / "catalogs"
    catalog_files = list(catalogs_directory.iterdir())
    assert len(catalog_files) == 2
    for file in catalog_files:
        catalog = QTable.read(file)
        assert len(catalog) > 0

    ccd_directory = tmp_orbit_directory / "cam1" / "ccd1"

    source_directory = ccd_directory / "source"
    source_files = list(source_directory.iterdir())
    assert len(source_files) == 1
    for source_file in source_files:
        assert source_file.suffix == ".fits"
        read_cutout_fits(source_file)

    epsf_directory = ccd_directory / "epsf"
    epsf_files = list(epsf_directory.iterdir())
    assert len(epsf_files) == 1
    for epsf_file in epsf_files:
        assert epsf_file.suffix == ".fits"
        read_epsf_fits(epsf_file)

    lc_directory = ccd_directory / "LC"
    lc_files = sorted(lc_directory.rglob("*.h5"))
    assert len(lc_files) > 0
    for lc_file in lc_files:
        with h5py.File(lc_file) as lc_data:
            assert "LightCurve" in lc_data.keys()


def test_full_pipeline_after_migration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_orbit_directory: Path,
    pyticdb_databases,
):
    """Run cutouts + epsfs, downgrade outputs to legacy pickle/.npy, then migrate
    forward and verify `tglc lightcurves` still produces light curves.
    """
    import pickle

    import numpy as np

    from tglc.io import migrate_cutout_pickle, migrate_epsf_npy

    ccd_directory = tmp_orbit_directory / "cam1" / "ccd1"

    for tglc_command in ("catalogs", "cutouts", "epsfs"):
        with monkeypatch.context() as m:
            m.setattr(
                sys,
                "argv",
                [
                    "tglc",
                    tglc_command,
                    "--tglc-data-dir",
                    str(tmp_path.resolve()),
                    "--orbit",
                    str(TEST_ORBIT),
                    "--ccd",
                    "1,1",
                ]
                + (["--cutout", "0,0"] if tglc_command != "catalogs" else []),
            )
            tglc_main()

    source_fits = next((ccd_directory / "source").glob("source_*.fits"))
    epsf_fits = next((ccd_directory / "epsf").glob("epsf_*.fits"))

    # Downgrade outputs to legacy format
    source_pkl = source_fits.with_suffix(".pkl")
    epsf_npy = epsf_fits.with_suffix(".npy")
    cutout = read_cutout_fits(source_fits)
    with source_pkl.open("wb") as fp:
        pickle.dump(cutout, fp, pickle.HIGHEST_PROTOCOL)
    epsf_array, epsf_metadata = read_epsf_fits(epsf_fits)
    np.save(epsf_npy, epsf_array)
    source_fits.unlink()
    epsf_fits.unlink()

    # Migrate forward using the library helpers under test
    migrate_cutout_pickle(source_pkl, delete_original=True)
    migrate_epsf_npy(
        epsf_npy,
        psf_size=epsf_metadata["psf_size"],
        oversample=epsf_metadata["oversample"],
        orbit=epsf_metadata["orbit"],
        sector=epsf_metadata["sector"],
        camera=epsf_metadata["camera"],
        ccd=epsf_metadata["ccd"],
        cutout_x=epsf_metadata["cutout_x"],
        cutout_y=epsf_metadata["cutout_y"],
        delete_original=True,
    )
    assert not source_pkl.exists()
    assert not epsf_npy.exists()

    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "tglc",
                "lightcurves",
                "--tglc-data-dir",
                str(tmp_path.resolve()),
                "--orbit",
                str(TEST_ORBIT),
                "--ccd",
                "1,1",
                "--cutout",
                "0,0",
            ],
        )
        tglc_main()

    lc_files = sorted((ccd_directory / "LC").rglob("*.h5"))
    assert len(lc_files) > 0
    for lc_file in lc_files:
        with h5py.File(lc_file) as lc_data:
            assert "LightCurve" in lc_data.keys()
