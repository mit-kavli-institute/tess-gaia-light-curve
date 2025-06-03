from tglc.utils import _optional_deps


def test_has_pyticdb():
    assert hasattr(_optional_deps, "HAS_PYTICDB")

def test_has_cupy():
    assert hasattr(_optional_deps, "HAS_CUPY")
