import os
import sys
from pathlib import Path

import pytest

# Make `qdmag` importable from the repository checkout, and the test helpers
# (cases.py, legacy.py) importable from the tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import OMP_NUM_THREADS

# Use the BLAS thread count of the references, before anything imports NumPy (see cases.py)
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = OMP_NUM_THREADS


@pytest.fixture
def workdir(tmp_path, request):
    """
    Directory for the files a test writes. Set QDMAG_TEST_DIR to keep runs out of
    the default temporary directory, e.g. on a scratch file system.
    """
    root = os.environ.get("QDMAG_TEST_DIR")
    if root is None:
        return tmp_path
    path = Path(root) / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def gpu_available():
    """True if CuPy imports and sees at least one CUDA device."""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs CuPy and a CUDA device; skipped otherwise")


def pytest_collection_modifyitems(config, items):
    if gpu_available():
        return
    skip = pytest.mark.skip(reason="CuPy or a CUDA device is not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
