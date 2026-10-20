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
