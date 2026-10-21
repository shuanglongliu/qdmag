"""
Generate the reference outputs in tests/reference/ for test_regression.py.

The references must come from the code before the GPU refactor (commit 14f7d02).
Export that commit as a package named qdmag and put its parent first on PYTHONPATH:

    git archive --prefix=qdmag/ 14f7d02 | tar -x -C /path/to/ref
    cd /path/to/workdir
    PYTHONPATH=/path/to/ref python /path/to/repo/tests/make_reference.py [case ...]

Runs are done in the current working directory (one subdirectory per case);
only the reference files are written into the repository.
"""
import os
import sys
import json
import time
import platform
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent))
from cases import OMP_NUM_THREADS

# Pin the BLAS threads before NumPy is imported (see cases.py)
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = OMP_NUM_THREADS

import numpy as np
import scipy

from cases import CASES, REFERENCE, run_case, save_reference

if __name__ == "__main__":
    import qdmag
    names = sys.argv[1:] or list(CASES)
    meta = {"qdmag": str(Path(qdmag.__file__).resolve().parent),
            "omp_num_threads": OMP_NUM_THREADS,
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "node": platform.node(), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": {}}
    meta_file = REFERENCE / "meta.json"
    if meta_file.exists():
        meta["cases"] = json.loads(meta_file.read_text())["cases"]
    for name in names:
        start = time.time()
        result = run_case(name, Path.cwd() / name)
        save_reference(name, result)
        meta["cases"][name] = {"seconds": round(time.time() - start, 1), "nsave": len(result["mag"])}
        print("{}: {:.1f} s".format(name, time.time() - start), file=sys.stderr)
    meta_file.write_text(json.dumps(meta, indent=2) + "\n")
