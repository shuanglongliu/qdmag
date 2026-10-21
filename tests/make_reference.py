"""
Generate the reference outputs in tests/reference/ for test_regression.py.

The references come from the code before the GPU refactor (commit 14f7d02), except for
the 3-spin cases: their effective basis changed with the Condon-Shortley pool basis
(core/effective_basis.canonical_pool_basis), and their references come from the commit
that introduced it (recorded per case in reference/meta.json). Export the commit as a
package named qdmag and put its parent first on PYTHONPATH:

    git archive --prefix=qdmag/ 14f7d02 | tar -x -C /path/to/ref
    cd /path/to/workdir
    PYTHONPATH=/path/to/ref python /path/to/repo/tests/make_reference.py --commit 14f7d02 [case ...]

Runs are done in the current working directory (one subdirectory per case);
only the reference files are written into the repository.
"""
import os
import sys
import json
import argparse
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

def cpu_model():
    """CPU model name from /proc/cpuinfo (the effective basis depends on it, see cases.py)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the regression references.")
    parser.add_argument("--commit", required=True, help="commit of the exported code on PYTHONPATH")
    parser.add_argument("cases", nargs="*", help="cases to generate (default: all)")
    args = parser.parse_args()
    run = {"commit": args.commit, "omp_num_threads": OMP_NUM_THREADS,
           "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "cpu": cpu_model(), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta_file = REFERENCE / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {"cases": {}}
    for name in args.cases or list(CASES):
        start = time.time()
        result = run_case(name, Path.cwd() / name)
        save_reference(name, result)
        meta["cases"][name] = dict(run, seconds=round(time.time() - start, 1), nsave=len(result["mag"]))
        print("{}: {:.1f} s".format(name, time.time() - start), file=sys.stderr)
    meta_file.write_text(json.dumps(meta, indent=2) + "\n")
