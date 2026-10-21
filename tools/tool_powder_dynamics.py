"""
Powder averaged spin dynamics in one job: all orientations of points_and_weights.txt are
evolved together, on the CPU or a GPU (core/powder_dynamics.py). For GPU runs this replaces
the one-job-per-orientation workflow of tool_powder.py (create_directories, get_inputs,
submit_job_array, get_M_dy_avg).

Run it in a directory that holds
  input.yaml               the input of a single run (as for tool_staircase.py), with the
                           molecule in its reference orientation. Set device: gpu in the
                           third dynamics block to run on a GPU. Supported: the staircase
                           method with exp_propagator Pade, saving the magnetization only.
  points_and_weights.txt   from tool_quadrature.py (Euler angles in degrees, weights).

    python tool_powder_dynamics.py [--batch-size N]

The result is written to output/.../magnetometry_powder/t...csv, with t, B, the powder
averaged Mx, My, Mz, and Mz_1, ..., Mz_n of each orientation, numbered as the rows of
points_and_weights.txt (as the directories of tool_powder.py are).

--batch-size: orientations evolved together (default: all on the CPU, as many as fit into
70% of the free memory on a GPU). Larger batches use the GPU better.

Every tensor rotates with the molecule (anisotropy, g and exchange frames); tool_powder.py
rotates the anisotropy only, which agrees when g and the exchange are isotropic.
"""
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import yaml

# Import qdmag from the parent of the repository
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qdmag.core.powder_dynamics import run_powder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=None, help="orientations evolved together")
    args = parser.parse_args()

    with open("input.yaml") as f:
        data = yaml.safe_load(f)
    points_and_weights = np.loadtxt("points_and_weights.txt", comments="#", ndmin=2)
    euler_angles, weights = points_and_weights[:, :3], points_and_weights[:, 3]

    start = time.time()
    run_powder(data, euler_angles, weights, batch_size=args.batch_size)
    print("Powder average over {} orientations took {:.1f} s".format(len(weights), time.time() - start))
