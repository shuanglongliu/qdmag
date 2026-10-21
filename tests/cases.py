"""
Short end-to-end cases used for regression testing, and a runner shared by
make_reference.py and test_regression.py.

Each case is an example input with some of its dynamics options overridden, so
that the run is short (tens to hundreds of stairs) and saves the magnetization
at every stair. The runner imports qdmag from sys.path, so the same case can be
run against any version of the code (see make_reference.py).
"""
import os
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
REFERENCE = Path(__file__).resolve().parent / "reference"

# BLAS threads used for the references. Before the Condon-Shortley pool basis
# (effective_basis.canonical_pool_basis), the effective basis depended on LAPACK's
# choice of basis, which changes with the threading and the CPU; the 14f7d02
# references were made at this thread count. Since then the basis is unique, and
# the count only fixes the rounding. It must be set before NumPy is imported; see
# conftest.py.
OMP_NUM_THREADS = "4"

# name -> (example directory, {index of the dynamics block: {key: value}})
CASES = {
    "1spin_linear_Pade": ("1-spin/linear", {
        2: {"tmax": 2.0e+06, "deltat": 1.0e+04},
        3: {"nt_mag": 1, "save_rho": False, "save_drdt": True, "nt_drdt": 50}}),
    "1spin_linear_Taylor": ("1-spin/linear", {
        2: {"tmax": 5.0e+03, "deltat": 5.0e+02, "exp_propagator": "Taylor"},
        3: {"nt_mag": 1, "save_rho": False, "save_drdt": False}}),
    "1spin_linear_Krylov": ("1-spin/linear", {
        2: {"tmax": 5.0e+03, "deltat": 5.0e+02, "exp_propagator": "Krylov"},
        3: {"nt_mag": 1, "save_rho": False, "save_drdt": False}}),
    "1spin_linear_Taylor_sparse": ("1-spin/linear", {
        2: {"tmax": 5.0e+03, "deltat": 5.0e+02, "exp_propagator": "Taylor_sparse"},
        3: {"nt_mag": 1, "save_rho": False, "save_drdt": False}}),
    "1spin_linear_Krylov_sparse": ("1-spin/linear", {
        2: {"tmax": 5.0e+03, "deltat": 5.0e+02, "exp_propagator": "Krylov_sparse"},
        3: {"nt_mag": 1, "save_rho": False, "save_drdt": False}}),
    "1spin_sin_RK4": ("1-spin/sinusoidal", {
        2: {"tmax": 0.5, "deltat": 0.001, "method": "RK4"},
        3: {"nt_mag": 10, "save_rho": False, "save_drdt": False}}),
    "2spin_Jzz_Pade": ("2-spin/Jzz/staircase", {
        2: {"tmax": 2.0e+06, "deltat": 1.0e+04},
        3: {"nt_mag": 1, "save_rho": False, "save_drdt": False}}),
    "3spin_16_Pade": ("3-spin/staircase/16", {
        2: {"tmax": 2.0e+06, "deltat": 2.0e+04},
        3: {"nt_mag": 1, "save_rho": False, "save_drdt": False}}),
    "3spin_26_Pade": ("3-spin/staircase/26", {
        2: {"tmax": 1.0e+06, "deltat": 2.0e+04},
        3: {"nt_mag": 1, "save_rho": False, "save_drdt": False}}),
}


def case_input(name, device=None):
    """The input of case `name` as a dict (as read from input.yaml), optionally with `device` set."""
    example, overrides = CASES[name]
    with open(EXAMPLES / example / "input.yaml") as f:
        data = yaml.safe_load(f)
    for iblock, options in overrides.items():
        data["dynamics"][iblock].update(options)
    if device is not None:
        data["dynamics"][2]["device"] = device
    return data


def write_input(name, rundir, device=None):
    """Write the input.yaml of case `name` into rundir, optionally with `device` set."""
    data = case_input(name, device)
    rundir.mkdir(parents=True, exist_ok=True)
    with open(rundir / "input.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


@contextlib.contextmanager
def working_directory(path):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def set_up_case(name, rundir, device=None):
    """Write the input of case `name` into rundir and return the liouville object at t = tmin."""
    from qdmag.core.common import read_input, many_spins
    from qdmag.core.effective_basis import effective_basis
    from qdmag.core.liouville import liouville

    rundir = Path(rundir)
    write_input(name, rundir, device)
    with working_directory(rundir):
        Ss, nS, exchange, anisotropy, gfactor, BT_Bgrid, BT_Tgrid, dynamics, states, n_threads = read_input()
        spins = many_spins(Ss, nS, gfactor)
        eff = effective_basis(spins, exchange, anisotropy, dynamics, states)
        lio = liouville(eff, dynamics)
        lio.get_initial_rho(from_file=False)
    return lio


def run_case(name, rundir, device=None):
    """
    Run case `name` in rundir, on `device` if given (else as in the input). Returns a dict with
      'mag'    : DataFrame of t, B, Mx, My, Mz at every save point,
      'risvrho': final RI-separated vectorized density matrix,
      'drdt'   : DataFrame of the saved drho/dt, or None,
      'basis'  : dict of h0, Mz and X in the effective basis.
    """
    rundir = Path(rundir)
    lio = set_up_case(name, rundir, device)
    with working_directory(rundir):
        lio.evolve_rho()
        drdt = pd.read_csv(lio.fdrdt) if lio.save_drdt else None
        mag = pd.read_csv(lio.fmag)
    # lio.backend does not exist in the code the references are made with (14f7d02)
    risvrho = lio.backend.to_host(lio.risvrho) if hasattr(lio, "backend") else np.asarray(lio.risvrho)
    return {"mag": mag, "risvrho": risvrho, "drdt": drdt,
            "basis": {"h0": lio.h0, "Mz": lio.Mz, "X": lio.X}}


def save_reference(name, result):
    outdir = REFERENCE / name
    outdir.mkdir(parents=True, exist_ok=True)
    result["mag"].to_csv(outdir / "mag.csv", index=False)
    np.save(outdir / "risvrho_final.npy", result["risvrho"])
    np.savez_compressed(outdir / "basis.npz", **result["basis"])
    if result["drdt"] is not None:
        result["drdt"].to_csv(outdir / "drdt.csv", index=False)


def load_reference(name):
    outdir = REFERENCE / name
    drdt = outdir / "drdt.csv"
    return {"mag": pd.read_csv(outdir / "mag.csv"),
            "risvrho": np.load(outdir / "risvrho_final.npy"),
            "drdt": pd.read_csv(drdt) if drdt.exists() else None,
            "basis": dict(np.load(outdir / "basis.npz"))}
