"""
Tests of the batched powder driver (core/powder_dynamics.py) against one single run per
orientation, which is what tools/tool_powder.py does.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from cases import case_input, working_directory
from qdmag.core.liouville import liouville
from qdmag.core.powder_dynamics import rotate_molecule, get_effective_basis, run_powder

# A few orientations (intrinsic ZYZ Euler angles in degrees) and weights summing to 1
ANGLES = np.array([[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 90.0, 90.0], [0.0, 35.0, 60.0], [0.0, 120.0, 200.0]])
WEIGHTS = np.array([0.3, 0.25, 0.2, 0.15, 0.1])


def short_input(name, nstairs, device=None, anisotropic_g=False):
    data = case_input(name, device)
    dyn = data["dynamics"]
    dyn[2]["tmax"] = dyn[2]["tmin"] + nstairs * dyn[2]["deltat"]
    dyn[3].update({"nt_mag": 1, "save_rho": False, "save_drdt": False})
    if anisotropic_g:
        # Mz then differs between orientations, not only h0
        for g in data["gfactor"]:
            g["gs"] = [1.20, 0.0, 0.0, 0.0, 1.30, 0.0, 0.0, 0.0, 1.50]
    return data


def single_runs(data, rundir):
    """Mz at every save point, one column per orientation, from one liouville run each."""
    columns = {}
    for i, angles in enumerate(ANGLES):
        d = rotate_molecule(data, angles)
        (rundir / str(i + 1)).mkdir(parents=True, exist_ok=True)
        with working_directory(rundir / str(i + 1)):
            lio = liouville(get_effective_basis(d), d["dynamics"])
            lio.get_initial_rho(from_file=False)
            lio.evolve_rho()
            mag = pd.read_csv(lio.fmag)
        for c in ("Mx", "My", "Mz"):
            columns["{}_{}".format(c, i + 1)] = mag[c].to_numpy()
        columns["t"] = mag["t"].to_numpy()
    return pd.DataFrame(columns)


def powder_run(data, rundir, batch_size=None):
    rundir.mkdir(parents=True, exist_ok=True)
    with working_directory(rundir):
        return run_powder(data, ANGLES, WEIGHTS, batch_size=batch_size)


def averages(single):
    n = len(ANGLES)
    return {c: np.stack([single["{}_{}".format(c, i + 1)] for i in range(n)], axis=1) @ WEIGHTS
            for c in ("Mx", "My", "Mz")}


def test_rotate_molecule_frames():
    """Every reference frame F becomes F R^T; for F = 1 that is the frame of tools/tool_powder.py."""
    data = case_input("3spin_16_Pade")
    angles = [10.0, 35.0, 60.0]
    R = Rotation.from_euler("ZYZ", angles, degrees=True).as_matrix()
    rotated = rotate_molecule(data, angles)
    for key in ("anisotropy", "gfactor", "exchange"):
        for old, new in zip(data[key], rotated[key]):
            F = np.array(old["reference_frame"]).reshape(3, 3)
            np.testing.assert_allclose(np.array(new["reference_frame"]).reshape(3, 3), F @ R.T, rtol=0, atol=1e-15)
    # tool_powder.py: emat = rotmat.T for an identity frame
    np.testing.assert_allclose(np.array(rotated["anisotropy"][0]["reference_frame"]).reshape(3, 3), R.T, atol=1e-15)
    # the input itself is left unchanged
    assert data["anisotropy"][0]["reference_frame"] == case_input("3spin_16_Pade")["anisotropy"][0]["reference_frame"]


@pytest.mark.parametrize("name, nstairs, anisotropic_g", [("1spin_linear_Pade", 20, True), ("3spin_16_Pade", 10, False)])
def test_powder_cpu_matches_single_runs(name, nstairs, anisotropic_g, workdir):
    """On the CPU, each orientation of a powder run is bit for bit its single run."""
    data = short_input(name, nstairs, "cpu", anisotropic_g)
    single = single_runs(data, workdir / "single")
    powder = powder_run(data, workdir / "powder")
    assert len(powder) == nstairs + 1
    np.testing.assert_array_equal(powder["t"], single["t"])
    for i in range(len(ANGLES)):
        np.testing.assert_array_equal(powder["Mz_{}".format(i + 1)], single["Mz_{}".format(i + 1)])
    for c, avg in averages(single).items():
        np.testing.assert_allclose(powder[c], avg, rtol=1e-14, atol=1e-16, err_msg=c)
    if anisotropic_g:
        # the orientations really differ
        assert np.ptp([single["Mz_{}".format(i + 1)].iloc[-1] for i in range(len(ANGLES))]) > 1e-3


def test_powder_batches(workdir):
    """
    Splitting the orientations into batches changes only the summation order of the average,
    so the averages agree to round-off of the largest term (atol 1e-15 muB).
    """
    data = short_input("1spin_linear_Pade", 10, "cpu", anisotropic_g=True)
    whole = powder_run(data, workdir / "whole")
    split = powder_run(data, workdir / "split", batch_size=2)
    assert list(split.columns) == list(whole.columns)
    for i in range(len(ANGLES)):
        np.testing.assert_array_equal(split["Mz_{}".format(i + 1)], whole["Mz_{}".format(i + 1)])
    for c in ("Mx", "My", "Mz"):
        np.testing.assert_allclose(split[c], whole[c], rtol=1e-14, atol=1e-15, err_msg=c)


def test_powder_rejects_unsupported(workdir):
    data = short_input("1spin_linear_Pade", 2, "cpu")
    data["dynamics"][2]["method"] = "RK4"
    with pytest.raises(ValueError, match="staircase"):
        powder_run(data, workdir)
    data = short_input("1spin_linear_Pade", 2, "cpu")
    data["dynamics"][3]["save_rho"] = True
    with pytest.raises(ValueError, match="save_rho"):
        powder_run(data, workdir)


@pytest.mark.gpu
@pytest.mark.parametrize("name, nstairs, anisotropic_g", [("1spin_linear_Pade", 20, True), ("3spin_16_Pade", 10, False)])
def test_powder_gpu_matches_single_runs(name, nstairs, anisotropic_g, workdir):
    """
    On the GPU, the batched expm agrees with scipy.linalg.expm per orientation to the per-stair
    tolerance of test_regression.py (1e-10 muB per stair; journal Section 13).
    """
    single = single_runs(short_input(name, nstairs, "cpu", anisotropic_g), workdir / "single")
    powder = powder_run(short_input(name, nstairs, "gpu", anisotropic_g), workdir / "powder")
    atol = 1e-10 * nstairs
    for i in range(len(ANGLES)):
        np.testing.assert_allclose(powder["Mz_{}".format(i + 1)], single["Mz_{}".format(i + 1)], rtol=0, atol=atol)
    for c, avg in averages(single).items():
        np.testing.assert_allclose(powder[c], avg, rtol=0, atol=atol, err_msg=c)
