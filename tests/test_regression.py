"""
End-to-end regression against the reference outputs in tests/reference/ (commit 14f7d02,
or for the 3-spin cases the commit of the Condon-Shortley pool basis; see make_reference.py).

Runs on the CPU, and for the Pade, Krylov and RK4 cases also on the GPU (marked gpu).
GPU and multithreaded BLAS results are not bitwise reproducible, so the contract is a
tolerance: 1e-10 muB on the magnetization and 1e-10 on the elements of rho, times the
number of stairs on the GPU and for the Pade staircase cases (see below).
"""
import numpy as np
import pytest

from cases import CASES, run_case, load_reference

# Cases whose exp_propagator / L_storage / method are available on the GPU
GPU_CASES = [name for name, (example, overrides) in CASES.items()
             if overrides[2].get("exp_propagator", "Pade") in ("Pade", "Krylov")
             and overrides[2].get("L_storage", "dense") == "dense"]
PARAMS = ([pytest.param(name, "cpu", id=name) for name in CASES] +
          [pytest.param(name, "gpu", id=name + "-gpu", marks=pytest.mark.gpu) for name in GPU_CASES])

ATOL_M = 1.0e-10
RTOL_GRID = 1.0e-12
ATOL_RHO = 1.0e-10
# drho/dt, relative to the largest element of each column. The dissipative part is a
# near-cancellation (|drdt_dis| ~ 1e-15 from terms ~ 1e-11), so it inherits a much
# larger relative error from the ~1e-12 differences in rho than the coherent part.
RTOL_DRDT = {"drdt_coh": 1.0e-8, "drdt_dis": 1.0e-6, "drdt_tot": 1.0e-8}
# On the GPU, exp(L deltat) comes from another implementation (cupyx.scipy.linalg.expm).
# Against exp(L deltat) in 80-bit extended precision, both it and scipy.linalg.expm are
# off by up to ~1e-10 on rho per stair for ||L deltat||_1 ~ 5e5 (journal Section 13), so
# the GPU tolerances are the CPU ones times the number of stairs.
GPU_FACTOR_PER_STAIR = 1
# The same holds on the CPU for the Pade staircase cases when the CPU or the BLAS build
# differs from that of the references (see reference/meta.json): with references made on
# an AMD EPYC 9655P (Zen 5) and the tests run on an EPYC 7702 (Zen 2) with the same Python,
# NumPy and SciPy, 1spin_linear_Pade differed by 3.6e-9 muB over 200 stairs and
# 3spin_16_Pade by 2.7e-10 muB over 100 stairs (the earlier, longer cases). These cases
# therefore get the per-stair tolerance on any device. Taylor, Krylov and RK4 agree to
# ~1e-12 anywhere.


def is_pade_staircase(name):
    options = CASES[name][1][2]
    return options.get("method", "staircase") == "staircase" and options.get("exp_propagator", "Pade") == "Pade"


def number_of_stairs(name, result):
    deltat = CASES[name][1][2]["deltat"]
    t = result["mag"]["t"]
    return int(round((t.iloc[-1] - t.iloc[0]) / deltat))


@pytest.mark.parametrize("name, device", PARAMS)
def test_regression(name, device, workdir):
    result = run_case(name, workdir, device)
    ref = load_reference(name)
    per_stair = device != "cpu" or is_pade_staircase(name)
    factor = GPU_FACTOR_PER_STAIR * number_of_stairs(name, result) if per_stair else 1

    # rho is compared in the effective basis, which must then be the same as for the
    # references. The Condon-Shortley pool basis is unique, so it is on any CPU and
    # thread count, up to rounding (h0 in cm^-1).
    for key in ("h0", "Mz", "X"):
        np.testing.assert_allclose(result["basis"][key], ref["basis"][key], rtol=0, atol=1e-10, err_msg=key)

    assert len(result["mag"]) == len(ref["mag"])
    # The reference CSVs were read back with the default float parser of pandas 2, which
    # can be off in the 16th significant digit (e.g. 0.00019999999999999998 is read as
    # 0.0001999999999999); pandas 3 parses exactly. rtol = 1e-12 still checks that the
    # time and field grids are the same.
    np.testing.assert_allclose(result["mag"]["t"], ref["mag"]["t"], rtol=RTOL_GRID, atol=0)
    np.testing.assert_allclose(result["mag"]["B"], ref["mag"]["B"], rtol=RTOL_GRID, atol=0)
    for col in ("Mx", "My", "Mz"):
        np.testing.assert_allclose(result["mag"][col], ref["mag"][col], rtol=0, atol=factor * ATOL_M, err_msg=col)
    np.testing.assert_allclose(result["risvrho"], ref["risvrho"], rtol=0, atol=factor * ATOL_RHO)

    if ref["drdt"] is not None:
        for col, rtol in RTOL_DRDT.items():
            scale = np.abs(ref["drdt"][col]).max()
            np.testing.assert_allclose(result["drdt"][col], ref["drdt"][col], rtol=0, atol=factor * rtol * scale, err_msg=col)

    # rho stays a density matrix (unit trace, Hermitian) at least as well as in the
    # reference. Neither holds exactly: each stair loses ~||L deltat|| * eps ~ 1e-11.
    # On the GPU the drift is ~10x larger (cupyx expm; journal Section 13), so the
    # per-stair tolerance is added there and for the Pade staircase cases.
    slack = factor * ATOL_RHO if per_stair else 1e-12
    rho, rho_ref = to_rho(result["risvrho"]), to_rho(ref["risvrho"])
    assert abs(np.trace(rho) - 1) <= 2 * abs(np.trace(rho_ref) - 1) + slack
    assert np.abs(rho - rho.conj().T).max() <= 2 * np.abs(rho_ref - rho_ref.conj().T).max() + slack


def to_rho(risvrho):
    dims = risvrho.size // 2
    dim = int(round(np.sqrt(dims)))
    return (risvrho[:dims] + 1j * risvrho[dims:]).reshape(dim, dim)
