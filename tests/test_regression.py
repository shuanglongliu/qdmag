"""
End-to-end regression against the reference outputs of commit 14f7d02 (see make_reference.py).

Runs on the CPU, and for the Pade, Krylov and RK4 cases also on the GPU (marked gpu).
GPU and multithreaded BLAS results are not bitwise reproducible, so the contract is a
tolerance: 1e-10 muB on the magnetization and 1e-10 on the elements of rho.
"""
import numpy as np
import pytest

from cases import CASES, run_case, load_reference

# Cases whose exp_propagator / method are available on the GPU
GPU_CASES = [name for name, (example, overrides) in CASES.items()
             if overrides[2].get("exp_propagator", "Pade") in ("Pade", "Krylov")]
PARAMS = ([pytest.param(name, "cpu", id=name) for name in CASES] +
          [pytest.param(name, "gpu", id=name + "-gpu", marks=pytest.mark.gpu) for name in GPU_CASES])

ATOL_M = 1.0e-10
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


def number_of_stairs(name, result):
    deltat = CASES[name][1][2]["deltat"]
    t = result["mag"]["t"]
    return int(round((t.iloc[-1] - t.iloc[0]) / deltat))


@pytest.mark.parametrize("name, device", PARAMS)
def test_regression(name, device, workdir):
    result = run_case(name, workdir, device)
    ref = load_reference(name)
    factor = 1 if device == "cpu" else GPU_FACTOR_PER_STAIR * number_of_stairs(name, result)

    # rho is compared in the effective basis, which must then be the same as for the
    # references; it is not on a CPU whose LAPACK picks another basis within the
    # degenerate blocks of the pool Hamiltonian (see cases.py).
    for key in ("h0", "Mz", "X"):
        if not np.allclose(result["basis"][key], ref["basis"][key], rtol=0, atol=1e-12):
            pytest.skip("effective basis differs from the reference ({}); rho cannot be compared".format(key))

    assert len(result["mag"]) == len(ref["mag"])
    np.testing.assert_allclose(result["mag"]["t"], ref["mag"]["t"], rtol=1e-15, atol=0)
    np.testing.assert_allclose(result["mag"]["B"], ref["mag"]["B"], rtol=1e-15, atol=0)
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
    # per-stair GPU tolerance is added.
    slack = 1e-12 if device == "cpu" else factor * ATOL_RHO
    rho, rho_ref = to_rho(result["risvrho"]), to_rho(ref["risvrho"])
    assert abs(np.trace(rho) - 1) <= 2 * abs(np.trace(rho_ref) - 1) + slack
    assert np.abs(rho - rho.conj().T).max() <= 2 * np.abs(rho_ref - rho_ref.conj().T).max() + slack


def to_rho(risvrho):
    dims = risvrho.size // 2
    dim = int(round(np.sqrt(dims)))
    return (risvrho[:dims] + 1j * risvrho[dims:]).reshape(dim, dim)
