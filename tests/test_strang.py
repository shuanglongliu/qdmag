"""
Tests of the 'Strang' propagator (liouville.propagate_risvrho_Strang): the symmetric splitting
of the coherent and dissipative parts in Hilbert space. It is approximate, so it is compared
with Pade runs made here, on the same basis and stairs, within a physical tolerance rather
than against the stored references.
"""
import numpy as np
import pytest

from cases import run_case, set_up_case

# Strang vs Pade on these short cases differs by ~1e-8 muB or less; production sweeps of
# Mn3 (10^4-2x10^5 stairs, 16-42 states) agree to < 1e-6 muB.
ATOL_M = 1.0e-6

CASES_STRANG = ["1spin_linear_Pade", "2spin_Jzz_Pade", "3spin_16_Pade"]
NO_DRDT = {3: {"save_drdt": False}}


def strang(**options):
    return {2: {"exp_propagator": "Strang", **options}, 3: {"save_drdt": False}}


@pytest.mark.parametrize("name", CASES_STRANG)
def test_strang_vs_pade(name, workdir):
    ref = run_case(name, workdir / "pade", extra=NO_DRDT)
    res = run_case(name, workdir / "strang", extra=strang())
    np.testing.assert_array_equal(res["mag"]["t"], ref["mag"]["t"])
    for col in ("Mx", "My", "Mz"):
        np.testing.assert_allclose(res["mag"][col], ref["mag"][col], rtol=0, atol=ATOL_M, err_msg=col)
    dims = res["risvrho"].size // 2
    dim = int(round(np.sqrt(dims)))
    rho = (res["risvrho"][:dims] + 1j * res["risvrho"][dims:]).reshape(dim, dim)
    assert abs(np.trace(rho) - 1) < 1e-12
    assert np.abs(rho - rho.conj().T).max() < 1e-12


def test_strang_substeps_and_order(workdir):
    """More substeps and a higher order change the result by no more than the tolerance."""
    a = run_case("3spin_16_Pade", workdir / "a", extra=strang())
    b = run_case("3spin_16_Pade", workdir / "b", extra=strang(strang_order=4, strang_nsub=0))
    np.testing.assert_allclose(a["mag"]["Mz"], b["mag"]["Mz"], rtol=0, atol=ATOL_M)


def test_strang_does_not_form_L(workdir):
    lio = set_up_case("3spin_16_Pade", workdir, extra=strang())
    assert not hasattr(lio, "L")


@pytest.mark.parametrize("extra, message", [
    ({3: {"save_drdt": True}}, "save_drdt"),
    ({2: {"method": "RK4"}}, "staircase"),
    ({2: {"strang_order": 0}}, "strang_order"),
    ({2: {"strang_nsub": -1}}, "strang_nsub"),
])
def test_strang_rejects(extra, message, workdir):
    opts = strang()
    for k, v in extra.items():
        opts.setdefault(k, {}).update(v)
    with pytest.raises(ValueError, match=message):
        set_up_case("3spin_16_Pade", workdir, extra=opts)
