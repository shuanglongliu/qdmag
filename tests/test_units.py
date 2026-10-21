"""
Unit tests of the per-stair routines against the code of commit 14f7d02 (legacy.py).
"""
import numpy as np
import pytest

import legacy
from cases import set_up_case
from qdmag.core import quantum_master as qm
from qdmag.core.constants import Tesla2wavenumber

# Small and medium systems from the regression cases
SYSTEMS = ["1spin_linear_Pade", "1spin_sin_RK4", "2spin_Jzz_Pade", "3spin_16_Pade"]
# Fields in T: zero, small, and large enough to cross levels
FIELDS = [0.0, 1.0e-3, 0.37, 2.5, 7.0]


@pytest.mark.parametrize("alpha", [2.0, 1.0, 1.5, 3.0])
@pytest.mark.parametrize("T", [0.001, 0.6, 2.0, 300.0])
def test_Phi_array_matches_scalar(T, alpha):
    """
    Phi on an array equals the scalar Phi of 14f7d02 element by element, in all three branches.
    For non-integer alpha, NumPy's array power (SIMD) and scalar power can differ by an ulp.
    """
    omega = np.concatenate([-np.logspace(-6, 3, 200), [0.0], np.logspace(-6, 3, 200)])
    expected = np.array([legacy.Phi(T, w, 1.0e-14, alpha=alpha) for w in omega])
    result = qm.Phi(T, omega, 1.0e-14, alpha=alpha)
    if alpha == int(alpha):
        np.testing.assert_array_equal(result, expected)
    else:
        np.testing.assert_allclose(result, expected, rtol=2 * np.finfo(float).eps, atol=0)


def test_Phi_scalar():
    """Phi of a scalar still returns a scalar."""
    assert np.ndim(qm.Phi(2.0, 0.5, 1.0e-14)) == 0
    assert qm.Phi(2.0, 0.5, 1.0e-14) == legacy.Phi(2.0, 0.5, 1.0e-14)


@pytest.mark.parametrize("name", SYSTEMS)
def test_Rhbar_matches_legacy(name, workdir):
    lio = set_up_case(name, workdir)
    for B in FIELDS:
        h = lio.h0 - Tesla2wavenumber * B * lio.Mz
        expected = legacy.get_Rhbar(h, lio.X, lio.I0, lio.T, alpha=lio.alpha)
        np.testing.assert_array_equal(qm.get_Rhbar(h, lio.X, lio.I0, lio.T, alpha=lio.alpha), expected)


@pytest.mark.parametrize("name", SYSTEMS)
def test_L_matches_legacy(name, workdir):
    """
    The dense L, and its coherent and dissipative parts LA and LC, agree with 14f7d02.
    The dissipative part is unchanged bit for bit. The Zeeman term is now B times a
    precomputed superoperator, which rounds differently, so LA and L agree to round-off
    of the largest element of LA.
    """
    lio = set_up_case(name, workdir)
    old = legacy.LegacyLiouville(lio)
    for B in FIELDS:
        lio.update_L(B)
        old.update_L_under_magnetic_field(B)
        LA, LC = lio.get_LA_LC()
        np.testing.assert_array_equal(LC, old.LC)
        atol = 4 * np.finfo(float).eps * np.abs(old.LA).max()
        np.testing.assert_allclose(LA, old.LA, rtol=0, atol=atol)
        np.testing.assert_allclose(lio.L, old.L, rtol=0, atol=atol)
        np.testing.assert_array_equal(lio.L, LA + LC)


def expm_extended(A):
    """exp(A) in 80-bit extended precision (trace shift, scaling, 30-term Taylor, squaring)."""
    A = A.astype(np.longdouble)
    n = A.shape[0]
    mu = np.trace(A) / n
    A = A - mu * np.eye(n, dtype=np.longdouble)
    s = max(0, int(np.ceil(np.log2(np.abs(A).sum(axis=0).max() / 0.25))))
    A = A / np.longdouble(2)**s
    E = term = np.eye(n, dtype=np.longdouble)
    for k in range(1, 31):
        term = term @ A / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E * np.exp(mu)


@pytest.mark.parametrize("name", ["1spin_linear_Pade", "2spin_Jzz_Pade", "3spin_16_Pade", "3spin_26_Pade"])
def test_expm_matches_scipy(name, workdir):
    """
    core/expm.py (the GPU expm, run here with NumPy) never squares more often than
    scipy.linalg.expm, and propagates rho as accurately: both are within ~1e-11 of an
    extended-precision exp(L deltat) rho (journal Section 14). The full matrices can differ
    far more, up to 6e-8 relative for 3-spin/16 at 5 T, in parts of exp(L deltat) that do
    not act on density matrices.
    """
    import scipy.linalg
    from scipy.linalg._matfuncs_expm import pick_pade_structure
    from qdmag.core import expm as qe

    lio = set_up_case(name, workdir)
    for B in [0.0, 2.5]:
        lio.update_L(B)
        A = lio.L * lio.deltat
        Am = np.empty((5,) + A.shape)
        Am[0] = A
        m, s = pick_pade_structure(Am)
        A2 = A @ A
        A4 = A2 @ A2
        assert qe.scaling_exponent(A, np, A4, A2 @ A4) <= s
        np.testing.assert_allclose(qe.expm(A) @ lio.risvrho, scipy.linalg.expm(A) @ lio.risvrho, rtol=0, atol=1e-10)


@pytest.mark.parametrize("B", [0.0, 1.0, 2.5, 5.0])
def test_expm_vs_extended_precision(B, workdir):
    """core/expm.py against exp(L deltat) in extended precision, for the small 2-spin system."""
    from qdmag.core import expm as qe

    lio = set_up_case("2spin_Jzz_Pade", workdir)
    lio.update_L(B)
    A = lio.L * lio.deltat
    exact = expm_extended(A)
    E = qe.expm(A)
    assert float(np.abs(E - exact).max() / np.abs(exact).max()) < 1e-11
    exact_v = (exact @ lio.risvrho.astype(np.longdouble)).astype(np.float64)
    np.testing.assert_allclose(E @ lio.risvrho, exact_v, rtol=0, atol=1e-12)


def test_expm_small_norm():
    """Without scaling (||A||_1 < theta_13) the approximant is accurate to round-off."""
    import scipy.linalg
    from qdmag.core import expm as qe
    A = np.random.default_rng(0).standard_normal((40, 40)) / 40
    assert qe.scaling_exponent(A) == 0
    np.testing.assert_allclose(qe.expm(A), scipy.linalg.expm(A), rtol=1e-13, atol=1e-14)


def test_kron_matches_numpy():
    """liouville.kron is np.kron for 2D arrays, bit for bit, and a batch of np.kron for stacks."""
    from qdmag.core.liouville import kron
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 5, 4)) + 1j * rng.standard_normal((3, 5, 4))
    B = rng.standard_normal((2, 6))
    np.testing.assert_array_equal(kron(A[0], B), np.kron(A[0], B))
    np.testing.assert_array_equal(kron(A, B), np.stack([np.kron(a, B) for a in A]))


def test_expm_stack_matches_expm(workdir):
    """expm_stack equals expm of each member, for L deltat of several fields (different s)."""
    from qdmag.core import expm as qe
    lio = set_up_case("2spin_Jzz_Pade", workdir)
    As = []
    for B in [0.0, 1.0, 2.5, 5.0]:
        lio.update_L(B)
        As.append(lio.L * lio.deltat)
    A = np.stack(As)
    E = qe.expm_stack(A)
    for b in range(len(As)):
        np.testing.assert_allclose(E[b], qe.expm(As[b]), rtol=0, atol=1e-13 * np.abs(E[b]).max())
