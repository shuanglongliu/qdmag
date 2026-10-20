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
