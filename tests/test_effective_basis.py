"""
The canonical (Condon-Shortley) pool basis of effective_basis: unique, independent of the
basis LAPACK returns, and so reproducible across BLAS thread counts.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cases import case_input
from qdmag.core.common import many_spins, eigen_handy, transform_O, get_h_exchange_iso, get_h_Zeeman_iso
from qdmag.core.constants import factor_ex
from qdmag.core import effective_basis as eb

G2 = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
FRAME = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def iso_exchange(pairs):
    return [{"pair": list(p), "coupling_matrix": [J, 0, 0, 0, J, 0, 0, 0, J], "reference_frame": FRAME}
            for p, J in pairs.items()]


# name -> (spins, exchange). The equilateral triangle has degenerate top states of the same S
# (two S = 1 triplets at each M); without exchange every (S, M) is degenerate.
SYSTEMS = {
    "1spin": ([8.0], []),
    "2spin_Jzz": (case_input("2spin_Jzz_Pade")["spins"], case_input("2spin_Jzz_Pade")["exchange"]),
    "3spin_16": (case_input("3spin_16_Pade")["spins"], case_input("3spin_16_Pade")["exchange"]),
    "triangle_S1": ([1.0, 1.0, 1.0], iso_exchange({(1, 2): -3.0, (2, 3): -3.0, (1, 3): -3.0})),
    "4spin_mixed": ([2.5, 2.0, 1.5, 0.5], iso_exchange({(1, 2): -2.0, (2, 3): -1.3, (3, 4): 0.7, (1, 4): -0.4})),
    "3spin_no_exchange": ([0.5, 1.0, 1.5], []),
}


def pool(name):
    """spins, the pool Hamiltonian, and its eigh, as in effective_basis.get_pool_and_pick_states."""
    Ss, exchange = SYSTEMS[name]
    spins = many_spins(Ss, len(Ss), [{"site": i + 1, "gs": G2, "reference_frame": FRAME} for i in range(len(Ss))])
    if spins.nS == 1:
        h = spins.Sv_tot[2]
    else:
        h = get_h_exchange_iso(spins, exchange, factor_ex) + get_h_Zeeman_iso(spins, [0.0, 0.0, 1e-4], "cartesian")
    return spins, h, eigen_handy(h)


@pytest.mark.parametrize("name", SYSTEMS)
def test_canonical_pool_basis(name):
    """Orthonormal eigenvectors of h, S^2 and Sz, with S- non-negative (Condon-Shortley)."""
    spins, h, eigen = pool(name)
    E_eigh = eigen.eigenvalues.copy()
    V = eb.canonical_pool_basis(h, eigen, spins).eigenvectors
    n = V.shape[0]
    scale = max(1.0, np.abs(E_eigh).max())
    np.testing.assert_allclose(V.conj().T @ V, np.eye(n), rtol=0, atol=1e-12)
    np.testing.assert_allclose(h @ V, V * eigen.eigenvalues, rtol=0, atol=1e-10 * scale)
    np.testing.assert_allclose(eigen.eigenvalues, E_eigh, rtol=0, atol=1e-10 * scale)
    for O in (spins.Sv_tot[2], spins.S2_tot):
        O_pool = transform_O(O, eigen)
        np.testing.assert_allclose(O_pool, np.diag(np.diag(O_pool)), rtol=0, atol=1e-10)
    Sm_pool = transform_O(spins.Sv_tot[0] - 1j * spins.Sv_tot[1], eigen)
    assert np.abs(Sm_pool.imag).max() < 1e-10
    assert Sm_pool.real.min() > -1e-10


@pytest.mark.parametrize("name", SYSTEMS)
def test_canonical_pool_basis_ignores_lapack_gauge(name):
    """Random phases, and random unitaries within degenerate eigenspaces, of the eigh basis change nothing."""
    spins, h, eigen = pool(name)
    rng = np.random.default_rng(7)
    other = copy.deepcopy(eigen)
    W = other.eigenvectors * np.exp(2j * np.pi * rng.random(other.dim))
    for c in eb.energy_clusters(other.eigenvalues):
        if len(c) > 1:
            Q, _ = np.linalg.qr(rng.normal(size=(len(c), len(c))) + 1j * rng.normal(size=(len(c), len(c))))
            W[:, c] = W[:, c] @ Q
    other.eigenvectors = W
    np.testing.assert_array_equal(eb.canonical_pool_basis(h, other, spins).eigenvectors,
                                  eb.canonical_pool_basis(h, eigen, spins).eigenvectors)


def test_single_spin_pool_is_unchanged():
    """For one spin the pool is eigh(Sz), already the standard basis: bit for bit the same."""
    spins, h, eigen = pool("1spin")
    expected = eigen.eigenvectors.copy()
    np.testing.assert_array_equal(eb.canonical_pool_basis(h, eigen, spins).eigenvectors, expected)


def test_partial_degenerate_selection_warns(capsys):
    """Selecting one state of a degenerate pair is flagged."""
    data = case_input("3spin_26_Pade")
    spins = many_spins(data["spins"], len(data["spins"]), data["gfactor"])
    data["states"] = [s for s in data["states"] if s != 21]
    eb.effective_basis(spins, data["exchange"], data["anisotropy"], data["dynamics"], data["states"])
    assert "contain 1 of the 2 degenerate pool states [20, 21]" in capsys.readouterr().out


SCRIPT = """
import json, sys
import numpy as np
sys.path[:0] = {paths!r}
from cases import case_input
from qdmag.core.common import many_spins
from qdmag.core.effective_basis import effective_basis
d = case_input("{name}")
spins = many_spins(d["spins"], len(d["spins"]), d["gfactor"])
eff = effective_basis(spins, d["exchange"], d["anisotropy"], d["dynamics"], d["states"])
print(json.dumps({{k: [np.real(v).tolist(), np.imag(v).tolist()] for k, v in
                  (("h0", eff.h0_eff), ("Mz", eff.Mz_eff), ("X", eff.X_eff))}}))
"""


@pytest.mark.parametrize("name", ["3spin_16_Pade", "3spin_26_Pade"])
def test_effective_basis_independent_of_blas_threads(name):
    """h0, Mz and X in the effective basis are the same with 1 and 4 BLAS threads (they were not before)."""
    tests = Path(__file__).resolve().parent
    code = SCRIPT.format(paths=[str(tests.parents[1]), str(tests)], name=name)
    results = []
    for threads in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
        results.append(json.loads(out.strip().splitlines()[-1]))
    for key in ("h0", "Mz", "X"):
        np.testing.assert_allclose(np.array(results[0][key]), np.array(results[1][key]), rtol=0, atol=1e-12, err_msg=key)
