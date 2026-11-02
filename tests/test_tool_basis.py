"""
tools/tool_basis.py suggests an effective basis from an energy cutoff (step 1 of the
procedure in Section 4.3 of the paper). For the Mn3 example, a cutoff of 25 cm-1 over
0-10 T gives the 26-state basis of Table 2 plus the S = 9/2, m_s = -9/2 state.
"""
import shutil
from pathlib import Path

import numpy as np

from cases import working_directory

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "3-spin" / "staircase" / "26" / "input.yaml"


def suggest(workdir, ecut):
    from qdmag.core.common import read_input, many_spins
    from qdmag.core.effective_basis import energy_clusters
    from qdmag.tools.tool_basis import pool_states, select_states
    shutil.copy(EXAMPLE, workdir / "input.yaml")
    with working_directory(workdir):
        Ss, nS, exchange, anisotropy, gfactor, *_ = read_input()
    spins = many_spins(Ss, nS, gfactor)
    eigen, E0, zee, S, ms = pool_states(spins, exchange)
    picked, gap_min, B_at_min = select_states(E0, zee, np.linspace(0.0, 10.0, 2001), ecut,
                                              energy_clusters(eigen.eigenvalues))
    return picked, S, ms


def test_ground_multiplet(workdir):
    picked, S, ms = suggest(workdir, 5.0)
    assert picked == list(range(6))
    np.testing.assert_allclose(S[picked], 2.5, atol=1e-8)


def test_table2_basis(workdir):
    picked, S, ms = suggest(workdir, 25.0)
    assert picked == list(range(26)) + [30]
    np.testing.assert_allclose([S[30], ms[30]], [4.5, -4.5], atol=1e-8)
