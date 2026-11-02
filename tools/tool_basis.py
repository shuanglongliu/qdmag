"""
Suggest an effective basis: the pool states that come within an energy cutoff of the
instantaneous ground state for some field of the sweep.

This is step 1 of the procedure for choosing an effective basis (Section 4.3 of the paper):
include all pool states whose energy E_i + g muB B m_s,i comes within E_cut of the
instantaneous ground state for some field B in the sweep. The energies are those of the pool
Hamiltonian (isotropic exchange plus the Zeeman term of the actual g tensors along z, taken
to first order in the pool basis), so the zero-field splitting is not included. Degenerate
pool states are kept together. The result should then be checked against the full
Hamiltonian (steps 2 and 3): compare the Zeeman diagram and the equilibrium magnetization,
and enlarge the basis until the nonequilibrium magnetization no longer changes.

Usage, in the directory of input.yaml (multispin systems only):
    python -m qdmag.tools.tool_basis --ecut 30
    python -m qdmag.tools.tool_basis --ecut 30 --bmin 0 --bmax 10 --nb 2001

--ecut is in cm-1. The field range defaults to [Bmin, Bmax] of the first BT_Bgrid entry.
The list printed after "states:" can be pasted into input.yaml.
"""
import argparse
import numpy as np
from qdmag.core.common import read_input, many_spins, eigen_handy, transform_O
from qdmag.core.common import get_h_exchange_iso, get_h_Zeeman_iso, get_h_Zeeman
from qdmag.core.constants import factor_ex
from qdmag.core.effective_basis import canonical_pool_basis, energy_clusters


def pool_states(spins, exchange):
    """Pool basis as in effective_basis: eigenstates of the isotropic exchange plus a small field along z."""
    h = get_h_exchange_iso(spins, exchange, factor_ex) + get_h_Zeeman_iso(spins, [0.0, 0.0, 1e-4], 'cartesian')
    eigen = canonical_pool_basis(h, eigen_handy(h), spins)
    ms = np.real(np.diag(transform_O(spins.Sv_tot[2], eigen)))
    S2 = np.real(np.diag(transform_O(spins.S2_tot, eigen)))
    S = (-1.0 + np.sqrt(4.0*S2 + 1.0)) / 2.0
    # Zeeman energy per tesla of each pool state, to first order, with the actual g tensors.
    zee = np.real(np.diag(transform_O(get_h_Zeeman(spins, [0.0, 0.0, 1.0], 'cartesian'), eigen)))
    zee_small = np.real(np.diag(transform_O(get_h_Zeeman_iso(spins, [0.0, 0.0, 1e-4], 'cartesian'), eigen)))
    E0 = eigen.eigenvalues - zee_small   # zero-field exchange energies
    return eigen, E0, zee, S, ms


def select_states(E0, zee, Bs, ecut, clusters):
    """Indices of the pool states within ecut of the ground state for some B in Bs."""
    E = E0[None, :] + Bs[:, None] * zee[None, :]          # (nB, npool)
    gap = E - E.min(axis=1, keepdims=True)
    gap_min = gap.min(axis=0)
    B_at_min = Bs[gap.argmin(axis=0)]
    selected = set(np.nonzero(gap_min <= ecut)[0].tolist())
    for c in clusters:                                    # keep degenerate states together
        if selected.intersection(c):
            selected.update(c)
    return sorted(selected), gap_min, B_at_min


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ecut", type=float, required=True, help="energy cutoff in cm-1")
    parser.add_argument("--bmin", type=float, default=None, help="lowest field of the sweep in T")
    parser.add_argument("--bmax", type=float, default=None, help="highest field of the sweep in T")
    parser.add_argument("--nb", type=int, default=2001, help="number of fields sampled")
    args = parser.parse_args()

    Ss, nS, exchange, anisotropy, gfactor, BT_Bgrid, BT_Tgrid, dynamics, states, n_threads = read_input()
    if nS < 2:
        raise SystemExit("tool_basis is for multispin systems; a single spin uses its full Hilbert space.")
    bmin = BT_Bgrid[0][0] if args.bmin is None else args.bmin
    bmax = BT_Bgrid[0][1] if args.bmax is None else args.bmax
    Bs = np.linspace(bmin, bmax, args.nb)

    spins = many_spins(Ss, nS, gfactor)
    eigen, E0, zee, S, ms = pool_states(spins, exchange)
    clusters = energy_clusters(eigen.eigenvalues)
    picked, gap_min, B_at_min = select_states(E0, zee, Bs, args.ecut, clusters)

    print("Pool states within {:g} cm-1 of the ground state for B in [{:g}, {:g}] T".format(args.ecut, bmin, bmax))
    print("(pool energies: isotropic exchange + Zeeman along z; zero-field splitting not included)")
    print("{:>6} {:>12} {:>6} {:>7} {:>14} {:>10}".format("index", "E (cm-1)", "S", "m_s", "min gap (cm-1)", "at B (T)"))
    for i in picked:
        print("{:6d} {:12.4f} {:6.2f} {:7.2f} {:14.4f} {:10.3f}".format(i, E0[i] - E0.min(), S[i], ms[i], gap_min[i], B_at_min[i]))
    print("{} states".format(len(picked)))
    print("states: [{}]".format(", ".join(str(i) for i in picked)))
