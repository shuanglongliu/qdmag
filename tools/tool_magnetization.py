import os
from qdmag.core.common import read_input, many_spins
from qdmag.core.common import get_h_exchange, get_h_anisotropy
from qdmag.core.common import get_M_vs_B, get_M_vs_B_Mv_tot
from qdmag.core.effective_basis import effective_basis

if __name__ == "__main__":

    do_full_space = False
    do_effective_space = True

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, states, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)

    if do_full_space:
        # Hamiltonian
        h_ex = get_h_exchange(spins, exchange, -2)
        h_ani = get_h_anisotropy(spins, anisotropy)
        
        # Magnetization in the full Hilbert space
        get_M_vs_B(spins, h_ex, h_ani, BT_Bgrid)

    if do_effective_space:
        # Set up the effective basis
        eff = effective_basis(spins, exchange, anisotropy, dynamics, states)
        
        # Magnetization in the effective Hilbert space
        get_M_vs_B_Mv_tot(eff.h0_eff, eff.Mv_eff, BT_Bgrid)

