import os
from spin_dynamics.core.common import read_input, many_spins
from spin_dynamics.core.common import get_h_exchange, get_h_anisotropy
from spin_dynamics.core.common import get_Zeeman_energy_levels, get_Zeeman_energy_levels_Mv_tot
from spin_dynamics.core.common import transform_O
from spin_dynamics.core.effective_basis import effective_basis

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
        
        # Zeeman diagram in the full Hilbert space
        get_Zeeman_energy_levels(spins, h_ex, h_ani, BT_Bgrid)

    if do_effective_space:
        # Set up the effective basis
        eff = effective_basis(spins, exchange, anisotropy, dynamics, states)
        
        # Zeeman diagram in the effective Hilbert space
        get_Zeeman_energy_levels_Mv_tot(eff.h0_eff, eff.Mv_eff, BT_Bgrid)

