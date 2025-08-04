import os
from spin_dynamics.core.common import read_input, many_spins
from spin_dynamics.core.common import get_h_exchange, get_h_anisotropy, get_h_Zeeman_Mv_tot
from spin_dynamics.core.common import get_effective_basis, transform_O, transform_Mv_tot
from spin_dynamics.core.common import get_projections
from spin_dynamics.core.effective_basis import get_effective_O, get_effective_Mv

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)

    # Hamiltonian
    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)

    # Chosen basis states for the effective system
    states = dynamics[5]['states']   

    # Get the effective Hamiltonian
    eigen_eff = get_effective_basis(spins, exchange, -2, 1e-4)
    h = transform_O(h_ex + h_ani, eigen_eff)
    h_eff = get_effective_O(h, states)

    # Get the effective Sz operator
    Sz = transform_O(spins.Sv_tot[2], eigen_eff)
    Sz_eff = get_effective_O(Sz, states)

    # Get the magnetization operators on the effective basis
    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen_eff)
    Mx_eff, My_eff, Mz_eff, Mv_eff = get_effective_Mv(Mv_tot, states)

    # State composition at the spedified B field
    B = 2.5 # Tesla
    h_eff = h_eff + get_h_Zeeman_Mv_tot(Mv_eff, [0,0,B], 'cartesian') 
    get_projections(h_eff, Sz_eff)

