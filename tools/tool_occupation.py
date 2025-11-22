import os
from qdmag.core.common import read_input, many_spins
from qdmag.core.common import get_h_exchange, get_h_anisotropy
from qdmag.core.common import get_effective_basis, transform_O, transform_Mv_tot
from qdmag.core.common import get_equilibrium_occupations, get_equilibrium_occupations_light
from qdmag.core.effective_basis import get_effective_O, get_effective_Mv

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, states, n_thread = read_input()

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

    # Get the magnetization operators on the effective basis
    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen_eff)
    Mx_eff, My_eff, Mz_eff, Mv_eff = get_effective_Mv(Mv_tot, states)

    # Get the equilibrium occupations at Bz and T
    Bz = 0.0 # Magnetic field in Tesla
    T = 2.0 # Temperature in Kelvin
    get_equilibrium_occupations_light(h_eff, Mv_eff, Bz, T)

    # Get the equilibrium occupations at T for a range of Bz
    # T = 2.0 # Temperature in Kelvin
    # get_equilibrium_occupations(h_eff, Mv_eff, BT_Bgrid, T)


