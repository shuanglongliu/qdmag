import os
from spin_dynamics.core.common import read_input, many_spins, save_eigenvalues, save_eigenvectors
from spin_dynamics.core.common import get_h_exchange, get_h_anisotropy, get_h_Zeeman, eigen_handy
from spin_dynamics.core.constants import factor_ex

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, states, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)

    # Set up the effective basis
    h_ex = get_h_exchange(spins, exchange, factor_ex)
    h_ani = get_h_anisotropy(spins, anisotropy)
    Bx, By, Bz = 0.0, 0.0, 0.0
    h_zee = get_h_Zeeman(spins, [Bx, By, Bz], 'Cartesian') # Unit for magnetic field is Tesla
    h = h_ex + h_ani + h_zee

    # Eigenvalues and eigenvectors
    eigen = eigen_handy(h) 

    # Save to files
    save_eigenvalues(eigen)
    save_eigenvectors(eigen)

