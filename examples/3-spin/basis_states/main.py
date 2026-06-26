import os
from qdmag.core.common import read_input, many_spins, save_spins
from qdmag.core.common import get_h_exchange_iso, get_h_Zeeman_iso, eigen_handy
from qdmag.core.constants import factor_ex

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactor, BT_Bgrid, BT_Tgrid, dynamics, states, n_threads = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    # Spin system
    spins = many_spins(Ss, nS, gfactor)

    # Set up the effective basis
    h_ex = get_h_exchange_iso(spins, exchange, factor_ex)
    h_zee = get_h_Zeeman_iso(spins, [0,0,1.0e-04], 'Cartesian') # Unit for magnetic field is Tesla
    h = h_ex + h_zee + 1e-06 * spins.S2_tot

    # Eigenvalues and eigenvectors
    eigen = eigen_handy(h) 

    # Spins
    save_spins(spins, eigen)
    
