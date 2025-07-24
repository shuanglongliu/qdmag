import os
import sys
import time
from spin_dynamics.dynamics.common import *



if __name__ == "__main__":

    # Read input parameters
    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, dynamics, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactor, dipole, positions)



    # Hamiltonian

    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)



    # Magnetization, polarization, magnetic susceptibility, electric susceptibility

    get_M_vs_B(spins, h_ex, h_ani, BET_Bgrid)
    # get_dMdB_vs_B(spins, h_ex, h_ani, BET_Bgrid, dB=0.001)


