import os
import sys
import time
from spin_dynamics.core.common import *



if __name__ == "__main__":

    # Read input parameters
    Ss, nS, positions, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)



    # Hamiltonian

    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)



    # Magnetization, polarization, magnetic susceptibility, electric susceptibility

    get_M_vs_B(spins, h_ex, h_ani, BT_Bgrid)
    # get_dMdB_vs_B(spins, h_ex, h_ani, BT_Bgrid, dB=0.001)


