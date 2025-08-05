import os
import sys
import time
from spin_dynamics.core.common import read_input, many_spins
from spin_dynamics.core.effective_basis import effective_basis
from spin_dynamics.core.liouville import liouville

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, states, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)

    # Set up the effective basis
    eff = effective_basis(spins, exchange, anisotropy, dynamics, states)

    # Set up the quantum master equation
    lio = liouville(eff, dynamics)
    lio.get_initial_rho(from_file=False)
    lio.evolve_rho(method="staircase")
        
