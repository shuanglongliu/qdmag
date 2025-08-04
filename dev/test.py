import os
import sys
import time
from spin_dynamics.core.common import read_input, many_spins
from spin_dynamics.core.common import get_h_exchange, get_h_anisotropy, get_h_Zeeman, eigen_handy
from spin_dynamics.core.common import transform_O, back_transform_O
from spin_dynamics.core.common import get_rhoe
from spin_dynamics.core.liouville import convert_rho_to_risvrho, evolve_rho_liouville_stairs
from spin_dynamics.core.liouville import liouville
from spin_dynamics.core.pulse import get_Bt
from spin_dynamics.core.hdf5 import get_rho_from_hdf5
from spin_dynamics.core.effective_basis import effective_basis

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
    lio.get_L_max_and_expLdeltat_max(5.0)
        

