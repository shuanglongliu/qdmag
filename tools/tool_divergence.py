import os
import numpy as np
from os import environ
from qdmag.core.common import read_input, many_spins
from qdmag.core.liouville import liouville
from qdmag.core.effective_basis import effective_basis


if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, states, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)


    # Set up the effective basis
    eff = effective_basis(spins, exchange, anisotropy, dynamics, states)

    # Sample the magnetic field and time step
    Bs = np.linspace(0, 10, 11, endpoint=True) # np.linspace(0, 50, 1001, endpoint=True)
    deltats = [1e3, 1e6 ] # [1e-3, 1e-2, 1e-1, 1., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 1e3, 1e4, 1e5, 1e6 ]

    # Set up the quantum master equation
    # And examine the biggest element of the time evolution operator
    lio = liouville(eff, dynamics)
    # lio.get_L_max_and_expLdeltat_max(10.0, 1e6, verbose=True)
    # lio.examine_L_max_and_expLdeltat_max(Bs, deltats, "0-10T")

