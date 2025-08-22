import os
import math
import numpy as np
from qmagnetic.core.common import read_input, many_spins
from qmagnetic.core.effective_basis import effective_basis
from qmagnetic.core.liouville import liouville
from qmagnetic.core.hdf5 import check_conditions_of_rho

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

    # Check the conditions of the density matrix
    fname = 'risvrho.h5'
    skip = 100 # The .h5 file is read for each time. Use a big skip to reduce the number of reads.
    for i in range(math.floor((dynamics[2]['tmax'] - dynamics[2]['tmin']) / (skip * dynamics[2]['deltat']))):
        t = dynamics[2]['tmin'] + i * skip * dynamics[2]['deltat']
        is_normalized, is_hermitian, is_positive, is_cschineq = check_conditions_of_rho(fname, t, lio)
        print(f"t = {t:15.3f} ps: is_normalized = {is_normalized}, is_hermitian = {is_hermitian}, is_positive = {is_positive}, is_cschineq = {is_cschineq}")

