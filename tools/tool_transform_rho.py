import os
import math
import h5py
import numpy as np
from qdmag.core.common import read_input, many_spins
from qdmag.core.common import eigen_simple
from qdmag.core.effective_basis import effective_basis
from qdmag.core.liouville import liouville
from qdmag.core.hdf5 import check_conditions_of_rho
from qdmag.core.constants import Tesla2wavenumber

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactor, BT_Bgrid, BT_Tgrid, dynamics, states, n_threads = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    # Spin system
    spins = many_spins(Ss, nS, gfactor)

    # Set up the effective basis
    eff = effective_basis(spins, exchange, anisotropy, dynamics, states)

    # Set up the quantum master equation
    lio = liouville(eff, dynamics)

    # Transform the density matrix from the S representation to the E representation
    f1name = 'risvrho_S.h5'
    f2name = 'risvrho_E.h5'
    with h5py.File(f1name, "r") as f1, h5py.File(f2name, "w") as f2:
        for i in range(math.floor((dynamics[2]['tmax'] - dynamics[2]['tmin']) / dynamics[2]['deltat'])):
            t = dynamics[2]['tmin'] + i * dynamics[2]['deltat']
            risvrho = f1["{:.3f}".format(t)][0:lio.dimds]
            rho = lio.convert_risvrho_to_rho(risvrho)
            h = lio.h0 - Tesla2wavenumber * lio.Bt(t) * lio.Mz
            eigen = eigen_simple(h)
            # Unitary transformation matrix
            M = eigen.eigenvectors
            M_dagger = np.conjugate(np.transpose(M))
            # Change the basis functions from the eigenstates of the spin operator Sz_tot (S representation)
            # to the eigenstates of the Hamiltonian H (E representation) 
            # using the unitary transformation matrix M
            rho_E = np.matmul(M_dagger, np.matmul(rho, M))
            risvrho_E = lio.convert_rho_to_risvrho(rho_E)
            f2.create_dataset("{:.3f}".format(t), data=risvrho_E)

