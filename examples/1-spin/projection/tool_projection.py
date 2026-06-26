import os
from qdmag.core.common import read_input, many_spins
from qdmag.core.effective_basis import effective_basis
from qdmag.core.common import get_projections

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactor, BT_Bgrid, BT_Tgrid, dynamics, states, n_threads = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    # Spin system
    spins = many_spins(Ss, nS, gfactor)

    # Set up the effective basis
    eff = effective_basis(spins, exchange, anisotropy, dynamics, states)

    # State composition at the spedified B field
    Bx, By, Bz = 0.0, 0.0, 0.1 # Tesla
    get_projections(eff, [Bx, By, Bz])

