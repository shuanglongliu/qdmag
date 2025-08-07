import os
from qmagnetic.core.common import read_input, many_spins
from qmagnetic.core.effective_basis import effective_basis
from qmagnetic.core.common import get_projections

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, states, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)

    # Set up the effective basis
    eff = effective_basis(spins, exchange, anisotropy, dynamics, states)

    # State composition at the spedified B field
    Bx, By, Bz = 0.0, 0.0, 0.0 # Tesla
    get_projections(eff, [Bx, By, Bz])

