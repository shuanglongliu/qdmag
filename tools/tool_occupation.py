import os
from qdmag.core.common import read_input, many_spins
from qdmag.core.effective_basis import effective_basis
from qdmag.core.common import get_equilibrium_occupations, get_equilibrium_occupations_light

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactor, BT_Bgrid, BT_Tgrid, dynamics, states, n_threads = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    # Spin system
    spins = many_spins(Ss, nS, gfactor)

    # Set up the effective basis
    eff = effective_basis(spins, exchange, anisotropy, dynamics, states)

    # Get the equilibrium occupations at B=[Bx, By, Bz] and T
    # Bx, By, Bz = 0.0, 0.0, 0.0 # Magnetic field in Tesla
    # T = 2.0 # Temperature in Kelvin
    # get_equilibrium_occupations_light(eff.h0_eff, eff.Mv_eff, [Bx, By, Bz], T)

    # Get the equilibrium occupations at T for a range of B values
    T = 2.0 # Temperature in Kelvin
    get_equilibrium_occupations(eff.h0_eff, eff.Mv_eff, BT_Bgrid, T)


