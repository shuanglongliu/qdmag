import os
import numpy as np
from qdmag.core.common import read_input, many_spins
from qdmag.core.common import get_h_exchange, get_h_anisotropy, get_h_Zeeman
from qdmag.core.constants import factor_ex
from qdmag.core.pulse import get_Bt
from qdmag.core.analysis import check_commutation

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactor, BT_Bgrid, BT_Tgrid, dynamics, states, n_threads = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_threads)

    # Spin system
    spins = many_spins(Ss, nS, gfactor)

    # Time-independent part of the Hamiltonian, H_ex + H_ani
    h_ex = get_h_exchange(spins, exchange, factor_ex)
    h_ani = get_h_anisotropy(spins, anisotropy)
    h0 = h_ex + h_ani

    # Direction of the pulsed magnetic field
    _, _, _, theta_B, phi_B = BT_Bgrid[0]

    # Time-dependent magnitude of the pulsed magnetic field
    Bt_params = dynamics[1]
    Bt = get_Bt(Bt_params)
    tmin = dynamics[2]['tmin']
    tmax = dynamics[2]['tmax']
    ts = np.linspace(tmin, tmax, 5)

    # H(t=0)
    h_zee_0 = get_h_Zeeman(spins, [Bt(tmin), theta_B, phi_B], 'spherical')
    h_0 = h0 + h_zee_0

    for t in ts:
        h_zee_t = get_h_Zeeman(spins, [Bt(t), theta_B, phi_B], 'spherical')
        h_t = h0 + h_zee_t

        print("\n t = {:15.6e} ps, B(t) = {:15.6e} T".format(t, Bt(t)))

        print(" Does H_ex + H_ani commute with H_zee(t)?")
        check_commutation(h0, h_zee_t)

        print(" Does H(t) commute with H(t=0)?")
        check_commutation(h_t, h_0)
