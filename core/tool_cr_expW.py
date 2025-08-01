import os
import sys
import time
from spin_dynamics.core.common import read_input, many_spins
from spin_dynamics.core.common import get_h_exchange, get_h_anisotropy, get_h_Zeeman, eigen_simple
from spin_dynamics.core.common import get_perturbed_basis, transform_O, transform_Mv_tot
from spin_dynamics.core.effective_basis import get_effective_O, get_effective_Mv
from spin_dynamics.core.pulse import get_Bt
from spin_dynamics.core.classical_rate import get_Pe, evolve_P_stairs

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)

    # Control parameters for time evolution
    T             = dynamics[0]['T']             # Temperature in K
    Bt_params     = dynamics[1]                  # Parameters for the pulsed magnetic field
    tmin          = dynamics[2]['tmin']          # Initial time in ps
    tmax          = dynamics[2]['tmax']          # Finial time in ps
    deltat        = dynamics[2]['deltat']        # Time step in ps
    states        = dynamics[5]['states']        # Chosen basis states for the effective system
    nu0           = 1e3                          # Attempt frequency in Hz
    symmetric     = False                        # Is the transition rate matrix W symmetric?
    nt_save       = 1                            # Save results every nt_save steps

    # Set up the pulsed magnetic field
    Bt = get_Bt(Bt_params)

    # Hamiltonian
    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)
    h_zee = get_h_Zeeman(spins, [0,0,Bt(tmin)], 'cartesian')

    # Get the basis states for the effective system
    eigen_p = get_perturbed_basis(spins, exchange, -2, 1e-4)

    # Get the effective Hamiltonian at t=0
    h = transform_O(h_ex + h_ani, eigen_p)
    h0_eff = get_effective_O(h, states)

    # Get the effective Hamiltonian at t=tmin
    h_tmin = transform_O(h_ex + h_ani + h_zee, eigen_p)
    h_eff = get_effective_O(h_tmin, states)

    # Get the magnetization operators on the effective basis
    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen_p)
    Mx_eff, My_eff, Mz_eff, Mv_eff = get_effective_Mv(Mv_tot, states)

    # Initial distribution probabilities on the eigenbasis of h_eff
    eigen_eff = eigen_simple(h_eff)
    P0_eff = get_Pe(eigen_eff.eigenvalues, T)

    # Time evolution
    start = time.time()
    tmax, P_eff = evolve_P_stairs(P0_eff, nu0, symmetric, tmin, tmax, deltat, Bt_params, T, h0_eff, Mz_eff, nt_save, dim)
    end   = time.time()
    print("Time used for evolution: {:8.3f} s\n".format(end - start) )

