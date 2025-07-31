import os
import sys
import time
from os import environ
from spin_dynamics.core.common import *
from spin_dynamics.core.pulse import *
from spin_dynamics.core.von_neumann import *
from spin_dynamics.core.schrodinger import *
from spin_dynamics.core.quantum_master import *

environ['OMP_NUM_THREADS'] = '2'



if __name__ == "__main__":

    # Spin system

    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, dynamics = read_input()
    spins = many_spins(Ss, nS, gfactor, dipole, positions)



    # Control parameters for time evolution

    T          = dynamics[0]['T']             # Temperature in K
    lambdaa    = dynamics[0]['lambdaa']       # Spin phonon coupling constant in cm-1
    I0         = dynamics[0]['I0']            # Prefactor for the phonon density of states
               
    tmin       = dynamics[1]['tmin']          # Initial time in ps
    tmax       = dynamics[1]['tmax']          # Finial time in ps
    deltat     = dynamics[1]['deltat']        # Time step in ps

    save_mag   = dynamics[2]['save_mag']      # Calculate magnetization during the dynamics ?
    deltat_mag = dynamics[2]['deltat_mag']    # Calculate magnetization every deltat_mag ps
    save_rho   = dynamics[2]['save_rho']      # Save rho ?
    deltat_rho = dynamics[2]['deltat_rho']    # Save rho every deltat_rho ps

    theta_B    = dynamics[3]['theta_B']       # Polar angle of magnetic field in deg
    phi_B      = dynamics[3]['phi_B']         # Azimuthal angle of magnetic field in deg



    # Hamiltonian

    h_ex = get_h_exchange(spins, exchange, -2)



    # Basis transformation

    eigen_p = get_perturbed_basis(h_ex, spins, [0,0,1e-4])

    h0 = transform_O(h_ex, eigen_p)
    Sz_tot = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Eigenvalues and eigenvectors of the initial Hamiltonian

    eigen0 = eigen_handy(h0)



    # Initial density matrix

    #rho0 = get_rho0(eigen0, T)

    rho0 = np.zeros(h_ex.shape, dtype = np.complex128)
    rho0[0, 0] = 1.0



    # Get the magnetic field pulse

    Bt = load_cs()
    nt, ts, Bs2, deltat = get_pulse_for_Runge_Kutta_double_grid(Bt, tmin, tmax, deltat)



    # Operators for constructing the \Gamma operator for spin-phonon coupling

    X = construct_X(Sz_tot)

    if True:
        # For a rigorous proof of the effective basis method.
        for i in range(32, eigen0.dim):
            X[i] = 0.
            X[:, i] = 0.

    Rhbar = construct_Rhbar(T, X, np.real( np.diag(h0) ), I0)

    #spy_XRhbar(X, Rhbar, Sz_tot)



    # Evolve the density matrix

    start = time.time()

    lambda1, lambda2 = get_constants(lambdaa)
    rho = evolve_rho_qme_Mv(h0, Mv_tot, rho0, nt, ts, deltat, Bs2, theta_B, phi_B, X, Rhbar, lambda1, lambda2, save_mag, deltat_mag, save_rho, deltat_rho)

    end   = time.time()

    print("Time used for evolution: {:8.3f} s".format(end - start) )



    # Initial magnetic moment

    M = get_Mv_from_rho(rho0, Mv_tot)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, initial M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))



    # Final magnetic moment as the system is driven

    M = get_Mv_from_rho(rho, Mv_tot)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps,   final M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))

