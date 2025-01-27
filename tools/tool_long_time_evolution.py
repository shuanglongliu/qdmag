import os
import sys
import time
from os import environ
from spin_dynamics.dynamics.common import *
from spin_dynamics.dynamics.von_neumann import *
from spin_dynamics.dynamics.schrodinger import *
from spin_dynamics.dynamics.quantum_master import *
from spin_dynamics.dynamics.effective_basis import * 
from spin_dynamics.dynamics.super_quantum_master import *
from spin_dynamics.dynamics.pulse import *


environ['OMP_NUM_THREADS'] = '16'

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

    save_mag   = dynamics[2]['save_mag']      # Calculate and save magnetization during the dynamics ?
    deltat_mag = dynamics[2]['deltat_mag']    # Calculate and save magnetization every deltat_mag ps
    save_rho   = dynamics[2]['save_rho']      # Save rho ?
    deltat_rho = dynamics[2]['deltat_rho']    # Save rho every deltat_rho ps

    theta_B    = dynamics[3]['theta_B']       # Polar angle of magnetic field in deg
    phi_B      = dynamics[3]['phi_B']         # Azimuthal angle of magnetic field in deg



    # Overwrite tmin, tmax, deltat if the number of arguments is 4
    # Stop the program if the number of arguments is not 4
    if len(sys.argv) == 4:
        tmin = float(sys.argv[1])
        tmax = float(sys.argv[2])
        deltat = float(sys.argv[3])

        # Print tmin, tmax, deltat
        # print('tmin =', tmin); print('tmax =', tmax); print('deltat =', deltat); exit()
    else:
        print("Usage: python3 tool_long_time_evolution.py tmin tmax deltat. Stopping the program."); exit()



    # Hamiltonian
    h_ex = get_h_exchange(spins, exchange, -2)
    B_tmin = get_B_sin(tmin)
    h_zee = get_h_Zeeman(spins, [0,0,B_tmin], 'cartesian')
    h_tmin = h_ex + h_zee



    # Basis transformation
    eigen_p = get_perturbed_basis(h_ex, spins, [0,0,1e-4])
    h_ex_p = transform_O(h_ex, eigen_p)
    h_tmin_p = transform_O(h_tmin, eigen_p)
    S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Get the effective system
    selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    h0_eff, h_tmin_eff, S2_eff, Sz_eff, Mv_eff, X_eff = set_up_the_effective_system(h_ex_p, h_tmin_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states)



    # Set up the double super quantum master equation
    D0_eff, D_eff, h0_eff_diag, h_tmin_eff_diag, Mz_eff_diag, Rhbar_eff, C_eff, CST_eff, dim, dims, dimds = set_up_double_super_qme(h0_eff, h_tmin_eff, Mv_eff[2], X_eff, I0, T, lambdaa)
    indices_nonzero_X_eff, indices_nonzero_C_eff = get_indices_nonzero_X_and_C(X_eff, dim)



    # Obtain the long time evolution operator
    t0, t1, U = get_U_dsqe_longtime(tmin, tmax, deltat, get_B_sin, D_eff, D0_eff, Mz_eff_diag, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds)

