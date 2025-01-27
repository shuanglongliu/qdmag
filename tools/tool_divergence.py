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


from scipy.optimize import fsolve
from spin_dynamics.dynamics.constants import Tesla2wavenumber


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

    # dim = 16
    selected_states = [200,150,88,30,10,0,1,2,3,4,5,17,41,99,173,215]

    h0_eff, h_tmin_eff, S2_eff, Sz_eff, Mv_eff, X_eff = set_up_the_effective_system(h_ex_p, h_tmin_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states)

    #spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff, Rhbar_eff); exit()



    # Set up the double super quantum master equation

    D0_eff, D_eff, h0_eff_diag, h_tmin_eff_diag, Mz_eff_diag, Rhbar_eff, C_eff, CST_eff, dim, dims, dimds = set_up_double_super_qme(h0_eff, h_tmin_eff, Mv_eff[2], X_eff, I0, T, lambdaa)

    indices_nonzero_X_eff, indices_nonzero_C_eff = get_indices_nonzero_X_and_C(X_eff, dim)

    #spy_M(D0_eff, "D0_eff", threshold=1e-3); exit()



    # Examine the biggest element of the time evolution operator

    # Find the initial magnetic field
    # B = get_B_sin(tmin)

    # Bs = np.linspace(3.49, 3.50, 11, endpoint=True)
    # deltats = [1e-3, 1e-2, 1e-1, 1., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 1e3, 1e4, 1e5, 1e6 ]
    # examine_D_max_and_expDdeltat_max(Bs, deltats, "near_3.49T", D_eff, D0_eff, Mz_eff_diag, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds)

    
    Bs = np.linspace(0, 50, 1001, endpoint=True)
    deltats = [1e-3, 1e-2, 1e-1, 1., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 1e3, 1e4, 1e5, 1e6 ]
    examine_D_max_and_expDdeltat_max(Bs, deltats, "0-50T", D_eff, D0_eff, Mz_eff_diag, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds)

    
