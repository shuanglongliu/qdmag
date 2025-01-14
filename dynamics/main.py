import os
import sys
import time
from os import environ
import matplotlib as mpl
import matplotlib.pyplot as plt
from spin_dynamics.dynamics.common import *
from spin_dynamics.dynamics.von_neumann import *
from spin_dynamics.dynamics.schrodinger import *
from spin_dynamics.dynamics.quantum_master import *
from spin_dynamics.dynamics.effective_basis import * 
from spin_dynamics.dynamics.super_quantum_master import *


environ['OMP_NUM_THREADS'] = '16'

if __name__ == "__main__":

    # Check setup

    #print(root_dir)

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
    #h_ani = get_h_anisotropy(spins, anisotropy)
    h_zee = get_h_Zeeman(spins, [0,0,1e-4], 'cartesian')

    # h_zee = get_h_Zeeman(spins, [0,0,3.49], 'cartesian')
    # h_zee = get_h_Zeeman(spins, [0,0,3.50], 'cartesian')
    # h_zee = get_h_Zeeman(spins, [0,0,3.4910], 'cartesian')
    # h_zee = get_h_Zeeman(spins, [0,0,3.491377], 'cartesian')

    # h = h_ex + h_zee

    h = h_ex
    h_tmin = h_ex + h_zee


    # Check commutation relation

    #check_commutation(h_ex, h_zee)


    # Solve the eigenvalue problem

    eigen = eigen_spin_hamiltonian(h)



    # Basis transformation

    eigen_p = get_perturbed_basis(h_ex, spins, [0,0,1e-4])

    h_ex_p = transform_O(h_ex, eigen_p)
    h_tmin_p = transform_O(h_tmin, eigen_p)
    S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Check results

    #print(eigen.eigenvalues)

    #check_eigen(h_ex, eigen_p)
    #check_eigen(spins.Sv_tot[2], eigen_p)
    #check_eigen(spins.S2_tot, eigen_p)

    #exit()


    # Save results

    #save_operator(h, "h")
    #save_operator(spins.Sv_tot[2], "Sz.dat")

    # save_eigenvalues(eigen_p, offset=True); # exit()
    # save_eigenvectors(eigen_p); # exit()

    # save_spins(spins, eigen_p); exit()



    # Get the effective Hamiltonian

    # dim = 32
    # selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    # dim = 6
    # selected_states = [ 0, 10, 12, 21, 24, 31 ]

    # dim = 8
    selected_states = [ 0, 10, 12, 21, 24, 31, 32, 36 ]

    h0_eff, h_tmin_eff, S2_eff, Sz_eff, Mv_eff, X_eff = set_up_the_effective_system(h_ex_p, h_tmin_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states)

    #spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff); exit()



    # Energy levels vs B field

    get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_eff, BET_Bgrid[0])
    exit()




    # Eigenvalues and eigenvectors of the effective Hamiltonian

    #eigen0_eff = eigen_spin_hamiltonian(h0_eff)

    #save_eigenvalues(eigen0_eff, offset=True)
    #save_eigenvectors(eigen0_eff)



    # Set up the double super quantum master equation

    D0_eff, D_eff, h0_eff_diag, h_tmin_eff_diag, Mz_eff_diag, Rhbar_eff, C_eff, CST_eff, dim, dims, dimds = set_up_double_super_qme(h0_eff, h_tmin_eff, Mv_eff[2], X_eff, I0, T, lambdaa)

    #spy_XRhbar(X_eff, Rhbar_eff, Sz_eff); exit()



    indices_nonzero_X_eff, indices_nonzero_C_eff = get_indices_nonzero_X_and_C(X_eff, dim)

    #print(indices_nonzero_X_eff); exit()

    D_eff = update_D_under_magnetic_field(D_eff, D0_eff, minus_Mz_eff_diag, 3.49, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds)

    # spy_M(D_eff, "D_eff")





    # Zeeman diagram

    #get_energy_levels_vs_B(spins, h_ex, h_ani, Bgrid)

    #get_energy_levels_vs_B_Mz_tot_diag(h0_eff, Mv_eff[2], BET_Bgrid[0])



    # Magnetization, polarization, magnetic susceptibility, electric susceptibility

    #get_M_vs_B(spins, h_ex, h_ani, Bgrid, Efield, Ts)


    # Get indices of rho_upper_triangle for plotting

    #get_indices_of_rho_upper(spins.dim) # Full system
    #get_indices_of_rho_upper(32) # Effective system


