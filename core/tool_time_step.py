import os
import sys
import time
from os import environ
from spin_dynamics.core.common import *
from spin_dynamics.core.von_neumann import *
from spin_dynamics.core.schrodinger import *
from spin_dynamics.core.quantum_master import *
from spin_dynamics.core.effective_basis import * 
from spin_dynamics.core.liouville import *
from spin_dynamics.core.pulse import *



environ['OMP_NUM_THREADS'] = '16'

def get_a_random_rho(dim):
    x = np.zeros(dim)
    x[24] = 5.
    x = x + np.random.rand(dim)
    x = x / np.sum(x)
    
    y = 2*(np.random.rand(dim, dim) - 0.5)
    y = ( np.transpose(y) + y )/2
    for i in range(dim):
        y[i, i] = x[i]

    return y

if __name__ == "__main__":

    # Spin system

    Ss, nS, positions, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics = read_input()
    spins = many_spins(Ss, nS, gfactors)



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

    eigen_p = get_perturbed_basis(spins, exchange, -2, 1e-4)

    h_ex_p = transform_O(h_ex, eigen_p)
    h_tmin_p = transform_O(h_tmin, eigen_p)
    S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Get the effective system

    selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    h0_eff, h_tmin_eff, S2_eff, Sz_eff, Mx_eff, My_eff, Mz_eff, Mv_eff, X_eff = set_up_the_effective_system(h_ex_p, h_tmin_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states)

    #spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff, Rhbar_eff); exit()



    # Eigenvalues and eigenvectors of the effective Hamiltonian

    #eigen0_eff = eigen_handy(h0_eff)

    #save_eigenvalues(eigen0_eff, offset=True)
    #save_eigenvectors(eigen0_eff)



    # Set up the double super quantum master equation

    D0_eff, D_eff, h0_eff_diag, h_tmin_eff_diag, Mz_eff_diag, Rhbar_eff, C_eff, CST_eff, dim, dims, dimds = set_up_double_super_qme(h0_eff, h_tmin_eff, Mv_eff[2], X_eff, I0, T, lambdaa)

    indices_nonzero_X_eff, indices_nonzero_C_eff = get_indices_nonzero_X_and_C(X_eff, dim)

    #print(indices_nonzero_X_eff); exit()

    #spy_M(D0_eff, "D0_eff", threshold=1e-3); exit()



    # Initial density matrix

    rho0_eff = get_rhoe(h_tmin_eff_diag, T)

    #rho0_eff = get_a_random_rho(dim)

    #rho0_eff = np.eye(dim) / dim

    



    #rho0_eff = get_rho0(eigen0_eff, T)

    #rho0_eff = np.zeros((dim, dim))
    #rho0_eff[0, 0] = 1.0

    double_super_rho0_eff = convert_rho_to_dsrho(rho0_eff)



    # Evolve the density matrix

    start = time.time()

    tmin = 0; tmax = 10; deltat = 1
    tmax, double_super_rho_eff1 = evolve_rho_dsqme_stairs_light(tmin, tmax, deltat, get_B_sin, double_super_rho0_eff, D_eff, D0_eff, Mz_eff, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds, save_rho, deltat_rho)

    tmin = 0; tmax = 10; deltat = 10
    tmax, double_super_rho_eff2 = evolve_rho_dsqme_stairs_light(tmin, tmax, deltat, get_B_sin, double_super_rho0_eff, D_eff, D0_eff, Mz_eff, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds, save_rho, deltat_rho)

    # Check the maximal difference between double_super_rho_eff1 and double_super_rho_eff2
    print("The maximal difference between dsrho1 and dsrho2 is", np.max(np.abs(double_super_rho_eff1 - double_super_rho_eff2)))

    tmin = 0; tmax = 10; deltat = 1
    t0, t1, U = get_U_dsqe_longtime(tmin, tmax, deltat, get_B_sin, D_eff, D0_eff, Mz_eff, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds)

    double_super_rho_eff3 = U @ double_super_rho0_eff

    # Check the maximal difference between double_super_rho_eff1 and double_super_rho_eff3
    print("The maximal difference between dsrho1 and dsrho3 is", np.max(np.abs(double_super_rho_eff1 - double_super_rho_eff3)))

    #spy_M(D_eff, "D_eff", threshold=1e-3); exit()

    # rho_eff = convert_dsrho_to_rho(double_super_rho_eff, dim, dims, dimds)
    
    #spy_M(rho_eff, "rho_eff", threshold=0); exit()
    
    end   = time.time()
    
    print("Time used for evolution: {:8.3f} s\n".format(end - start) )
    
    
