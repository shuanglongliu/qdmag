import os
import sys
import time
from os import environ
from spin_dynamics.core.common import *
from spin_dynamics.core.von_neumann import *
from spin_dynamics.core.schrodinger import *
from spin_dynamics.core.quantum_master import *
from spin_dynamics.core.effective_basis import * 
from spin_dynamics.core.super_quantum_master import *
from spin_dynamics.core.pulse import *


from scipy.optimize import fsolve
from spin_dynamics.core.constants import Tesla2wavenumber


environ['OMP_NUM_THREADS'] = '2'

def get_crossing_magnetic_fields(indices_nonzero_X_eff, h0_eff_diag, Mz_eff_diag):
    epsilon1 = 1.e-6
    epsilon2 = 1.e-4
    crossing_magnetic_fields = []
    for k in range(indices_nonzero_X_eff[0].shape[0]):
        i = indices_nonzero_X_eff[0][k]
        j = indices_nonzero_X_eff[1][k]
        B = (h0_eff_diag[i] - h0_eff_diag[j]) / (Mz_eff_diag[i] - Mz_eff_diag[j]) / Tesla2wavenumber
        if B > epsilon1:
            crossing_magnetic_fields.append(B)
    crossing_magnetic_fields = np.sort(crossing_magnetic_fields)
    unique_crossing_magnetic_fields = [crossing_magnetic_fields[0]]
    for i in range(1, crossing_magnetic_fields.shape[0]):
        if crossing_magnetic_fields[i] - crossing_magnetic_fields[i-1] > epsilon2:
            unique_crossing_magnetic_fields.append(crossing_magnetic_fields[i])
    return unique_crossing_magnetic_fields


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

    selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff = set_up_the_effective_system(h_ex_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states)
    h_tmin_eff, _, _, _, _ = set_up_the_effective_system(h_tmin_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states)

    #spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff, Rhbar_eff); exit()



    # Energy levels vs B field

    #get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_eff, BET_Bgrid[0])



    # Eigenvalues and eigenvectors of the effective Hamiltonian

    #eigen0_eff = eigen_spin_hamiltonian(h0_eff)

    #save_eigenvalues(eigen0_eff, offset=True)
    #save_eigenvectors(eigen0_eff)



    # Set up the double super quantum master equation

    D0_eff, D_eff, h0_eff_diag, Mz_eff_diag, Rhbar_eff, C_eff, CST_eff, dim, dims, dimds = set_up_double_super_qme(h0_eff, h_tmin_eff, Mv_eff[2], X_eff, I0, T)

    indices_nonzero_X_eff, indices_nonzero_C_eff = get_indices_nonzero_X_and_C(X_eff, dim)

    #print(indices_nonzero_X_eff); exit()

    #spy_M(D0_eff, "D0_eff", threshold=1e-3); exit()



    # Initial density matrix

    energies = h0_eff_diag - Mz_eff_diag * Tesla2wavenumber * get_B_sin(tmin)
    rho0_eff = get_rhoe(energies, T)

    #rho0_eff = get_a_random_rho(dim)

    #rho0_eff = np.eye(dim) / dim

    



    #rho0_eff = get_rho0(eigen0_eff, T)

    #rho0_eff = np.zeros((dim, dim))
    #rho0_eff[0, 0] = 1.0

    double_super_rho0_eff = convert_rho_to_dsrho(rho0_eff)



    # Evolve the density matrix

    start = time.time()

    #tmax = tmin + 5*deltat

    tmax, double_super_rho_eff = evolve_rho_dsqme_stairs(tmin, tmax, deltat, get_B_sin, double_super_rho0_eff, D_eff, D0_eff, Mz_eff_diag, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds, Mv_eff, save_mag, deltat_mag, save_rho, deltat_rho)

    #spy_M(D_eff, "D_eff", threshold=1e-3); exit()

    rho_eff = convert_dsrho_to_rho(double_super_rho_eff, dim, dims, dimds)
    
    #spy_M(rho_eff, "rho_eff", threshold=0); exit()
    
    end   = time.time()
    
    print("Time used for evolution: {:8.3f} s\n".format(end - start) )
    
    
