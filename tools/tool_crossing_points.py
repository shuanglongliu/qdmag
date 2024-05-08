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



def get_crossing_times(B0, t0):
    f = lambda x: get_B_sin(x) - B0
    crossing_time = fsolve(f, [t0])
    print("{:20.3f}".format(crossing_time[0]))


def get_max_evolution_operator(t, deltat):
    B = get_B_sin(t)
    D = update_D_under_magnetic_field(D_eff, D0_eff, Mz_eff_diag, B, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds)
    max_evolution_operator = np.max(np.abs(expm(D*deltat)))
    print("t , deltat , B , max_evolution_operator = {:20.3f} , {:20.3f} , {:20.3f} , {:20.3f}".format(t, deltat[0], B, max_evolution_operator))
    return max_evolution_operator

def get_max_time_step(t, deltat0, max_evolution_operator0=10000.):

    f = lambda x: get_max_evolution_operator(t, x) - max_evolution_operator0
    max_deltat = fsolve(f, [deltat0])
    print("{:20.3f}".format(max_deltat[0]))

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



    # Basis transformation

    eigen_p = get_perturbed_basis(h_ex, spins, [0,0,1e-4])

    h_ex_p = transform_O(h_ex, eigen_p)
    S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Get the effective system

    selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff = set_up_the_effective_system(h_ex_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states)

    #spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff, Rhbar_eff); exit()



    # Energy levels vs B field

    #get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_eff, BET_Bgrid[0])



    # Set up the double super quantum master equation

    D_eff, D0_eff, h0_eff_diag, Mz_eff_diag, Rhbar_eff, C_eff, CST_eff, dim, dims, dimds = set_up_double_super_qme(h0_eff, Mv_eff[2], X_eff, I0, T)

    indices_nonzero_X_eff, indices_nonzero_C_eff = get_indices_nonzero_X_and_C(X_eff, dim)

    #print(indices_nonzero_X_eff); exit()

    #crossing_magnetic_fields = get_crossing_magnetic_fields(indices_nonzero_X_eff, h0_eff_diag, Mz_eff_diag)
    #print(crossing_magnetic_fields); exit() # [1.7135593106969351, 1.7778177848482146, 3.491377095545602]

    #get_crossing_times(1.7135593106969351, 0.1e9)   # 139878972.288
    #get_crossing_times(1.7778177848482146, 0.1e9)   # 145125705.173
    #get_crossing_times(3.491377095545602 , 0.1e9)   # 285106489.578



    # Get the maximal reasonable time step

    #get_max_evolution_operator(0, 1e0)

    #get_max_time_step(0, 1e1) # 10.021
    #get_max_time_step(1e6, 1e1) # 10.036
    #get_max_time_step(1e7, 1e1) # 12.038
    #get_max_time_step(1e8, 1e1) # 12.038
    get_max_time_step(5e8, 1e3) # 12.038



