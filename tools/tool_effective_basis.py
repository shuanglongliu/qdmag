import os
import sys
from os import environ
from timeit import default_timer as timer
from spin_dynamics.core.common import *
from spin_dynamics.core.pulse import *
from spin_dynamics.core.von_neumann import *
from spin_dynamics.core.schrodinger import *
from spin_dynamics.core.quantum_master import *
from spin_dynamics.core.effective_basis import * 
from spin_dynamics import __file__ as root_dir

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

    save_mag   = dynamics[2]['save_mag']      # Calculate and save magnetization during the dynamics ?
    deltat_mag = dynamics[2]['deltat_mag']    # Calculate and save magnetization every deltat_mag ps
    save_rho   = dynamics[2]['save_rho']      # Save rho ?
    deltat_rho = dynamics[2]['deltat_rho']    # Save rho every deltat_rho ps

    theta_B    = dynamics[3]['theta_B']       # Polar angle of magnetic field in deg
    phi_B      = dynamics[3]['phi_B']         # Azimuthal angle of magnetic field in deg

    print(dynamics); exit()



    # Hamiltonian

    h_ex = get_h_exchange(spins, exchange, -2)



    # Basis transformation

    eigen_p = get_perturbed_basis(h_ex, spins, [0,0,1e-4])

    h_ex_p = transform_O(h_ex, eigen_p)
    S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Get the effective Hamiltonian

    selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff, Rhbar_eff = set_up_the_effective_system(h_ex_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states, T, I0)

    #spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff, Rhbar_eff); exit()



    # Eigenvalues and eigenvectors of the effective Hamiltonian

    #eigen0_eff = eigen_spin_hamiltonian(h0_eff)

    #save_eigenvalues(eigen0_eff, offset=True)
    #save_eigenvectors(eigen0_eff)



    # Energy levels vs B field

    #get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_eff, BET_Bgrid[0])



    # Initial density matrix

    #rho0_eff = get_rho0(eigen0_eff, T)

    rho0_eff = np.zeros(h0_eff.shape, dtype = np.complex128)
    rho0_eff[0, 0] = 1.0



    # Get the magnetic field pulse

    #Bt = load_cs()
    Bt = get_B_sin

    nt, ts, Bs2, deltat = get_pulse_for_Runge_Kutta_double_grid(Bt, tmin, tmax, deltat)



    # Evolve the density matrix

    start = timer()

    lambda1, lambda2 = get_constants(lambdaa)
    if theta_B == 0.0 and phi_B == 0.0:
        rho_eff = evolve_rho_qme_Mz(h0_eff, Mv_eff[2], rho0_eff, nt, ts, deltat, Bs2, X_eff, Rhbar_eff, lambda1, lambda2, save_mag, deltat_mag, save_rho, deltat_rho)
    else:
        rho_eff = evolve_rho_qme_Mv(h0_eff, Mv_eff, rho0_eff, nt, deltat, Bs2, theta_B, phi_B, X_eff, Rhbar_eff, lambda1, lambda2)

    end = timer()

    print("Time used for evolution: {:8.3f} s".format(end - start) )


    # Initial magnetic moment

    M = get_Mv_from_rho(rho0_eff, Mv_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, initial M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))



    # Final magnetic moment as the system is driven

    M = get_Mv_from_rho(rho_eff, Mv_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps,   final M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))


