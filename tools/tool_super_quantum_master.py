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

    h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff, nonzero_indices_X_eff = set_up_the_effective_system(h_ex_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states, T, I0)

    #spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff); exit()



    #Rhbar_eff = construct_Rhbar(T, X_eff, nonzero_indices_X_eff, np.real(np.diag(h0_eff)), I0)
    #, Rhbar_eff
    #with open(root_dir + "output/Rhbar_eff.dat", "w") as f:
        #for i in range(n):
            #for j in range(n):
                #f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, Rhbar_eff[i, j]))
 
    #max_Rhbar_eff = np.max(np.absolute(Rhbar_eff))
    #spy_sparsity(Rhbar_eff, "Rhbar_eff", precision = 0.01*max_Rhbar_eff, figsize=(10, 10), markersize=5)


    # Construct the superoperator B_eff from X_eff, and Rhbar_eff

    B_eff = construct_B(X_eff, Rhbar_eff, dim, dims)

    # Construct the superoperator D0 that corresponds to h0_eff/A0_eff using the diagonal elements of A0_eff

    D0_eff = construct_D_using_A_diag(A0_eff_diag, B_eff, lambdaa, dims)



    # Energy levels vs B field

    #get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_eff, BET_Bgrid[0])



    # Eigenvalues and eigenvectors of the effective Hamiltonian

    #eigen0_eff = eigen_spin_hamiltonian(h0_eff)

    #save_eigenvalues(eigen0_eff, offset=True)
    #save_eigenvectors(eigen0_eff)



    # Set up the double super quantum master equation

    D0_eff, Mz_eff_diag, dim, dims, dimds = set_up_double_super_qme(h0_eff, Mv_eff[2], X_eff, Rhbar_eff, lambdaa)

    #spy_double_super_system(D0_eff, "D0_eff", dimds); exit()



    # Initial density matrix

    #rho0_eff = get_rho0(eigen0_eff, T)

    rho0_eff = np.zeros((dim, dim))
    rho0_eff[0, 0] = 1.0

    double_super_rho0_eff = convert_rho_to_dsrho(rho0_eff)



    # Check commutation relation

    #D1_eff = construct_D_using_Bfield(D0_eff, -1*Mz_eff_diag, 1, dim, dims)
    #check_commutation(D0_eff, D1_eff); exit()



    # Get the magnetic field pulse

    Bt = load_cs()
    nt, ts, Bs2, deltat = get_pulse_for_Runge_Kutta_double_grid(Bt, tmin, tmax, deltat)



    # Evolve the density matrix

    start = time.time()

    double_super_rho_eff = evolve_rho_dsqme(D0_eff, Mz_eff_diag, double_super_rho0_eff, nt, deltat, Bs2, dim, dims)

    rho_eff = convert_dsrho_to_rho(double_super_rho_eff, dim, dims, dimds)

    end   = time.time()

    print("Time used for evolution: {:8.3f} s\n".format(end - start) )



    # Initial magnetic moment

    M = get_Mv_from_rho(rho0_eff, Mv_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, initial M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))



    # Final magnetic moment as the system is driven

    M = get_Mv_from_rho(rho_eff, Mv_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps,   final M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))


