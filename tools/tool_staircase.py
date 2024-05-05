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

    h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff = set_up_the_effective_system(h_ex_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states)

    #spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff, Rhbar_eff); exit()



    # Energy levels vs B field

    #get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_eff, BET_Bgrid[0])



    # Eigenvalues and eigenvectors of the effective Hamiltonian

    eigen0_eff = eigen_spin_hamiltonian(h0_eff)

    #save_eigenvalues(eigen0_eff, offset=True)
    #save_eigenvectors(eigen0_eff)



    # Set up the double super quantum master equation

    D_eff, D0_eff, h0_eff_diag, Mz_eff_diag, Rhbar_eff, C_eff, CST_eff, dim, dims, dimds = set_up_double_super_qme(h0_eff, Mv_eff[2], X_eff, I0, T)

    indices_nonzero_X_eff, indices_nonzero_C_eff = get_indices_nonzero_X_and_C(X_eff, dim)

    norm = np.linalg.norm(D0_eff, 2)
    print("Norm of the initial D matrix = {:12.6f}\n".format(norm))

    #spy_M(D0_eff, "D0_eff", threshold=1e-3); exit()



    # Initial density matrix

    rho0_eff = get_rho0(eigen0_eff, T)

    #rho0_eff = np.zeros((dim, dim))
    #rho0_eff[0, 0] = 1.0

    double_super_rho0_eff = convert_rho_to_dsrho(rho0_eff)



    # Check commutation relation

    #D1_eff = construct_D_using_Bfield(D0_eff, -1*Mz_eff_diag, 1, dim, dims)
    #check_commutation(D0_eff, D1_eff); exit()



    # Get the magnetic field pulse

    #Bt = load_cs()
    #nt, ts, Bs2, deltat = get_pulse_for_Runge_Kutta_double_grid(Bt, tmin, tmax, deltat)

    Bt = get_B_sin



    # Evolve the density matrix

    start = time.time()

    print("Trace of initial density matrix = {:20.16f}\n".format( np.real( np.trace(rho0_eff) ) ))

    #double_super_rho_eff = evolve_rho_dsqme(D0_eff, Mz_eff_diag, double_super_rho0_eff, nt, deltat, Bs2, dim, dims)

    B = 3.49
    D_eff = update_D_under_magnetic_field(D_eff, D0_eff, Mz_eff_diag, B, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds)

    norm = np.linalg.norm(D_eff, 2)
    print("Norm of the D matrix at {:8.3f} Tesla = {:12.6f}\n".format(B, norm))

    #spy_M(D_eff, "D_eff", threshold=1e-3); exit()

    for deltat in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:

        double_super_rho_eff = expm(D_eff * deltat) @ double_super_rho0_eff
    
        rho_eff = convert_dsrho_to_rho(double_super_rho_eff, dim, dims, dimds)
    
        #spy_M(rho_eff, "rho_eff", threshold=1e-9)
    
        print("Trace of final density matrix = {:20.16f}\n".format( np.real( np.trace(rho_eff) ) ))
    
        end   = time.time()
    
        print("Time used for evolution: {:8.3f} s\n".format(end - start) )
    
    
    
        # Initial magnetic moment
    
        #M = get_Mv_from_rho(rho0_eff, Mv_eff)
        #print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, initial M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))
    
    
    
        # Final magnetic moment as the system is driven
    
        M = get_Mv_from_rho(rho_eff, Mv_eff)
        print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps,   final M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))


