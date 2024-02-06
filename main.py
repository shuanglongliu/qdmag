import os
import sys
import time
import ray
from common import *
from fitting import fit_magnetization
from von_neumann import *
from schrodinger import *
from quantum_master import *
from effective_basis import * 





if __name__ == "__main__":

    use_ray = False

    # Ray initialization

    if use_ray:
        num_cpus = 16 # int(os.getenv('SLURM_CPUS_PER_TASK'))
        ray.init(num_cpus=num_cpus)


    # Spin system

    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, fit_problem = read_input()
    spins = many_spins(Ss, nS, gfactor, dipole, positions)

    #np.savetxt("Sz.dat", spins.Sv_tot[2], fmt="%6.2f")


    # Hamiltonian

    #h_ex = spins.zero
    h_ex = get_h_exchange(spins, exchange, -2)
    #h_ani = spins.zero
    #h_ani = get_h_anisotropy(spins, anisotropy)
    h_zee = get_h_Zeeman(spins, [0,0,1e-9], 'cartesian')
    #h_stark = get_h_Stark(spins, [1,0,0], 'cartesian')

    #h = h_ex + h_ani + h_zee + h_stark
    #h = h_ex + h_ani + h_zee 
    #h = h_ex + h_ani
    h = h_ex + h_zee 
    #h = h_ex
    #h = h_ani


    # Check commutation relation

    #check_commutation(h_ex, h_zee)


    start = time.time()

    # Control parameters for time evolution

    T = 2.0 # Temperature in K
    tmin = 0.0 # Initial time in ps
    tmax = 1e3 # Finial time in ps
    deltat = 0.001 # Time step in ps
    theta_B = 90.0 # Polar angle of magnetic field in deg
    phi_B = 0.0 # Azimuthal angle of magnetic field in deg
    lambda_ = 0.0 # Spin phonon coupling constant in cm-1


    # Eigenvalues and eigenvectors of the initial Hamiltonian

    eigen0 = eigen_spin_hamiltonian(h)
    #save_eigenvalues(eigen0, True)
    #save_eigenvectors(spins, eigen0)
    #np.savetxt("GS.dat", eigen0.eigenvectors[:, 0], fmt="%6.2f")


    # Initial Hamiltonian in the basis of eigenvectors of h0

    h0 = transform_O(h, eigen0)
    #np.savetxt("h0.dat", h0, fmt="%6.2f")


    # Magnetic moment operator in the basis of eigenvectors of h0

    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen0)
    #np.savetxt("Mz.dat", Mv_tot[2], fmt="%6.2f")


    # Expectation of Sz_tot for all states

    total_Sz_for_all_eigenstates = get_total_Sz_for_all_eigenstates(spins, eigen0)


    # Get the effective Hamiltonian

    selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    h0_eff, Mv_tot_eff = set_up_the_effective_system(h0, Mv_tot, selected_states)

    Mz_tot_eff_re = np.real(Mv_tot_eff[2])
    Mz_tot_eff_im = np.imag(Mv_tot_eff[2])

    np.savetxt("./output/Mz_tot_eff_re.dat", Mz_tot_eff_re, fmt="%8.4f")
    np.savetxt("./output/Mz_tot_eff_im.dat", Mz_tot_eff_im, fmt="%8.4f")

    exit()


    # Energy levels vs B field

    #get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_tot_eff, BET_Bgrid[0])


    # Eigenvalues and eigenvectors of the effective Hamiltonian

    eigen0_eff = eigen_spin_hamiltonian(h0_eff)



    # Initial density matrix

    rho0_eff = get_rho0(eigen0_eff, T)
    #np.savetxt("./output/rho0_eff.dat", rho0_eff, fmt="%12.6f")


    # Initial magnetic moment

    #M = get_M(rho0_eff, Mv_tot_eff)
    #print("Initial M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(*M))



    # Operators for constructing the \Gamma operator for spin-phonon coupling

    X_eff = construct_X_eff(total_Sz_for_all_eigenstates, selected_states)
    Rhbar_eff = construct_Rhbar(T, X_eff, eigen0_eff)



    # Get the magnetic field pulse

    cs = load_cs()
    #nt, ts, Bs, deltat = get_pulse(cs, tmin, tmax, deltat)
    nt, ts, Bs, deltat = get_pulse_for_Runge_Kutta(cs, tmin, tmax, deltat)
    #print("The last magnetic field is {:8.4f} T".format(Bs[-1]))


    # Final magnetic moment if the system is in equilibrium

    #M =  get_M_at_BET_Mv_tot((h0_eff, Mv_eff, Bs[-1], theta_B, phi_B, 0, 0, 0, T))
    #print("  Final M = {:12.4E} {:12.4E} {:12.4E} mu_B (if in equilibrium)".format(*M))

    #M =  get_M_at_BET_Mv_tot((h0_eff, Mv_tot_eff, 14, theta_B, phi_B, 0, 0, 0, T))
    #print("  M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(*M))


    # Evolve the density matrix

    rho_eff = evolve_rho_qme(h0_eff, Mv_tot_eff, rho0_eff, nt, ts, deltat, Bs, theta_B, phi_B, cs, X_eff, Rhbar_eff, lambda_)
    np.savetxt("./output/rho_eff.dat", rho_eff, fmt="%12.6f")



    # Final magnetic moment as the system is driven

    M = get_M(rho_eff, Mv_tot_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, final M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(tmin, tmax, deltat, *M))



    end   = time.time()
    print("Time: {:8.3f} s".format(end - start) )




    # Solve the eigenvalue problem

    #eigen = eigen_spin_hamiltonian(h)

    # Save results

    #save_operator(h, "h")

    #save_eigenvalues(eigen, True)
    #save_eigenvectors(spins, eigen)

    #save_spins(spins, eigen)


    # Zeeman diagram

    #get_energy_levels_vs_B(spins, h_ex, h_ani, Bgrid)


    # Level crossing

    #check_energy_level_crossing_B1_vs_B2(spins, h_ex, h_ani, 1.3, 1.4, 0, 0)
    #check_energy_level_crossing(spins, h_ex, h_ani, 0.0001, 10, 0.02, 0, 0)
    #check_energy_level_crossing(spins, h_ex, h_ani, 40, 110, 0.02, 0, 0)


    # Magnetization, polarization, magnetic susceptibility, electric susceptibility

    #get_M_vs_B(spins, h_ex, h_ani, Bgrid, Efield, Ts)
    #get_M_and_P_vs_B_and_E(spins, h_ex, h_ani, BET_BEgrid)
    #get_chim_and_chie_vs_T(spins, h_ex, h_ani, BET_Tgrid, 0.0001, 0.0001, chim_unit, chie_unit, n_u, V_u)
    #get_chim_and_chie_vs_B_and_E(spins, h_ex, h_ani, BET_BEgrid, 0.0001, 0.0001, chim_unit, chie_unit, n_u, V_u)
    #get_dMdB_and_dM2dB2_vs_B_and_E(spins, h_ex, h_ani, BET_BEgrid, 0.0001)


    # Transition probability for EPR

    #get_transition_strength_vs_B_for_one_pair(spins, h, BET_Bgrid, i=0, j=1) # h = h_ani or h_ex + h_ani


    # Fit spin Hamiltonian parameters

    #fit_magnetization(exp_technique, Ts_exp, spins, exchange, anisotropy, ext_field, fit_problem)


    # Ray finalization

    if use_ray:
        ray.shutdown()



