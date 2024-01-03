import os
import sys
import time
import ray
from common import *
from fitting import fit_magnetization
from von_neumann import *





if __name__ == "__main__":

    use_ray = False

    # Ray initialization

    if use_ray:
        num_cpus = 4 # int(os.getenv('SLURM_CPUS_PER_TASK'))
        ray.init(num_cpus=num_cpus)


    # Spin system

    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, fit_problem = read_input()
    spins = many_spins(Ss, nS, gfactor, dipole, positions)



    # Hamiltonian

    #h_ex = spins.zero
    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = spins.zero
    #h_ani = get_h_anisotropy(spins, anisotropy)
    h_zee = get_h_Zeeman(spins, [0,0,10], 'cartesian')
    #h_stark = get_h_Stark(spins, [1,0,0], 'cartesian')

    #h = h_ex + h_ani + h_zee + h_stark
    #h = h_ex + h_ani + h_zee 
    #h = h_ex + h_ani
    #h = h_ex + h_zee 
    #h = h_ex
    #h = h_ani


    # Check commutation relation

    #check_commutation(h_ex, h_zee)


    # Solve the eigenvalue problem

    #eigen = eigen_spin_hamiltonian(h_ex)



    start = time.time()

    # Control parameters fo time evolution

    T = 2.0
    deltat = 1.0; tmin = 0.0; tmax = 10.0
    theta_B = 90.0; phi_B = 0.0

    # Eigenvalues and eigenvectors of the initial Hamiltonian

    eigen0 = eigen_spin_hamiltonian(h_ex)

    # Initial density matrix

    rho0 = get_rho0(eigen0, T)

    # Initial Hamiltonian in the basis of eigenvectors of h0

    h0 = transform_O(h_ex, eigen0)

    # Magnetic moment operator in the basis of eigenvectors of h0

    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen0)

    # Initial magnetic moment

    M = get_M(rho0, Mv_tot)

    print("Initial M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(*M))



    # Get the magnetic field pulse

    cs = load_cs()
    Bs, ts = get_pulse(cs, tmin, tmax, deltat)
    print("The last magnetic field is {:8.4f} T".format(Bs[-1]))

    # Final magnetic moment if the system is in equilibrium

    M = get_M_at_BET_plain((spins, h_ex, h_ani, Bs[-1], theta_B, phi_B, 0, 0, 0, T))
    print("  Final M = {:12.4E} {:12.4E} {:12.4E} mu_B (if in equilibrium)".format(*M))

    # Get a series of Hamiltonians as time passes by

    hs = get_hs(h0, Mv_tot, Bs, theta_B, phi_B)

    # Get time evolution operators

    deltaUs = get_deltaUs(hs, deltat)

    # Evolve the density matrix due to the magnetic field pulse

    rho = evolve_n_deltat(rho0, deltaUs)




    ## Evolve the density matrix due to a constant magnetic field

    #h = h_ex + h_zee 
    #h = transform_O(h, eigen0)
    #deltaU = get_deltaU(h, 1e6)
    #rho = evolve_deltat(rho0, deltaU)

    #print(np.diagonal(rho0)[0:20])
    #print(np.real(np.diagonal(rho))[0:20])
    #print(np.real(np.diagonal(rho))[0:20] - np.diagonal(rho0)[0:20])

    ##np.savetxt("tmp.real", np.real(rho), fmt="%8.4f")
    ##np.savetxt("tmp.imag", np.imag(rho), fmt="%8.4f")



    # Final magnetic moment as the system is driven

    M = get_M(rho, Mv_tot)
    print("  Final M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(*M))

    end   = time.time()
    print("Time: ", end - start)




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



