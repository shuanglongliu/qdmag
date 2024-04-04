import os
import sys
from timeit import default_timer as timer
import ray
from common import *
from pulse import *
from von_neumann import *
from schrodinger import *
from quantum_master import *
from effective_basis import * 
from os import environ

environ['OMP_NUM_THREADS'] = '2'




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
    h_zee = get_h_Zeeman(spins, [0,0,1e-4], 'cartesian')
    #h_stark = get_h_Stark(spins, [1,0,0], 'cartesian')

    # See the comments in the function set_up_the_effective_system in the effective_basis.py for the choice of the perturbative magnetic field. 
    h = h_ex + h_zee 



    # Check commutation relation

    #check_commutation(h_ex, h_zee)



    # Control parameters for time evolution

    T = 2.0 # Temperature in K
    tmin = 0.0 # Initial time in ps
    tmax = 1.0 # Finial time in ps
    deltat = 0.01 # Time step in ps
    Deltat = 0.01 # Save rho every Deltat ps
    theta_B = 0.0 # Polar angle of magnetic field in deg
    phi_B = 0.0 # Azimuthal angle of magnetic field in deg
    lambda_ = 10.0 # Spin phonon coupling constant in cm-1



    # Eigenvalues and eigenvectors of the initial Hamiltonian

    eigen0 = eigen_spin_hamiltonian(h)
    #save_eigenvalues(eigen0, offset=True, sort=False); exit()
    #save_eigenvectors(eigen0, sort=False)
    #np.savetxt("./output/GS.dat", eigen0.eigenvectors[:, 0], fmt="%6.2f")
    #save_spins(spins, eigen0); exit()



    # Initial Hamiltonian in the basis of eigenvectors of h0

    h0 = transform_O(h, eigen0)
    #np.savetxt("h0.dat", h0, fmt="%6.2f")



    # Magnetic moment operator in the basis of eigenvectors of h0

    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen0)
    #np.savetxt("Mz.dat", Mv_tot[2], fmt="%6.2f")
    #check_commutation(h0, Mv_tot[0]) # No
    #check_commutation(h0, Mv_tot[1]) # No
    #check_commutation(h0, Mv_tot[2]) # Yes



    # Expectation of Sz_tot for all states

    total_Sz_for_all_eigenstates = get_total_Sz_for_all_eigenstates(spins, eigen0)



    # Get the effective Hamiltonian

    selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    h0_eff, Mv_eff = set_up_the_effective_system(h0, Mv_tot, selected_states, save_to_file=False)



    # Energy levels vs B field

    #get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_eff, BET_Bgrid[0])



    # Eigenvalues and eigenvectors of the effective Hamiltonian

    eigen0_eff = eigen_spin_hamiltonian(h0_eff)
    #save_eigenvalues(eigen0_eff, offset=True, sort=False); exit()



    # Initial density matrix

    #rho0_eff = get_rho0(eigen0_eff, T)
    rho0_eff = np.zeros(h0_eff.shape, dtype = np.complex128)
    rho0_eff[0, 0] = 1.0
    #np.savetxt("./output/rho0_eff.dat", rho0_eff, fmt="%12.6f")



    # Operators for constructing the \Gamma operator for spin-phonon coupling

    X_eff = construct_X_eff(total_Sz_for_all_eigenstates, selected_states, save_to_file=False)
    Rhbar_eff = construct_Rhbar(T, X_eff, eigen0_eff, save_to_file=False)



    # Get the magnetic field pulse

    cs = load_cs()
    nt, ts, Bs2, deltat = get_pulse_for_Runge_Kutta_double_grid(cs, tmin, tmax, deltat)
    #print("The last magnetic field is {:8.4f} T".format(Bs2[-1]))


    # Final magnetic moment if the system is in equilibrium

    #M =  get_M_at_BET_Mv_tot((h0_eff, Mv_eff, Bs2[-1], theta_B, phi_B, 0, 0, 0, T))
    #print("  Final M = {:12.4E} {:12.4E} {:12.4E} mu_B (if in equilibrium)".format(*M))

    #M =  get_M_at_BET_Mv_tot((h0_eff, Mv_eff, 14, theta_B, phi_B, 0, 0, 0, T))
    #print("  M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(*M))


    # Evolve the density matrix

    start = timer()

    list_of_ts = []
    list_of_rhos = []
    lambda1, lambda2 = get_constants(lambda_)
    if theta_B == 0.0 and phi_B == 0.0:
        if save_rho:
            rho_eff = evolve_rho_qme_Bz(h0_eff, Mv_eff[2], rho0_eff, nt, deltat, Bs2, X_eff, Rhbar_eff, lambda1, lambda2)
            #rho_eff, list_of_ts, list_of_rhos = evolve_rho_qme_Bz_save_rho(h0_eff, Mv_eff[2], rho0_eff, nt, ts, deltat, Bs2, X_eff, Rhbar_eff, lambda1, lambda2, list_of_rhos, Deltat)
        else:
            rho_eff = evolve_rho_qme_Bz(h0_eff, Mv_eff[2], rho0_eff, nt, deltat, Bs2, X_eff, Rhbar_eff, lambda1, lambda2)
    else:
        rho_eff = evolve_rho_qme(h0_eff, Mv_eff, rho0_eff, nt, deltat, Bs2, theta_B, phi_B, X_eff, Rhbar_eff, lambda1, lambda2)
    #np.savetxt("./output/rho_eff.dat", rho_eff, fmt="%12.6f")

    end = timer()

    print("Time: {:8.3f} s".format(end - start) )



    # Initial magnetic moment

    M = get_M(rho0_eff, Mv_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, initial M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))



    # Final magnetic moment as the system is driven

    M = get_M(rho_eff, Mv_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps,   final M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))



    # Ray finalization

    if use_ray:
        ray.shutdown()



