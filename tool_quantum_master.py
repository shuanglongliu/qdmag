import os
import sys
import time
import ray
from common import *
from fitting import fit_magnetization
from pulse import *
from von_neumann import *
from schrodinger import *
from quantum_master import *
from os import environ

environ['OMP_NUM_THREADS'] = '16'





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

    h = h_ex + h_zee 


    # Check commutation relation

    #check_commutation(h_ex, h_zee)


    # Control parameters for time evolution

    T = 2.0 # Temperature in K
    tmin = 0.0 # Initial time in ps
    tmax = 1.0 # Finial time in ps
    deltat = 0.001 # Time step in ps
    theta_B = 0.0 # Polar angle of magnetic field in deg
    phi_B = 0.0 # Azimuthal angle of magnetic field in deg
    lambda_ = 0.5 # Spin phonon coupling constant in cm-1


    # Eigenvalues and eigenvectors of the initial Hamiltonian

    eigen0 = eigen_spin_hamiltonian(h)
    #save_eigenvalues(eigen0, True)
    #save_eigenvectors(spins, eigen0)
    #np.savetxt("GS.dat", eigen0.eigenvectors[:, 0], fmt="%6.2f")
    #print(eigen0.indices); exit()


    # Initial Hamiltonian in the basis of eigenvectors of h0

    h0 = transform_O(h, eigen0)
    #np.savetxt("h0.dat", h0, fmt="%6.2f")


    # Magnetic moment operator in the basis of eigenvectors of h0

    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen0)
    #np.savetxt("Mz.dat", Mv_tot[2], fmt="%6.2f")



    # Expectation of Sz_tot for all states

    total_Sz_for_all_eigenstates = get_total_Sz_for_all_eigenstates(spins, eigen0)


    # Initial density matrix

    #rho0 = get_rho0(eigen0, T)
    rho0 = np.zeros(h0.shape, dtype = np.complex128)
    rho0[2, 2] = 1.0
    #np.savetxt("rho0.dat", rho0, fmt="%12.6f")



    # Get the magnetic field pulse

    cs = load_cs()
    nt, ts, Bs2, deltat = get_pulse_for_Runge_Kutta_double_grid(cs, tmin, tmax, deltat)
    #print("The last magnetic field is {:8.4f} T".format(Bs2[-1]))


    # Final magnetic moment if the system is in equilibrium

    #M = get_M_at_BET_plain((spins, h_ex, h_ani, Bs2[-1], theta_B, phi_B, 0, 0, 0, T))
    #print("  Final M = {:12.4E} {:12.4E} {:12.4E} mu_B (if in equilibrium)".format(*M))


    # Operators for constructing the \Gamma operator for spin-phonon coupling

    X = construct_X(total_Sz_for_all_eigenstates)
    # For rigorous proof of the effective basis method.
    for i in range(32, eigen0.dim):
        X[i] = 0.
        X[:, i] = 0.
    Rhbar = construct_Rhbar(T, X, eigen0)


    # Evolve the density matrix

    start = time.time()

    lambda1, lambda2 = get_constants(lambda_)
    rho = evolve_rho_qme(h0, Mv_tot, rho0, nt, deltat, Bs2, theta_B, phi_B, X, Rhbar, lambda1, lambda2)
    #np.savetxt("rho.dat", rho, fmt="%12.6f")

    end   = time.time()

    print("Time: {:8.3f} s".format(end - start) )



    # Initial magnetic moment

    M = get_M(rho0, Mv_tot)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, initial M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))



    # Final magnetic moment as the system is driven

    M = get_M(rho, Mv_tot)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps,   final M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))



    # Ray finalization

    if use_ray:
        ray.shutdown()



