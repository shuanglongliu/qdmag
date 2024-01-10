import os
import sys
import time
import ray
from common import *
from fitting import fit_magnetization
from von_neumann import *
from schrodinger import *





if __name__ == "__main__":

    use_ray = False

    # Ray initialization

    if use_ray:
        num_cpus = 16 # int(os.getenv('SLURM_CPUS_PER_TASK'))
        ray.init(num_cpus=num_cpus)


    # Spin system

    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, fit_problem = read_input()
    spins = many_spins(Ss, nS, gfactor, dipole, positions)



    # Hamiltonian

    #h_ex = spins.zero
    h_ex = get_h_exchange(spins, exchange, -2)
    #h_ani = spins.zero
    #h_ani = get_h_anisotropy(spins, anisotropy)
    #h_zee = get_h_Zeeman(spins, [0,0,10], 'cartesian')
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

    # Control parameters for time evolution

    T = 2.0 # Temperature
    tmin = 0.0 # Initial time
    tmax = 1.0 # Finial time
    deltat = 0.001 # Time step
    theta_B = 0.0 # Polar angle of magnetic field
    phi_B = 0.0 # Azimuthal angle of magnetic field

    # Eigenvalues and eigenvectors of the initial Hamiltonian

    eigen0 = eigen_spin_hamiltonian(h_ex)

    # Initial density matrix

    rho0 = get_rho0(eigen0, T)

    # Initial Hamiltonian in the basis of eigenvectors of h0

    h0 = transform_O(h_ex, eigen0)

    # Magnetic moment operator in the basis of eigenvectors of h0

    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen0)

    # Initial magnetic moment

    #M = get_M(rho0, Mv_tot)
    #print("Initial M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(*M))



    # Get the magnetic field pulse

    cs = load_cs()
    #nt, ts, Bs, deltat = get_pulse(cs, tmin, tmax, deltat)
    nt, ts, Bs, deltat = get_pulse_for_schrodinger(cs, tmin, tmax, deltat)
    #print("The last magnetic field is {:8.4f} T".format(Bs[-1]))


    # Final magnetic moment if the system is in equilibrium

    #M = get_M_at_BET_plain((spins, h_ex, h_ani, Bs[-1], theta_B, phi_B, 0, 0, 0, T))
    #print("  Final M = {:12.4E} {:12.4E} {:12.4E} mu_B (if in equilibrium)".format(*M))


    # Get initial eigen vectors

    eigenvectors0 = get_initial_eigenvectors(h0)


    # Check if get_rho is written correctly

    #rho = get_rho_from_eigenvectors(rho0, eigenvectors)
    #M = get_M(rho, Mv_tot)
    #print("Initial M = {:12.4E} {:12.4E} {:12.4E} mu_B (after calling get_rho_from_eigenvectors)".format(*M))


    # Evolve eigenvectors

    eigenvectors = evolve_psi_by_nt_steps(h0, Mv_tot, eigenvectors0, nt, ts, deltat, Bs, theta_B, phi_B, cs)


    # Final magnetic moment as the system is driven

    rho = get_rho_from_eigenvectors(rho0, eigenvectors)
    M = get_M(rho, Mv_tot)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, final M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(tmin, tmax, deltat, *M))



    end   = time.time()
    print("Time: {:8.3f} s".format(end - start) )


    # Ray finalization

    if use_ray:
        ray.shutdown()



