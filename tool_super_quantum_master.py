import os
import sys
import time
import ray
from common import *
from von_neumann import *
from schrodinger import *
from quantum_master import *
from effective_basis import * 
from super_quantum_master import *





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

    h0 = h_ex + h_zee 


    # Check commutation relation

    #check_commutation(h_ex, h_zee)


    start = time.time()

    # Control parameters for time evolution

    T = 2.0 # Temperature in K
    tmin = 0.0 # Initial time in ps
    tmax = 1.0 # Finial time in ps
    deltat = 0.01 # Time step in ps
    theta_B = 0.0 # Polar angle of magnetic field in deg
    phi_B = 0.0 # Azimuthal angle of magnetic field in deg
    lambda_ = 10.0 # Spin phonon coupling constant in cm-1


    # Eigenvalues and eigenvectors of the initial Hamiltonian

    eigen0 = eigen_spin_hamiltonian(h0)
    #save_eigenvalues(eigen0, True)
    #save_eigenvectors(spins, eigen0)
    #np.savetxt("GS.dat", eigen0.eigenvectors[:, 0], fmt="%6.2f")


    # Initial Hamiltonian in the basis of eigenvectors of h0. It is real and diagonal.

    h0 = transform_h0_using_eigen0(h0, eigen0) 
    #np.savetxt("./output/h0.dat", h0, fmt="%6.2f"); exit()


    # Magnetic moment operator in the basis of eigenvectors of h0

    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen0)

    # Check if Mz_tot is real. If it is, use a real matrix instead of a complex matrix for it.
    # Mz_tot is real when the exchange couplings are isotropic, and the g tensors are isotropic and identical.

    Mz_tot = convert_cmatrix_to_rmatrix(Mv_tot[2], "Important note: Mz_tot")

    #np.savetxt("./output/Mz.dat", Mz_tot, fmt="%6.2f"); exit()



    # Expectation of Sz_tot for all states

    total_Sz_for_all_eigenstates = get_total_Sz_for_all_eigenstates(spins, eigen0)
    #np.savetxt("./output/Sz_tot.dat", total_Sz_for_all_eigenstates, fmt="%6.2f"); exit()


    # Get the effective Hamiltonian

    selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    dim   = len(selected_states) # Dimension of effective Hilbert space
    dims  = dim*dim              # Dimension of superoperators
    dimds = 2*dims               # Dimension of double superoperators

    print()
    print("Dimension of the effective Hamiltonian: ", dim)
    print("Dimension of superoperators: ", dims)
    print("Dimension of double superoperators: ", dimds)
    print()


    ## Matrix form

    ## h0_eff will be used to obtain eigen0_eff
    ## Mv_tot_eff will be used to calculate the Zeeman energy levels and the magnetic moment

    h0_eff = get_effective_operator(h0, selected_states, is_real=True)
    Mv_tot_eff = get_effective_Mv(Mv_tot, selected_states)

    ## Diagonal matrix elements only

    ## h0_eff_diag will be used to construct the superoperator A0_eff_diag
    ## Mz_tot_eff_diag will be used to construct the superoprator D (Azee_eff_diag more specifically)

    h0_eff_diag = get_effective_operator_diag(h0, selected_states, is_real=True)
    #np.savetxt("./output/h0_eff_diag.dat", h0_eff_diag, fmt="%12.6f")

    Mz_tot_eff_diag = get_effective_operator_diag(Mz_tot, selected_states, is_real=True)
    #np.savetxt("./output/Mz_tot_eff_diag.dat", Mz_tot_eff_diag, fmt="%12.6f")


    # Energy levels vs B field

    #get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_tot_eff, BET_Bgrid[0]); exit()


    # Eigenvalues and eigenvectors of the effective Hamiltonian

    eigen0_eff = eigen_spin_hamiltonian(h0_eff)



    # Initial density matrix

    rho0_eff = get_rho0(eigen0_eff, T)
    rho0_eff = np.zeros(h0_eff.shape)
    rho0_eff[2, 2] = 1.0
    #np.savetxt("./output/rho0_eff.dat", rho0_eff, fmt="%12.6f")

    super_rho0_eff = rho0_eff.flatten()
    super_rho0_eff_re = np.real(super_rho0_eff)
    super_rho0_eff_im = np.imag(super_rho0_eff)
    double_super_rho0_eff = np.concatenate((super_rho0_eff_re, super_rho0_eff_im))


    # Initial magnetic moment

    M = get_M(rho0_eff, Mv_tot_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, initial M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))


    # Construct the superoperator A0 from h0. 
    # Only the diagonal matrix elements of A0 are computed since A0 is diagonal.

    A0_eff_diag = construct_A_diag_from_H_diag(h0_eff_diag)



    # Operators for constructing the \Gamma operator for spin-phonon coupling

    X_eff = construct_X_eff(total_Sz_for_all_eigenstates, selected_states)
    Rhbar_eff = construct_Rhbar(T, X_eff, eigen0_eff)
    #np.savetxt("./output/Rhbar.dat", Rhbar_eff, fmt="%12.4e"); exit()
    #spy_sparsity(X_eff, "X_eff", precision=1.0e-20, figsize=(20, 20), markersize=1); exit()
    #spy_sparsity(Rhbar_eff, "Rhbar_eff", precision=1.0e-20, figsize=(20, 20), markersize=1); exit()



    # Construct the superoperator B_eff from X_eff, and Rhbar_eff

    B_eff = construct_B(X_eff, Rhbar_eff, is_real=True)
    #np.savetxt("./output/B_eff.dat", B_eff, fmt="%12.4e"); exit()
    #spy_sparsity(B_eff, "B_eff", precision=1.0e-20, figsize=(20, 20), markersize=1); exit()



    # Construct the superoperator D0 that corresponds to h0_eff/A0_eff using the diagonal elements of A0_eff

    D0_eff = construct_D_using_A_diag(A0_eff_diag, B_eff, lambda_)
    #np.savetxt("./output/D0_eff.dat", D0_eff, fmt="%12.4e"); exit()
    #spy_sparsity(D0_eff, "D0_eff", precision=1.0e-20, figsize=(20, 20), markersize=1); exit()

    D1_eff = construct_D_from_D0_and_Bfield(D0_eff, -1*Mz_tot_eff_diag, 1)
    check_commutation(D0_eff, D1_eff); exit()



    # Get the magnetic field pulse

    cs = load_cs()
    nt, ts, Bs2, deltat = get_pulse_for_Runge_Kutta_double_grid(cs, tmin, tmax, deltat)
    #print("The last magnetic field is {:9.4e} T".format(Bs[-1]))



    # Final magnetic moment if the system is in equilibrium

    #M =  get_M_at_BET_Mv_tot((h0_eff, Mv_eff, Bs[-1], theta_B, phi_B, 0, 0, 0, T))
    #print("  Final M = {:12.4E} {:12.4E} {:12.4E} mu_B (if in equilibrium)".format(*M))

    #M =  get_M_at_BET_Mv_tot((h0_eff, Mv_tot_eff, 14, theta_B, phi_B, 0, 0, 0, T))
    #print("  M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(*M))


    # Evolve the density matrix

    rho_eff = evolve_rho_sqme(D0_eff, Mz_tot_eff_diag, double_super_rho0_eff, nt, deltat, Bs2)



    # Final magnetic moment as the system is driven

    M = get_M(rho_eff, Mv_tot_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps,   final M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))



    end   = time.time()
    print()
    print("Time: {:8.3f} s".format(end - start) )



    # Ray finalization

    if use_ray:
        ray.shutdown()



