import os
from timeit import default_timer as timer
from spin_dynamics.core.common import *
from spin_dynamics.core.pulse import *
from spin_dynamics.core.von_neumann import *
from spin_dynamics.core.schrodinger import *
from spin_dynamics.core.quantum_master import *
from spin_dynamics.core.effective_basis import * 



if __name__ == "__main__":

    # Read input parameters
    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, dynamics, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactor, dipole, positions)



    # Control parameters for time evolution

    T             = dynamics[0]['T']             # Temperature in K
    lambdaa       = dynamics[0]['lambdaa']       # Spin phonon coupling constant in cm-1
    I0            = dynamics[0]['I0']            # Prefactor for the phonon density of states

    Bt_params     = dynamics[1]                  # Parameters for the pulsed magnetic field

    tmin          = dynamics[2]['tmin']          # Initial time in ps
    tmax          = dynamics[2]['tmax']          # Finial time in ps
    deltat        = dynamics[2]['deltat']        # Time step in ps
                  
    save_mag      = dynamics[3]['save_mag']      # Save magnetization ?
    nt_mag        = dynamics[3]['nt_mag']        # Calculate and save magnetization every nt_mag*deltat ps
    save_rho      = dynamics[3]['save_rho']      # Save rho ?
    nt_rho        = dynamics[3]['nt_rho']        # Save rho every nt_rho*deltat ps

    multiphonon   = dynamics[4]['multiphonon']   # Include multiphonon processes ?
    imbalance     = dynamics[4]['imbalance']     # Make X unsymmetric ? 

    states        = dynamics[5]['states']        # Chosen basis states for the effective system



    # Set up the pulsed magnetic field
    Bt = get_Bt(Bt_params)



    # Hamiltonian

    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)
    h = h_ex + h_ani



    # Basis transformation

    # The basis functions are the common eigenstates of the isotropic exchange interaction and the Sz_tot operator
    # A perturbation is added to the isotropic exchange interaction to void mixing of different Sz states

    h_ex_iso = get_h_exchange_iso(spins, exchange, -2)
    eigen_p = get_perturbed_basis(h_ex_iso, spins, [0,0,1e-4])

    h_p = transform_O(h, eigen_p)
    S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Get the effective Hamiltonian

    dim = len(states)
    h_eff = get_effective_O(h_p, states)
    S2_eff = get_effective_O(S2_tot_p, states)
    Sz_eff = get_effective_O(Sz_tot_p, states)
    Mx_eff, My_eff, Mz_eff, Mv_eff = get_effective_Mv(Mv_tot_p, states)
    X_eff = construct_X_eff(Sz_tot_p, states, multiphonon=multiphonon, imbalance=imbalance)
    Rhbar_eff = get_Rhbar(h_eff, X_eff, T, I0)



    # Check the matrix elements of the above operators

    # spy_the_effective_system(h_eff, S2_eff, Sz_eff, Mv_eff, X_eff)



    # Eigenvalues and eigenvectors of the effective Hamiltonian

    # eigen_eff = eigen_handy(h_eff)

    # save_eigenvalues(eigen_eff, offset=True, suffix="eff")
    # save_eigenvectors(eigen_eff, suffix="eff")



    # Get Zeeman energy levels

    get_Zeeman_energy_levels_Mv_tot(h_eff, Mv_eff, BET_Bgrid)



    # Initial density matrix
    # h0_eff = h_eff
    # eigen0_eff = eigen_handy(h0_eff)
    # rho0_eff = get_rho0(eigen0_eff, T)

    # rho0_eff = np.zeros(h0_eff.shape, dtype = np.complex128)
    # rho0_eff[0, 0] = 1.0



    # Set up the time grid for the Runge-Kutta method

    # nt, ts, Bs2, deltat = get_pulse_for_Runge_Kutta_double_grid(Bt, tmin, tmax, deltat)



    # Evolve the density matrix

    # start = timer()

    # lambda1, lambda2 = get_constants(lambdaa)
    # if Bt_params["theta_B"] == 0.0 and Bt_params["phi_B"] == 0.0:
    #     rho_eff = evolve_rho_qme_Mz(h0_eff, Mv_eff[2], rho0_eff, nt, ts, deltat, Bs2, X_eff, Rhbar_eff, lambda1, lambda2, save_mag, nt_mag, save_rho, nt_rho)
    # else:
    #     rho_eff = evolve_rho_qme_Mv(h0_eff, Mv_eff, rho0_eff, nt, ts, deltat, Bs2, Bt_params["theta_B"], Bt_params["phi_B"], X_eff, Rhbar_eff, lambda1, lambda2, save_mag, nt_mag, save_rho, nt_rho)

    # end = timer()

    # print("Time used for evolution: {:8.3f} s".format(end - start) )


    # Initial magnetic moment

    # M = get_Mv_from_rho(rho0_eff, Mv_eff)
    # print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, initial M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))



    # Final magnetic moment as the system is driven

    # M = get_Mv_from_rho(rho_eff, Mv_eff)
    # print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps,   final M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))


