import os
import sys
import time
from os import environ
import matplotlib as mpl
import matplotlib.pyplot as plt
from spin_dynamics.core.common import *
from spin_dynamics.core.von_neumann import *
from spin_dynamics.core.schrodinger import *
from spin_dynamics.core.quantum_master import *
from spin_dynamics.core.effective_basis import * 
from spin_dynamics.core.liouville import *
from spin_dynamics.core.pulse import get_Bt
from spin_dynamics.core.hdf5 import get_rho_from_hdf5



if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)



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

    states        = dynamics[5]['states']        # Make X unsymmetric ? 



    # Set up the pulsed magnetic field
    Bt = get_Bt(Bt_params)



    # Hamiltonian

    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)
    h_zee = get_h_Zeeman(spins, [0,0,1e-4], 'cartesian')

    # Spin Hamiltonian at t=0
    h_t0 = h_ex + h_ani

    # Spin Hamiltonian at t=tmin
    h_tmin = h_ex + h_zee


    # Check commutation relation

    # check_commutation(h_ex, h_zee); exit()
    # check_commutation(h_ex, spins.Sv_tot[2]); exit()
    # check_commutation(h_ex, spins.S2_tot); exit()



    # Solve the eigenvalue problem

    eigen = eigen_handy(h_ex + h_ani + h_zee)



    # Basis transformation
    # The basis functions are the common eigenstates of the isotropic exchange interaction and the Sz_tot operator
    # A perturbation is added to the isotropic exchange interaction to void mixing of different Sz states

    eigen_p = get_perturbed_basis(spins, exchange, -2, 1e-4)

    h_t0_p = transform_O(h_t0, eigen_p)
    h_tmin_p = transform_O(h_tmin, eigen_p)
    S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Save results

    #save_operator(h, "h")
    #save_operator(spins.Sv_tot[2], "Sz.dat")

    # save_eigenvalues(eigen, offset=True)
    # save_eigenvectors(eigen_p)

    # save_spins(spins, eigen)



    # Zeeman diagram for the full system

    # get_Zeeman_energy_levels(spins, h_ex, h_ani, BT_Bgrid)



    # Magnetization, polarization, magnetic susceptibility, electric susceptibility

    # get_M_vs_B(spins, h_ex, h_ani, BT_Bgrid)



    # Get the effective Hamiltonian

    h_t0_eff, h_tmin_eff, S2_eff, Sz_eff, Mx_eff, My_eff, Mz_eff, Mv_eff, X_eff, dim = \
      set_up_the_effective_system(h_t0_p, h_tmin_p, S2_tot_p, Sz_tot_p, Mv_tot_p, states, multiphonon=multiphonon, imbalance=imbalance)

    # spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff); exit()





    # Eigenvalues and eigenvectors of the effective Hamiltonian

    #eigen0_eff = eigen_handy(h0_eff)

    #save_eigenvalues(eigen0_eff, offset=True)
    #save_eigenvectors(eigen0_eff)



    # Zeeman diagram for the effective system

    # get_Zeeman_energy_levels_Mv_tot(h_t0_eff, Mv_eff, BT_Bgrid)




    # State composition

    h_eff = h_t0_eff + get_h_Zeeman_Mv_tot(Mv_eff, [0,0,6.4134], 'cartesian') # 2.5, 3.731
    eigen_eff = eigen_handy(h_eff)
    save_projections(eigen_eff, 15/2)


    # Set up the double super quantum master equation

    # D0_eff, D_eff, h0_eff_diag, h_tmin_eff_diag, Mz_eff_diag, Rhbar_eff, C_eff, CST_eff, dim, dims, dimds = set_up_liouville(h0_eff, h_tmin_eff, Mv_eff[2], X_eff, I0, T, lambdaa)

    #spy_XRhbar(X_eff, Rhbar_eff, Sz_eff); exit()



    # indices_nonzero_X_eff, indices_nonzero_C_eff = get_indices_nonzero_X_and_C(X_eff, dim)

    #print(indices_nonzero_X_eff); exit()

    # D_eff = update_D_under_magnetic_field(D_eff, D0_eff, minus_Mz_eff_diag, 3.49, C_eff, CST_eff, X_eff, Rhbar_eff, h0_eff_diag, indices_nonzero_X_eff, indices_nonzero_C_eff, lambdaa, I0, T, dim, dims, dimds)

    # spy_M(D_eff, "D_eff")





