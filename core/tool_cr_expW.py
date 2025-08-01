import os
import sys
import time
from spin_dynamics.core.common import *
from spin_dynamics.core.von_neumann import *
from spin_dynamics.core.schrodinger import *
from spin_dynamics.core.quantum_master import *
from spin_dynamics.core.effective_basis import * 
from spin_dynamics.core.liouville import *
from spin_dynamics.core.pulse import get_Bt
from spin_dynamics.core.classical_rate import get_Pe, evolve_P_stairs



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

    states        = dynamics[5]['states']        # Chosen basis states for the effective system



    # Set up the pulsed magnetic field
    Bt = get_Bt(Bt_params)



    # Hamiltonian

    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)
    h_zee_tmin = get_h_Zeeman(spins, [0,0,Bt(tmin)], 'cartesian')

    # Spin Hamiltonian at t=0
    h_t0 = h_ex + h_ani 

    # Spin Hamiltonian at t=tmin
    h_tmin = h_ex + h_ani + h_zee_tmin



    # Basis transformation
    # The basis functions are the common eigenstates of the isotropic exchange interaction and the Sz_tot operator
    # A perturbation is added to the isotropic exchange interaction to void mixing of different Sz states

    eigen_p = get_perturbed_basis(spins, exchange, -2, 1e-4)

    h_t0_p = transform_O(h_t0, eigen_p)
    h_tmin_p = transform_O(h_tmin, eigen_p)
    S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Get the effective system

    h_t0_eff, h_tmin_eff, S2_eff, Sz_eff, Mx_eff, My_eff, Mz_eff, Mv_eff, X_eff, dim = \
      set_up_the_effective_system(h_t0_p, h_tmin_p, S2_tot_p, Sz_tot_p, Mv_tot_p, states, multiphonon=multiphonon, imbalance=imbalance)



    # Initial distribution probabilities

    ## Construct the initial density matrix

    eigen_tmin_eff = eigen_simple(h_tmin_eff)

    ### Construct the initial density matrix on the eigenbasis of h_tmin_eff
    P0_eff = get_Pe(eigen_tmin_eff.eigenvalues, T)


    # Time evolution

    start = time.time()

    # Evolve the double super density matrix
    nu0 = 1e3  # Hz
    symmetric = True  # Symmetric evolution
    nt_save = 1
    tmax, P_eff = evolve_P_stairs(P0_eff, nu0, symmetric, tmin, tmax, deltat, Bt_params, T, h_t0_eff, Mz_eff, nt_save, dim)

    end   = time.time()
    
    print("Time used for evolution: {:8.3f} s\n".format(end - start) )

