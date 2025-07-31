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



    # Get Zeeman energy levels

    get_Zeeman_energy_levels_Mv_tot(h_eff, Mv_eff, BET_Bgrid)



