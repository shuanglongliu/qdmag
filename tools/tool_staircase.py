import os
import sys
import time
from spin_dynamics.core.common import read_input, many_spins
from spin_dynamics.core.common import get_h_exchange, get_h_anisotropy, get_h_Zeeman, eigen_handy
from spin_dynamics.core.common import get_effective_basis, transform_O, back_transform_O, transform_Mv_tot
from spin_dynamics.core.common import get_rhoe
from spin_dynamics.core.effective_basis import set_up_the_effective_system
from spin_dynamics.core.liouville import convert_rho_to_risvrho, set_up_liouville, evolve_rho_liouville_stairs
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

    eigen_eff = get_effective_basis(spins, exchange, -2, 1e-4)

    h_t0_p = transform_O(h_t0, eigen_eff)
    h_tmin_p = transform_O(h_tmin, eigen_eff)
    S2_tot_p = transform_O(spins.S2_tot, eigen_eff)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_eff)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_eff)



    # Get the effective system

    h_t0_eff, h_tmin_eff, S2_eff, Sz_eff, Mx_eff, My_eff, Mz_eff, Mv_eff, X_eff, dim = \
      set_up_the_effective_system(h_t0_p, h_tmin_p, S2_tot_p, Sz_tot_p, Mv_tot_p, states, multiphonon=multiphonon, imbalance=imbalance)



    # Set up the quantum master equation in Liouville form

    L0_eff, L_eff, Rhbar_eff, C_eff, CST_eff, dims, dimds = set_up_liouville(h_t0_eff, h_tmin_eff, X_eff, dim, I0, T, lambdaa)



    # Initial density matrix

    ## Solve the eigenvalue problem for the Hamiltonian at the time tmin
    eigen_tmin_eff = eigen_handy(h_tmin_eff)

    ## Construct the initial density matrix on the eigenbasis of h_tmin_eff
    rho0_eff = get_rhoe(eigen_tmin_eff.eigenvalues, T)

    ## Transform the density matrix from the eigenbasis of h_tmin_eff to the common eigenbasis (the perturbed basis)
    rho0_eff = back_transform_O(rho0_eff, eigen_tmin_eff)

    ## Convert the density matrix to the double super density matrix
    risvrho0_eff = convert_rho_to_risvrho(rho0_eff)

    ## Read the initial density matrix from a file
    # fname = "/blue/m2qm-efrc/shuan.liu.neu/projects/spin_dynamics/output/risvrho_0.000-10.000_step0.001ps.hdf5"
    # risvrho0_eff = get_rho_from_hdf5(fname, tmin, dimds)



    # Time evolution

    start = time.time()

    # Evolve the double super density matrix
    tmax, risvrho_eff = evolve_rho_liouville_stairs(tmin, tmax, deltat, Bt_params, risvrho0_eff, L_eff, L0_eff, h_t0_eff, Mz_eff, C_eff, CST_eff, X_eff, Rhbar_eff, lambdaa, I0, T, dim, dims, dimds, save_mag, nt_mag, save_rho, nt_rho)

    end   = time.time()
    
    print("Time used for evolution: {:8.3f} s\n".format(end - start) )

