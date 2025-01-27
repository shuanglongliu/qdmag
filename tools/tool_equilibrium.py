import os
import sys
import time
from spin_dynamics.dynamics.common import *
from spin_dynamics.dynamics.von_neumann import *
from spin_dynamics.dynamics.schrodinger import *
from spin_dynamics.dynamics.quantum_master import *
from spin_dynamics.dynamics.effective_basis import * 
from spin_dynamics.dynamics.super_quantum_master import *
from spin_dynamics.dynamics.pulse import get_Bt
from hdf5_functions import get_rho_from_hdf5



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



    # Set up the pulsed magnetic field
    Bt = get_Bt(Bt_params)



    # Hamiltonian

    h_ex = get_h_exchange(spins, exchange, -2)
    h_zee = get_h_Zeeman(spins, [0,0,Bt(tmin)], 'cartesian')

    h_tmin = h_ex + h_zee



    # Basis transformation

    eigen_p = get_perturbed_basis(h_ex, spins, [0,0,1e-4])
    h_ex_p = transform_O(h_ex, eigen_p)
    h_tmin_p = transform_O(h_tmin, eigen_p)
    S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    Sz_tot_p = transform_O(spins.Sv_tot[2], eigen_p)
    Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)



    # Get the effective system

    # dim = 16
    selected_states = [200,150,88,30,10,0,1,2,3,4,5,17,41,99,173,215]
    dim = len(selected_states)

    h0_eff, h_tmin_eff, S2_eff, Sz_eff, Mx_eff, My_eff, Mz_eff, Mv_eff, X_eff = set_up_the_effective_system(h_ex_p, h_tmin_p, S2_tot_p, Sz_tot_p, Mv_tot_p, selected_states)

    h0_eff_diag = np.real( h0_eff.diagonal() )
    Mz_eff_diag = np.real( Mz_eff.diagonal() )



    Bmin, Bmax, Bstep, theta_B, phi_B = BET_Bgrid[0]
    nB = int((Bmax-Bmin)/Bstep) + 1

    Bs = []
    rho_diag_s = []
    for i in range(nB):
        B = Bmin + i*Bstep
        h_eff_diag = h0_eff_diag - Tesla2wavenumber*B*Mz_eff_diag
        rho_eff = get_rhoe(h_eff_diag, T)
        rho_diag = np.real( rho_eff.diagonal() )
        Bs.append(B)
        rho_diag_s.append(rho_diag)

    # Save the results
    ostring = "{:8.3f}" + dim*" {:12.3e}" + "\n"
    with open(root_dir + '/output/rho_diag_T{:.1f}K.dat'.format(T), 'w') as f:
        f.write('# B (Tesla) rho_ii, i=0,1,...,15\n')
        for i in range(nB):
            f.write(ostring.format(Bs[i], *rho_diag_s[i]))

