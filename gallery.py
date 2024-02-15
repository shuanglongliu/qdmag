import os
import sys
import time
import ray
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import *
from fitting import fit_magnetization
from von_neumann import *
from schrodinger import *
from quantum_master import *
from effective_basis import * 
from super_quantum_master import *
from constants import Tesla2wavenumber





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
    h_zee = get_h_Zeeman(spins, [0,0,1e-8], 'cartesian')
    #h_stark = get_h_Stark(spins, [1,0,0], 'cartesian')

    #h = h_ex + h_ani + h_zee + h_stark
    #h = h_ex + h_ani + h_zee 
    #h = h_ex + h_ani
    h = h_ex + h_zee 
    #h = h_ex
    #h = h_ani


    # Check commutation relation

    check_commutation(h_ex, h_zee)

    # Solve the eigenvalue problem

    #eigen = eigen_spin_hamiltonian(h)

    # Save results

    #save_operator(h, "h")

    #save_eigenvalues(eigen, True)
    #save_eigenvectors(spins, eigen)

    #save_spins(spins, eigen)


    # Zeeman diagram

    #get_energy_levels_vs_B(spins, h_ex, h_ani, Bgrid)


    # Level crossing

    #check_energy_level_crossing_B1_vs_B2(spins, h_ex, h_ani, 1.3, 1.4, 0, 0)
    #check_energy_level_crossing(spins, h_ex, h_ani, 0.0001, 10, 0.02, 0, 0)
    #check_energy_level_crossing(spins, h_ex, h_ani, 40, 110, 0.02, 0, 0)


    # Magnetization, polarization, magnetic susceptibility, electric susceptibility

    #get_M_vs_B(spins, h_ex, h_ani, Bgrid, Efield, Ts)
    #get_M_and_P_vs_B_and_E(spins, h_ex, h_ani, BET_BEgrid)
    #get_chim_and_chie_vs_T(spins, h_ex, h_ani, BET_Tgrid, 0.0001, 0.0001, chim_unit, chie_unit, n_u, V_u)
    #get_chim_and_chie_vs_B_and_E(spins, h_ex, h_ani, BET_BEgrid, 0.0001, 0.0001, chim_unit, chie_unit, n_u, V_u)
    #get_dMdB_and_dM2dB2_vs_B_and_E(spins, h_ex, h_ani, BET_BEgrid, 0.0001)


    # Transition probability for EPR

    #get_transition_strength_vs_B_for_one_pair(spins, h, BET_Bgrid, i=0, j=1) # h = h_ani or h_ex + h_ani


    # Fit spin Hamiltonian parameters

    #fit_magnetization(exp_technique, Ts_exp, spins, exchange, anisotropy, ext_field, fit_problem)


    # Ray finalization

    if use_ray:
        ray.shutdown()



