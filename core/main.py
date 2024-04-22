import os
import sys
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import *
from von_neumann import *
from schrodinger import *
from quantum_master import *
from effective_basis import * 
from super_quantum_master import *





if __name__ == "__main__":

    # Spin system

    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, dynamics = read_input()
    spins = many_spins(Ss, nS, gfactor, dipole, positions)


    # Hamiltonian

    #h_ex = get_h_exchange(spins, exchange, -2)
    #h_ani = get_h_anisotropy(spins, anisotropy)
    #h_zee = get_h_Zeeman(spins, [0,0,1e-4], 'cartesian')

    #h = h_ex
    #h = h_ani
    #h = h_ex + h_ani
    #h = h_ex + h_zee 
    #h = h_ex + h_ani + h_zee 


    # Check commutation relation

    #check_commutation(h_ex, h_zee)


    # Solve the eigenvalue problem

    #eigen = eigen_spin_hamiltonian(h)

    #eigen_p = get_perturbed_basis(h_ex, spins, [0,0,1e-4])


    # Check results

    #print(eigen.eigenvalues)

    #check_eigen(h_ex, eigen)
    #check_eigen(spins.Sv_tot[2], eigen)
    #check_eigen(spins.S2_tot, eigen)


    # Basis transformation

    #h_ex_p = transform_O(h_ex, eigen_p)
    #S2_tot_p = transform_O(spins.S2_tot, eigen_p)
    #Sv_tot_p = transform_Sv_tot(spins.Sv_tot, eigen_p)
    #Mv_tot_p = transform_Mv_tot(spins.Mv_tot, eigen_p)


    # Save results

    #save_operator(h, "h")
    #save_operator(spins.Sv_tot[2], "Sz.dat")

    #save_eigenvalues(eigen, offset=True)
    #save_eigenvectors(eigen)

    #save_spins(spins, eigen)


    # Zeeman diagram

    #get_energy_levels_vs_B(spins, h_ex, h_ani, Bgrid)


    # Magnetization, polarization, magnetic susceptibility, electric susceptibility

    #get_M_vs_B(spins, h_ex, h_ani, Bgrid, Efield, Ts)


    # Get indices of rho_upper_triangle for plotting

    #get_indices_of_rho_upper(spins.dim) # Full system
    get_indices_of_rho_upper(32) # Effective system


