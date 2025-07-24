import os
from spin_dynamics.dynamics.common import *



if __name__ == "__main__":

    # Read input parameters
    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, dynamics, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactor, dipole, positions)



    # Hamiltonian

    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)
    h_zee = get_h_Zeeman(spins, [0,0,1e-4], 'cartesian')

    # h = h_ex + h_ani + h_zee
    h = h_ex + h_ani 


    # Check commutation relation

    # check_commutation(h_ex, h_zee); exit()
    # check_commutation(h_ex, spins.Sv_tot[2]); exit()
    # check_commutation(h_ex, spins.S2_tot); exit()



    # Save operators

    # save_operator(h, "h")
    # save_operator(spins.Sv_tot[2], "Sz")



    # Solve the eigenvalue problem

    # eigen = eigen_spin_hamiltonian(h)

    # save_eigenvalues(eigen, offset=True)
    # save_eigenvectors(eigen)



    # Save the local and total spins

    # save_spins(spins, eigen)

