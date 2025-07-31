import os
import sys
import time
from common import *
from fitting import fit_magnetization
from pulse import *
from von_neumann import *





if __name__ == "__main__":

    # Spin system

    Ss, nS, positions, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, fit_problem = read_input()
    spins = many_spins(Ss, nS, gfactors)



    # Hamiltonian

    #h_ex = spins.zero
    h_ex = get_h_exchange(spins, exchange, -2)
    #h_ani = spins.zero
    #h_ani = get_h_anisotropy(spins, anisotropy)
    h_zee = get_h_Zeeman(spins, [0,0,1e-9], 'cartesian')
    #h_stark = get_h_Stark(spins, [1,0,0], 'cartesian')

    #h = h_ex + h_ani + h_zee + h_stark
    #h = h_ex + h_ani + h_zee 
    #h = h_ex + h_ani
    h = h_ex + h_zee 
    #h = h_ex
    #h = h_ani



    start = time.time()

    # Control parameters for time evolution

    T = 2.0 # Temperature
    tmin = 0.0 # Initial time
    tmax = 1.0 # Finial time
    deltat = 0.0001 # Time step
    theta_B = 90.0 # Polar angle of magnetic field
    phi_B = 0.0 # Azimuthal angle of magnetic field
    nperiod = 8 # Number of time periods

    # Eigenvalues and eigenvectors of the initial Hamiltonian

    eigen0 = eigen_handy(h)

    # Initial density matrix

    rho0 = get_rho0(eigen0, T)

    # Initial Hamiltonian in the basis of eigenvectors of h0

    h0 = transform_O(h, eigen0)

    # Magnetic moment operator in the basis of eigenvectors of h0

    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen0)

    # Initial magnetic moment

    #M = get_M(rho0, Mv_tot)

    #print("Initial M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(*M))



    # Get the magnetic field pulse

    Bt = load_cs()
    nt, ts, Bs, deltat = get_pulse_for_TEO(Bt, tmin, tmax, deltat)
    #print("The last magnetic field is {:8.4f} T".format(Bs[-1]))



    # Get time evolution operators

    DeltaUs = get_DeltaUs(h0, Mv_tot, nt, ts, deltat, Bs, theta_B, phi_B, nperiod)

    # Evolve the density matrix due to the magnetic field pulse

    rho = evolve_Deltats(rho0, DeltaUs)

    # Save the density matrix at time tmax

    #save_rho(rho, tmax)




    ## Evolve the density matrix due to a constant magnetic field

    #h = h_ex + h_zee 
    #h = transform_O(h, eigen0)
    #deltaU = get_deltaU(h, 1e6)
    #rho = evolve_deltat(rho0, deltaU)

    #print(np.diagonal(rho0)[0:20])
    #print(np.real(np.diagonal(rho))[0:20])
    #print(np.real(np.diagonal(rho))[0:20] - np.diagonal(rho0)[0:20])

    ##np.savetxt("tmp.real", np.real(rho), fmt="%8.4f")
    ##np.savetxt("tmp.imag", np.imag(rho), fmt="%8.4f")



    # Final magnetic moment as the system is driven

    M = get_M(rho, Mv_tot)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, final M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(tmin, tmax, deltat, *M))

    end   = time.time()
    print("Time: {:8.3f} s".format(end - start) )


