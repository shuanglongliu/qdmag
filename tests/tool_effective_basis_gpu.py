import os
import sys
from timeit import default_timer as timer
import cupy as cp
from constants import Tesla2wavenumber
from common import *
from pulse import *
from von_neumann import *
from schrodinger import *
from quantum_master import *
from effective_basis import * 



def get_Gammarho_gpu(rho, X, Rhbar, lambda2):
    """
    lambda2 = lambda_^2 * pi * const1^2
    """

    Rhbarrho = cp.matmul(Rhbar, rho)
    commutation = cp.matmul(X, Rhbarrho) - cp.matmul(Rhbarrho, X)
    Gammarho = lambda2 * ( commutation + cp.transpose(cp.conjugate(commutation)) )

    return Gammarho

def evolve_rho_by_deltat_qme_gpu(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat):
    """
    Evolve rho by deltat using the Runge-Kutta method according to the quantum master equation
    d rho / d t = -i * const1 * [H, rho] - { [X, Rhbar rho] + [X, Rhbar rho]^\dagger } * lambda_^2 * pi * const1^2

    Input: 
      ha: h(t), the total Hamiltonian including the Zeeman interaction at time t. Unit: cm-1.
      hb: h(t+deltat/2)
      hc: h(t+deltat). ha, hb, and hc are  written on the basis of the eigenvectors of h0, i.e. the zero-field spin Hamiltonian.
      rho: density matrix at time t rho(t) on the basis of the eigenvectors of h0.
      X: the operator in the spin space, which couples to phonons. See construct_X in this file. Unitless.
      Rhbar: an auxiliary operator R multiplied by hbar. See construct_Rhbar in this file. Unitless.
      lambda1 = -1j * const1
      lambda2 = lambda_^2 * pi * const1^2, lambda_: spin-phonon coupling constant in cm-1.
      deltat: time step in ps. 
    """

    k1                             = lambda1 * ( cp.matmul(ha, rho ) - cp.matmul(rho , ha) ) - get_Gammarho_gpu(rho , X, Rhbar, lambda2)
    rho1 = rho + 0.5*deltat*k1; k2 = lambda1 * ( cp.matmul(hb, rho1) - cp.matmul(rho1, hb) ) - get_Gammarho_gpu(rho1, X, Rhbar, lambda2)
    rho2 = rho + 0.5*deltat*k2; k3 = lambda1 * ( cp.matmul(hb, rho2) - cp.matmul(rho2, hb) ) - get_Gammarho_gpu(rho2, X, Rhbar, lambda2)
    rho3 = rho +     deltat*k3; k4 = lambda1 * ( cp.matmul(hc, rho3) - cp.matmul(rho3, hc) ) - get_Gammarho_gpu(rho3, X, Rhbar, lambda2)

    rho_new = rho + deltat * (k1 + 2*k2 + 2*k3 + k4) / 6

    return rho_new

def get_h_Zeeman_Mz_tot_gpu(Mz_tot, Bz):
    """ 
    Zeeman term H_Zee = - \vec{\mu} \cdot \vec{B} = - Mz_tot * Bz when the magnetic field is parallel to the z axis.

    In the last line, B takes unit of energy (cm^-1 per \mu_B), and mu takes unit of \mu_b.
    
    Units: Tesla for B.

    Mz_tot can be given in arbitrary basis. 
    """

    return -1 * Tesla2wavenumber*Bz * Mz_tot

def get_h_h0basis_Bz_gpu(h0, Mz_tot, B):
    """
    Both h0 and Mz_tot should be on the basis of eigenvectors of h0.
    Return h under the magnetic field B.

    Assumptions:
      The magnetic field is along the z direction.
    """

    h_zee = get_h_Zeeman_Mz_tot_gpu(Mz_tot, B)
    h = h0 + h_zee

    return h

def get_habc_Bz_gpu(h0, Mz_tot, it, deltat, Bs2):
    """
    Obtain ha = h(ts[it])
           hb = h(ts[it] + deltat/2)
           hc = h(ts[it] + deltat)

    h0 and Mv_tot are on the basis of the eigenvectors of h0.

    ts:  A list of time with a time step of deltat
    Bs2: A list of B fields with a time step of deltat/2
         B(t = ts[it]) = Bs2[2*it]

    Assumptions:
      The magnetic field is along the z direction.
    """

    ha = get_h_h0basis_Bz_gpu(h0, Mz_tot, Bs2[2*it  ])
    hb = get_h_h0basis_Bz_gpu(h0, Mz_tot, Bs2[2*it+1])
    hc = get_h_h0basis_Bz_gpu(h0, Mz_tot, Bs2[2*it+2])

    return (ha, hb, hc)

def get_habc_reuse_ha_Bz_gpu(h0, Mz_tot, it, deltat, Bs2, ha, hb, hc):
    """
    Obtain ha = h(ts[it])
           hb = h(ts[it] + deltat/2)
           hc = h(ts[it] + deltat)

    h0 and Mv_tot are on the basis of the eigenvectors of h0.

    ts:  A list of time with a time step of deltat
    Bs2: A list of B fields with a time step of deltat/2
         B(t = ts[it]) = Bs2[2*it]

    Assumptions:
      The magnetic field is along the z direction.
    """

    hb = get_h_h0basis_Bz_gpu(h0, Mz_tot, Bs2[2*it+1])
    hc = get_h_h0basis_Bz_gpu(h0, Mz_tot, Bs2[2*it+2])

    return (ha, hb, hc)

def evolve_rho_qme_Bz_gpu(h0, Mz_tot, rho, nt, deltat, Bs2, X, Rhbar, lambda1, lambda2):
    """
    Evolve the density matrix using the Runge-Kutta method according to the quantum master equation.

    h0 and Mz_tot are written on the basis of the eigenvectors of h0.

    Input:
      h0: Hamiltonian at t0 = ts[0] written on the basis of the eigenvectors of h0.
      Mz_tot: Magnetization operators written on the basis of the eigenvectors of h0.
      rho: initial density matrix at t0.
      nt: number of time steps.
      ts: list of time in unit of ps.
      delta: time step in unit of ps.
      Bs2: a list of B fields with a time step of deltat/2
           B(t = ts[it]) = Bs2[2*it]
      theta_B: polar angle of the magnetic field.
      phi_B: azimuthal angle of the magnetic field.
      cs: monotone cubic spline object for the pulse field.
      X, Rhbar: Auxiliary operators for constructing the \Gamma operator for spin-phonon coupling.
      lambda1 = -1j * const1
      lambda2 = lambda_^2 * pi * const1^2, lambda_: spin-phonon coupling constant in cm-1.

    Assumptions:
      The magnetic field is along the z direction.
    """

    it = 0; ha, hb, hc = get_habc_Bz_gpu(h0, Mz_tot, it, deltat, Bs2)
    rho = evolve_rho_by_deltat_qme_gpu(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
    #print("it = {:d}, max(rho) = {:12.4e}".format(it, np.max(np.absolute(rho))))

    for it in range(1, nt):
        ha, hb, hc = get_habc_reuse_ha_Bz_gpu(h0, Mz_tot, it, deltat, Bs2, hc, ha, hb)
        rho = evolve_rho_by_deltat_qme_gpu(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
        #print("it = {:d}, max(rho) = {:12.4e}".format(it, np.max(np.absolute(rho))))

    return rho

if __name__ == "__main__":
    
    # Spin system
    
    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, fit_problem = read_input()
    spins = many_spins(Ss, nS, gfactor, dipole, positions)
    
    #np.savetxt("Sz.dat", spins.Sv_tot[2], fmt="%6.2f")
    
    
    # Hamiltonian
    
    #h_ex = spins.zero
    h_ex = get_h_exchange(spins, exchange, -2)
    #h_ani = spins.zero
    #h_ani = get_h_anisotropy(spins, anisotropy)
    h_zee = get_h_Zeeman(spins, [0,0,1e-4], 'cartesian')
    #h_stark = get_h_Stark(spins, [1,0,0], 'cartesian')
    
    # See the comments in the function set_up_the_effective_system in the effective_basis.py for the choice of the perturbative magnetic field.
    h = h_ex + h_zee
    
    
    # Check commutation relation
    
    #check_commutation(h_ex, h_zee)
    
    
    
    # Eigenvalues and eigenvectors of the initial Hamiltonian
    
    eigen0 = eigen_spin_hamiltonian(h)
    #save_eigenvalues(eigen0, offset=True, sort=False)
    #save_eigenvectors(eigen0, sort=False)
    #np.savetxt("./output/GS.dat", eigen0.eigenvectors[:, 0], fmt="%6.2f")
    # save_spins(spins, eigen0)
    
    
    
    # Control parameters for time evolution
    
    T = 2.0 # Temperature in K
    tmin = 0.0 # Initial time in ps
    tmax = 10.0 # Finial time in ps
    deltat = 0.01 # Time step in ps
    theta_B = 0.0 # Polar angle of magnetic field in deg
    phi_B = 0.0 # Azimuthal angle of magnetic field in deg
    lambda_ = 10.0 # Spin phonon coupling constant in cm-1
    
    
    # Initial Hamiltonian in the basis of eigenvectors of h0
    
    h0 = transform_O(h, eigen0)
    #np.savetxt("h0.dat", h0, fmt="%6.2f")
    
    
    # Magnetic moment operator in the basis of eigenvectors of h0
    
    Mv_tot = transform_Mv_tot(spins.Mv_tot, eigen0)
    #np.savetxt("Mz.dat", Mv_tot[2], fmt="%6.2f")
    
    
    # Expectation of Sz_tot for all states
    
    total_Sz_for_all_eigenstates = get_total_Sz_for_all_eigenstates(spins, eigen0)
    
    
    # Get the effective Hamiltonian
    
    selected_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    
    h0_eff, Mv_eff = set_up_the_effective_system(h0, Mv_tot, selected_states, save_to_file=False)
    
    
    # Energy levels vs B field
    
    #get_energy_levels_vs_B_Mv_tot(h0_eff, Mv_eff, BET_Bgrid[0])
    
    
    # Eigenvalues and eigenvectors of the effective Hamiltonian
    
    eigen0_eff = eigen_spin_hamiltonian(h0_eff)
    save_eigenvalues(eigen0_eff, offset=True, sort=False)
    
    
    
    # Initial density matrix
    
    #rho0_eff = get_rho0(eigen0_eff, T)
    rho0_eff = np.zeros(h0_eff.shape, dtype = np.complex128)
    rho0_eff[2, 2] = 1.0
    #np.savetxt("./output/rho0_eff.dat", rho0_eff, fmt="%12.6f")
    
    
    # Operators for constructing the \Gamma operator for spin-phonon coupling
    
    X_eff = construct_X_eff(total_Sz_for_all_eigenstates, selected_states, save_to_file=False)
    Rhbar_eff = construct_Rhbar(T, X_eff, eigen0_eff, save_to_file=False)
    
    
    
    # Get the magnetic field pulse
    
    cs = load_cs()
    nt, ts, Bs2, deltat = get_pulse_for_Runge_Kutta_double_grid(cs, tmin, tmax, deltat)
    #print("The last magnetic field is {:8.4f} T".format(Bs2[-1]))
    
    
    
    
    
    h0_eff_gpu = cp.asarray(h0_eff)
    Mz_eff_gpu = cp.asarray(Mv_eff[2])
    X_eff_gpu = cp.asarray(X_eff)
    Rhbar_eff_gpu = cp.asarray(Rhbar_eff)
    rho0_eff_gpu = cp.asarray(rho0_eff)
    Bs2_gpu = cp.asarray(Bs2)
    
    lambda1, lambda2 = get_constants(lambda_)
    
    
    
    
    start = timer()
    
    rho_eff_gpu = evolve_rho_qme_Bz_gpu(h0_eff_gpu, Mz_eff_gpu, rho0_eff_gpu, nt, deltat, Bs2_gpu, X_eff_gpu, Rhbar_eff_gpu, lambda1, lambda2)
    
    end = timer()
    
    print("Time on CPU: {:8.3f} s".format(end - start) )
    
    
    
    
    
    
    rho_eff = cp.asnumpy(rho_eff_gpu)
    
    
    # Initial magnetic moment
    
    M = get_M(rho0_eff, Mv_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps, initial M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))
    
    
    
    # Final magnetic moment as the system is driven
    
    M = get_M(rho_eff, Mv_eff)
    print("tmin = {:8.4f} ps, tmax = {:8.4f} ps, deltat = {:8.4f} ps,   final M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(tmin, tmax, deltat, *np.real(M)))
    
    
    
    # Final magnetic moment if the system is in equilibrium
    
    #M =  get_M_at_BET_Mv_tot((h0_eff, Mv_eff, Bs2[-1], theta_B, phi_B, 0, 0, 0, T))
    #print("  Final M = {:12.4E} {:12.4E} {:12.4E} mu_B (if in equilibrium)".format(*M))
    
    #M =  get_M_at_BET_Mv_tot((h0_eff, Mv_eff, 14, theta_B, phi_B, 0, 0, 0, T))
    #print("  M = {:12.4E} {:12.4E} {:12.4E} mu_B".format(*M))
    
    
    
    
