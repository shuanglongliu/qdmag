import numpy as np
from constants import const1, Kelvin2wavenumber
from common import get_total_Sz_for_all_eigenstates
from common import get_commutation
from common import spy_sparsity
from schrodinger import get_habc, get_habc_reuse_ha

"""
Codes for solving the quantum master equation described in the Eq. 2.7 of
J. Phys. Soc. Jpn. 2001.70:2151-2157 by Hiroki Nakano and Seiji Miyashita.

The quantum master equation reads (in the units given later)
    d rho / d t = -i * const1 * [H, rho] - { [X, Rhbar rho] + [X, Rhbar rho]^\dagger } * lambda_^2 * pi * const1^2
 
    X_{ij} = 1 if | M_{iz} - M_{jz} | = 1. Otherwise, X_{ij} = 0. 
    Rhbar_{ij} = X_{ij} * Phi_{ij}
    Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
    omega_ij = ( E_i - E_j ) / hbar.
    I(omega) = I0 omega^2 theta(omega): spectral density for phonons
    theta(omega) is the step function.

Units: 
    cm-1 for energy
    ps for time

    cm-1 for the spin-phonon coupling constant lambda_
    ps for spectral density I and and its prefactor I0
    ps-1 for frequency omega
    X and Rhbar are unitless
"""

def get_constants(lambda_):
    lambda1 = -1j * const1
    lambda2 = lambda_**2 * np.pi * const1**2
    return (lambda1, lambda2)

def energy2omega(energy):
    """
    Convert enrgy (in cm-1) to angular frequency (radian ps-1) using E = hbar omega.
    """

    return const1 * energy

def omega2energy(omega):
    """
    Convert angular frequency (radian ps-1) to enrgy (in cm-1) using E = hbar omega.
    """

    return omega / const1


def spectral_density(omega, I0=1e-6):
    """
    The spectral density for phonons: I(ω) = I0 * omega^2 * θ(omega) where θ(omega) is the step function.
    Units: ps for spectral density I and and its prefactor I0.

    Numerically, I0*lambda_^2 is the only tunable parameter in the quantum master equation. 
    We will fix I0 and tune lambda_. 

    If I0 = 1, then Rhbar have huge matrix elements. Then, rho diverges.
    """

    return I0 * omega**2 * np.heaviside(omega, 0.5)

def Phi(T, omega):

    beta = 1/(Kelvin2wavenumber * T)
    energy = omega2energy(omega)

    return ( spectral_density(omega) - spectral_density(-omega) ) / ( np.exp(beta * energy) - 1 )

def construct_X(total_Sz_for_all_eigenstates, save_to_file=False):
    """
    Construct the operator in the spin space, which couples to phonons.
    X_{ij} = 1 if | M_{iz} - M_{jz} | = 1. Otherwise, X_{ij} = 0. 
    """

    n = len(total_Sz_for_all_eigenstates)
    X = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = abs(total_Sz_for_all_eigenstates[i] - total_Sz_for_all_eigenstates[j])
            # When a B field of 1e-9 T is added to the isotropic qunatum Heisenberg Hamiltonian, 
            # the deviation in Sz from half integers is within 1e-8 mu_B.
            if abs(diff - 1.0) < 1e-6:
                X[i, j] = 1.0

    if save_to_file:
        with open("./output/X.dat", "w") as f:
            for i in range(n):
                for j in range(n):
                    f.write("i   j   Sz_tot_i   Sz_tot_j   X_ij   = {:5d}   {:5d}   {:8.3f}   {:8.3f}   {:5.1f}\n".format( \
                         i, j, total_Sz_for_all_eigenstates[selected_states[i]], total_Sz_for_all_eigenstates[selected_states[j]], X[i, j]))
        spy_sparsity(X, "X", precision=1.0e-20, figsize=(10, 10), markersize=5)

    return X


def construct_Rhbar(T, X, eigen, save_to_file=False):
    """
    Construct the auxiliary operator R multiplied by hbar.

    Rhbar_{ij} = X_{ij} * Phi_{ij}
    Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
    omega_ij = ( E_i - E_j ) / hbar.
    I(omega) = I0 omega^2 theta(omega): spectral density for phonons
    theta(omega) is the step function.
    """

    omegas = energy2omega(eigen.eigenvalues)

    Rhbar = np.zeros((eigen.dim, eigen.dim), dtype=np.float64)

    for i in range(eigen.dim):
        for j in range(eigen.dim):
            omega_ij = omegas[i] - omegas[j]
            if X[i, j] != 0.0:
                #print(i, j, omega_ij)
                Rhbar[i, j] = X[i, j] * Phi(T, omega_ij)

    if save_to_file:
        with open("./output/Rhbar.dat", "w") as f:
            for i in range(eigen.dim):
                for j in range(eigen.dim):
                    f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, Rhbar[i, j]))

        max_Rhbar = np.max(np.absolute(Rhbar))
        spy_sparsity(Rhbar, "Rhbar", precision = 0.1*max_Rhbar, figsize=(10, 10), markersize=5)

    return Rhbar


def get_Gammarho(rho, X, Rhbar, lambda2):
    """
    lambda2 = lambda_^2 * pi * const1^2
    """

    Rhbarrho = np.matmul(Rhbar, rho)
    commutation = np.matmul(X, Rhbarrho) - np.matmul(Rhbarrho, X)
    Gammarho = lambda2 * ( commutation + np.transpose(np.conjugate(commutation)) )

    return Gammarho


def evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat):
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

    k1                             = lambda1 * ( np.matmul(ha, rho ) - np.matmul(rho , ha) ) - get_Gammarho(rho , X, Rhbar, lambda2)
    rho1 = rho + 0.5*deltat*k1; k2 = lambda1 * ( np.matmul(hb, rho1) - np.matmul(rho1, hb) ) - get_Gammarho(rho1, X, Rhbar, lambda2)
    rho2 = rho + 0.5*deltat*k2; k3 = lambda1 * ( np.matmul(hb, rho2) - np.matmul(rho2, hb) ) - get_Gammarho(rho2, X, Rhbar, lambda2)
    rho3 = rho +     deltat*k3; k4 = lambda1 * ( np.matmul(hc, rho3) - np.matmul(rho3, hc) ) - get_Gammarho(rho3, X, Rhbar, lambda2)

    rho_new = rho + deltat * (k1 + 2*k2 + 2*k3 + k4) / 6

    return rho_new

def evolve_rho_qme(h0, Mv_tot, rho, nt, deltat, Bs2, theta_B, phi_B, X, Rhbar, lambda1, lambda2):
    """
    Evolve the density matrix using the Runge-Kutta method according to the quantum master equation.

    h0 and Mv_tot are written on the basis of the eigenvectors of h0.

    Input:
      h0: Hamiltonian at t0 = ts[0] written on the basis of the eigenvectors of h0.
      Mv_tot: Magnetization operators written on the basis of the eigenvectors of h0.
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
    """

    it = 0; ha, hb, hc = get_habc(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B)
    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
    #print("it = {:d}, max(rho) = {:12.4e}".format(it, np.max(np.absolute(rho))))

    for it in range(1, nt):
        ha, hb, hc = get_habc_reuse_ha(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B, hc, ha, hb)
        rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
        #print("it = {:d}, max(rho) = {:12.4e}".format(it, np.max(np.absolute(rho))))

    return rho


