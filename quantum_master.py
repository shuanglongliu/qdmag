import numpy as np
from constants import const1, Kelvin2wavenumber
from common import get_total_Sz_for_all_eigenstates
from common import get_commutation
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

lambda_ = 1.0

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


def spectral_density(omega):
    """
    The spectral density for phonons: I(ω) = I0 * omega^2 * θ(omega) where θ(omega) is the step function.
    Units: ps for spectral density I and and its prefactor I0.

    Numerically, I0*lambda_^2 is the only tunable parameter in the quantum master equation. 
    We will fix I0 and tune lambda_. 
    """

    I0 = 1e-6

    # If I0 = 1, then Rhbar have huge matrix elements. Then, rho diverges.

    return I0 * omega**2 * np.heaviside(omega, 0.5)

def Phi(T, omega):

    beta = 1/(Kelvin2wavenumber * T)
    energy = omega2energy(omega)

    return ( spectral_density(omega) - spectral_density(-omega) ) / ( np.exp(beta * energy) - 1 )

def construct_X(total_Sz_for_all_eigenstates):
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
            #print("{:5d}::{:8.3f},   {:5d}::{:8.3f},   {:5.1f}".format(i, total_Sz_for_all_eigenstates[i], j, total_Sz_for_all_eigenstates[j], X[i, j]))
    return X


def construct_Rhbar(T, X, eigen):
    """
    Construct the auxiliary operator R multiplied by hbar.

    Rhbar_{ij} = X_{ij} * Phi_{ij}
    Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
    omega_ij = ( E_i - E_j ) / hbar.
    I(omega) = I0 omega^2 theta(omega): spectral density for phonons
    theta(omega) is the step function.
    """

    omegas = energy2omega(eigen.eigenvalues)

    n = eigen.dim

    Rhbar = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            omega_ij = omegas[i] - omegas[j]
            if X[i, j] != 0.0:
                print(i, j, omega_ij)
                Rhbar[i, j] = X[i, j] * Phi(T, omega_ij)

    #print(np.max(np.absolute(Rhbar)))

    #np.savetxt("./output/Rhbar.dat", Rhbar, fmt="%12.6f")

    return Rhbar


def get_Gammarho(rho, X, Rhbar, lambda_):
    Rhbarrho = np.matmul(Rhbar, rho)
    #print("max(Rhbarrho) = ", np.max(np.absolute(Rhbarrho)))
    commutation = np.matmul(X, Rhbarrho) - np.matmul(Rhbarrho, X)
    tmp = commutation + np.transpose(np.conjugate(commutation))
    result = tmp * lambda_**2 * np.pi * const1**2
    #print("max(Gamma rho) = ", np.max(np.absolute(result)))
    return result


def evolve_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda_, deltat):
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
      lambda_: spin-phonon coupling constant. Unit: cm-1.
      deltat: time step in ps.
    """

    k1 = -1j * const1 * ( np.matmul(ha, rho) - np.matmul(rho, ha) ) - get_Gammarho(rho, X, Rhbar, lambda_)
    k2 = -1j * const1 * ( np.matmul(hb, rho + 0.5*deltat*k1) - np.matmul(rho + 0.5*deltat*k1, hb) ) - get_Gammarho(rho + 0.5*deltat*k1, X, Rhbar, lambda_)
    k3 = -1j * const1 * ( np.matmul(hb, rho + 0.5*deltat*k2) - np.matmul(rho + 0.5*deltat*k2, hb) ) - get_Gammarho(rho + 0.5*deltat*k2, X, Rhbar, lambda_)
    k4 = -1j * const1 * ( np.matmul(hc, rho +     deltat*k3) - np.matmul(rho +     deltat*k3, hc) ) - get_Gammarho(rho +     deltat*k3, X, Rhbar, lambda_)

    rho_new = rho + deltat * (k1 + 2*k2 + 2*k3 + k4) / 6

    return rho_new

def evolve_rho_qme(h0, Mv_tot, rho, nt, ts, deltat, Bs, theta_B, phi_B, cs, X, Rhbar, lambda_):
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
      Bs: list of magnetic field.
      theta_B: polar angle of the magnetic field.
      phi_B: azimuthal angle of the magnetic field.
      cs: monotone cubic spline object for the pulse field.
      X, Rhbar: Auxiliary operators for constructing the \Gamma operator for spin-phonon coupling.
      lambda_: spin-phonon coupling constant in cm-1.
    """

    ha, hb, hc = get_habc(h0, Mv_tot, ts, 0, deltat, Bs, theta_B, phi_B, cs)

    rho = evolve_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda_, deltat)

    #print("i = 0, max(rho) = ", np.max(np.absolute(rho)))

    for i in range(1, nt):
        ha, hb, hc = get_habc_reuse_ha(h0, Mv_tot, ts, i, deltat, Bs, theta_B, phi_B, cs, hc)
        rho = evolve_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda_, deltat)

        #print("i = {:d}, max(rho) = ".format(i), np.max(np.absolute(rho)))

    return rho


