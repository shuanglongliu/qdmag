import os
import subprocess
import numpy as np
from spin_dynamics.core.constants import const1, Kelvin2wavenumber
from spin_dynamics.core.common import eigen_simple

r"""
Codes for solving the quantum master equation described in the Eq. 2.7 of
J. Phys. Soc. Jpn. 2001.70:2151-2157 by Hiroki Nakano and Seiji Miyashita.

The quantum master equation reads (in the units given later)
    d rho / d t = -i * const1 * [H, rho] - { [X, Rhbar rho] + [X, Rhbar rho]^\dagger } * lambdaa^2 * pi * const1^2
 
    X_{ij} = 1 if | M_{iz} - M_{jz} | = 1. Otherwise, X_{ij} = 0. 
    Rhbar_{ij} = X_{ij} * Phi_{ij}
    Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
    omega_ij = ( E_i - E_j ) / hbar.
    I(omega) = I0 omega^2 theta(omega): spectral density for phonons
    theta(omega) is the step function.

Units: 
    cm-1 for energy
    ps for time

    cm-1 for the spin-phonon coupling constant lambdaa
    ps for spectral density I and and its prefactor I0
    ps-1 for frequency omega
    X and Rhbar are unitless
"""

def get_constants(lambdaa):
    lambda1 = -1j * const1
    lambda2 = lambdaa**2 * np.pi * const1**2
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

def spectral_density(omega, I0=1e-4):
    """
    The spectral density for phonons: I(ω) = I0 * omega^2 * θ(omega) where θ(omega) is the step function.
      Units: ps for spectral density I and and its prefactor I0.
    Numerically, I0*lambdaa^2 is the only tunable parameter in the quantum master equation. 
      We will fix I0 and tune lambdaa. 
    If I0 = 1, then Rhbar have huge matrix elements. Then, rho diverges.
      int_0^10 I(ω) dω = 1 => I0 * 10^3 / 3 = 1 => I0 = 3*10-3.
    """
    return I0 * omega**2 * np.heaviside(omega, 0.5)

def Phi(T, omega, I0):
    """
    Construct the function Phi in the quantum master equation.
    """
    beta = 1/(Kelvin2wavenumber * T)
    energy = omega2energy(omega)
    if abs( beta*energy ) < 1e-3:
        # Use Taylor expansion to avoid divergence
        result = abs( -I0*omega*omega/2 + I0*const1*omega/beta )
    else:
        numerator = spectral_density(omega, I0=I0) - spectral_density(-omega, I0=I0)
        # Print beta, energy, and their product to check the divergence
        # print("beta, energy, beta*energy = {:12.4e} {:12.4e} {:12.4e}".format(beta, energy, beta*energy))
        denominator = np.exp(beta * energy) - 1
        result = numerator / denominator
    return result

def construct_X(Sz_tot):
    """
    Construct the operator in the spin space, which couples to phonons.
    X_{ij} = 1 if | M_{iz} - M_{jz} | = 1. Otherwise, X_{ij} = 0. 
    """
    n = Sz_tot.shape[0]
    X = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = abs(Sz_tot[i, i] - Sz_tot[j, j])
            # When a B field of 1e-9 T is added to the isotropic qunatum Heisenberg Hamiltonian, 
            # the deviation in Sz from half integers is within 1e-8 mu_B.
            if abs(diff - 1.0) < 1e-6:
                X[i, j] = 1.0
    return X

def get_Rhbar(h, X, I0, T):
    """
    Construct the auxiliary operator R multiplied by hbar in both the E and S representations.
    The E representation is spanned by the eigenstates of the Hamiltonian. 
    The S representation is spanned by the eigenstates of the spin operator Sz_tot.
    Only the Rhbar in the S representation is used in the quantum master equation.

    h: Hamiltonian at a certain time in the S representation.
    T: temperature in Kelvin.
    X: the operator in the spin space in the S representation, which couples to phonons. 
    I0: prefactor for the spectral density for phonons. 
    dim: dimension of the Hilbert space or the Hamiltonian, or the X operator.

    Rhbar_E_{ij} = X_E_{ij} * Phi_{ij}
    Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
    omega_ij = ( E_i - E_j ) / hbar.
    I(omega) = I0 omega^2 theta(omega): spectral density for phonons
    theta(omega) is the step function.
    """
    # Diagonalize the Hamiltonian to find eigenvalues and eigenvectors
    eigen = eigen_simple(h)
    # Unitary transformation matrix
    M = eigen.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))
    # Change the basis functions from the eigenstates of the spin operator Sz_tot (S representation)
    # to the eigenstates of the Hamiltonian H (E representation) 
    X_E = np.matmul(M_dagger, np.matmul(X, M))
    # Construct Rhbar in the E representation
    Rhbar = np.zeros(X.shape, dtype=np.complex128)
    omegas = energy2omega(eigen.eigenvalues)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            omega_ij = omegas[i] - omegas[j]
            Rhbar[i, j] = X_E[i, j] * Phi(T, omega_ij, I0)
    # Change the basis functions from the eigenstates of the Hamiltonian H (E representation)
    # to the eigenstates of the spin operator Sz_tot (S representation)
    Rhbar = np.matmul(M, np.matmul(Rhbar, M_dagger))
    return Rhbar

def update_Rhbar(Rhbar, h, X, I0, T):
    """
    Construct the auxiliary operator R multiplied by hbar in both the E and S representations.
    The E representation is spanned by the eigenstates of the Hamiltonian. 
    The S representation is spanned by the eigenstates of the spin operator Sz_tot.

    h: Hamiltonian at a certain time in the S representation.
    X: the operator in the spin space in the S representation, which couples to phonons. 
    I0: prefactor for the spectral density for phonons. 
    T: temperature in Kelvin.

    In the E representation,
      Rhbar_E_{ij} = X_E_{ij} * Phi_{ij}
      Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
      omega_ij = ( E_i - E_j ) / hbar.
      I(omega) = I0 omega^2 theta(omega): spectral density for phonons
      theta(omega) is the step function.
    """
    # Diagonalize the Hamiltonian to find eigenvalues and eigenvectors
    eigen = eigen_simple(h)
    # Unitary transformation matrix
    M = eigen.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))
    # Change the basis functions from the eigenstates of the spin operator Sz_tot (S representation)
    # to the eigenstates of the Hamiltonian H (E representation) 
    # using the unitary transformation matrix M
    # for the X_eff operator.
    X_E = np.matmul(M_dagger, np.matmul(X, M))
    # Construct Rhbar in the E representation
    omegas = energy2omega(eigen.eigenvalues)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            omega_ij = omegas[i] - omegas[j]
            Rhbar[i, j] = X_E[i, j] * Phi(T, omega_ij, I0)
    # Change the basis functions from the eigenstates of the Hamiltonian H (E representation)
    # to the eigenstates of the spin operator Sz_tot (S representation)
    Rhbar = np.matmul(M, np.matmul(Rhbar, M_dagger))
    return Rhbar

