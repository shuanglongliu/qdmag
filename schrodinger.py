import numpy as np
from constants import const1
from von_neumann import get_h_h0basis, get_h_h0basis_Bz

def get_habc(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B):
    """
    Obtain ha = h(ts[it])
           hb = h(ts[it] + deltat/2)
           hc = h(ts[it] + deltat)

    h0 and Mv_tot are on the basis of the eigenvectors of h0.

    ts:  A list of time with a time step of deltat
    Bs2: A list of B fields with a time step of deltat/2
         B(t = ts[it]) = Bs2[2*it]
    """

    ha = get_h_h0basis(h0, Mv_tot, Bs2[2*it  ], theta_B, phi_B)
    hb = get_h_h0basis(h0, Mv_tot, Bs2[2*it+1], theta_B, phi_B)
    hc = get_h_h0basis(h0, Mv_tot, Bs2[2*it+2], theta_B, phi_B)

    return (ha, hb, hc)

def get_habc_Bz(h0, Mz_tot, it, deltat, Bs2):
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

    ha = get_h_h0basis_Bz(h0, Mz_tot, Bs2[2*it  ])
    hb = get_h_h0basis_Bz(h0, Mz_tot, Bs2[2*it+1])
    hc = get_h_h0basis_Bz(h0, Mz_tot, Bs2[2*it+2])

    return (ha, hb, hc)

def get_habc_reuse_ha(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B, ha, hb, hc):
    """
    Obtain ha = h(ts[it])
           hb = h(ts[it] + deltat/2)
           hc = h(ts[it] + deltat)

    h0 and Mv_tot are on the basis of the eigenvectors of h0.

    ts:  A list of time with a time step of deltat
    Bs2: A list of B fields with a time step of deltat/2
         B(t = ts[it]) = Bs2[2*it]
    """

    hb = get_h_h0basis(h0, Mv_tot, Bs2[2*it+1], theta_B, phi_B)
    hc = get_h_h0basis(h0, Mv_tot, Bs2[2*it+2], theta_B, phi_B)

    return (ha, hb, hc)

def get_habc_reuse_ha_Bz(h0, Mz_tot, it, deltat, Bs2, ha, hb, hc):
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

    hb = get_h_h0basis_Bz(h0, Mz_tot, Bs2[2*it+1])
    hc = get_h_h0basis_Bz(h0, Mz_tot, Bs2[2*it+2])

    return (ha, hb, hc)



def evolve_psi_by_deltat(ha, hb, hc, psi, deltat):
    """
    Evolve psi by deltat using the Runge-Kutta method according to the time dependent Schrodinger equation
    i \partial \psi / \partial t = const1 * h(t) \psi or \partial \psi / \partial t = -i * const1 * h(t) \psi

    Input: 
      ha: h(t), the total Hamiltonian including the Zeeman interaction at time t. 
      hb: h(t+deltat/2)
      hc: h(t+deltat). ha, hb, and hc are  written on the basis of the eigenvectors of h0, i.e. the zero-field spin Hamiltonian.
      psi: wavefunction at time t psi(t) on the basis of the eigenvectors of h0.
           or an array of eigenvectors which are arranged in columns
           in particular, it can be the square matrix "eigenvectors" as returned by np.linalg.eigh

    Units: ps for time, cm-1 for h. psi is unitless.
    """

    k1 = -1j * const1 * np.matmul(ha, psi)
    k2 = -1j * const1 * np.matmul(hb, psi + 0.5*deltat*k1)
    k3 = -1j * const1 * np.matmul(hb, psi + 0.5*deltat*k2)
    k4 = -1j * const1 * np.matmul(hc, psi +     deltat*k3)

    psi_new = psi + deltat * (k1 + 2*k2 + 2*k3 + k4) / 6

    return psi_new

def evolve_psi_by_nt_steps(h0, Mv_tot, eigenvectors, nt, deltat, Bs2, theta_B, phi_B):
    """
    Evolve psi by nt time steps using the Runge-Kutta method according to the time dependent Schrodinger equation..

    h0 and Mv_tot are written on the basis of the eigenvectors of h0.

    Input:
      h0: Hamiltonian at t0 = ts[0] written on the basis of the eigenvectors of h0.
      Mv_tot: Magnetization operators written on the basis of the eigenvectors of h0.
      eigenvectors: initial wavefunctions at t0, each column is one wavefunction.
      nt: number of time steps.
      delta: time step in unit of ps.
      Bs: list of magnetic field.
      theta_B: polar angle of the magnetic field.
      phi_B: azimuthal angle of the magnetic field.
    """

    it = 0; ha, hb, hc = get_habc(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B)

    eigenvectors = evolve_psi_by_deltat(ha, hb, hc, eigenvectors, deltat)
    #print(eigenvectors)

    for it in range(1, nt):
        ha, hb, hc = get_habc_reuse_ha(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B, hc, ha, hb)
        eigenvectors = evolve_deltat(ha, hb, hc, eigenvectors, deltat)

    return eigenvectors

def get_initial_eigenvectors(h0):
    """
    Initial eigenvectors on the basis of the eigenvectors of h0.
    """

    return np.eye(h0.shape[0])

def get_rho_from_eigenvectors(rho0, eigenvectors):
    """
    Construct the density matrix corresponding to eigenvectors based on the inital thermal occupation as stored in rho0.
    See get_rho0 in von_neumann.py for the matrix elements of rho0.
    """

    rho = np.zeros( rho0.shape , dtype=complex)

    eigenvectors_conj = np.conjugate(eigenvectors)

    for i in range(rho0.shape[0]):
        for j in range(rho0.shape[0]):
            for k in range(rho0.shape[0]):
                rho[i, j] = rho[i, j] + rho0[k, k] * eigenvectors[i, k] * eigenvectors_conj[j, k]

    return rho

