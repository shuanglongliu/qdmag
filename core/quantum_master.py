import numpy as np
from spin_dynamics.core.constants import const1, Kelvin2wavenumber
from spin_dynamics.core.common import get_commutation
from spin_dynamics.core.common import spy_sparsity
from spin_dynamics.core.common import get_habc_Mv, get_habc_Mz, get_habc_reuse_ha_Mv, get_habc_reuse_ha_Mz
from spin_dynamics.core.common import get_rho_upper
from spin_dynamics.core.pulse import get_partial_double_grid, get_partial_double_grid_left
from spin_dynamics import __file__ as root_dir

"""
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

    # Numerical parameter to avoid divergence
    eta = 5.e-2

    beta = 1/(Kelvin2wavenumber * T)
    energy = omega2energy(omega)

    numerator = spectral_density(omega, I0=I0) - spectral_density(-omega, I0=I0)

    if abs( energy ) < eta:
        if energy >= 0:
            energy = eta
        else:
            energy = -1 * eta
        denominator = np.exp(beta * energy) - 1
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

def construct_Rhbar(T, X, energies, I0):
    """
    Construct the auxiliary operator R multiplied by hbar.

    energies: energies of perturbed basis under any finite magnetic field (along z direction)

    Rhbar_{ij} = X_{ij} * Phi_{ij}
    Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
    omega_ij = ( E_i - E_j ) / hbar.
    I(omega) = I0 omega^2 theta(omega): spectral density for phonons
    theta(omega) is the step function.
    """

    Rhbar = np.zeros(X.shape, dtype=np.float64)

    omegas = energy2omega(energies)

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            omega_ij = omegas[i] - omegas[j]
            Rhbar[i, j] = X[i, j] * Phi(T, omega_ij, I0)

    return Rhbar

def update_Rhbar(Rhbar, T, X, indices_nonzero_X, energies, I0):
    """
    Update the auxiliary operator R multiplied by hbar.
    Save time by avoiding memory allocation for Rhbar.

    indices_nonzero_X: indices of nonzero matrix elements of X
    energies: energies of perturbed basis under any finite magnetic field (along z direction)

    Rhbar_{ij} = X_{ij} * Phi_{ij}
    Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
    omega_ij = ( E_i - E_j ) / hbar.
    I(omega) = I0 omega^2 theta(omega): spectral density for phonons
    theta(omega) is the step function.

    Assumption:
      1. The perturbed basis are the eigenvectors under zero and finite magnetic fields.
      2. The order of the perturbed basis functions does not change, although the order of their energies do change.
    """

    omegas = energy2omega(energies)

    n = indices_nonzero_X[0].shape[0] # Number of nonzero matrix elements of X
    for k in range(n):
        i = indices_nonzero_X[0][k]
        j = indices_nonzero_X[1][k]
        omega_ij = omegas[i] - omegas[j]
        Rhbar[i, j] = X[i, j] * Phi(T, omega_ij, I0)

    return Rhbar

def spy_XRhbar(X, Rhbar, Sz_tot):
    n = X.shape[0]

    with open(root_dir + "output/X.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("i   j   Sz_tot_i   Sz_tot_j   X_ij   = {:5d}   {:5d}   {:8.3f}   {:8.3f}   {:5.1f}\n".format( \
                     i, j, np.real(Sz_tot[i, i]), np.real(Sz_tot[j, j]), X[i, j]))

    with open(root_dir + "output/Rhbar.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, Rhbar[i, j]))
 
    spy_sparsity(X, "X", precision=1.0e-8, figsize=(10, 10), markersize=5)

    max_Rhbar = np.max(np.absolute(Rhbar))
    spy_sparsity(Rhbar, "Rhbar", precision = 0.1*max_Rhbar, figsize=(10, 10), markersize=5)

    return


def get_Gammarho(rho, X, Rhbar, lambda2):
    """
    lambda2 = lambdaa^2 * pi * const1^2
    """

    Rhbarrho = np.matmul(Rhbar, rho)
    commutation = np.matmul(X, Rhbarrho) - np.matmul(Rhbarrho, X)
    Gammarho = lambda2 * ( commutation + np.transpose(np.conjugate(commutation)) )

    return Gammarho


def evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat):
    """
    Evolve rho by deltat using the Runge-Kutta method according to the quantum master equation
    d rho / d t = -i * const1 * [H, rho] - { [X, Rhbar rho] + [X, Rhbar rho]^\dagger } * lambdaa^2 * pi * const1^2

    Input: 
      ha: h(t), the total Hamiltonian including the Zeeman interaction at time t. Unit: cm-1.
      hb: h(t+deltat/2)
      hc: h(t+deltat). ha, hb, and hc are  written on the basis of the eigenvectors of h0, i.e. the zero-field spin Hamiltonian.
      rho: density matrix at time t rho(t) on the basis of the eigenvectors of h0.
      X: the operator in the spin space, which couples to phonons. See construct_X in this file. Unitless.
      Rhbar: an auxiliary operator R multiplied by hbar. See construct_Rhbar in this file. Unitless.
      lambda1 = -1j * const1
      lambda2 = lambdaa^2 * pi * const1^2, lambdaa: spin-phonon coupling constant in cm-1.
      deltat: time step in ps.
    """

    k1                             = lambda1 * ( np.matmul(ha, rho ) - np.matmul(rho , ha) ) - get_Gammarho(rho , X, Rhbar, lambda2)
    rho1 = rho + 0.5*deltat*k1; k2 = lambda1 * ( np.matmul(hb, rho1) - np.matmul(rho1, hb) ) - get_Gammarho(rho1, X, Rhbar, lambda2)
    rho2 = rho + 0.5*deltat*k2; k3 = lambda1 * ( np.matmul(hb, rho2) - np.matmul(rho2, hb) ) - get_Gammarho(rho2, X, Rhbar, lambda2)
    rho3 = rho +     deltat*k3; k4 = lambda1 * ( np.matmul(hc, rho3) - np.matmul(rho3, hc) ) - get_Gammarho(rho3, X, Rhbar, lambda2)

    rho_new = rho + deltat * (k1 + 2*k2 + 2*k3 + k4) / 6

    return rho_new

def evolve_rho_qme_Mv_light(h0, Mv_tot, rho, nt, deltat, Bs2, theta_B, phi_B, X, Rhbar, lambda1, lambda2):
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
      X, Rhbar: Auxiliary operators for constructing the \Gamma operator for spin-phonon coupling.
      lambda1 = -1j * const1
      lambda2 = lambdaa^2 * pi * const1^2, lambdaa: spin-phonon coupling constant in cm-1.
    """

    it = 0; ha, hb, hc = get_habc_Mv(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B)
    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
    #print("it = {:d}, max(rho) = {:12.4e}".format(it, np.max(np.absolute(rho))))

    for it in range(1, nt):
        ha, hb, hc = get_habc_reuse_ha_Mv(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B, hc, ha, hb)
        rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
        #print("it = {:d}, max(rho) = {:12.4e}".format(it, np.max(np.absolute(rho))))

    return rho


def evolve_rho_qme_Mz_light(h0, Mz_tot, rho, nt, deltat, Bs2, X, Rhbar, lambda1, lambda2):
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
      X, Rhbar: Auxiliary operators for constructing the \Gamma operator for spin-phonon coupling.
      lambda1 = -1j * const1
      lambda2 = lambdaa^2 * pi * const1^2, lambdaa: spin-phonon coupling constant in cm-1.

    Assumptions:
      The magnetic field is along the z direction.
    """

    it = 0; ha, hb, hc = get_habc_Mz(h0, Mz_tot, it, deltat, Bs2)
    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
    #print("it = {:d}, max(rho) = {:12.4e}".format(it, np.max(np.absolute(rho))))

    for it in range(1, nt):
        ha, hb, hc = get_habc_reuse_ha_Mz(h0, Mz_tot, it, deltat, Bs2, hc, ha, hb)
        rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
        #print("it = {:d}, max(rho) = {:12.4e}".format(it, np.max(np.absolute(rho))))

    return rho

def evolve_rho_qme_Mv(h0, Mv_tot, rho, nt, ts, deltat, Bs2, theta_B, phi_B, X, Rhbar, lambda1, lambda2, save_mag, deltat_mag, save_rho, deltat_rho):
    """
    Evolve the density matrix using the Runge-Kutta method according to the quantum master equation.

    h0 and Mv_tot are written on the perturbed basis.

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
      X, Rhbar: Auxiliary operators for constructing the \Gamma operator for spin-phonon coupling.
      lambda1 = -1j * const1
      lambda2 = lambdaa^2 * pi * const1^2, lambdaa: spin-phonon coupling constant in cm-1.
      save_mag: calculate and save magnetization?
      deltat_mag: calculate and save magnetization per deltat_mag ps
      save_rho: save the density matrix?
      deltat_rho: save the density matrix per deltat_rho ps
    """



    nt_mag = round( deltat_mag / deltat )
    nt_rho = round( deltat_rho / deltat )

    if nt_rho > nt_mag:
        ratio = round(nt_rho/nt_mag)
        nt_rho = ratio * nt_mag
    else:
        ratio = round(nt_mag/nt_rho)
        nt_mag = ratio * nt_rho

    deltat_mag = nt_mag * deltat
    deltat_rho = nt_rho * deltat

    print("Rounded time step for saving the magnetization : {:12.4E} ps".format(deltat_mag))
    print("Rounded time step for saving the density matrix: {:12.4E} ps".format(deltat_rho))

    nround_mag = nt // nt_mag
    nround_rho = nt // nt_rho

    if save_mag:
        ostring1 = "{:18.3f} {:18.10e} {:12.6f} {:12.6f} {:12.6f}\n"
        f1 = open(root_dir + "output/Mv_vs_t.dat", "w")

    if save_rho:
        indices_upper = np.triu_indices(rho.shape[0])
        ostring2 = "{:18.3f} {:18.10e}" + indices_upper[0].shape[0] * " {:18.10e}" + "\n"
        f2 = open(root_dir + "output/rho_vs_t.dat", "w")

    if nt_rho > nt_mag:
        nt_part = nt_mag
        n_part = nt // nt_part

        for iround_rho in range(1):
            for iround_mag in range(1):
                i_part = iround_rho*ratio + iround_mag
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, nround_rho, iround_mag, ratio))

                it_part = 0; ha, hb, hc = get_habc_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B)
                rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
    
                for it_part in range(1, nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
                
                if save_mag:
                    Mx = np.real( np.trace( np.matmul(rho, Mv_tot[0]) ) )
                    My = np.real( np.trace( np.matmul(rho, Mv_tot[1]) ) )
                    Mz = np.real( np.trace( np.matmul(rho, Mv_tot[2]) ) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))
                    print(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))        #delete#
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
        for iround_rho in range(1):
            for iround_mag in range(1, ratio):
                i_part = iround_rho*ratio + iround_mag
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, nround_rho, iround_mag, ratio))

                for it_part in range(nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

                if save_mag:
                    Mx = np.real( np.trace( np.matmul(rho, Mv_tot[0]) ) )
                    My = np.real( np.trace( np.matmul(rho, Mv_tot[1]) ) )
                    Mz = np.real( np.trace( np.matmul(rho, Mv_tot[2]) ) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))
                    print(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))        #delete#
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
        for iround_rho in range(1, nround_rho):
            for iround_mag in range(ratio):
                i_part = iround_rho*ratio + iround_mag
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, nround_rho, iround_mag, ratio))

                for it_part in range(nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

                if save_mag:
                    Mx = np.real( np.trace( np.matmul(rho, Mv_tot[0]) ) )
                    My = np.real( np.trace( np.matmul(rho, Mv_tot[1]) ) )
                    Mz = np.real( np.trace( np.matmul(rho, Mv_tot[2]) ) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))
                    print(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))        #delete#
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))

        nt_left = nt - nround_rho * nt_rho

        if nt_left != 0:
            ts_part, Bs2_part = get_partial_double_grid_left(nt, ts, Bs2, nt_left)

            for it_part in range(nt_left):
                ha, hb, hc = get_habc_reuse_ha_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B, hc, ha, hb)
                rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

            if save_mag:
                Mx = np.real( np.trace( np.matmul(rho, Mv_tot[0]) ) )
                My = np.real( np.trace( np.matmul(rho, Mv_tot[1]) ) )
                Mz = np.real( np.trace( np.matmul(rho, Mv_tot[2]) ) )
                f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
    else:
        nt_part = nt_rho
        n_part = nt // nt_part

        for iround_mag in range(1):
            for iround_rho in range(1):
                i_part = iround_mag*ratio + iround_rho
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, ratio, iround_mag, nround_mag))

                it_part = 0; ha, hb, hc = get_habc_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B)
                rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
    
                for it_part in range(1, nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
                
                if save_mag:
                    Mx = np.real( np.trace( np.matmul(rho, Mv_tot[0]) ) )
                    My = np.real( np.trace( np.matmul(rho, Mv_tot[1]) ) )
                    Mz = np.real( np.trace( np.matmul(rho, Mv_tot[2]) ) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))
                    print(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))        #delete#
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
        for iround_mag in range(1):
            for iround_rho in range(1, ratio):
                i_part = iround_mag*ratio + iround_rho
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, ratio, iround_mag, nround_mag))

                for it_part in range(nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

                if save_mag:
                    Mx = np.real( np.trace( np.matmul(rho, Mv_tot[0]) ) )
                    My = np.real( np.trace( np.matmul(rho, Mv_tot[1]) ) )
                    Mz = np.real( np.trace( np.matmul(rho, Mv_tot[2]) ) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))
                    print(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))        #delete#
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
        for iround_mag in range(1, nround_mag):
            for iround_rho in range(ratio):
                i_part = iround_mag*ratio + iround_rho
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, ratio, iround_mag, nround_mag))

                for it_part in range(nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

                if save_mag:
                    Mx = np.real( np.trace( np.matmul(rho, Mv_tot[0]) ) )
                    My = np.real( np.trace( np.matmul(rho, Mv_tot[1]) ) )
                    Mz = np.real( np.trace( np.matmul(rho, Mv_tot[2]) ) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))
                    print(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))        #delete#
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))

        nt_left = nt - nround_mag * nt_mag

        if nt_left != 0:
            ts_part, Bs2_part = get_partial_double_grid_left(nt, ts, Bs2, nt_left)

            for it_part in range(nt_left):
                ha, hb, hc = get_habc_reuse_ha_Mv(h0, Mv_tot, it_part, deltat, Bs2_part, theta_B, phi_B, hc, ha, hb)
                rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

            if save_mag:
                Mx = np.real( np.trace( np.matmul(rho, Mv_tot[0]) ) )
                My = np.real( np.trace( np.matmul(rho, Mv_tot[1]) ) )
                Mz = np.real( np.trace( np.matmul(rho, Mv_tot[2]) ) )
                f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], Mx, My, Mz))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
    if save_mag:
        f1.close()

    if save_rho:
        f2.close()

    return rho




def evolve_rho_qme_Mz(h0, Mz_tot, rho, nt, ts, deltat, Bs2, X, Rhbar, lambda1, lambda2, save_mag, deltat_mag, save_rho, deltat_rho):
    """
    Evolve the density matrix using the Runge-Kutta method according to the quantum master equation.

    h0 and Mz_tot are written on the perturbed basis.

    Input:
      h0: Hamiltonian at t0 = ts[0] written on the basis of the eigenvectors of h0.
      Mz_tot: Magnetization operators written on the perturbed basis.
      rho: initial density matrix at t0.
      nt: number of time steps.
      ts: list of time in unit of ps.
      delta: time step in unit of ps.
      Bs2: a list of B fields with a time step of deltat/2
           B(t = ts[it]) = Bs2[2*it]
      theta_B: polar angle of the magnetic field.
      phi_B: azimuthal angle of the magnetic field.
      X, Rhbar: Auxiliary operators for constructing the \Gamma operator for spin-phonon coupling.
      lambda1 = -1j * const1
      lambda2 = lambdaa^2 * pi * const1^2, lambdaa: spin-phonon coupling constant in cm-1.
      save_mag: calculate and save magnetization?
      deltat_mag: calculate and save magnetization per deltat_mag ps
      save_rho: save the density matrix?
      deltat_rho: save the density matrix per deltat_rho ps

    Assumptions:
      The magnetic field is along the z direction.
    """



    nt_mag = round( deltat_mag / deltat )
    nt_rho = round( deltat_rho / deltat )

    if nt_rho > nt_mag:
        ratio = round(nt_rho/nt_mag)
        nt_rho = ratio * nt_mag
    else:
        ratio = round(nt_mag/nt_rho)
        nt_mag = ratio * nt_rho

    deltat_mag = nt_mag * deltat
    deltat_rho = nt_rho * deltat

    print("Rounded time step for saving the magnetization : {:12.4E} ps".format(deltat_mag))
    print("Rounded time step for saving the density matrix: {:12.4E} ps".format(deltat_rho))

    nround_mag = nt // nt_mag
    nround_rho = nt // nt_rho

    if save_mag:
        ostring1 = "{:18.3f} {:18.10e} {:12.6f} {:12.6f}\n"
        f1 = open(root_dir + "output/Mz_vs_t.dat", "w")

    if save_rho:
        indices_upper = np.triu_indices(rho.shape[0])
        ostring2 = "{:18.3f} {:18.10e}" + indices_upper[0].shape[0] * " {:18.10e}" + "\n"
        f2 = open(root_dir + "output/rho_vs_t.dat", "w")

    if nt_rho > nt_mag:
        nt_part = nt_mag
        n_part = nt // nt_part

        for iround_rho in range(1):
            for iround_mag in range(1):
                i_part = iround_rho*ratio + iround_mag
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, nround_rho, iround_mag, ratio))

                it_part = 0; ha, hb, hc = get_habc_Mz(h0, Mz_tot, it_part, deltat, Bs2_part)
                rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
    
                for it_part in range(1, nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mz(h0, Mz_tot, it_part, deltat, Bs2_part, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
                
                if save_mag:
                    Mz = np.trace( np.matmul(rho, Mz_tot) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], np.real(Mz), np.imag(Mz)))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
        for iround_rho in range(1):
            for iround_mag in range(1, ratio):
                i_part = iround_rho*ratio + iround_mag
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, nround_rho, iround_mag, ratio))

                for it_part in range(nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mz(h0, Mz_tot, it_part, deltat, Bs2_part, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

                if save_mag:
                    Mz = np.trace( np.matmul(rho, Mz_tot) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], np.real(Mz), np.imag(Mz)))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
        for iround_rho in range(1, nround_rho):
            for iround_mag in range(ratio):
                i_part = iround_rho*ratio + iround_mag
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, nround_rho, iround_mag, ratio))

                for it_part in range(nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mz(h0, Mz_tot, it_part, deltat, Bs2_part, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

                if save_mag:
                    Mz = np.trace( np.matmul(rho, Mz_tot) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], np.real(Mz), np.imag(Mz)))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))

        nt_left = nt - nround_rho * nt_rho

        if nt_left != 0:
            ts_part, Bs2_part = get_partial_double_grid_left(nt, ts, Bs2, nt_left)

            for it_part in range(nt_left):
                ha, hb, hc = get_habc_reuse_ha_Mz(h0, Mz_tot, it_part, deltat, Bs2_part, hc, ha, hb)
                rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

            if save_mag:
                Mz = np.trace( np.matmul(rho, Mz_tot) )
                f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], np.real(Mz), np.imag(Mz)))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
    else:
        nt_part = nt_rho
        n_part = nt // nt_part

        for iround_mag in range(1):
            for iround_rho in range(1):
                i_part = iround_mag*ratio + iround_rho
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, ratio, iround_mag, nround_mag))

                it_part = 0; ha, hb, hc = get_habc_Mz(h0, Mz_tot, it_part, deltat, Bs2_part)
                rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
    
                for it_part in range(1, nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mz(h0, Mz_tot, it_part, deltat, Bs2_part, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)
                
                if save_mag:
                    Mz = np.trace( np.matmul(rho, Mz_tot) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], np.real(Mz), np.imag(Mz)))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
        for iround_mag in range(1):
            for iround_rho in range(1, ratio):
                i_part = iround_mag*ratio + iround_rho
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, ratio, iround_mag, nround_mag))

                for it_part in range(nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mz(h0, Mz_tot, it_part, deltat, Bs2_part, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

                if save_mag:
                    Mz = np.trace( np.matmul(rho, Mz_tot) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], np.real(Mz), np.imag(Mz)))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
        for iround_mag in range(1, nround_mag):
            for iround_rho in range(ratio):
                i_part = iround_mag*ratio + iround_rho
                ts_part, Bs2_part = get_partial_double_grid(nt, ts, Bs2, nt_part, i_part)
    
                print("i_part / n_part = {:6d} / {:6d}, iround_rho / nround_rho = {:6d} / {:6d}, iround_mag / nround_mag = {:6d} / {:6d}".format(i_part, n_part, iround_rho, ratio, iround_mag, nround_mag))

                for it_part in range(nt_part):
                    ha, hb, hc = get_habc_reuse_ha_Mz(h0, Mz_tot, it_part, deltat, Bs2_part, hc, ha, hb)
                    rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

                if save_mag:
                    Mz = np.trace( np.matmul(rho, Mz_tot) )
                    f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], np.real(Mz), np.imag(Mz)))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))

        nt_left = nt - nround_mag * nt_mag

        if nt_left != 0:
            ts_part, Bs2_part = get_partial_double_grid_left(nt, ts, Bs2, nt_left)

            for it_part in range(nt_left):
                ha, hb, hc = get_habc_reuse_ha_Mz(h0, Mz_tot, it_part, deltat, Bs2_part, hc, ha, hb)
                rho = evolve_rho_by_deltat_qme(ha, hb, hc, rho, X, Rhbar, lambda1, lambda2, deltat)

            if save_mag:
                Mz = np.trace( np.matmul(rho, Mz_tot) )
                f1.write(ostring1.format(ts_part[-1], Bs2_part[-1], np.real(Mz), np.imag(Mz)))
            if save_rho:
                rho_upper = get_rho_upper(rho, indices_upper)
                f2.write(ostring2.format(ts_part[-1], Bs2_part[-1], *rho_upper))
    
    if save_mag:
        f1.close()

    if save_rho:
        f2.close()

    return rho


