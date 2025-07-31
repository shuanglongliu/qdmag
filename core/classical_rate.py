import numpy as np
import pandas as pd
from scipy.linalg import expm
from constants import Kelvin2wavenumber, Tesla2wavenumber
from common import eigen_simple, get_Mz_from_rho
from pulse import get_Bt
import copy
import os

def get_Pe(energies, T):
    """
    Get the density matrix on the perturbed basis for the thermal equilibrium
    rho0_kk = p_k which is the probability of occupying the state |k>
    T: Temperature in Kelvin
    """

    dim = energies.shape[0]

    Pe = np.zeros(dim, dtype=np.float64)

    e_ref = np.min(energies)

    beta = 1/(Kelvin2wavenumber * T)

    for i in range(dim):
        eigenvalue = energies[i] - e_ref
        Pe[i] = np.exp(-beta*eigenvalue)

    Pe = Pe / np.sum(Pe)

    return Pe


def transform_rep_S2E(h, O):
    """
    h is in the S representation
    O is in the S representation
    Transform O into the E representation
    """
    # Diagonalize the Hamiltonian to find eigenvalues and eigenvectors
    eigen = eigen_simple(h)

    # Unitary transformation matrix
    M = eigen.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))

    # Change the basis functions from the eigenstates of the spin operator Sz_tot (S representation)
    # to the eigenstates of the Hamiltonian H (E representation) 
    O_E = np.matmul(M_dagger, np.matmul(O, M))

    return O_E

def transform_rep_E2S(h, O):
    """
    h is in the S representation
    O is in the E representation
    Transform O into the S representation
    """
    # Diagonalize the Hamiltonian to find eigenvalues and eigenvectors
    eigen = eigen_simple(h)

    # Unitary transformation matrix
    M = eigen.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))

    # Change the basis functions from the eigenstates of the Hamiltonian H (E representation)
    # to the eigenstates of the spin operator Sz_tot (S representation)
    O_S = np.matmul(M, np.matmul(O, M_dagger))

    return O_S

def get_rate_matrix(energies, T, dim, nu0, symmetric):
    """
    Build the rate matrix W

    To ensure thermal equilibrium is reached, the rates must satisfy detailed balance:
    W_{m→n}/W_{n→m} = exp(-(E_n - E_m)/k_B T)

    Symmetric form of W:
      W_{m→n} = nu0 * exp(-(E_n - E_m)/(2*k_B T))
      W_{n→m} = nu0 * exp(-(E_m - E_n)/(2*k_B T))

    Unsymmetric form of W:
      W_{m→n} = nu0 * exp(-max(0, E_n - E_m)/(k_B T))
      W_{n→m} = nu0 * exp(-max(0, E_m - E_n)/(k_B T))

    Parameters:
        energies: Sorted list or energies (ascending) of the states in cm-1
        T: Temperature in Kelvin
        dim: Number of states (length of energies)
        nu0: Characteristic frequency in Hz (it ranges from 1 to 1e12)
        symmetric: If True, use symmetric form of W; otherwise, use unsymmetric form

    Returns:
        W: Rate matrix of shape (dim, dim)
           W[i,j] is the rate from state j to state i

           Conservation of probability is ensured:
               dP_i/dt = Σ_j W[i,j] P_j
               Σ_i (dP_i/dt) = 0 or Σ_i Σ_j W[i,j] P_j = 0 or Σ_j (Σ_i W[i,j]) P_j = 0
               Σ_i W[i,j] = 0  for all j
    """

    W = np.zeros((dim, dim))
    
    kB_T = Kelvin2wavenumber * T  # cm-1

    if symmetric:
        # W[i,j] transition rate from state j to state i
        for j in range(dim):
            for i in range(dim):
                if i != j:
                    W[i,j] = nu0 * np.exp(-(energies[i] - energies[j]) / (2*kB_T))
            # Diagonal elements: -sum of outgoing rates
            W[j,j] = -np.sum(W[:,j])
    else:
        # Unsymmetric form
        for j in range(dim):
            for i in range(dim):
                if i != j:
                    W[i,j] = nu0 * np.exp(-max(0, energies[i] - energies[j]) / kB_T)
            # Diagonal elements: -sum of outgoing rates
            W[j,j] = -np.sum(W[:,j])
    
    return W

def update_rate_matrix(W, energies, T, dim, nu0, symmetric):
    """
    Build the rate matrix W

    To ensure thermal equilibrium is reached, the rates must satisfy detailed balance:
    W_{m→n}/W_{n→m} = exp(-(E_n - E_m)/k_B T)

    Symmetric form of W:
      W_{m→n} = nu0 * exp(-(E_n - E_m)/(2*k_B T))
      W_{n→m} = nu0 * exp(-(E_m - E_n)/(2*k_B T))

    Unsymmetric form of W:
      W_{m→n} = nu0 * exp(-max(0, E_n - E_m)/(k_B T))
      W_{n→m} = nu0 * exp(-max(0, E_m - E_n)/(k_B T))

    Parameters:
        energies: Sorted list or energies (ascending) of the states in cm-1
        T: Temperature in Kelvin
        dim: Number of states (length of energies)
        nu0: Characteristic frequency in Hz (it ranges from 1 to 1e12)
        symmetric: If True, use symmetric form of W; otherwise, use unsymmetric form

    Returns:
        W: Rate matrix of shape (dim, dim)
           W[i,j] is the rate from state j to state i

           Conservation of probability is ensured:
               dP_i/dt = Σ_j W[i,j] P_j
               Σ_i (dP_i/dt) = 0 or Σ_i Σ_j W[i,j] P_j = 0 or Σ_j (Σ_i W[i,j]) P_j = 0
               Σ_i W[i,j] = 0  for all j
    """

    W = np.zeros((dim, dim))
    
    kB_T = Kelvin2wavenumber * T  # cm-1

    if symmetric:
        # W[i,j] transition rate from state j to state i
        for j in range(dim):
            for i in range(dim):
                if i != j:
                    W[i,j] = nu0 * np.exp(-(energies[i] - energies[j]) / (2*kB_T))
            # Diagonal elements: -sum of outgoing rates
            W[j,j] = -np.sum(W[:,j])
    else:
        # Unsymmetric form
        for j in range(dim):
            for i in range(dim):
                if i != j:
                    W[i,j] = nu0 * np.exp(-max(0, energies[i] - energies[j]) / kB_T)
            # Diagonal elements: -sum of outgoing rates
            W[j,j] = -np.sum(W[:,j])
    
    return W

def evolve_P_onestair(P, W, h_t0, B, Mz_op, dt, T, dim, nu0, symmetric):
    """
    Evolve the probability vector P over a time step dt
    according to the classical rate equations dP/dt = W @ P where W is the rate matrix

    dP_n/dt = Σ_m [W_{m→n} P_m - W_{n→m} P_n]

    Parameters:
        P: Probability vector of shape (dim,) where dim is the number of states
           P[i] = probability of being in state i
        W: Rate matrix of shape (dim, dim) where W[i,j] is the rate from state j to state i
        dt: Time step for evolution in ps
    """
    h = h_t0 - Tesla2wavenumber * B * Mz_op
    eigen = eigen_simple(h)
    W = update_rate_matrix(W, eigen.eigenvalues, T, dim, nu0, symmetric)
    P = expm(W*dt*1e-12) @ P
    return P

def get_outdirs(T, nu0, Bt_params):
    """
    Get the output directory for the distribution probability and the magnetic moment.
    T: temperature in Kelvin.
    nu0: attempt frequency in Hz.
    Bt_params: parameters for the magnetic fields. See pulse.py for details.
    """
    if Bt_params['Bt_type'] == 'linear':
        outdir = './output/T_{:.1f}K_nu0_{:.2e}/Bt_linear_sweep_rate_{:.1f}'.format(T, nu0, Bt_params['sweep_rate'])
    elif Bt_params['Bt_type'] == 'pwlinear':
        times = Bt_params['times']
        fields = Bt_params['fields']
        outdir = './output/T_{:.1f}K_nu0_{:.2e}/'.format(T, nu0)
        outdir += 'Bt_pwlinear_t{:.1e}ps-B{:.1f}T'.format(times[0], fields[0])
        for i in range(1, len(times)):
            outdir += '_t{:.1e}ps-B{:.1f}T'.format(times[i], fields[i])
        outdir = outdir.replace('+', '')
    elif Bt_params['Bt_type'] == 'pwlinear_by_slope':
        outdir = './output/T_{:.1f}K_nu0_{:.2e}/Bt_pwlinear_average_sweep_rate_{:.1f}'.format(T, nu0, Bt_params['sweep_rate_ave'])
    elif Bt_params['Bt_type'] == 'sin':
        outdir = './output/T_{:.1f}K_nu0_{:.2e}/Bt_sin_amplitude_{:.1f}_omega_{:.2f}'.format(T, nu0, Bt_params['amplitude'], Bt_params['omega'])
    elif Bt_params['Bt_type'] == 'cs':
        outdir = './output/T_{:.1f}K_nu0_{:.2e}/Bt_cs'.format(T, nu0)
    else:
        raise ValueError("Invalid Bt_type: {}".format(Bt_params['Bt_type']))
    return outdir

def get_Mz_from_P(P, h_t0, B, Mz_op):
    """
    Calculate the magnetic moment Mz from the probability vector P

    Parameters:
        P: Probability vector of shape (dim,)
        h_t0: Hamiltonian at time t0 (S representation)
        B: Magnetic field at time t
        Mz_op: Magnetic moment operator of shape (dim, dim) (S representation)
        dim: Dimension of the system

    Returns:
        Mz: Magnetic moment in units of Bohr magneton
    """
    # Density matrix on the basis of the eigenstates of the Hamiltonian at time t
    # which is diagonal with rho[i,i] = P[i]
    # The states are ordered according to the eigenvalues of the Hamiltonian at time t
    rho_E = np.diag(P)

    # Construct the Hamiltonian at time t
    h = h_t0  - Tesla2wavenumber * B * Mz_op

    # Transform rho_E to the S representation
    rho_S = transform_rep_E2S(h, rho_E)

    # Calculate the magnetic moment Mz
    Mz = get_Mz_from_rho(rho_S, Mz_op)

    return Mz

def get_csv_header(dim):
    columns = ['t', 'B']
    for i in range(dim):
        columns.append('P{:d}'.format(i+1))
    columns.append('Mz')
    return columns

def get_csv_row(columns, t, B, P, Mz, dim):
    dict_data = {col: [] for col in columns}
    dict_data['t'].append(t)
    dict_data['B'].append(B)
    for i in range(dim):
        dict_data['P{:d}'.format(i+1)].append(P[i])
    dict_data['Mz'].append(Mz)
    return dict_data

def evolve_P_stairs(P, nu0, symmetric, t0, t1, deltat, Bt_params, T, h_t0, Mz_op, nt_save, dim):
    """
    Evolve the probability vector P from time t0 to t1 in steps of deltat
    """
    # Make a copy of the hamiltonian h_t0 to store the Hamiltonian at time t
    # This is to avoid repeated memory allocation for h.
    h = copy.deepcopy(h_t0)

    # Specify the magnetic field as a function of time
    Bt = get_Bt(Bt_params)

    # Set the initial t and B
    t = t0 + 0.0
    B = Bt(t)

    # Initial magnetic moment
    Mz = get_Mz_from_P(P, h_t0, B, Mz_op)

    # nt should be a multiple of nt_save
    nround = int( max((t1 - t0)//deltat // nt_save, 1) )
    nt = nround * nt_save
    
    # Adjust the final time
    t1 = t0 + nt*deltat

    # Columns of the csv file
    columns = get_csv_header(dim)

    # Create a DataFrame to store the results
    row = get_csv_row(columns, t, B, P, Mz, dim)
    df = pd.DataFrame(row)

    # Get the initial transition rate matrix W
    eigen = eigen_simple(h_t0)
    W = get_rate_matrix(eigen.eigenvalues, T, dim, nu0, True)

    # Loop over the rounds for saving the P and Mz
    for iround in range(nround):
        # Loop over the nt_mag time steps
        for it_ in range(nt_save):
            P = evolve_P_onestair(P, W, h_t0, B, Mz_op, deltat, T, dim, nu0, symmetric)
            it = iround * nt_save + it_
            t = t0 + it*deltat
            B = Bt(t)
        # Calculate the magnetic moment
        Mz = get_Mz_from_P(P, h_t0, B, Mz_op)
        print(P)
        print(Mz)
        row = get_csv_row(columns, t, B, P, Mz, dim)
        # Append the row to the DataFrame
        df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)

    # Output directories
    outdir = get_outdirs(T, nu0, Bt_params)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Output file
    fname = outdir + '/{:.3f}-{:.3f}ps_dt{:.3f}ps.csv'.format(t0, t1, deltat)

    # Save the DataFrame to a CSV file
    df.to_csv(fname, index=False)

    return t, P

