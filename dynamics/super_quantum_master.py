import os
import copy
import numpy as np
import h5py
from filelock import FileLock
from scipy.linalg import expm
from spin_dynamics.dynamics.constants import const1, Tesla2wavenumber
from spin_dynamics.dynamics.common import kronecker_delta
from spin_dynamics.dynamics.common import spy_sparsity
from spin_dynamics.dynamics.common import get_Mv_from_rho, get_Mz_from_rho
from spin_dynamics.dynamics.common import eigen_simple
from spin_dynamics.dynamics.quantum_master import get_Rhbar, update_Rhbar
from spin_dynamics.dynamics.pulse import get_Bt

r"""
Codes for solving the quantum master equation described in the Eq. 2.7 of
J. Phys. Soc. Jpn. 2001.70:2151-2157 by Hiroki Nakano and Seiji Miyashita.
The quantum master equation is cast into the Liouville form with the real 
and imaginary parts of the density matrix treated as independent variables.
The need to separate the real and imaginary parts comes from the Hermitian
conjugate operation in the \Gamma\rho term, which cannot be written as a
matrix product. After the separation of the real and imaginary parts, the
Liouville superoperator can be merely written as a matrix, which should 
allow speed up with vectorization and parallelization.

The quantum master equation reads (in the units given later)
    d (rhore, rhoim)^T / d t = np.array([[D11, D12], [D21, D22]]) (rhore, rhoim)^T

    D11 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre + CSTre)
    D12 =  const1 * Are + lambdaa**2 pi const1**2 (Cim + CSTim)
    D21 = -const1 * Are - lambdaa**2 pi const1**2 (Cim - CSTim)
    D22 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre - CSTre)
 
Units: 
    cm-1 for energy
    ps for time

    cm-1 for the spin-phonon coupling constant lambdaa
    ps for spectral density I and and its prefactor I0
    ps-1 for frequency omega
    X and Rhbar are unitless

Dimensions:
    dim            : Dimension of the effective Hamiltonian
    dims  = dim^2  : Dimension of superoperators
    dimds = 2*dim^2: Dimension of double superoperators
"""

# ============================================================================ #
# Functions for formatting the density matrix.
# ============================================================================ #

def convert_rho_to_dsrho(rho):
    """
    Convert density matrix to double super density matrix.
    """
    super_rho = rho.flatten()
    super_rho_re = np.real(super_rho)
    super_rho_im = np.imag(super_rho)
    dsrho = np.concatenate((super_rho_re, super_rho_im))
    return dsrho

def convert_dsrho_to_rho(dsrho, dim, dims, dimds):
    """
    Convert double super density matrix to density matrix.
    """
    super_rho_re = dsrho[0:dims]
    super_rho_im = dsrho[dims:dimds]
    super_rho = super_rho_re + 1j * super_rho_im
    rho = super_rho.reshape((dim, dim))
    return rho



# ============================================================================ #
# Functions for calculating magnetic moment. 
# ============================================================================ #

def get_Mv_from_dsrho(dsrho, Mv, dim, dims, dimds):

    rho = convert_dsrho_to_rho(dsrho, dim, dims, dimds)

    M = get_Mv_from_rho(rho, Mv)

    return M

def get_Mz_from_dsrho(dsrho, Mz, dim, dims, dimds):

    rho = convert_dsrho_to_rho(dsrho, dim, dims, dimds)

    return get_Mz_from_rho(rho, Mz)

# ============================================================================ #
# Functions for calculating magnetic susceptibility.
# ============================================================================ #

def get_chimz_from_rho(h, Bt, t, Mz_tot, rho, X, Rhbar, lambdaa, dt=1e+3):
    r"""
    Assume that the magnetic field is along the z direction.

    I do not know why but this function only works for isotropic exchange interaction.

    chimz = dMz/dBz
          = d ( Tr(rho Mz) ) / dBz
          = Tr( drho/dBz Mz ) # Use the same basis functions/states throughout the calculation.
          = Tr( drho/dt dt/dBz Mz )
          = Tr( drho/dt (dBz/dt)^-1 Mz )
          = Tr( (-i * const1 * [H, rho] - { [X, Rhbar rho] + [X, Rhbar rho]^\dagger } * lambdaa^2 * pi* const1^2) (dBz/dt)^-1 Mz )
    """
    drhodt = -1j * const1 * (np.matmul(h, rho) - np.matmul(rho, h))
    commutator = np.matmul(X, np.matmul(Rhbar, rho)) - np.matmul(np.matmul(Rhbar, rho), X)
    drhodt = drhodt - (commutator + np.conjugate(np.transpose(commutator))) * lambdaa**2 * np.pi * const1**2
    dBzdt = (Bt(t+dt) - Bt(t-dt))/(2*dt) # Tesla/ps
    chimz = np.real( np.trace( np.matmul(drhodt, Mz_tot) ) / dBzdt )
    # print("    chimz = {:15.6e}".format(chimz))
    return chimz

def get_chimz_from_dsrho(h, h_t0, Bt, t, Mz, dsrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds):
    """
    Wrapper for get_chimz_from_rho
    h_t0: Hamiltonian at time t=0 ps
    Bt: Magnetic field as a function of time t
    t: Time
    Mz: z component of the total magnetic moment operator
    dsrho: Double super density matrix
    X: Spin transition matrix
    Rhbar: Spin-phonon coupling superoperator
    lambdaa: Spin-phonon coupling constant
    dim: Dimension of the effective Hamiltonian
    dims: Dimension of superoperators
    dimds: Dimension of double superoperators
    """

    # Hamiltonian at time t
    h = h_t0 - Tesla2wavenumber * Bt(t) * Mz

    # Rhbar at time t
    Rhbar = update_Rhbar(Rhbar, h, X, I0, T)

    # Get the density matrix rho from the double super density matrix dsrho
    rho = convert_dsrho_to_rho(dsrho, dim, dims, dimds)

    # Call get_chimz_from_rho
    chimz = get_chimz_from_rho(h, Bt, t, Mz, rho, X, Rhbar, lambdaa)

    return chimz


def get_chimz_finite_diff(dsrho, t1, dt, Bt, D, D0, h, h_t0, Mz_op, Mz_diag, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    """
    dsrho: Double super density matrix at time t1
    """

    B1 = Bt(t1)
    B2 = Bt(t1 + dt)

    M1 = get_Mz_from_dsrho(dsrho, Mz_op, dim, dims, dimds)

    dsrho2 = evolve_rho_dsqme_onestair(dsrho, dt, D, D0, h, h_t0, Mz_diag, B1, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    M2 = get_Mz_from_dsrho(dsrho2, Mz_op, dim, dims, dimds)

    chimz = (M2 - M1) / (B2 - B1)

    # print("    chimz = {:15.6e}".format(chimz))

    return chimz


# ============================================================================ #
# Functions for constructing the Liouville superoperator.
# ============================================================================ #

def get_composite_index(i, j, N):
    I = i * N + j
    return I

def dec_composite_index(I, N):
    i = I // N
    j = I % N
    return (i, j)

def construct_A(H, dim, dims, diagonal_H=False, dtype=np.complex128):
    """
    [H, rho]_I = A_{IJ} rho_J
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    A_{IJ} = H_{ik} delta_{lj} - delta_{ik} H_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N
    dim: dimension of the Hilbert space. N = dim in the above comments.
    dims: dimension of a super operator. dims = dim**2.
    diagonal_H: If True, then H is diagonal and construct A from the diagonal elements of H.
    dype = np.complex128 or np.float64
    If H is diagonal, then A is diagonal with 
      A_{II} = Hii - Hjj
      i = I // N
      j = I % N
    If H is diagonal, then H (and thus A) is real.
    """

    A = np.zeros((dims, dims), dtype=dtype)
    
    if diagonal_H:
        #for I in range(dims):
            #i, j = dec_composite_index(I, dim)
            #A[I, I] = H[i, i] - H[j, j]
        
        # Faster
        #for i in range(dim):
            #for j in range(dim):
                #I = i * dim + j
                #A[I, I] = H[i, i] - H[j, j]
        
        # Even faster
        I = 0
        for i in range(dim):
            for j in range(dim):
                A[I, I] = H[i, i] - H[j, j]
                I = I + 1
    else:
        for I in range(dims):
            i, j = dec_composite_index(I, dim)
            for J in range(dims):
                k, l = dec_composite_index(J, dim)
                A[I, J] = H[i, k] * kronecker_delta(l, j) - kronecker_delta(i, k) * H[l, j]

    result = A

    return result

def construct_A_diag(H_diag, dim, dims):
    """
    Assumption: H is diagonal and real.
    Get the diagonal elements of the superoperator A from the diagonal elements of the Hamiltonian H.
    """
    A_diag = np.zeros(dims, dtype=np.float64)

    I = 0
    for i in range(dim):
        for j in range(dim):
            A_diag[I] = H_diag[i] - H_diag[j]
            I = I + 1

    return A_diag

def construct_C(X, Rhbar, dim, dims):
    """
    Both X and Rhbar are in the S representation.

    [X, Rhbar rho]_{I} = C_{IJ} rho_{J}
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    C_{IJ} = (X Rhbar)_{ik} delta_{lj} - Rhbar_{ik} X_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N

    Note: 
        X_S (where S denotes the representation) is a matrix of real numbers. 
        X_E = M^dagger X_S M can be a matrix of complex numbers.
        The matrix element Rhbar_E(m, n) = X_E(m, n) * Phi(m, n) thus can be a complex number.
        So are the matrix elments of Rhbar_S = M Rhbar_E M^dagger.
    """

    XRhbar = np.matmul(X, Rhbar)

    C = np.zeros((dims, dims), dtype=np.complex128)

    for I in range(dims):
        i, j = dec_composite_index(I, dim)
        for J in range(dims):
            k, l = dec_composite_index(J, dim)
            C[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - Rhbar[i, k] * X[l, j]

    return C

def get_indices_nzC(X, dim):
    """
    Get indices of nonzero matrix elements of C
    X: a matrix of dimension dim x dim for the spin transitions
    C: a super operator of dimension dims due to the interaction with bath
       see the function construct_C for the definition of C
    """

    # Get indices of nonzero matrix elements of X
    indices_nzX = np.nonzero(X)

    # Convert the list [[i1, i2, ..., in], [j1, j2, ..., jn]] to the set {(i1, j1), (i2, j2), ..., (in, jn)}
    set_index_tuples_nzX = set( [(indices_nzX[0][i], indices_nzX[1][i]) for i in range(indices_nzX[0].shape[0])] )

    # Get indices of nonzero matrix elements of C
    indices_nzC = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    if (l == j) or  ( (l, j) in set_index_tuples_nzX ):
                        indices_nzC.append( (i, j, k, l) )
    n_nzC = len(indices_nzC)

    return (n_nzC, indices_nzC)

def construct_CST(C, dim, dims):
    """
    Get super transpose of the superoperator C.
    CST_{I J} = C_{It J}
      I -> ij -> ji -> It
    """

    CST = np.zeros((dims, dims), dtype=np.complex128)

    for I in range(dims):
        i, j = dec_composite_index(I, dim)
        It = get_composite_index(j, i, dim)
        CST[I] = C[It]

    return CST

def construct_D(A, C, dim, dims, lambdaa):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[D11, D12], [D21, D22]]) (rhore, rhoim)^T
    
        D11 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre + CSTre)
        D12 =  const1 * Are + lambdaa**2 pi const1**2 (Cim + CSTim)
        D21 = -const1 * Are - lambdaa**2 pi const1**2 (Cim - CSTim)
        D22 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre - CSTre)

    A: Superoperator for coherent evolution. See the function construct_A.
    C: Superoperator for spin-phonon coupling (incoherent evolution). See the function construct_C.
    dim: Dimension of the effective Hamiltonian
    dims: Dimension of superoperators
    lambdaa: spin-phonon coupling constant
    only_from_A: If True, then only the contribution from A are included in D.

    Note:
        A is real if H is diagonal (and thus real).
    """

    Are = np.real(A)
    Aim = np.imag(A)

    if C is None:
        # Construct D from A only
        D11 =  const1 * Aim
        D12 =  const1 * Are
        D21 = -const1 * Are
        D22 =  const1 * Aim
    else:
        CST = construct_CST(C, dim, dims)
        Cre = np.real(C)
        Cim = np.imag(C)
        CSTre = np.real(CST)
        CSTim = np.imag(CST)
        D11 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (Cre + CSTre)
        D12 =  const1 * Are + lambdaa**2 * np.pi * const1**2 * (Cim + CSTim)
        D21 = -const1 * Are - lambdaa**2 * np.pi * const1**2 * (Cim - CSTim)
        D22 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (Cre - CSTre)
        
    D1 = np.hstack((D11, D12))
    D2 = np.hstack((D21, D22))
    D  = np.vstack((D1 , D2 ))

    return D

def construct_D_from_A_diag(A_diag, dims):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[D11, D12], [D21, D22]]) (rhore, rhoim)^T
    
        D11 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre + CSTre)
        D12 =  const1 * Are + lambdaa**2 pi const1**2 (Cim + CSTim)
        D21 = -const1 * Are - lambdaa**2 pi const1**2 (Cim - CSTim)
        D22 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre - CSTre)

    A_diag = A.diagonal()

    Assume that A is diagonal and thus real. 

    dims: dimension of a super operator. dims = dim**2, where dim is the dimenion of the Hilbert space.
    """

    D11 = np.zeros((dims, dims), dtype=np.float64)
    D12 = np.zeros((dims, dims), dtype=np.float64)
    D21 = np.zeros((dims, dims), dtype=np.float64)
    D22 = np.zeros((dims, dims), dtype=np.float64)

    c1A_diag = const1 * A_diag
    for i in range(dims):
        D12[i, i] =   c1A_diag[i]
        D21[i, i] = - c1A_diag[i]

    D1 = np.hstack((D11, D12))
    D2 = np.hstack((D21, D22))
    D  = np.vstack((D1 , D2 ))

    #D = csr_array(D)

    return D

def set_up_double_super_qme(h_t0_eff, h_tmin_eff, X_eff, dim, I0, T, lambdaa):
    """
    h_t0: Hamiltonian at t = 0
    h_tmin: Hamiltonian at t = tmin

    D0_eff: D matrix at t = 0 ps without spin-phonon coupling
    D_eff: D matrix at t = tmin ps with spin-phonon coupling

    dim: dimension of the effective Hilbert space.
    """

    # Dimensions

    dims  = dim*dim              # Dimension of superoperators
    dimds = 2*dims               # Dimension of double superoperators

    print("Dimension of the effective Hamiltonian: {:6d}".format(dim))
    print("Dimension of superoperators: {:6d}".format(dims))
    print("Dimension of double superoperators: {:6d}\n".format(dimds))

    # Construct the superoperator D_t0_eff at t = 0 ps without spin-phonon coupling
    A_t0_eff = construct_A(h_t0_eff, dim, dims, diagonal_H=False, dtype=np.complex128)
    D_t0_eff = construct_D(A_t0_eff, None, dim, dims, lambdaa)

    # Construct the superoperator A_tmin_eff at t = tmin ps
    A_tmin_eff = construct_A(h_tmin_eff, dim, dims, diagonal_H=False, dtype=np.complex128)

    # Construct Rhbar, C, and CST using the energy spectrum at t = tmin
    # We set them up at the beginning to reuse the memory and save time.
    # Rhbar_eff will also be used to calculate chimz at t = tmin.

    # Construct the superoperator Rhbar_eff in the S representation at t = tmin
    Rhbar_eff = get_Rhbar(h_tmin_eff, X_eff, T, I0)

    # Construct the superoperator C_eff using the energy spectrum at t = tmin
    C_eff = construct_C(X_eff, Rhbar_eff, dim, dims)
    CST_eff = construct_CST(C_eff, dim, dims)

    # Construct the superoperator D_tmin_eff at t = tmin ps with spin-phonon coupling
    D_tmin_eff = construct_D(A_tmin_eff, C_eff, dim, dims, lambdaa)

    return (D_t0_eff, D_tmin_eff, Rhbar_eff, C_eff, CST_eff, dims, dimds)

# ============================================================================ #
# Functions for updating the Liouville superoperator.
# ============================================================================ #

def update_C_and_CST(C, CST, X, Rhbar, dim, n_nzC, indices_nzC):
    """
    Save time by avoiding memory allocation for C.
    Rhbar is time dependent, and so is C.
    C and CST are recomputed completely at each time step.
    
    Numba njit version is 4-15 timems slower than the pure Python version for this function when dim=16.
    """

    XRhbar = np.matmul(X, Rhbar)

    for idx in range(n_nzC):
        i, j, k, l = indices_nzC[idx]
        I = i * dim + j
        J = k * dim + l
        C[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - Rhbar[i, k] * X[l, j]
        It = j * dim + i
        CST[It, J] = C[I, J]

    return (C, CST)

def update_D_under_magnetic_field(D, D0, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    """
    D: D matrix to be updated.
    D0: D matrix at t = 0 ps.
    h: Hamiltonian at t = t ps
      h is recalculated at each time step.
      save memory by avoiding memory allocation for h.
      The off-diagonal elements of h do not change with time.
    h_t0: spin Hamiltonian at t = 0 ps
    Mz_diag: diagonal matrix element of the z component of the total magnetic moment operator M = (Mx, My, Mz).
      Mz is diagonal and real in the S representation where the basis functions are the eigenstates of the spin operator Sz_tot.
    B: Magnetic field in Tesla.
    C: The superoperator for spin-phonon coupling.
    CST: super transpose of C
    X: The matrix that encodes possible spin transitions
    Rhbar: The superoperator for the spin-phonon coupling in the S representation.
      In the E representation
        Rhbar_{ij} = X_{ij} * Phi_{ij} 
        Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
        omega_ij = ( E_i - E_j ) / hbar.
        I(omega) = I0 omega^2 theta(omega): spectral density for phonons
        theta(omega) is the step function.
    n_nzC: number of nonzero matrix elements of C
    indices_nzC: indices of nonzero matrix elements of C
    lambdaa: spin-phonon coupling constant
    I0: prefactor for the phonon spectral density
    T: Temperature
    dim            : Dimension of the effective Hamiltonian
    dims  = dim^2  : Dimension of superoperators
    dimds = 2*dim^2: Dimension of double superoperators
    """

    # In general, the D operator is as follows
    # D11 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (Cre + CSTre)
    # D12 =  const1 * Are + lambdaa**2 * np.pi * const1**2 * (Cim + CSTim)
    # D21 = -const1 * Are - lambdaa**2 * np.pi * const1**2 * (Cim - CSTim)
    # D22 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (Cre - CSTre)

    # Since Azee is diagonal and real, only the following update is needed
    # D = [[D11, D12], [D21, D22]]
    #       D12 = D012 + const1 * Azee
    #       D21 = D021 - const1 * Azee

    # Calculate the A operator corresponding to the Zeeman interaction
    hzee_diag = -1 * Mz_diag * Tesla2wavenumber * B
    Azee_diag = construct_A_diag(hzee_diag, dim, dims)

    # Add Zeeman interaction to the D matrix
    c1Azee_diag = const1 * Azee_diag
    for i in range(dims):
        D[     i, dims+i] = D0[     i, dims+i] + c1Azee_diag[i]
        D[dims+i,      i] = D0[dims+i,      i] - c1Azee_diag[i]

    # Update the diagonal elements of the Hamiltonian h
    for i in range(dim):
        h[i, i] = h_t0[i, i] + hzee_diag[i]

    # Update the superoperator C and CST
    Rhbar = update_Rhbar(Rhbar, h, X, I0, T)
    C, CST = update_C_and_CST(C, CST, X, Rhbar, dim, n_nzC, indices_nzC)

    # Update the spin-phonon coupling of the D matrix. 
    Cre = np.real(C)
    Cim = np.imag(C)
    CSTre = np.real(CST)
    CSTim = np.imag(CST)
    D[0:dims, 0:dims]         = D0[0:dims, 0:dims]         - lambdaa**2 * np.pi * const1**2 * (Cre + CSTre)
    D[0:dims, dims:dimds]     = D[0:dims, dims:dimds]      + lambdaa**2 * np.pi * const1**2 * (Cim + CSTim)
    D[dims:dimds, 0:dims]     = D[dims:dimds, 0:dims]      - lambdaa**2 * np.pi * const1**2 * (Cim - CSTim)
    D[dims:dimds, dims:dimds] = D0[dims:dimds, dims:dimds] - lambdaa**2 * np.pi * const1**2 * (Cre - CSTre)

    return (D, h, Rhbar)

# ====================================================================================================== #
# Functions for examining the maximum of the absolute values of the elements of D and exp(D * deltat).
# ====================================================================================================== #

def get_D_max_and_expDdeltat_max(D, D0, deltat, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    D, h, Rhbar = update_D_under_magnetic_field(D, D0, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    D_max, expDdeltat_max = np.max(np.abs(D)), np.max(np.abs(expm(D*deltat)))
    # print("B = {:15.6f} T, deltat = {:15.3f}  ps, max(|D|) = {:12.6f}, max(|exp(D * deltat)|) = {:12.6f}\n".format(B, deltat, D_max, expDdeltat_max))
    return (D_max, expDdeltat_max)
    
def examine_D_max_and_expDdeltat_max(Bs, deltats, tag, D, D0, Mz_diag, C, CST, X, Rhbar, h0_diag, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    """
    Examine the maximum of the absolute values of the elements of D and exp(D * deltat)
    at various B fields and time steps.
    Bs: list of magnetic fields in Tesla.
    deltats: list of time steps in ps.

    """

    # Create an array of zeros of dimension m x n, where m is the number of B fields and n is the number of time steps. 
    # The array will be filled with the maximum of the absolute values of the elements of D and exp(D * deltat).

    m = len(Bs)
    n = len(deltats)
    D_max = np.zeros((m, n))
    expDdeltat_max = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            # Get the current magnetic field and time step
            B = Bs[i]
            deltat = deltats[j]

            # Print the magnetic field and the time step
            print("B = {:15.6f} T, deltat = {:15.3f}  ps".format(B, deltat))

            # Get the maximum of the absolute values of the elements of D and exp(D * deltat)
            D_max[i, j], expDdeltat_max[i, j] = get_D_max_and_expDdeltat_max(D, D0, deltat, Mz_diag, B, C, CST, X, Rhbar, h0_diag, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)

    if not os.path.exists("./output"):
        os.makedirs("./output")

    # Save D_max in the file D_max.dat, the first column is the magnetic field, and the first row is the time step.
    # Ths first row should be a comment line starting with #.
    with open("./output/D_max_" + tag + ".dat", "w") as f:
        f.write("# ")
        for j in range(n):
            f.write("{:15.3f}".format(deltats[j]))
        f.write("\n")
        for i in range(m):
            f.write("{:15.6f}".format(Bs[i]))
            for j in range(n):
                f.write(" {:15.6f}".format(D_max[i, j]))
            f.write("\n")

    # Save expDdeltat_max in the file expDdeltat_max.dat, the first column is the magnetic field, and the first row is the time step.
    # Ths first row should be a comment line starting with #. 
    with open("./output/expDdeltat_max_" + tag + ".dat", "w") as f:
        f.write("# ")
        for j in range(n):
            f.write("{:15.3f}".format(deltats[j]))
        f.write("\n")
        for i in range(m):
            f.write("{:15.6f}".format(Bs[i]))
            for j in range(n):
                f.write(" {:15.6f}".format(expDdeltat_max[i, j]))
            f.write("\n")

    return



# ============================================================================ #
# Functions for calculating the long-time evolution operator.
# ============================================================================ #

def get_U_dsqe_longtime(t0, t1, deltat, D, D0, h, h_t0, Mz_diag, Bt, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    r"""
    Get the long-time evolution operator \Pi_i U = exp(D_i * deltat), where \Pi is the product operator.
    The operators exp(D_i * deltat) are time-ordered such that later times are to the left of earlier times.
    """

    half_deltat = deltat/2
    nt = int( np.round((t1 - t0)/deltat) )
    t1 = t0 + nt*deltat

    # Initialize the long-time evolution operator
    U = np.identity(dimds, dtype=np.float64)

    # Loop over the time steps and multiply the short-time evolution operators to get the long-time evolution operator
    for it in range(nt):
        t = t0 + it*deltat + half_deltat
        B = Bt(t)
        # Print t and B for debugging
        # print("it = {:6d}, t = {:18.3f}, B = {:15.3e}".format(it, t, B))
        D, h, Rhbar = update_D_under_magnetic_field(D, D0, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
        U = np.matmul(expm(D*deltat), U)

    if not os.path.exists("./output"):
        os.makedirs("./output")

    # file name for saving the long-time evolution operator
    ta = int( t0 / 1e6 // 1 ) # micro second 
    fname      = './output/double_super_U_{:d}-{:d}us.hdf5'.format(ta, ta+1)
    fname_lock = './output/double_super_U_{:d}-{:d}us.hdf5.lock'.format(ta, ta+1)

    # Lock the file to prevent other processes from writing to it
    with FileLock(fname_lock):
        # The tag is uique and accurate if t < 5e12 ps.
        tag = "{:.3f}-{:.3f}".format(t0, t1)
        # Save the long-time double super evolution operator
        with h5py.File(fname, "a") as f1:
            # Write data safely. 
            if tag in f1:
                # print("The dataset {} already exists in the file {}. Deleting the dataset ...".format(tag, fname))
                del f1[tag]
            # print("Creating the dataset {} in the file {} ...".format(tag, fname))
            dset = f1.create_dataset("{:.3f}-{:.3f}".format(t0, t1), data=U)

    return (t0, t1, U)



# ============================================================================ #
# Functions for solving the quantum master equation.
# ============================================================================ #

def get_Dabc(Bs2_wavenumber, it, h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nzC):
    """
    Obtain Da = D(ts[it])
           Db = D(ts[it] + deltat/2)
           Dc = D(ts[it] + deltat)

    Bs2_wavenumber are the fields on the time grid with a time step of helf deltat.

    dim: dimension of the Hilbert space.
    dims: dimension of a super operator. dims = dim**2.
    """

    Da = get_D_at_Bfield(Bs2_wavenumber[2*it  ], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nzC)
    Db = get_D_at_Bfield(Bs2_wavenumber[2*it+1], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nzC)
    Dc = get_D_at_Bfield(Bs2_wavenumber[2*it+2], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nzC)

    return (Da, Db, Dc)

def get_Dabc_reuse_Da(Bs2_wavenumber, it, h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nzC, Da, Db, Dc):
    """
    Obtain Da = D(ts[it])
           Db = D(ts[it] + deltat/2)
           Dc = D(ts[it] + deltat)

    Bs2_wavenumber are the fields on the time grid with a time step of helf deltat.

    dim: dimension of the Hilbert space.
    dims: dimension of a super operator. dims = dim**2.
    """

    Db = get_D_at_Bfield(Bs2_wavenumber[2*it+1], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nzC)
    Dc = get_D_at_Bfield(Bs2_wavenumber[2*it+2], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nzC)

    return (Da, Db, Dc)

def evolve_deltat_dsqme(Da, Db, Dc, rho, deltat):
    """
    Evolve rho by deltat using the Runge-Kutta method according to the quantum master equation
        d rho / d t = D rho
        rho = np.vstack(rhore, rhoim)

    Input: 
      Da: D(t), the Liouville superoperator at time t. Unit: cm-1.
      Db: D(t+deltat/2)
      Dc: D(t+deltat). ha, hb, and hc are written on the basis of the eigenvectors of h0, i.e. the zero-field spin Hamiltonian.
      rho: vectorized density matrix at time t rho(t) on the basis of the eigenvectors of h0.
      deltat: time step in ps.
    """

    k1 = Da @ rho                   # np.matmul(Da, rho)
    k2 = Db @ (rho + 0.5*deltat*k1) # np.matmul(Db, rho + 0.5*deltat*k1)
    k3 = Db @ (rho + 0.5*deltat*k2) # np.matmul(Db, rho + 0.5*deltat*k2)
    k4 = Dc @ (rho +     deltat*k3) # np.matmul(Dc, rho +     deltat*k3)

    rho_new = rho + deltat * (k1 + 2*k2 + 2*k3 + k4) / 6

    return rho_new

def evolve_rho_dsqme(D0, Mz_tot_diag, dsrho, nt, deltat, Bs2, dim, dims):
    """
    Evolve the density matrix using the Runge-Kutta method according to the quantum master equation.

    h0 and Mv_tot are written on the basis of the eigenvectors of h0.

    Input:
      D0: double superoperator at t0 = ts[0]
      Mz_tot_diag: diagnal matrix elements of the z component of the magnetization operators
      dsrho: initial double super density matrix at t0.
      nt: number of time steps.
      delta: time step in unit of ps.
      Bs2: list of magnetic field on the double grid.
      dim: dimension of the Hilbert space.
      dims: dimension of superoperators

    Assumptions: 
      The magnetic field is along the z direction.
      The exchange couplings are isotropic.
      The g tensors are isotropic and identical.

    As a result, Mz_tot is real and diagonal.
    """

    # Evolve the density matrix

    ## It is more efficient to use minus_Mz_tot_diag and Bs2_wavenumber. 
    minus_Mz_tot_diag = -1 * Mz_tot_diag
    Bs2_wavenumber = Bs2 * Tesla2wavenumber 

    Da, Db, Dc = get_Dabc(D0, minus_Mz_tot_diag, 0, deltat, Bs2_wavenumber, dim, dims)

    dsrho = evolve_deltat_dsqme(Da, Db, Dc, dsrho, deltat)
    #print("i = 0, max(dsrho) = ", np.max(np.absolute(dsrho)))

    for i in range(1, nt):
        Da, Db, Dc = get_Dabc_reuse_Da(D0, minus_Mz_tot_diag, i, deltat, Bs2_wavenumber, Dc, Da, Db, dim, dims)
        dsrho = evolve_deltat_dsqme(Da, Db, Dc, dsrho, deltat)

    return dsrho

def evolve_rho_dsqme_onestair(dsrho, deltat, D, D0, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    """
    Evolve rho by deltat of constant Bfield using the analytical solution of the quantum master equation
        d rho / d t = D rho
        rho = np.vstack(rhore, rhoim)
        rho_new = exp(int_t1^t2 D dt) rho = exp(D deltat) rho

    Input: 
        dsrho: double super density matrix
        deltat: time period for evolution
        D: double superoperator
        D0: double superoperator at zero magnetic field.
        h: Hamiltonian at time t
        h_t0: the initial Hamiltonian on the perturbed basis
        Mz_diag: diagnal matrix elements of the z component of the magnetization operators
        B: magnetic field in Tesla
        C: super operator for spin-phonon coupling
        CST: super transpose of C
        X: spin operator that encodes possible spin transitions
        Rhbar: auxiliary operator for spin phonon coupling
        indices_nzC: indices of the nonzero matrix elements of C
        lambdaa: spin-phonon coupling constant in wavenumbers
        I0: prefactor for the phonon density of states
        T: temperature in Kelvin
        rho: vectorized initial density matrix on the perturbed basis.
        deltat: time step in ps.
        dim: dimension of the Hilbert space.
        dims: dimension of superoperators
    """

    D, h, Rhbar = update_D_under_magnetic_field(D, D0, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    
    dsrho = expm(D * deltat) @ dsrho

    return dsrho

def get_outdirs(T, I0, lambdaa, Bt_params):
    """
    Get the output directory for the double super density matrix and the magnetic moment.
    T: temperature in Kelvin.
    I0: prefactor for the phonon density of states.
    lambdaa: spin-phonon coupling constant in wavenumbers.
    Bt_params: parameters for the magnetic fields. See pulse.py for details.
    """
    if Bt_params['Bt_type'] == 'linear':
        outdir_rho = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_linear_sweep_rate_{:.1f}/dsrho'.format(T, I0, lambdaa, Bt_params['sweep_rate'])
        outdir_mag = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_linear_sweep_rate_{:.1f}/magnetometry'.format(T, I0, lambdaa, Bt_params['sweep_rate'])
    elif Bt_params['Bt_type'] == 'pwlinear':
        times = Bt_params['times']
        fields = Bt_params['fields']
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/'.format(T, I0, lambdaa)
        outdir += 'Bt_pwlinear_t{:.1e}ps-B{:.1f}T'.format(times[0], fields[0])
        for i in range(1, len(times)):
            outdir += '_t{:.1e}ps-B{:.1f}T'.format(times[i], fields[i])
        outdir = outdir.replace('+', '')
        outdir_rho = outdir + '/dsrho'
        outdir_mag = outdir + '/magnetometry'
    elif Bt_params['Bt_type'] == 'pwlinear_by_slope':
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_pwlinear_average_sweep_rate_{:.1f}'.format(T, I0, lambdaa, Bt_params['sweep_rate_ave'])
        outdir_rho = outdir + '/dsrho'
        outdir_mag = outdir + '/magnetometry'
    elif Bt_params['Bt_type'] == 'sin':
        outdir_rho = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_sin_amplitude_{:.1f}_omega_{:.2f}/dsrho'.format(T, I0, lambdaa, Bt_params['amplitude'], Bt_params['omega'])
        outdir_mag = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_sin_amplitude_{:.1f}_omega_{:.2f}/magnetometry'.format(T, I0, lambdaa, Bt_params['amplitude'], Bt_params['omega'])
    elif Bt_params['Bt_type'] == 'cs':
        outdir_rho = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_cs/dsrho'.format(T, I0, lambdaa)
        outdir_mag = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_cs/magnetometry'.format(T, I0, lambdaa)
    else:
        raise ValueError("Invalid Bt_type: {}".format(Bt_params['Bt_type']))
    return (outdir_rho, outdir_mag)

def evolve_rho_dsqme_stairs(t0, t1, deltat, Bt_params, dsrho, D, D0, h_t0, Mz_op, Mz_diag, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, save_mag, nt_mag, save_rho, nt_rho):
    """
    Evolve the double super density matrix using the staircase method according to the quantum master equation.
    Multiple staircases are used. All stairs have the same width deltat.

    t0: initial time
    t1: final time
    deltat: time step in ps
    turning_points: turning points of the magnetic field as a function of time. turning_points = [times, fields]
    dsrho: initial double super density matrix at t0
    D: The double super operator D at tmin
    D0: The exchange-only double super operator at tmin
    h_t0: the initial Hamiltonian on the perturbed basis
    Mz_op: The z component of the magnetization operators
    Mz_diag: diagnal matrix elements of the z component of the magnetization operators
    C: The superoperator for spin-phonon coupling
    CST: super transpose of C
      C and CST will be reconstructed completely from X and Rhbar.
    X: The matrix that encodes possible spin transitions
    Rhbar: auxiliary operator for spin phonon coupling at t0
      Rhbar will be reconstructed completely from X and the eigenvalues.
    n_nzC: number of nonzero matrix elements of C
    indices_nzC: indices of the nonzero matrix elements of C
    lambdaa: spin-phonon coupling constant in wavenumbers
    I0: prefactor for the phonon density of states
    T: temperature in Kelvin
    dim: dimension of the Hilbert space
    dims: dimension of superoperators
    dimds: dimension of double superoperators
    save_mag: logical, save the magnetic moment at chosen time steps?
    nt_mag: save the magnetic moment per nt_mag*delta ps
    save_rho: logical, save the double super density matrix at chosen time steps?
    nt_rho: save the double super density matrix per nt_rho*delta ps. nt_rho will be adjusted to be a multiple of nt_mag.
    """

    # Make a copy of the hamiltonian h_t0 to store the Hamiltonian at time t
    # This is to avoid repeated memory allocation for h.
    h = copy.deepcopy(h_t0)

    half_deltat = deltat/2

    # Specify the magnetic field as a function of time
    Bt = get_Bt(Bt_params)

    if save_rho and save_mag:
        # nt_rho should be a multiple of nt_mag
        nround_mag = int( max(nt_rho // nt_mag, 1) )
        nt_rho = nround_mag * nt_mag
        
        # nt should be a multiple of nt_rho
        nround_rho = int( max((t1 - t0)//deltat // nt_rho, 1) )
        nt = nround_rho * nt_rho
        
        # Adjust the final time
        t1 = t0 + nt*deltat

        # Output directories
        outdir_rho, outdir_mag = get_outdirs(T, I0, lambdaa, Bt_params)
        if not os.path.exists(outdir_rho):
            os.makedirs(outdir_rho)
        if not os.path.exists(outdir_mag):
            os.makedirs(outdir_mag)

        # Output files
        fname1 = outdir_rho + '/{:.3f}-{:.3f}ps_dt{:.3f}ps.hdf5'.format(t0, t1, deltat)
        fname2 = outdir_mag + '/{:.3f}-{:.3f}ps_dt{:.3f}ps.dat'.format(t0, t1, deltat)

        with h5py.File(fname1, 'w') as f1,  open(fname2, 'w') as f2:
            tag = "{:.3f}".format(t0)
            dset = f1.create_dataset(tag, data=dsrho)

            Mz = get_Mz_from_dsrho(dsrho, Mz_op, dim, dims, dimds)
            # chimz = get_chimz_from_dsrho(h, h_t0, Bt, t0, Mz_op, dsrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
            # chimz = get_chimz_finite_diff(dsrho, t0, 1e6, Bt, D, D0, h, h_t0, Mz_op, Mz_diag, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
            chimz = 0.0 # I do not know how to calculate the magnetic susceptibility correctly.
            f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t0, Bt(t0), Mz, chimz))

            # Save the initial magnetic moment for calculating the magnetic susceptibility using finite difference
            Mz_old = Mz + 0.0

            # Step of magnetic field for calculating the magnetic susceptibility using finite difference
            dB = Bt(deltat) - Bt(0) # Assume that the magnetic field changes linearly with time.
    
            # Loop over the rounds for saving the double super density matrix
            for iround_rho in range(nround_rho):
                # Loop over the rounds for saving the magnetic moment
                for iround_mag in range(nround_mag):
                    # Loop over the nt_mag time steps
                    for it_mag in range(nt_mag):
                        it = iround_rho * nt_rho + iround_mag * nt_mag + it_mag
                        t = t0 + it*deltat + half_deltat
                        B = Bt(t)
                        print("it/nt = {:9d}/{:9d}, t = {:18.3f}, B = {:15.3e}".format(it, nt, t, B))
                        # h and Rhbar are updated at each time step
                        dsrho = evolve_rho_dsqme_onestair(dsrho, deltat, D, D0, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
                    # Calculate the magnetic moment
                    Mz = get_Mz_from_dsrho(dsrho, Mz_op, dim, dims, dimds)

                    # Calculate the magnetic susceptibility
                    # Attempt 1 
                    # chimz = get_chimz_from_dsrho(h, h_t0, Bt, t+half_deltat, Mz_op, dsrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
                    # Attempt 2
                    # chimz = get_chimz_finite_diff(dsrho, t0, 1e3, Bt, D, D0, h, h_t0, Mz_op, Mz_diag, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
                    # Attempt 3
                    chimz = (Mz - Mz_old) / dB; Mz_old = Mz + 0.0

                    # Save the magnetic moment and the magnetic susceptibility
                    f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t+half_deltat, Bt(t+half_deltat), Mz, chimz))
                # Save the double super density matrix
                tag = "{:.3f}".format(t + half_deltat)
                dset = f1.create_dataset(tag, data=dsrho)
    elif save_rho and (not save_mag):
        # nt should be a multiple of nt_rho
        nround_rho = int( (t1 - t0)//deltat // nt_rho )
        nt = nround_rho * nt_rho
        
        # Adjust the final time
        t1 = t0 + nt*deltat

        # Output directory
        outdir_rho, outdir_mag = get_outdirs(T, I0, lambdaa, Bt_params)
        if not os.path.exists(outdir_rho):
            os.makedirs(outdir_rho)

        # Output files
        fname1 = outdir_rho + '/{:.3f}-{:.3f}ps_dt{:.3f}ps.hdf5'.format(t0, t1, deltat)

        with h5py.File(fname1, 'w') as f1:
            tag = "{:.3f}".format(t0)
            dset = f1.create_dataset(tag, data=dsrho)

            # Loop over the rounds for saving the double super density matrix
            for iround_rho in range(nround_rho):
                # Loop over the nt_rho time steps
                for it_rho in range(nt_rho):
                    it = iround_rho * nt_rho + it_rho
                    t = t0 + it*deltat + half_deltat
                    B = Bt(t)
                    print("it/nt = {:9d}/{:9d}, t = {:18.3f}, B = {:15.3e}".format(it, nt, t, B))
                    dsrho = evolve_rho_dsqme_onestair(dsrho, deltat, D, D0, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
                # Save the double super density matrix
                tag = "{:.3f}".format(t + half_deltat)
                dset = f1.create_dataset(tag, data=dsrho)
    elif (not save_rho) and save_mag:
        # nt should be a multiple of nt_mag
        nround_mag = int( (t1 - t0)//deltat // nt_mag )
        nt = nround_mag * nt_mag
        
        # Adjust the final time
        t1 = t0 + nt*deltat

        # Output directory
        outdir_rho, outdir_mag = get_outdirs(T, I0, lambdaa, Bt_params)
        if not os.path.exists(outdir_mag):
            os.makedirs(outdir_mag)

        # Output files
        fname2 = outdir_mag + '/{:.3f}-{:.3f}ps_dt{:.3f}ps.dat'.format(t0, t1, deltat)

        with open(fname2, 'w') as f2:
            Mz = get_Mz_from_dsrho(dsrho, Mz_op, dim, dims, dimds)
            chimz = get_chimz_from_dsrho(h, h_t0, Bt, t0, Mz_op, dsrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
            f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t0, Bt(t0), Mz, chimz))
    
            # Loop over the rounds for saving the magnetic moment
            for iround_mag in range(nround_mag):
                # Loop over the nt_mag time steps
                for it_mag in range(nt_mag):
                    it = iround_mag * nt_mag + it_mag
                    t = t0 + it*deltat + half_deltat
                    B = Bt(t)
                    print("it/nt = {:9d}/{:9d}, t = {:18.3f}, B = {:15.3e}".format(it, nt, t, B))
                    dsrho = evolve_rho_dsqme_onestair(dsrho, deltat, D, D0, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
                # Save magnetic moment
                Mz = get_Mz_from_dsrho(dsrho, Mz_op, dim, dims, dimds)
                chimz = get_chimz_from_dsrho(h, h_t0, Bt, t+half_deltat, Mz_op, dsrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
                f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t+half_deltat, Bt(t+half_deltat), Mz, chimz))
    else:
        # the time period should be a multiple of deltat
        nt = int( (t1 - t0)//deltat )
        
        # Adjust the final time
        t1 = t0 + nt*deltat

        # Loop over the nt time steps
        for it in range(nt):
            t = t0 + it*deltat + half_deltat
            B = Bt(t)
            print("it/nt = {:9d}/{:9d}, t = {:18.3f}, B = {:15.3e}".format(it, nt, t, B))
            dsrho = evolve_rho_dsqme_onestair(dsrho, deltat, D, D0, h, h_t0, Mz_diag, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)

    return ( t1, dsrho )

