import os
import copy
import numpy as np
import h5py
from filelock import FileLock
from scipy.linalg import expm
from spin_dynamics.core.constants import const1, Tesla2wavenumber
from spin_dynamics.core.common import kronecker_delta
from spin_dynamics.core.common import spy_sparsity
from spin_dynamics.core.common import get_Mv_from_rho, get_Mz_from_rho
from spin_dynamics.core.common import eigen_simple
from spin_dynamics.core.quantum_master import get_Rhbar, update_Rhbar
from spin_dynamics.core.pulse import get_Bt

r"""
Codes for solving the quantum master equation described in the Eq. 2.7 of
J. Phys. Soc. Jpn. 2001.70:2151-2157 by Hiroki Nakano and Seiji Miyashita.
The quantum master equation is cast into the Liouville form with the real 
and imaginary parts of the density matrix treated as independent variables.
The need to separate the real and imaginary parts comes from the Hermitian
conjugate operation in the \Gamma\rho term, which cannot be written as a
matrix product. After the separation of the real and imaginary parts, the
Liouville superoperator can be written as a matrix, which allows the appl-
ication of the analytical solution on each step of the stairs.

The quantum master equation reads (in the units given later)
    d (rhore, rhoim)^T / d t = np.array([[L11, L12], [L21, L22]]) (rhore, rhoim)^T

    L11 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre + CSTre)
    L12 =  const1 * Are + lambdaa**2 pi const1**2 (Cim + CSTim)
    L21 = -const1 * Are - lambdaa**2 pi const1**2 (Cim - CSTim)
    L22 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre - CSTre)
 
Units: 
    cm-1 for energy
    ps for time

    cm-1 for the spin-phonon coupling constant lambdaa
    ps for spectral density I and and its prefactor I0
    ps-1 for frequency omega
    X and Rhbar are unitless

Dimensions:
    dim            : Dimension of the effective Hamiltonian
    dims  = dim^2  : Dimension of superoperators for the vectorized density matrix
    dimds = 2*dim^2: Dimension of superoperators for the RI-separated vectorized density matrix
"""

# ============================================================================ #
# Functions for formatting the density matrix.
# ============================================================================ #

def convert_rho_to_risvrho(rho):
    """
    Convert density matrix to vectorized density matrix with the real and imaginary parts separated.
    """
    vrho = rho.flatten()
    vrho_re = np.real(vrho)
    vrho_im = np.imag(vrho)
    risvrho = np.concatenate((vrho_re, vrho_im))
    return risvrho

def convert_risvrho_to_rho(risvrho, dim, dims, dimds):
    """
    Convert RI-separated vectorized density matrix to density matrix.
    """
    vrho_re = risvrho[0:dims]
    vrho_im = risvrho[dims:dimds]
    vrho = vrho_re + 1j * vrho_im
    rho = vrho.reshape((dim, dim))
    return rho



# ============================================================================ #
# Functions for calculating magnetic moment. 
# ============================================================================ #

def get_Mv_from_risvrho(risvrho, Mv, dim, dims, dimds):

    rho = convert_risvrho_to_rho(risvrho, dim, dims, dimds)

    M = get_Mv_from_rho(rho, Mv)

    return M

def get_Mz_from_risvrho(risvrho, Mz, dim, dims, dimds):

    rho = convert_risvrho_to_rho(risvrho, dim, dims, dimds)

    return get_Mz_from_rho(rho, Mz)

# ============================================================================ #
# Functions for calculating magnetic susceptibility.
# ============================================================================ #

def get_chimz_from_rho(h, Bt, t, Mz_tot, rho, X, Rhbar, lambdaa, dt=1e+3):
    r"""
    Assume that the magnetic field is along the z direction.

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

def get_chimz_from_risvrho(h, h_t0, Bt, t, Mz, risvrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds):
    """
    Wrapper for get_chimz_from_rho
    h_t0: Hamiltonian at time t=0 ps
    Bt: Magnetic field as a function of time t
    t: Time
    Mz: z component of the total magnetic moment operator
    risvrho: vectorized density matrix with the real and imaginary parts separated
    X: Spin transition matrix
    Rhbar: An operator that accounts for spin-phonon coupling
    lambdaa: Spin-phonon coupling constant
    dim: Dimension of the effective Hamiltonian
    dims: Dimension of superoperators for the vectorized density matrix
    dimds: Dimension of superoperators for the RI-separated vectorized density matrix
    """

    # Hamiltonian at time t
    h = h_t0 - Tesla2wavenumber * Bt(t) * Mz

    # Rhbar at time t
    Rhbar = update_Rhbar(Rhbar, h, X, I0, T)

    # Get the density matrix rho from the RI-separated vectorized density matrix risvrho
    rho = convert_risvrho_to_rho(risvrho, dim, dims, dimds)

    # Call get_chimz_from_rho
    chimz = get_chimz_from_rho(h, Bt, t, Mz, rho, X, Rhbar, lambdaa)

    return chimz


def get_chimz_finite_diff(risvrho, t1, dt, Bt, L, L0, h, h_t0, Mz_op, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    """
    risvrho: RI-separated vectorized density matrix at time t1
    """

    B1 = Bt(t1)
    B2 = Bt(t1 + dt)

    M1 = get_Mz_from_risvrho(risvrho, Mz_op, dim, dims, dimds)

    risvrho2 = evolve_rho_liouville_onestair(risvrho, dt, L, L0, h, h_t0, Mz_op, B1, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    M2 = get_Mz_from_risvrho(risvrho2, Mz_op, dim, dims, dimds)

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
    dims = dim**2.: dimension of superoperators for the vectorized density matrix.
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
    C: an operator of dimension dims due to the interaction with bath
       see the function construct_C for the definition of C
    n_nzC: number of nonzero matrix elements of C
    indices_nzC: indices of the nonzero matrix elements of C
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

def construct_L(A, C, dim, dims, lambdaa):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[L11, L12], [L21, L22]]) (rhore, rhoim)^T
    
        L11 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre + CSTre)
        L12 =  const1 * Are + lambdaa**2 pi const1**2 (Cim + CSTim)
        L21 = -const1 * Are - lambdaa**2 pi const1**2 (Cim - CSTim)
        L22 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre - CSTre)

    A: Superoperator for coherent evolution. See the function construct_A.
    C: Superoperator for spin-phonon coupling (incoherent evolution). See the function construct_C.
    dim: Dimension of the effective Hamiltonian
    dims: Dimension of superoperators
    lambdaa: spin-phonon coupling constant
    only_from_A: If True, then only the contribution from A are included in L.

    Note:
        A is real if H is diagonal (and thus real).
    """

    Are = np.real(A)
    Aim = np.imag(A)

    if C is None:
        # Construct L from A only
        L11 =  const1 * Aim
        L12 =  const1 * Are
        L21 = -const1 * Are
        L22 =  const1 * Aim
    else:
        CST = construct_CST(C, dim, dims)
        Cre = np.real(C)
        Cim = np.imag(C)
        CSTre = np.real(CST)
        CSTim = np.imag(CST)
        L11 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (Cre + CSTre)
        L12 =  const1 * Are + lambdaa**2 * np.pi * const1**2 * (Cim + CSTim)
        L21 = -const1 * Are - lambdaa**2 * np.pi * const1**2 * (Cim - CSTim)
        L22 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (Cre - CSTre)
        
    L1 = np.hstack((L11, L12))
    L2 = np.hstack((L21, L22))
    L  = np.vstack((L1 , L2 ))

    return L

def construct_L_from_A_diag(A_diag, dims):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[L11, L12], [L21, L22]]) (rhore, rhoim)^T
    
        L11 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre + CSTre)
        L12 =  const1 * Are + lambdaa**2 pi const1**2 (Cim + CSTim)
        L21 = -const1 * Are - lambdaa**2 pi const1**2 (Cim - CSTim)
        L22 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre - CSTre)

    A_diag = A.diagonal()

    Assume that A is diagonal and thus real. 

    dims: dimension of superoperators for the vectorized density matrix. dims = dim**2, where dim is the dimenion of the Hilbert space.
    """

    L11 = np.zeros((dims, dims), dtype=np.float64)
    L12 = np.zeros((dims, dims), dtype=np.float64)
    L21 = np.zeros((dims, dims), dtype=np.float64)
    L22 = np.zeros((dims, dims), dtype=np.float64)

    c1A_diag = const1 * A_diag
    for i in range(dims):
        L12[i, i] =   c1A_diag[i]
        L21[i, i] = - c1A_diag[i]

    L1 = np.hstack((L11, L12))
    L2 = np.hstack((L21, L22))
    L  = np.vstack((L1 , L2 ))

    return L

def set_up_liouville(h_t0_eff, h_tmin_eff, X_eff, dim, I0, T, lambdaa):
    """
    h_t0: Hamiltonian at t = 0
    h_tmin: Hamiltonian at t = tmin

    L0_eff: L matrix at t = 0 ps without spin-phonon coupling
    L_eff: L matrix at t = tmin ps with spin-phonon coupling

    dim: dimension of the effective Hilbert space.
    """

    # Dimensions
    dims  = dim*dim              # Dimension of superoperators for the vectorized density matrix
    dimds = 2*dims               # Dimension of superoperators for the RI-separated vectorized density matrix

    print("Dimension of the effective Hamiltonian: {:6d}".format(dim))
    print("Dimension of superoperators for the vectorized density matrix: {:6d}".format(dims))
    print("Dimension of superoperators for the RI-separated vectorized density matrix: {:6d}\n".format(dimds))

    # Construct the superoperator L_t0_eff at t = 0 ps without spin-phonon coupling
    A_t0_eff = construct_A(h_t0_eff, dim, dims, diagonal_H=False, dtype=np.complex128)
    L_t0_eff = construct_L(A_t0_eff, None, dim, dims, lambdaa)

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

    # Construct the superoperator L_tmin_eff at t = tmin ps with spin-phonon coupling
    L_tmin_eff = construct_L(A_tmin_eff, C_eff, dim, dims, lambdaa)

    return (L_t0_eff, L_tmin_eff, Rhbar_eff, C_eff, CST_eff, dims, dimds)

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

def update_L_under_magnetic_field(L, L0, h, h_t0, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    r"""
    L: L matrix to be updated.
    L0: L matrix at t = 0 ps.
    h: Hamiltonian at t = t ps
      h is recalculated at each time step.
      save memory by avoiding memory allocation for h.
      The off-diagonal elements of h do not change with time.
    h_t0: spin Hamiltonian at t = 0 ps
    Mz_op: the z component of the total magnetic moment operator M = (Mx, My, Mz).
      In general, Mz is a matrix of complex numbers.
      If the g tensor is isotropic, Mz is diagonal and real in the S representation where the basis functions are the eigenstates of the spin operator Sz_tot.
    B: Magnetic field in Tesla.
    C: The superoperator for spin-phonon coupling.
    CST: an operator derived from C
    X: The matrix that encodes possible spin transitions
    Rhbar: An operator for the spin-phonon coupling in the S representation.
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
    dims  = dim^2  : Dimension of superoperators for the vectorized density matrix
    dimds = 2*dim^2: Dimension of superoperators for the RI-separated vectorized density matrix
    """

    # In general, the L operator is as follows
    # L11 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (Cre + CSTre)
    # L12 =  const1 * Are + lambdaa**2 * np.pi * const1**2 * (Cim + CSTim)
    # L21 = -const1 * Are - lambdaa**2 * np.pi * const1**2 * (Cim - CSTim)
    # L22 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (Cre - CSTre)

    # If Azee is diagonal and real, which is true when the g tensors are isotropic, 
    # only the following update is needed. (used in early versions of the code)
    # L = [[L11, L12], [L21, L22]]
    #       L12 = L012 + const1 * Azee
    #       L21 = L021 - const1 * Azee

    # Calculate the A operator corresponding to the Zeeman interaction

    hzee = -1 * Mz_op * Tesla2wavenumber * B
    Azee = construct_A(hzee, dim, dims, diagonal_H=False, dtype=np.complex128)
    c1Azee = const1 * Azee
    c1Azeere = np.real(c1Azee)
    c1Azeeim = np.imag(c1Azee)

    # Update the diagonal elements of the Hamiltonian h
    h = h_t0 + hzee

    # Update the operators C and CST
    Rhbar = update_Rhbar(Rhbar, h, X, I0, T)
    C, CST = update_C_and_CST(C, CST, X, Rhbar, dim, n_nzC, indices_nzC)
    factorC = lambdaa**2 * np.pi * const1**2 * C
    factorCST = lambdaa**2 * np.pi * const1**2 * CST
    factorCre = np.real(factorC)
    factorCim = np.imag(factorC)
    factorCSTre = np.real(factorCST)
    factorCSTim = np.imag(factorCST)

    # Add the Zeeman interaction and the spin-phonon coupling to the L matrix
    L[   0:dims,     0: dims] = L0[   0:dims,     0: dims] + c1Azeeim - (factorCre + factorCSTre)
    L[   0:dims,  dims:dimds] = L0[   0:dims,  dims:dimds] + c1Azeere + (factorCim + factorCSTim)
    L[dims:dimds,    0: dims] = L0[dims:dimds,    0: dims] - c1Azeere - (factorCim - factorCSTim)
    L[dims:dimds, dims:dimds] = L0[dims:dimds, dims:dimds] + c1Azeeim - (factorCre - factorCSTre)

    return (L, h, Rhbar)

# ====================================================================================================== #
# Functions for examining the maximum of the absolute values of the elements of L and exp(L * deltat).
# ====================================================================================================== #

def get_L_max_and_expLdeltat_max(L, L0, deltat, h_t0, h, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    L, h, Rhbar = update_L_under_magnetic_field(L, L0, h, h_t0, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    L_max, expLdeltat_max = np.max(np.abs(L)), np.max(np.abs(expm(L*deltat)))
    # print("B = {:15.6f} T, deltat = {:15.3f}  ps, max(|L|) = {:12.6f}, max(|exp(L * deltat)|) = {:12.6f}\n".format(B, deltat, L_max, expLdeltat_max))
    return (L_max, expLdeltat_max)
    
def examine_L_max_and_expLdeltat_max(Bs, deltats, tag, L, L0, h_t0, h, Mz_op, C, CST, X, Rhbar, lambdaa, I0, T, dim, dims, dimds):
    """
    Examine the maximum of the absolute values of the elements of L and exp(L * deltat)
    at various B fields and time steps.
    Bs: list of magnetic fields in Tesla.
    deltats: list of time steps in ps.
    """

    # Create an array of zeros of dimension m x n, where m is the number of B fields and n is the number of time steps. 
    # The array will be filled with the maximum of the absolute values of the elements of L and exp(L * deltat).

    m = len(Bs)
    n = len(deltats)
    L_max = np.zeros((m, n))
    expLdeltat_max = np.zeros((m, n))

    # Which elements of C are nonzero?
    n_nzC, indices_nzC = get_indices_nzC(X, dim)

    for i in range(m):
        for j in range(n):
            # Get the current magnetic field and time step
            B = Bs[i]
            deltat = deltats[j]

            # Print the magnetic field and the time step
            print("B = {:15.6f} T, deltat = {:15.3f}  ps".format(B, deltat))

            # Get the maximum of the absolute values of the elements of L and exp(L * deltat)
            L_max[i, j], expLdeltat_max[i, j] = get_L_max_and_expLdeltat_max(L, L0, deltat, h_t0, h, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)

    if not os.path.exists("./output"):
        os.makedirs("./output")

    # Save L_max in the file L_max.dat, the first column is the magnetic field, and the first row is the time step.
    # Ths first row should be a comment line starting with #.
    with open("./output/L_max_" + tag + ".dat", "w") as f:
        f.write("# ")
        for j in range(n):
            f.write("{:15.3f}".format(deltats[j]))
        f.write("\n")
        for i in range(m):
            f.write("{:15.6f}".format(Bs[i]))
            for j in range(n):
                f.write(" {:15.6f}".format(L_max[i, j]))
            f.write("\n")

    # Save expLdeltat_max in the file expLdeltat_max.dat, the first column is the magnetic field, and the first row is the time step.
    # Ths first row should be a comment line starting with #. 
    with open("./output/expLdeltat_max_" + tag + ".dat", "w") as f:
        f.write("# ")
        for j in range(n):
            f.write("{:15.3f}".format(deltats[j]))
        f.write("\n")
        for i in range(m):
            f.write("{:15.6f}".format(Bs[i]))
            for j in range(n):
                f.write(" {:15.6f}".format(expLdeltat_max[i, j]))
            f.write("\n")

    return



# ============================================================================ #
# Functions for solving the quantum master equation.
# ============================================================================ #

# To do: Modify the RK4 codes to deal with the ZFS term
# To do: Modify the RK4 codes to save the intermediate results

def get_Labc(Bs2_wavenumber, it, h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, L0, indices_nzC):
    """
    Obtain La = L(ts[it])
           Lb = L(ts[it] + deltat/2)
           Lc = L(ts[it] + deltat)

    Bs2_wavenumber are the fields on the time grid with a time step of helf deltat.

    dim: dimension of the Hilbert space.
    dims: dimension of superoperators for the vectorized density matrix. dims = dim**2.
    """

    La = get_L_at_Bfield(Bs2_wavenumber[2*it  ], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, L0, indices_nzC)
    Lb = get_L_at_Bfield(Bs2_wavenumber[2*it+1], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, L0, indices_nzC)
    Lc = get_L_at_Bfield(Bs2_wavenumber[2*it+2], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, L0, indices_nzC)

    return (La, Lb, Lc)

def get_Labc_reuse_La(Bs2_wavenumber, it, h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, L0, indices_nzC, La, Lb, Lc):
    """
    Obtain La = L(ts[it])
           Lb = L(ts[it] + deltat/2)
           Lc = L(ts[it] + deltat)

    Bs2_wavenumber are the fields on the time grid with a time step of helf deltat.

    dim: dimension of the Hilbert space.
    dims: dimension of a super operator for the vectorized density matrix. dims = dim**2.
    """

    Lb = get_L_at_Bfield(Bs2_wavenumber[2*it+1], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, L0, indices_nzC)
    Lc = get_L_at_Bfield(Bs2_wavenumber[2*it+2], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, L0, indices_nzC)

    return (La, Lb, Lc)

def evolve_deltat_liouville(La, Lb, Lc, rho, deltat):
    """
    Evolve rho by deltat using the Runge-Kutta method according to the quantum master equation
        d rho / d t = L rho
        rho = np.vstack(rhore, rhoim)

    Input: 
      La: L(t), the Liouville superoperator at time t. Unit: cm-1.
      Lb: L(t+deltat/2)
      Lc: L(t+deltat). ha, hb, and hc are written on the basis of the eigenvectors of h0, i.e. the zero-field spin Hamiltonian.
      rho: vectorized density matrix at time t rho(t) on the basis of the eigenvectors of h0.
      deltat: time step in ps.
    """

    k1 = La @ rho                   # np.matmul(La, rho)
    k2 = Lb @ (rho + 0.5*deltat*k1) # np.matmul(Lb, rho + 0.5*deltat*k1)
    k3 = Lb @ (rho + 0.5*deltat*k2) # np.matmul(Lb, rho + 0.5*deltat*k2)
    k4 = Lc @ (rho +     deltat*k3) # np.matmul(Lc, rho +     deltat*k3)

    rho_new = rho + deltat * (k1 + 2*k2 + 2*k3 + k4) / 6

    return rho_new

def evolve_rho_liouville(L0, Mz_tot_diag, risvrho, nt, deltat, Bs2, dim, dims):
    """
    Evolve the density matrix using the Runge-Kutta method according to the quantum master equation.

    h0 and Mv_tot are written on the basis of the eigenvectors of h0.

    Input:
      L0: Liouvillian at t0 = ts[0]
      Mz_tot_diag: diagnal matrix elements of the z component of the magnetization operators
      risvrho: initial vertorized RI-separated density matrix at t0.
      nt: number of time steps.
      delta: time step in unit of ps.
      Bs2: list of magnetic field on the double grid.
      dim: dimension of the Hilbert space.
      dims: dimension of superoperators for the vectorized density matrix. dims = dim**2.

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

    La, Lb, Lc = get_Labc(L0, minus_Mz_tot_diag, 0, deltat, Bs2_wavenumber, dim, dims)

    risvrho = evolve_deltat_liouville(La, Lb, Lc, risvrho, deltat)
    #print("i = 0, max(risvrho) = ", np.max(np.absolute(risvrho)))

    for i in range(1, nt):
        La, Lb, Lc = get_Labc_reuse_La(L0, minus_Mz_tot_diag, i, deltat, Bs2_wavenumber, Lc, La, Lb, dim, dims)
        risvrho = evolve_deltat_liouville(La, Lb, Lc, risvrho, deltat)

    return risvrho

def evolve_rho_liouville_onestair(risvrho, deltat, L, L0, h, h_t0, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds):
    """
    Evolve rho by deltat of constant Bfield using the analytical solution of the quantum master equation
        d rho / d t = L rho
        rho = np.vstack(rhore, rhoim)
        rho_new = exp(int_t1^t2 L dt) rho = exp(L deltat) rho

    Input: 
        risvrho: RI-separated vectorized density matrix
        deltat: time period for evolution
        L: the Liouvillian
        L0: the Liouvillian at zero magnetic field.
        h: Hamiltonian at time t
        h_t0: the initial Hamiltonian on the perturbed basis
        Mz_op: the z component of the magnetization operators
        B: magnetic field in Tesla
        C: the operator for spin-phonon coupling
        CST: an opeator derived from C
        X: spin operator that encodes possible spin transitions
        Rhbar: auxiliary operator for spin phonon coupling
        indices_nzC: indices of the nonzero matrix elements of C
        lambdaa: spin-phonon coupling constant in wavenumbers
        I0: prefactor for the phonon density of states
        T: temperature in Kelvin
        rho: vectorized initial density matrix on the perturbed basis.
        deltat: time step in ps.
        dim: dimension of the Hilbert space.
        dims: dimension of superoperators for the vectorized density matrix. dims = dim**2.
    """

    L, h, Rhbar = update_L_under_magnetic_field(L, L0, h, h_t0, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    
    risvrho = expm(L * deltat) @ risvrho

    return risvrho

def get_outdirs(T, I0, lambdaa, Bt_params):
    """
    Get the output directory for the RI-separated vectorized density matrix and the magnetic moment.
    T: temperature in Kelvin.
    I0: prefactor for the phonon density of states.
    lambdaa: spin-phonon coupling constant in wavenumbers.
    Bt_params: parameters for the magnetic fields. See pulse.py for details.
    """
    if Bt_params['Bt_type'] == 'linear':
        outdir_rho = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_linear_sweep_rate_{:.1f}/risvrho'.format(T, I0, lambdaa, Bt_params['sweep_rate'])
        outdir_mag = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_linear_sweep_rate_{:.1f}/magnetometry'.format(T, I0, lambdaa, Bt_params['sweep_rate'])
    elif Bt_params['Bt_type'] == 'pwlinear':
        times = Bt_params['times']
        fields = Bt_params['fields']
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/'.format(T, I0, lambdaa)
        outdir += 'Bt_pwlinear_t{:.1e}ps-B{:.1f}T'.format(times[0], fields[0])
        for i in range(1, len(times)):
            outdir += '_t{:.1e}ps-B{:.1f}T'.format(times[i], fields[i])
        outdir = outdir.replace('+', '')
        outdir_rho = outdir + '/risvrho'
        outdir_mag = outdir + '/magnetometry'
    elif Bt_params['Bt_type'] == 'pwlinear_by_slope':
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_pwlinear_average_sweep_rate_{:.1f}'.format(T, I0, lambdaa, Bt_params['sweep_rate_ave'])
        outdir_rho = outdir + '/risvrho'
        outdir_mag = outdir + '/magnetometry'
    elif Bt_params['Bt_type'] == 'sin':
        outdir_rho = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_sin_amplitude_{:.1f}_omega_{:.2f}/risvrho'.format(T, I0, lambdaa, Bt_params['amplitude'], Bt_params['omega'])
        outdir_mag = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_sin_amplitude_{:.1f}_omega_{:.2f}/magnetometry'.format(T, I0, lambdaa, Bt_params['amplitude'], Bt_params['omega'])
    elif Bt_params['Bt_type'] == 'cs':
        outdir_rho = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_cs/risvrho'.format(T, I0, lambdaa)
        outdir_mag = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_cs/magnetometry'.format(T, I0, lambdaa)
    else:
        raise ValueError("Invalid Bt_type: {}".format(Bt_params['Bt_type']))
    return (outdir_rho, outdir_mag)

def evolve_rho_liouville_stairs(t0, t1, deltat, Bt_params, risvrho, L, L0, h_t0, Mz_op, C, CST, X, Rhbar, lambdaa, I0, T, dim, dims, dimds, save_mag, nt_mag, save_rho, nt_rho):
    """
    Evolve the RI-separated vectorized density matrix using the staircase method according to the quantum master equation.
    Multiple staircases are used. All stairs have the same width deltat.

    t0: initial time
    t1: final time
    deltat: time step in ps
    turning_points: turning points of the magnetic field as a function of time. turning_points = [times, fields]
    risvrho: initial RI-separated vectorized density matrix at t0
    L: The Liouvillian at tmin
    L0: The Liouvillian under zero magnetic field 
    h_t0: the initial Hamiltonian on the perturbed basis
    Mz_op: The z component of the magnetization operators
    C: An operator for spin-phonon coupling
    CST: an operator derived from C
      C and CST will be reconstructed completely from X and Rhbar.
    X: The matrix that encodes possible spin transitions
    Rhbar: auxiliary operator for spin phonon coupling at t0
      Rhbar will be reconstructed completely from X and the eigenvalues.
    lambdaa: spin-phonon coupling constant in wavenumbers
    I0: prefactor for the phonon density of states
    T: temperature in Kelvin
    dim: dimension of the Hilbert space
    dims: dimension of superoperators for the vectorized density matrix
    dimds: dimension of superoperators for the RI-separated vectorized density matrix
    save_mag: logical, save the magnetic moment at chosen time steps?
    nt_mag: save the magnetic moment per nt_mag*delta ps
    save_rho: logical, save the RI-separated vectorized density matrix at chosen time steps?
    nt_rho: save the RI-separated vectorized density matrix per nt_rho*delta ps. nt_rho will be adjusted to be a multiple of nt_mag.
    """

    # Make a copy of the hamiltonian h_t0 to store the Hamiltonian at time t
    # This is to avoid repeated memory allocation for h.
    h = copy.deepcopy(h_t0)

    half_deltat = deltat/2

    # Specify the magnetic field as a function of time
    Bt = get_Bt(Bt_params)

    # Which elements of the C operator are non-zero ?
    n_nzC, indices_nzC = get_indices_nzC(X, dim)

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
        fname1 = outdir_rho + '/{:.3f}-{:.3f}ps_dt{:.3f}ps.h5'.format(t0, t1, deltat)
        fname2 = outdir_mag + '/{:.3f}-{:.3f}ps_dt{:.3f}ps.dat'.format(t0, t1, deltat)

        with h5py.File(fname1, 'w') as f1,  open(fname2, 'w') as f2:
            tag = "{:.3f}".format(t0)
            dset = f1.create_dataset(tag, data=risvrho)

            Mz = get_Mz_from_risvrho(risvrho, Mz_op, dim, dims, dimds)
            chimz = get_chimz_from_risvrho(h, h_t0, Bt, t0, Mz_op, risvrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
            # chimz = get_chimz_finite_diff(risvrho, t0, 1e6, Bt, L, L0, h, h_t0, Mz_op, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
            # chimz = np.nan 
            f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t0, Bt(t0), Mz, chimz))

            # Save the initial magnetic moment for calculating the magnetic susceptibility using finite difference
            Mz_old = Mz + 0.0

            # Step of magnetic field for calculating the magnetic susceptibility using finite difference
            dB = Bt(deltat) - Bt(0) # Assume that the magnetic field changes linearly with time.
    
            # Loop over the rounds for saving the RI-separated vectorized density matrix
            for iround_rho in range(nround_rho):
                # Loop over the rounds for saving the magnetic moment
                for iround_mag in range(nround_mag):
                    # Loop over the nt_mag time steps
                    for it_mag in range(nt_mag):
                        it = iround_rho * nt_rho + iround_mag * nt_mag + it_mag
                        t = t0 + it*deltat + half_deltat
                        B = Bt(t)
                        # print("it/nt = {:9d}/{:9d}, t = {:18.3f}, B = {:15.3e}".format(it, nt, t, B))
                        # h and Rhbar are updated at each time step
                        risvrho = evolve_rho_liouville_onestair(risvrho, deltat, L, L0, h, h_t0, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
                    # Calculate the magnetic moment
                    Mz = get_Mz_from_risvrho(risvrho, Mz_op, dim, dims, dimds)

                    # Calculate the magnetic susceptibility
                    # Attempt 1 
                    chimz = get_chimz_from_risvrho(h, h_t0, Bt, t+half_deltat, Mz_op, risvrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
                    # Attempt 2
                    # chimz = get_chimz_finite_diff(risvrho, t0, 1e3, Bt, L, L0, h, h_t0, Mz_op, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
                    # Attempt 3
                    # chimz = (Mz - Mz_old) / dB; Mz_old = Mz + 0.0

                    # Save the magnetic moment and the magnetic susceptibility
                    f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t+half_deltat, Bt(t+half_deltat), Mz, chimz))
                # Save the RI-separated vectorized density matrix
                tag = "{:.3f}".format(t + half_deltat)
                dset = f1.create_dataset(tag, data=risvrho)
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
        fname1 = outdir_rho + '/{:.3f}-{:.3f}ps_dt{:.3f}ps.h5'.format(t0, t1, deltat)

        with h5py.File(fname1, 'w') as f1:
            tag = "{:.3f}".format(t0)
            dset = f1.create_dataset(tag, data=risvrho)

            # Loop over the rounds for saving the RI-separated vectorized density matrix
            for iround_rho in range(nround_rho):
                # Loop over the nt_rho time steps
                for it_rho in range(nt_rho):
                    it = iround_rho * nt_rho + it_rho
                    t = t0 + it*deltat + half_deltat
                    B = Bt(t)
                    # print("it/nt = {:9d}/{:9d}, t = {:18.3f}, B = {:15.3e}".format(it, nt, t, B))
                    risvrho = evolve_rho_liouville_onestair(risvrho, deltat, L, L0, h, h_t0, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
                # Save the RI-separated vectorized density matrix
                tag = "{:.3f}".format(t + half_deltat)
                dset = f1.create_dataset(tag, data=risvrho)
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
            Mz = get_Mz_from_risvrho(risvrho, Mz_op, dim, dims, dimds)
            chimz = get_chimz_from_risvrho(h, h_t0, Bt, t0, Mz_op, risvrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
            f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t0, Bt(t0), Mz, chimz))
    
            # Loop over the rounds for saving the magnetic moment
            for iround_mag in range(nround_mag):
                # Loop over the nt_mag time steps
                for it_mag in range(nt_mag):
                    it = iround_mag * nt_mag + it_mag
                    t = t0 + it*deltat + half_deltat
                    B = Bt(t)
                    # print("it/nt = {:9d}/{:9d}, t = {:18.3f}, B = {:15.3e}".format(it, nt, t, B))
                    risvrho = evolve_rho_liouville_onestair(risvrho, deltat, L, L0, h, h_t0, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
                # Save magnetic moment
                Mz = get_Mz_from_risvrho(risvrho, Mz_op, dim, dims, dimds)
                chimz = get_chimz_from_risvrho(h, h_t0, Bt, t+half_deltat, Mz_op, risvrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
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
            # print("it/nt = {:9d}/{:9d}, t = {:18.3f}, B = {:15.3e}".format(it, nt, t, B))
            risvrho = evolve_rho_liouville_onestair(risvrho, deltat, L, L0, h, h_t0, Mz_op, B, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)

    return ( t1, risvrho )

