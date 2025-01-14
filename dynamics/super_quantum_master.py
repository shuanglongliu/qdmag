import copy
import numpy as np
import h5py
from filelock import FileLock
from scipy.linalg import expm
from spin_dynamics.dynamics.constants import const1, Tesla2wavenumber
from spin_dynamics.dynamics.common import kronecker_delta
from spin_dynamics.dynamics.common import convert_cmatrix_to_rmatrix
from spin_dynamics.dynamics.common import spy_sparsity
from spin_dynamics.dynamics.common import get_Mv_from_rho, get_Mz_from_rho
from spin_dynamics.dynamics.quantum_master import construct_Rhbar, update_Rhbar
from spin_dynamics import __file__ as root_dir

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
# Auxiliary functions
# ============================================================================ #

def get_hdf5_file_size():
    t = 17000000 # Total time
    span = 100000 # Time span for each data set of U
    dimds = 2048
    # The factor 8 converts bits to bytes
    # The factor 1024 / 1024 /1024 converts bytes to Gb
    fsize = ( t / span ) * 2048 * 2048 * 8 / 1024 / 1024 /1024 
    print("File size of double_super_U.hdf5: {:.2f} Gb".format(fsize))



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
    double_super_rho = np.concatenate((super_rho_re, super_rho_im))
    return double_super_rho

def convert_dsrho_to_rho(double_super_rho, dim, dims, dimds):
    """
    Convert double super density matrix to density matrix.
    """
    super_rho_re = double_super_rho[0:dims]
    super_rho_im = double_super_rho[dims:dimds]
    super_rho = super_rho_re + 1j * super_rho_im
    rho = super_rho.reshape((dim, dim))
    return rho



# ============================================================================ #
# Functions for calculating magnetic moment. 
# ============================================================================ #

def get_magnetic_moment_dsrho(double_super_rho, Mv, dim, dims, dimds):

    rho = convert_dsrho_to_rho(double_super_rho, dim, dims, dimds)

    #spy_M(rho, "rho", threshold=0)
    #print("Trace of the density matrix = {:20.16f}".format( np.real( np.trace(rho) ) ))
    
    M = get_Mv_from_rho(rho, Mv)

    return M

def get_Mz_from_dsrho(double_super_rho, Mz, dim, dims, dimds):

    rho = convert_dsrho_to_rho(double_super_rho, dim, dims, dimds)

    return get_Mz_from_rho(rho, Mz)

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

def get_indices_nonzero_X_and_C(X, dim):
    """
    Get indices of nonzero matrix elements of X and C
    X: a spin operator of dimension dim storing information of possible spin transitions
    C: a super operator of dimension dims due to the interaction with bath
    """

    indices_nonzero_X = np.nonzero(X)

    X2 = X @ X
    indices_nonzero_X2 = np.nonzero(X2)

    set_index_tuples_nonzero_X = set( [(indices_nonzero_X[0][i], indices_nonzero_X[1][i]) for i in range(indices_nonzero_X[0].shape[0])] )
    set_index_tuples_nonzero_X2 = set( [(indices_nonzero_X2[0][i], indices_nonzero_X2[1][i]) for i in range(indices_nonzero_X2[0].shape[0])] )

    indices_nonzero_C = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    if ( (i, k) in set_index_tuples_nonzero_X2 and l == j) or  ( (i, k) in set_index_tuples_nonzero_X and (l, j) in set_index_tuples_nonzero_X ):
                        indices_nonzero_C.append( (i, j, k, l) )

    return (indices_nonzero_X, indices_nonzero_C)

def construct_A_from_H(H, dtype=np.complex128):
    """
    [H, rho]_I = A_{IJ} rho_J
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    A_{IJ} = H_{ik} delta_{lj} - delta_{ik} H_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N
    dype = np.complex128 or np.float64
    """

    N = H.shape[0]
    dim = N**2

    A = np.zeros((dim, dim), dtype=dtype)
    for I in range(dim):
        i, j = dec_composite_index(I, N)
        for J in range(dim):
            k, l = dec_composite_index(J, N)
            A[I, J] = H[i, k] * kronecker_delta(l, j) - kronecker_delta(i, k) * H[l, j]

    return A

def construct_A_from_diagonalH(H):
    """
    [H, rho]_I = A_{IJ} rho_J
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    A_{IJ} = H_{ik} delta_{lj} - delta_{ik} H_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N

    If H is diagonal, then A is diagonal with 
      A_{II} = Hii - Hjj
      i = I // N
      j = I % N

    If H is diagonal, then H (and thus A) is real.
    """

    N = H.shape[0]
    dim = N**2

    A = np.zeros((dim, dim), dtype=np.float64)

    #for I in range(dim):
        #i, j = dec_composite_index(I, N)
        #A[I, I] = H[i, i] - H[j, j]

    # Faster
    #for i in range(N):
        #for j in range(N):
            #I = i * N + j
            #A[I, I] = H[i, i] - H[j, j]

    # Even faster
    I = 0
    for i in range(N):
        for j in range(N):
            A[I, I] = H[i, i] - H[j, j]
            I = I + 1

    return A

def construct_A_diag_from_diagonalH(H):
    """
    [H, rho]_I = A_{IJ} rho_J
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    A_{IJ} = H_{ik} delta_{lj} - delta_{ik} H_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N

    If H is diagonal, then A is diagonal with 
      A_{II} = Hii - Hjj
      i = I // N
      j = I % N

    Let A_diag = A.diagonal()

    If H is diagonal, then H (and thus A) is real.
    """

    N = H.shape[0]
    dim = N**2

    A_diag = np.zeros(dim, dtype=np.float64)

    I = 0
    for i in range(N):
        for j in range(N):
            A_diag[I] = H[i, i] - H[j, j]
            I = I + 1

    return A_diag

def construct_A_diag_from_H_diag(H_diag, dim, dims):
    """
    [H, rho]_I = A_{IJ} rho_J
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    A_{IJ} = H_{ik} delta_{lj} - delta_{ik} H_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N

    If H is diagonal, then A is diagonal with 
      A_{II} = Hii - Hjj
      i = I // N
      j = I % N

    Let H_diag = H.diagonal()
        A_diag = A.diagonal()

    If H is diagonal, then H (and thus A) is real.

    dim: dimension of the Hilbert space.
    dims: dimension of a super operator. dims = dim**2.
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
    [X, Rhbar rho]_{I} = C_{IJ} rho_{J}
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    C_{IJ} = (X Rhbar)_{ik} delta_{lj} - Rhbar_{ik} X_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N

    Note: X and Rhbar are matrices of real numbers, and thus C is a matrix of real numbers.
    """

    XRhbar = np.matmul(X, Rhbar)

    C = np.zeros((dims, dims), dtype=np.float64)

    for I in range(dims):
        i, j = dec_composite_index(I, dim)
        for J in range(dims):
            k, l = dec_composite_index(J, dim)
            C[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - Rhbar[i, k] * X[l, j]

    return C

def construct_CST(C, dim, dims):
    """
    Get super transpose of the superoperator C.
    CST_{I J} = C_{It J}
      I -> ij -> ji -> It
    """

    CST = np.zeros((dims, dims), dtype=np.float64)

    for I in range(dims):
        i, j = dec_composite_index(I, dim)
        It = get_composite_index(j, i, dim)
        CST[I] = C[It]

    return CST

def construct_D_from_A_and_C(A, C, lambdaa, is_real=True):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[D11, D12], [D21, D22]]) (rhore, rhoim)^T
    
        D11 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre + CSTre)
        D12 =  const1 * Are + lambdaa**2 pi const1**2 (Cim + CSTim)
        D21 = -const1 * Are - lambdaa**2 pi const1**2 (Cim - CSTim)
        D22 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre - CSTre)

    is_real: Is H (and thus A) real? C is always real.
    """

    CST = construct_CST(C)

    if is_real:
        D11 = - lambdaa**2 * np.pi * const1**2 * (C + CST)
        D12 =   const1 * A                                             
        D21 = - const1 * A                                             
        D22 = - lambdaa**2 * np.pi * const1**2 * (C - CST)
    else:
        Are = np.real(A)
        Aim = np.imag(A)
        
        D11 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (C + CST)
        D12 =  const1 * Are 
        D21 = -const1 * Are 
        D22 =  const1 * Aim - lambdaa**2 * np.pi * const1**2 * (C - CST)
        
    D1 = np.hstack((D11, D12))
    D2 = np.hstack((D21, D22))
    D  = np.vstack((D1 , D2 ))

    return D

def construct_D_from_A_diag_and_C(A_diag, C, lambdaa, dims):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[D11, D12], [D21, D22]]) (rhore, rhoim)^T
    
        D11 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre + CSTre)
        D12 =  const1 * Are + lambdaa**2 pi const1**2 (Cim + CSTim)
        D21 = -const1 * Are - lambdaa**2 pi const1**2 (Cim - CSTim)
        D22 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre - CSTre)

    A_diag = A.diagonal()

    Assume that A is diagonal and thus real. C is always real.

    dims: dimension of a super operator. dims = dim**2, where dim is the dimenion of the Hilbert space.
    """

    CST = construct_CST(C)

    D11 = - lambdaa**2 * np.pi * const1**2 * (C + CST)

    D12 = np.zeros(C.shape, dtype=np.float64)
    D21 = np.zeros(C.shape, dtype=np.float64)
    c1A_diag = const1 * A_diag
    for i in range(dims):
        D12[i, i] =   c1A_diag[i]
        D21[i, i] = - c1A_diag[i]

    D22 = - lambdaa**2 * np.pi * const1**2 * (C - CST)

    D1 = np.hstack((D11, D12))
    D2 = np.hstack((D21, D22))
    D  = np.vstack((D1 , D2 ))

    # Sparse matrix
    #D = csr_array(D)

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

# ============================================================================ #
# Functions for updating the Liouville superoperator.
# ============================================================================ #

def update_C_and_CST(C, CST, X, Rhbar, dim, indices_nonzero_C):
    """
    Save time by avoiding memory allocation for C.
    Rhbar is time dependent, and so is C.
    C and CST are recomputed completely at each time step.
    """

    XRhbar = np.matmul(X, Rhbar)

    for m in indices_nonzero_C:
        i, j, k, l = m
        I = get_composite_index(i, j, dim)
        J = get_composite_index(k, l, dim)
        C[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - Rhbar[i, k] * X[l, j]
        It = get_composite_index(j, i, dim)
        CST[It, J] = C[I, J]

    return (C, CST)

def update_D_under_magnetic_field(D, D0, Mz_tot_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds):
    """
    D: D matrix to be updated.
    D0: D matrix at t = 0 ps.
    Mz_tot_diag: np.diagonal(Mz_tot), converted into a vector of real number in set_up_double_super_qme
    B: Magnetic field in Tesla.
    C: The superoperator for spin-phonon coupling.
    CST: super transpose of C
    X: The matrix that encodes possible spin transitions
    Rhbar_{ij} = X_{ij} * Phi_{ij}
      Phi_{ij} = ( I(omega_ij) - I(-omega_ij) ) / ( exp(beta * hbar * omega_ij ) - 1 )
      omega_ij = ( E_i - E_j ) / hbar.
      I(omega) = I0 omega^2 theta(omega): spectral density for phonons
      theta(omega) is the step function.
    h0_diag: diagonal matrix elements of h0, i.e. the initial energy spectrum 
    indices_nonzero_X: indices of nonzero matrix elements of X
    indices_nonzero_C: indices of nonzero matrix elements of C
    lambdaa: spin-phonon coupling constant
    I0: prefactor for the phonon spectral density
    T: Temperature
    dim            : Dimension of the effective Hamiltonian
    dims  = dim^2  : Dimension of superoperators
    dimds = 2*dim^2: Dimension of double superoperators
    """

    # Add Zeeman interaction to the D matrix

    # D = [[D11, D12], [D21, D22]]
    #       D12 = D012 + const1 * Azee
    #       D21 = D021 - const1 * Azee

    hzee_diag = -1 * Mz_tot_diag * Tesla2wavenumber * B
    Azee_diag = construct_A_diag_from_H_diag(hzee_diag, dim, dims)

    c1Azee_diag = const1 * Azee_diag
    for i in range(dims):
        D[     i, dims+i] = D0[     i, dims+i] + c1Azee_diag[i]
        D[dims+i,      i] = D0[dims+i,      i] - c1Azee_diag[i]

    # Update the spin-phonon coupling of the D matrix. Both A and C are real.

    # D = [[D11, D12], [D21, D22]]
    #       D11 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre + CSTre)
    #       D22 =  const1 * Aim - lambdaa**2 pi const1**2 (Cre - CSTre)

    Rhbar = update_Rhbar(Rhbar, T, X, indices_nonzero_X, h0_diag + hzee_diag, I0)
    C, CST = update_C_and_CST(C, CST, X, Rhbar, dim, indices_nonzero_C)

    D[0:dims, 0:dims]         = - lambdaa**2 * np.pi * const1**2 * (C + CST)
    D[dims:dimds, dims:dimds] = - lambdaa**2 * np.pi * const1**2 * (C - CST)

    return D

# ====================================================================================================== #
# Functions for examining the maximum of the absolute values of the elements of D and exp(D * deltat).
# ====================================================================================================== #

def get_D_max_and_expDdeltat_max(D, D0, deltat, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds):
    D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
    D_max, expDdeltat_max = np.max(np.abs(D)), np.max(np.abs(expm(D*deltat)))
    # print("B = {:15.6f} T, deltat = {:15.3f}  ps, max(|D|) = {:12.6f}, max(|exp(D * deltat)|) = {:12.6f}\n".format(B, deltat, D_max, expDdeltat_max))
    return (D_max, expDdeltat_max)
    
def examine_D_max_and_expDdeltat_max(Bs, deltats, tag, D, D0, Mz_diag, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds):
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
            D_max[i, j], expDdeltat_max[i, j] = get_D_max_and_expDdeltat_max(D, D0, deltat, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)

    # Save D_max in the file D_max.dat, the first column is the magnetic field, and the first row is the time step.
    # Ths first row should be a comment line starting with #.
    with open(root_dir + "./output/D_max_" + tag + ".dat", "w") as f:
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
    with open(root_dir + "./output/expDdeltat_max_" + tag + ".dat", "w") as f:
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
# Functions for solving the quantum master equation.
# ============================================================================ #

def set_up_double_super_qme(h0_eff, h1_eff, Mz_eff, X_eff, I0, T, lambdaa):
    """
    h0: Hamiltonian at t = 0
    h1: Hamiltonian at t = tmin

    D0_eff: D matrix at t = 0 ps without spin-phonon coupling
    D_eff: D matrix at t = tmin ps with spin-phonon coupling
    """

    # Dimensions

    dim   = h0_eff.shape[0]      # Dimension of effective Hilbert space
    dims  = dim*dim              # Dimension of superoperators
    dimds = 2*dims               # Dimension of double superoperators

    print("Dimension of the effective Hamiltonian: {:6d}".format(dim))
    print("Dimension of superoperators: {:6d}".format(dims))
    print("Dimension of double superoperators: {:6d}\n".format(dimds))

    # Build real matrices to speed up the construction of the D matrix, which will be done at each time step.

    h0_eff = convert_cmatrix_to_rmatrix(h0_eff, "h0_eff")
    h1_eff = convert_cmatrix_to_rmatrix(h1_eff, "h1_eff")
    Mz_eff = convert_cmatrix_to_rmatrix(Mz_eff, "Mz_eff")

    # Diagonal matrix elements

    h0_eff_diag = np.diagonal(h0_eff)
    h1_eff_diag = np.diagonal(h1_eff)
    Mz_eff_diag = np.diagonal(Mz_eff)

    # Construct the superoperator A0 from h0. 

    A0_eff_diag = construct_A_diag_from_H_diag(h0_eff_diag, dim, dims)
    A1_eff_diag = construct_A_diag_from_H_diag(h1_eff_diag, dim, dims)

    # Construct the superoperator D0 that corresponds to h0_eff/A0_eff using the diagonal elements of A0_eff

    D0_eff = construct_D_from_A_diag(A0_eff_diag, dims)
    D_eff  = construct_D_from_A_diag(A1_eff_diag, dims)

    # Construct Rhbar, C, and CST using the energy spectrum at t = tmin
    # The matrix elements of Rhbar_eff, C, and CST_eff do not matter here, since they will be reconstructed completely later.
    # We set them up at the beginning to reuse the memory and save time.

    Rhbar_eff = construct_Rhbar(T, X_eff, h1_eff_diag, I0)
    C_eff = construct_C(X_eff, Rhbar_eff, dim, dims)
    CST_eff = construct_CST(C_eff, dim, dims)

    # Add the spin-phonon cuopling to the D_eff matrix

    D_eff[0:dims, 0:dims]         = - lambdaa**2 * np.pi * const1**2 * (C_eff + CST_eff)
    D_eff[dims:dimds, dims:dimds] = - lambdaa**2 * np.pi * const1**2 * (C_eff - CST_eff)

    return (D0_eff, D_eff, h0_eff_diag, h1_eff_diag, Mz_eff_diag, Rhbar_eff, C_eff, CST_eff, dim, dims, dimds)

def get_Dabc(Bs2_wavenumber, it, h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nonzero_X, indices_nonzero_C):
    """
    Obtain Da = D(ts[it])
           Db = D(ts[it] + deltat/2)
           Dc = D(ts[it] + deltat)

    Bs2_wavenumber are the fields on the time grid with a time step of helf deltat.

    dim: dimension of the Hilbert space.
    dims: dimension of a super operator. dims = dim**2.
    """

    Da = get_D_at_Bfield(Bs2_wavenumber[2*it  ], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nonzero_X, indices_nonzero_C)
    Db = get_D_at_Bfield(Bs2_wavenumber[2*it+1], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nonzero_X, indices_nonzero_C)
    Dc = get_D_at_Bfield(Bs2_wavenumber[2*it+2], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nonzero_X, indices_nonzero_C)

    return (Da, Db, Dc)

def get_Dabc_reuse_Da(Bs2_wavenumber, it, h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nonzero_X, indices_nonzero_C, Da, Db, Dc):
    """
    Obtain Da = D(ts[it])
           Db = D(ts[it] + deltat/2)
           Dc = D(ts[it] + deltat)

    Bs2_wavenumber are the fields on the time grid with a time step of helf deltat.

    dim: dimension of the Hilbert space.
    dims: dimension of a super operator. dims = dim**2.
    """

    Db = get_D_at_Bfield(Bs2_wavenumber[2*it+1], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nonzero_X, indices_nonzero_C)
    Dc = get_D_at_Bfield(Bs2_wavenumber[2*it+2], h0_diag, minus_Mz_tot_diag, X, lambdaa, I0, T, dim, dims, D0, indices_nonzero_X, indices_nonzero_C)

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

def evolve_rho_dsqme(D0, Mz_tot_diag, double_super_rho, nt, deltat, Bs2, dim, dims):
    """
    Evolve the density matrix using the Runge-Kutta method according to the quantum master equation.

    h0 and Mv_tot are written on the basis of the eigenvectors of h0.

    Input:
      D0: double superoperator at t0 = ts[0]
      Mz_tot_diag: diagnal matrix elements of the z component of the magnetization operators
      double_super_rho: initial double super density matrix at t0.
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

    double_super_rho = evolve_deltat_dsqme(Da, Db, Dc, double_super_rho, deltat)
    #print("i = 0, max(double_super_rho) = ", np.max(np.absolute(double_super_rho)))

    for i in range(1, nt):
        Da, Db, Dc = get_Dabc_reuse_Da(D0, minus_Mz_tot_diag, i, deltat, Bs2_wavenumber, Dc, Da, Db, dim, dims)
        double_super_rho = evolve_deltat_dsqme(Da, Db, Dc, double_super_rho, deltat)

    return double_super_rho

def evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds):
    """
    Evolve rho by deltat of constant Bfield using the analytical solution of the quantum master equation
        d rho / d t = D rho
        rho = np.vstack(rhore, rhoim)
        rho_new = exp(int_t1^t2 D dt) rho = exp(D deltat) rho

    Input: 
        double_super_rho: double super density matrix
        deltat: time period for evolution
        D: double superoperator
        D0: double superoperator at zero magnetic field.
        Mz_diag: diagnal matrix elements of the z component of the magnetization operators
        B: magnetic field in Tesla
        C: super operator for spin-phonon coupling
        CST: super transpose of C
        X: spin operator that encodes possible spin transitions
        Rhbar: auxiliary operator for spin phonon coupling
        h0_diag: diagonal matrix elements of the initial Hamiltonian on the perturbed basis
        indices_nonzero_X: indices of the nonzero matrix elements of X
        indices_nonzero_C: indices of the nonzero matrix elements of C
        lambdaa: spin-phonon coupling constant in wavenumbers
        I0: prefactor for the phonon density of states
        T: temperature in Kelvin
        rho: vectorized initial density matrix on the perturbed basis.
        deltat: time step in ps.
        dim: dimension of the Hilbert space.
        dims: dimension of superoperators
    """

    D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)

    double_super_rho = expm(D * deltat) @ double_super_rho

    return double_super_rho

def evolve_rho_dsqme_stairs(t0, t1, deltat, Bt, double_super_rho, D, D0, Mz_diag, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds, Mv, save_mag, deltat_mag, save_rho, deltat_rho):
    """
    Evolve the double super density matrix using the staircase method according to the quantum master equation.
    Multiple staircases are used. All stairs have the same width deltat.

    t0: initial time
    t1: final time
    deltat: time step in ps
    Bt: magnetic field as a function of time
    double_super_rho: initial double super density matrix at t0
    D: The double super operator D at tmin
    D0: The exchange-only double super operator at tmin
    Mz_diag: diagnal matrix elements of the z component of the magnetization operators
    C: The superoperator for spin-phonon coupling
    CST: super transpose of C
      C and CST will be reconstructed completely from X and Rhbar.
    X: The matrix that encodes possible spin transitions
    Rhbar: auxiliary operator for spin phonon coupling
      Rhbar will be reconstructed completely from X and the eigenvalues.
    h0_diag: diagonal matrix elements of the initial Hamiltonian on the perturbed basis
    indices_nonzero_X: indices of the nonzero matrix elements of X
    indices_nonzero_C: indices of the nonzero matrix elements of C
    lambdaa: spin-phonon coupling constant in wavenumbers
    I0: prefactor for the phonon density of states
    T: temperature in Kelvin
    dim: dimension of the Hilbert space
    dims: dimension of superoperators
    dimds: dimension of double superoperators
    Mv: magnetic moment operator
    save_mag: logical, save the magnetic moment at chosen time steps?
    deltat_mag: time step for saving the magnetic moment
    save_rho: logical, save the double super density matrix at chosen time steps?
    deltat_rho: time step for saving the double super density matrix
    """

    half_deltat = deltat/2
    nt = int( np.round((t1 - t0)/deltat) )
    t1 = t0 + nt*deltat

    B0 = Bt(t0)
    B1 = Bt(t1)

    print("deltat = {:18.3f}".format(deltat))
    print("nt = {:18d}".format(nt))
    print("t0/t1 = {:18.3f}/{:18.3f}".format(t0, t1))
    print("B0/B1 = {:18.3f}/{:18.3f}\n".format(B0, B1))

    nt_mag = round( deltat_mag / deltat )
    nt_rho = round( deltat_rho / deltat )

    # For debugging
    # M = get_magnetic_moment_dsrho(double_super_rho, Mv, dim, dims, dimds)
    # print("it = {:6d}, t = {:18.3f}, B = {:15.3e}, max(|D|) = {:12.6f}, max(|exp(D * deltat)|) = {:12.6f}, M = {:20.8E} {:20.8E} {:20.8E} mu_B\n".format(-1, t0, Bt(t0), np.max(np.abs(D)), np.max(np.abs(expm(D*deltat))), *np.real(M)))

    fname1 = root_dir + 'output/double_super_rho_{:.3f}-{:.3f}.hdf5'.format(t0, t1)
    fname2 = root_dir + 'output/M-t_{:.3f}-{:.3f}.dat'.format(t0, t1)

    with h5py.File(fname1, 'w') as f1,  open(fname2, 'w') as f2:

        if save_rho:
            tag = "{:.3f}".format(t0)
            dset = f1.create_dataset(tag, data=double_super_rho)

        if save_mag:
            Mz = get_Mz_from_dsrho(double_super_rho, Mv[2], dim, dims, dimds)
            f2.write("{:20.3f} {:20.6E} {:20.8E}\n".format(t0, Bt(t0), Mz))
    
        for it in range(nt):
            t = t0 + it*deltat + half_deltat
            B = Bt(t)
    
            print("it = {:6d}, t = {:18.3f}, B = {:15.3e}".format(it, t, B))

            D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
    
            double_super_rho = evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
    
            if save_rho and it%nt_rho == 0:
                # This tag is uique and accurate if t < 5e12 ps.
                tag = "{:.3f}".format(t + half_deltat)
                dset = f1.create_dataset(tag, data=double_super_rho)

            if save_mag and it%nt_mag == 0:
                Mz = get_Mz_from_dsrho(double_super_rho, Mv[2], dim, dims, dimds)
                f2.write("{:20.3f} {:20.6E} {:20.8E}\n".format(t + half_deltat, B, Mz))

            # For debugging
            # M = get_magnetic_moment_dsrho(double_super_rho, Mv, dim, dims, dimds)
            # print("it / nt = {:6d} / {:6d}, t = {:18.3f}, B = {:15.3e}, max(|D|) = {:12.6f}, max(|exp(D * deltat)|) = {:12.6f}, M = {:20.8E} {:20.8E} {:20.8E} mu_B".format(it, nt, t+half_deltat, B, np.max(np.abs(D)), np.max(np.abs(expm(D*deltat))), *np.real(M)))

    return ( t1, double_super_rho )

def evolve_rho_dsqme_stairs_rhomag(t0, t1, deltat, Bt, double_super_rho, D, D0, Mz_diag, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds, Mv, save_rhomag, deltat_rhomag):
    """
    Evolve the double super density matrix using the staircase method according to the quantum master equation.
    Multiple staircases are used. All stairs have the same width deltat.

    t0: initial time
    t1: final time
    deltat: time step in ps
    Bt: magnetic field as a function of time
    double_super_rho: initial double super density matrix at t0
    D: The double super operator D at tmin
    D0: The exchange-only double super operator at tmin
    Mz_diag: diagnal matrix elements of the z component of the magnetization operators
    C: The superoperator for spin-phonon coupling
    CST: super transpose of C
      C and CST will be reconstructed completely from X and Rhbar.
    X: The matrix that encodes possible spin transitions
    Rhbar: auxiliary operator for spin phonon coupling
      Rhbar will be reconstructed completely from X and the eigenvalues.
    h0_diag: diagonal matrix elements of the initial Hamiltonian on the perturbed basis
    indices_nonzero_X: indices of the nonzero matrix elements of X
    indices_nonzero_C: indices of the nonzero matrix elements of C
    lambdaa: spin-phonon coupling constant in wavenumbers
    I0: prefactor for the phonon density of states
    T: temperature in Kelvin
    dim: dimension of the Hilbert space
    dims: dimension of superoperators
    dimds: dimension of double superoperators
    save_mag: logical, save the magnetic moment at chosen time steps?
    deltat_mag: time step for saving the magnetic moment
    save_rhomag: logical, save the double super density matrix and the magnetic moment at chosen time steps?
    deltat_rhomag: time step for saving the double super density matrix and the magnetic moment
    """

    half_deltat = deltat/2
    nt = int( np.round((t1 - t0)/deltat) )
    t1 = t0 + nt*deltat

    B0 = Bt(t0)
    B1 = Bt(t1)

    print("deltat = {:18.3f}".format(deltat))
    print("nt = {:18d}".format(nt))
    print("t0/t1 = {:18.3f}/{:18.3f}".format(t0, t1))
    print("B0/B1 = {:18.3f}/{:18.3f}\n".format(B0, B1))

    nt_rhomag = round( deltat_rhomag / deltat )

    nround = nt // nt_rhomag
    nremain = nt % nt_rhomag

    fname1 = root_dir + 'output/double_super_rho_{:.3f}-{:.3f}.hdf5'.format(t0, t1)
    fname2 = root_dir + 'output/M-t_{:.3f}-{:.3f}.dat'.format(t0, t1)

    # Save the double super density matrix peirodically
    if save_rhomag:
        with h5py.File(fname1, 'w') as f1, open(fname2, 'w') as f2:
            # Save the double super density matrix at t0
            dset = f1.create_dataset("{:.3f}".format(t0), data=double_super_rho)

            # Save the magnetic moment at t0
            Mz = get_Mz_from_dsrho(double_super_rho, Mv[2], dim, dims, dimds)
            f2.write("{:20.3f} {:20.6E} {:20.8E}\n".format(t0, Bt(t0), Mz))

            # Loop over the rounds of nt_rhomag steps
            for iround in range(nround):
                for it in range(nt_rhomag):
                    t = t0 + (iround*nt_rhomag + it)*deltat + half_deltat
                    B = Bt(t)
                    # Print t and B for debugging
                    print("it = {:6d}, t = {:18.3f}, B = {:15.3e}".format(iround*nt_rhomag + it, t, B))
                    D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
                    double_super_rho = evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
                # Save the double super density matrix at t. The tag is uique and accurate if t < 5e12 ps.
                dset = f1.create_dataset("{:.3f}".format(t + half_deltat), data=double_super_rho)

                # Save the magnetic moment at t
                Mz = get_Mz_from_dsrho(double_super_rho, Mv[2], dim, dims, dimds)
                f2.write("{:20.3f} {:20.6E} {:20.8E}\n".format(t + half_deltat, B, Mz))

        # Loop over the remaining steps
        for it in range(nremain):
            t = t0 + (nround*nt_rhomag + it)*deltat + half_deltat
            B = Bt(t)
            D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
            double_super_rho = evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
    # Do not save the double super density matrix
    else:
        # Loop over the rounds of nt_rhomag steps
        for iround in range(nround):
            for it in range(nt_rhomag):
                t = t0 + (iround*nt_rhomag + it)*deltat + half_deltat
                B = Bt(t)
                # Print t and B for debugging
                print("it = {:6d}, t = {:18.3f}, B = {:15.3e}".format(iround*nt_rhomag + it, t, B))
                D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
                double_super_rho = evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)

        # Loop over the remaining steps
        for it in range(nremain):
            t = t0 + (nround*nt_rhomag + it)*deltat + half_deltat
            B = Bt(t)
            D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
            double_super_rho = evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)

    return ( t1, double_super_rho )

def evolve_rho_dsqme_stairs_light(t0, t1, deltat, Bt, double_super_rho, D, D0, Mz_diag, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds, save_rho, deltat_rho):
    """
    Evolve the double super density matrix using the staircase method according to the quantum master equation.
    Multiple staircases are used. All stairs have the same width deltat.

    t0: initial time
    t1: final time
    deltat: time step in ps
    Bt: magnetic field as a function of time
    double_super_rho: initial double super density matrix at t0
    D: The double super operator D at tmin
    D0: The exchange-only double super operator at tmin
    Mz_diag: diagnal matrix elements of the z component of the magnetization operators
    C: The superoperator for spin-phonon coupling
    CST: super transpose of C
      C and CST will be reconstructed completely from X and Rhbar.
    X: The matrix that encodes possible spin transitions
    Rhbar: auxiliary operator for spin phonon coupling
      Rhbar will be reconstructed completely from X and the eigenvalues.
    h0_diag: diagonal matrix elements of the initial Hamiltonian on the perturbed basis
    indices_nonzero_X: indices of the nonzero matrix elements of X
    indices_nonzero_C: indices of the nonzero matrix elements of C
    lambdaa: spin-phonon coupling constant in wavenumbers
    I0: prefactor for the phonon density of states
    T: temperature in Kelvin
    dim: dimension of the Hilbert space
    dims: dimension of superoperators
    dimds: dimension of double superoperators
    save_mag: logical, save the magnetic moment at chosen time steps?
    deltat_mag: time step for saving the magnetic moment
    save_rho: logical, save the double super density matrix at chosen time steps?
    deltat_rho: time step for saving the double super density matrix
    """

    half_deltat = deltat/2
    nt = int( np.round((t1 - t0)/deltat) )
    t1 = t0 + nt*deltat

    B0 = Bt(t0)
    B1 = Bt(t1)

    print("deltat = {:18.3f}".format(deltat))
    print("nt = {:18d}".format(nt))
    print("t0/t1 = {:18.3f}/{:18.3f}".format(t0, t1))
    print("B0/B1 = {:18.3f}/{:18.3f}\n".format(B0, B1))

    nt_rho = round( deltat_rho / deltat )

    nround = nt // nt_rho
    nremain = nt % nt_rho

    fname1 = root_dir + 'output/double_super_rho_{:.3f}-{:.3f}.hdf5'.format(t0, t1)

    # Save the double super density matrix peirodically
    if save_rho:
        with h5py.File(fname1, 'w') as f1:
            # Save the double super density matrix at t0
            dset = f1.create_dataset("{:.3f}".format(t0), data=double_super_rho)

            # Loop over the rounds of nt_rho steps
            for iround in range(nround):
                for it in range(nt_rho):
                    t = t0 + (iround*nt_rho + it)*deltat + half_deltat
                    B = Bt(t)
                    # Print t and B for debugging
                    print("it = {:6d}, t = {:18.3f}, B = {:15.3e}".format(iround*nt_rho + it, t, B))
                    D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
                    double_super_rho = evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
                # Save the double super density matrix at t. The tag is uique and accurate if t < 5e12 ps.
                dset = f1.create_dataset("{:.3f}".format(t + half_deltat), data=double_super_rho)

        # Loop over the remaining steps
        for it in range(nremain):
            t = t0 + (nround*nt_rho + it)*deltat + half_deltat
            B = Bt(t)
            D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
            double_super_rho = evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
    # Do not save the double super density matrix
    else:
        # Loop over the rounds of nt_rho steps
        for iround in range(nround):
            for it in range(nt_rho):
                t = t0 + (iround*nt_rho + it)*deltat + half_deltat
                B = Bt(t)
                # Print t and B for debugging
                print("it = {:6d}, t = {:18.3f}, B = {:15.3e}".format(iround*nt_rho + it, t, B))
                D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
                double_super_rho = evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)

        # Loop over the remaining steps
        for it in range(nremain):
            t = t0 + (nround*nt_rho + it)*deltat + half_deltat
            B = Bt(t)
            D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
            double_super_rho = evolve_rho_dsqme_onestair(double_super_rho, deltat, D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)

    return ( t1, double_super_rho )

# ============================================================================ #
# Functions for calculating the long-time evolution operator.
# ============================================================================ #

def get_U_dsqe_longtime(t0, t1, deltat, Bt, D, D0, Mz_diag, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds):
    r"""
    Get the long-time evolution operator \Pi_i U = exp(D_i * deltat), where \Pi is the product operator.
    The operators exp(D_i * deltat) are time-ordered such that later times are to the left of earlier times.
    """

    half_deltat = deltat/2
    nt = int( np.round((t1 - t0)/deltat) )
    t1 = t0 + nt*deltat

    B0 = Bt(t0)
    B1 = Bt(t1)

    print("deltat = {:18.3f}".format(deltat))
    print("nt = {:18d}".format(nt))
    print("t0/t1 = {:18.3f}/{:18.3f}".format(t0, t1))
    print("B0/B1 = {:18.3f}/{:18.3f}\n".format(B0, B1))

    # Initialize the long-time evolution operator
    U = np.identity(dimds, dtype=np.float64)

    # Loop over the time steps and multiply the short-time evolution operators to get the long-time evolution operator
    for it in range(nt):
        t = t0 + it*deltat + half_deltat
        B = Bt(t)
        # Print t and B for debugging
        # print("it = {:6d}, t = {:18.3f}, B = {:15.3e}".format(it, t, B))
        D = update_D_under_magnetic_field(D, D0, Mz_diag, B, C, CST, X, Rhbar, h0_diag, indices_nonzero_X, indices_nonzero_C, lambdaa, I0, T, dim, dims, dimds)
        U = np.matmul(expm(D*deltat), U)

    # file name for saving the long-time evolution operator
    # fname      = root_dir + 'output/double_super_U.hdf5'
    # fname_lock = root_dir + 'output/double_super_U.hdf5.lock'

    ta = int( t0 / 1e6 // 1 ) # micro second 
    fname      = root_dir + 'output/double_super_U_{:d}-{:d}us.hdf5'.format(ta, ta+1)
    fname_lock = root_dir + 'output/double_super_U_{:d}-{:d}us.hdf5.lock'.format(ta, ta+1)

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



