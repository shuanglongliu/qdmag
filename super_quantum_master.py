import numpy as np
from constants import const1, Kelvin2wavenumber
from common import get_total_Sz_for_all_eigenstates
from common import get_commutation
from common import kronecker_delta
from schrodinger import get_habc, get_habc_reuse_ha
from quantum_master import energy2omega, omega2energy, spectral_density, Phi, construct_X, construct_Rhbar

"""
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

    D11 =  const1 * Aim - lambda_**2 pi const1**2 (Bre + BSTre)
    D12 =  const1 * Are + lambda_**2 pi const1**2 (Bim + BSTim)
    D21 = -const1 * Are - lambda_**2 pi const1**2 (Bim - BSTim)
    D22 =  const1 * Aim - lambda_**2 pi const1**2 (Bre - BSTre)
 
Units: 
    cm-1 for energy
    ps for time

    cm-1 for the spin-phonon coupling constant lambda_
    ps for spectral density I and and its prefactor I0
    ps-1 for frequency omega
    X and Rhbar are unitless
"""

lambda_ = 1.0

def get_composite_index(i, j, N):
    I = i * N + j
    return I

def dec_composite_index(I, N):
    i = I // N
    j = I % N
    return (i, j)

def construct_A(H):
    """
    [H, rho]_I = A_{IJ} rho_J
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    A_{IJ} = H_{ik} delta_{lj} - delta_{ik} H_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N
    """

    N = H.shape[0]
    dim = N**2

    A = np.zeros((dim, dim), dtype=np.complex64)
    for I in range(dim):
        i, j = dec_composite_index(I, N)
        for J in range(dim):
            k, l = dec_composite_index(J, N)
            A[I, J] = H[i, k] * kronecker_delta(l, j) - kronecker_delta(i, k) * H[l, j]

    return A

def construct_A_diag(H):
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
    """

    N = H.shape[0]
    dim = N**2

    A = np.zeros((dim, dim), dtype=np.complex64)

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

def construct_A_real_and_diag(H):
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

def construct_Avec_real(H):
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

    Let Avec = A.diagonal()
    """

    N = H.shape[0]
    dim = N**2

    Avec = np.zeros(dim, dtype=np.float64)

    I = 0
    for i in range(N):
        for j in range(N):
            Avec[I] = H[i, i] - H[j, j]
            I = I + 1

    return Avec

def construct_B(X, Rhbar):
    """
    [X, Rhbar rho]_{I} = B_{IJ} rho_{J}
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    B_{IJ} = (X Rhbar)_{ik} delta_{lj} - Rhbar_{ik} X_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N
    """

    N = X.shape[0]
    dim = N**2

    XRhbar = np.matmul(X, Rhbar)

    B = np.zeros((dim, dim), dtype=np.complex64)
    for I in range(dim):
        i, j = dec_composite_index(I, N)
        for J in range(dim):
            k, l = dec_composite_index(J, N)
            B[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - Rhbar(i, k) * X[l, j]

    return B

def construct_B_real(X, Rhbar):
    """
    [X, Rhbar rho]_{I} = B_{IJ} rho_{J}
      I = i * N + j, N = dim(H)
      J = k * N + l, N = dim(H)
    B_{IJ} = (X Rhbar)_{ik} delta_{lj} - Rhbar_{ik} X_{lj}
      i = I // N
      j = I % N
      k = J // N
      l = J % N

    Assumption: X and Rhbar are matrices of real numbers.
    """

    N = X.shape[0]
    dim = N**2

    XRhbar = np.matmul(X, Rhbar)

    B = np.zeros((dim, dim), dtype=np.float64)
    for I in range(dim):
        i, j = dec_composite_index(I, N)
        for J in range(dim):
            k, l = dec_composite_index(J, N)
            B[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - Rhbar(i, k) * X[l, j]

    return B

def construct_BST(B):
    """
    Get super transpose of the superoperator B.
    BST_{I J} = B_{It J}
      I -> ij -> ji -> It
    """

    dim = B.shape[0]
    N = int(np.sqrt(dim))

    BST = np.zeros((dim, dim), dtype=np.complex64)
    for I in range(dim):
        i, j = dec_composite_index(I, N)
        It = get_composite_index(j, i, N)
        BST[I] = B[It]

    return BST

def construct_BST_real(B):
    """
    Get super transpose of the superoperator B.
    BST_{I J} = B_{It J}
      I -> ij -> ji -> It

    Assumption: B is a matrix of real numbers.
    """

    dim = B.shape[0]
    N = int(np.sqrt(dim))

    BST = np.zeros((dim, dim), dtype=np.float64)
    for I in range(dim):
        i, j = dec_composite_index(I, N)
        It = get_composite_index(j, i, N)
        BST[I] = B[It]

    return BST

def construct_D(A, B, lambda_):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[D11, D12], [D21, D22]]) (rhore, rhoim)^T
    
        D11 =  const1 * Aim - lambda_**2 pi const1**2 (Bre + BSTre)
        D12 =  const1 * Are + lambda_**2 pi const1**2 (Bim + BSTim)
        D21 = -const1 * Are - lambda_**2 pi const1**2 (Bim - BSTim)
        D22 =  const1 * Aim - lambda_**2 pi const1**2 (Bre - BSTre)
    """

    BST = construct_BST(B)

    Are = np.real(A)
    Aim = np.imag(A)
    Bre = np.real(B)
    Bim = np.imag(B)
    BSTre = np.real(BST)
    BSTim = np.imag(BST)

    D11 =  const1 * Aim - lambda_**2 * np.pi * const1**2 * (Bre + BSTre)
    D12 =  const1 * Are + lambda_**2 * np.pi * const1**2 * (Bim + BSTim)
    D21 = -const1 * Are - lambda_**2 * np.pi * const1**2 * (Bim - BSTim)
    D22 =  const1 * Aim - lambda_**2 * np.pi * const1**2 * (Bre - BSTre)

    D1 = np.hstack(D11, D12)
    D2 = np.hstack(D21, D22)
    D  = np.vstack(D1 , D2 )

    return D

def construct_D_real(A, B, lambda_):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[D11, D12], [D21, D22]]) (rhore, rhoim)^T
    
        D11 =  const1 * Aim - lambda_**2 pi const1**2 (Bre + BSTre)
        D12 =  const1 * Are + lambda_**2 pi const1**2 (Bim + BSTim)
        D21 = -const1 * Are - lambda_**2 pi const1**2 (Bim - BSTim)
        D22 =  const1 * Aim - lambda_**2 pi const1**2 (Bre - BSTre)

    Assumption: A and B are matrices of real numbers.
    """

    BST = construct_BST(B)

    D11 = - lambda_**2 * np.pi * const1**2 * (B + BST)
    D12 =   const1 * A                                             
    D21 = - const1 * A                                             
    D22 = - lambda_**2 * np.pi * const1**2 * (B - BST)

    D1 = np.hstack(D11, D12)
    D2 = np.hstack(D21, D22)
    D  = np.vstack(D1 , D2 )

    return D

def construct_D_by_adding_D0_and_diagonalAzee(D0, Azee):
    """
    D0:   The superoperator for the initial Hamiltonian (with a small Zeeman perturbation)
          D0 = [[D011, D012], [D021, D022]]
    Azee: The A matrix from the Zeeman term

    D = [[D11, D12], [D21, D22]]
          D11 = D011
          D12 = D012 + const1 * Azee
          D21 = D021 - const1 * Azee
          D22 = D022

    Assumption: Azee is diagonal.
    """

    D = D0.copy()

    #c1diagAzee = const1 * Azee.diagonal()
    #Dzee = np.hstack(([c1diagAzee], [-c1diagAzee]))[0]

    #dim = D.shape[0]

    #for I in range(dim):
        #D[I, I] += Dzee[I]

    c1diagAzee = const1 * Azee.diagonal()
    dim = D.shape[0]//2
    for i in range(dim):
        D[    i,     i] += c1diagAzee[i]
        D[dim+i, dim+i] -= c1diagAzee[i]

    return D

def construct_D_by_adding_D0_and_Azeevec(D0, Azeevec):
    """
    D0:   The superoperator for the initial Hamiltonian (with a small Zeeman perturbation)
          D0 = [[D011, D012], [D021, D022]]
    Azee: The A matrix from the Zeeman term

    D = [[D11, D12], [D21, D22]]
          D11 = D011
          D12 = D012 + const1 * Azee
          D21 = D021 - const1 * Azee
          D22 = D022

    Azeevec = Azee.diagonal().

    Assumption: Azee is diagonal.
    """

    D = D0.copy()

    c1Azeevec = const1 * Azeevec
    dim = D.shape[0]//2
    for i in range(dim):
        D[    i,     i] += c1Azeevec[i]
        D[dim+i, dim+i] -= c1Azeevec[i]

    return D

# Next step: get_Dabc
def get_Dabc(D0, Mv_tot):
    return # (Da, Db, Dc)

# Next next step: get_Dabc_reuse reuse the matrices to save time in allocating memory.

def evolve_deltat_sqme(Da, Db, Dc, rho, deltat):
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

    k1 = np.matmul(Da, rho)
    k2 = np.matmul(Db, rho + 0.5*deltat*k1)
    k3 = np.matmul(Db, rho + 0.5*deltat*k2)
    k4 = np.matmul(Dc, rho +     deltat*k3)

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


