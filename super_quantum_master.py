import numpy as np
from constants import const1, Kelvin2wavenumber, Tesla2wavenumber
from common import get_total_Sz_for_all_eigenstates
from common import get_commutation
from common import kronecker_delta
from schrodinger import get_habc, get_habc_reuse_ha
from quantum_master import energy2omega, omega2energy, spectral_density, Phi, construct_X, construct_Rhbar
from scipy.sparse import csr_array

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

def construct_A_diag_real(H):
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

def construct_A_diag_from_H_diag(Hvec):
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

    Let Hvec = H.diagonal()
        A_diag = A.diagonal()
    """

    N = Hvec.shape[0]
    dim = N**2

    A_diag = np.zeros(dim, dtype=np.float64)

    I = 0
    for i in range(N):
        for j in range(N):
            A_diag[I] = Hvec[i] - Hvec[j]
            I = I + 1

    return A_diag

def construct_B(X, Rhbar, is_real=True):
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

    if is_real:
        B = np.zeros((dim, dim), dtype=np.float64)
    else:
        B = np.zeros((dim, dim), dtype=np.complex64)

    for I in range(dim):
        i, j = dec_composite_index(I, N)
        for J in range(dim):
            k, l = dec_composite_index(J, N)
            B[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - Rhbar[i, k] * X[l, j]

    return B

def construct_BST(B, is_real=True):
    """
    Get super transpose of the superoperator B.
    BST_{I J} = B_{It J}
      I -> ij -> ji -> It
    """

    dim = B.shape[0]
    N = int(np.sqrt(dim))

    if is_real:
        BST = np.zeros((dim, dim), dtype=np.float64)
    else:
        BST = np.zeros((dim, dim), dtype=np.complex64)

    for I in range(dim):
        i, j = dec_composite_index(I, N)
        It = get_composite_index(j, i, N)
        BST[I] = B[It]

    return BST

def construct_D(A, B, lambda_, is_real=True):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[D11, D12], [D21, D22]]) (rhore, rhoim)^T
    
        D11 =  const1 * Aim - lambda_**2 pi const1**2 (Bre + BSTre)
        D12 =  const1 * Are + lambda_**2 pi const1**2 (Bim + BSTim)
        D21 = -const1 * Are - lambda_**2 pi const1**2 (Bim - BSTim)
        D22 =  const1 * Aim - lambda_**2 pi const1**2 (Bre - BSTre)

    is_real: Are A and B real?
    """

    BST = construct_BST(B, is_real=is_real)

    if is_real:
        D11 = - lambda_**2 * np.pi * const1**2 * (B + BST)
        D12 =   const1 * A                                             
        D21 = - const1 * A                                             
        D22 = - lambda_**2 * np.pi * const1**2 * (B - BST)
    else:
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
        
    D1 = np.hstack((D11, D12))
    D2 = np.hstack((D21, D22))
    D  = np.vstack((D1 , D2 ))

    return D

def construct_D_using_A_diag(A_diag, B, lambda_):
    """
    The quantum master equation reads (in the units given at the beginning of this file)
        d (rhore, rhoim)^T / d t = np.array([[D11, D12], [D21, D22]]) (rhore, rhoim)^T
    
        D11 =  const1 * Aim - lambda_**2 pi const1**2 (Bre + BSTre)
        D12 =  const1 * Are + lambda_**2 pi const1**2 (Bim + BSTim)
        D21 = -const1 * Are - lambda_**2 pi const1**2 (Bim - BSTim)
        D22 =  const1 * Aim - lambda_**2 pi const1**2 (Bre - BSTre)

    Assumption:
        A is diagonal
        A_diag = A.diagonal()
        A and B are real
    """

    BST = construct_BST(B, is_real=True)

    D11 = - lambda_**2 * np.pi * const1**2 * (B + BST)

    D12 = np.zeros( B.shape , dtype=np.float64)
    D21 = np.zeros( B.shape , dtype=np.float64)
    for i in range(B.shape[0]):
        D12[i, i] =   const1 * A_diag[i] + 1.0
        D21[i, i] = - const1 * A_diag[i] + 1.0

    D22 = - lambda_**2 * np.pi * const1**2 * (B - BST)

    #spy_sparsity(D11, "D11", precision=1.0e-20, figsize=(20, 20), markersize=1)
    #spy_sparsity(D22, "D22", precision=1.0e-20, figsize=(20, 20), markersize=1)

    D1 = np.hstack((D11, D12))
    D2 = np.hstack((D21, D22))
    D  = np.vstack((D1 , D2 ))

    #D = csr_array(D)

    # The +1.0 and -1.0 operations are for creating the desired sparsity.
    dim = B.shape[0]
    for i in range(dim):
        D[    i, dim+i] -= 1.0
        D[dim+i,     i] -= 1.0

    return D

def construct_D_by_adding_D0_and_Azee_diag(D0, Azee_diag):
    """
    D0:   The superoperator for the initial Hamiltonian (with a small Zeeman perturbation)
          D0 = [[D011, D012], [D021, D022]]
    Azee: The A matrix from the Zeeman term

    D = [[D11, D12], [D21, D22]]
          D11 = D011
          D12 = D012 + const1 * Azee
          D21 = D021 - const1 * Azee
          D22 = D022

    Assumption:
        Azee is diagonal.
        Azee_diag = Azee.diagonal().
    """

    D = D0.copy()

    c1Azee_diag = const1 * Azee_diag
    dim = D.shape[0]//2
    for i in range(dim):
        D[    i, dim+i] += c1Azee_diag[i]
        D[dim+i,     i] -= c1Azee_diag[i]

    return D

def construct_D_from_D0_and_Bfield(D0, minus_Mz_tot_eff_diag, B):
    """
    B in wavenumber.
    """

    Hzee_diag = minus_Mz_tot_eff_diag * B
    Azee_diag = construct_A_diag_from_H_diag(Hzee_diag)
    D = construct_D_by_adding_D0_and_Azee_diag(D0, Azee_diag)

    return D

def get_Dabc(D0, minus_Mz_tot_eff_diag, it, deltat, Bs_wavenumber):
    """
    Obtain Da = D(ts[it])
           Db = D(ts[it] + deltat/2)
           Dc = D(ts[it] + deltat)

    Bs_wavenumber are the fields on the time grid with a time step of helf deltat.
    """

    Da = construct_D_from_D0_and_Bfield(D0, minus_Mz_tot_eff_diag, Bs_wavenumber[2*it])
    Db = construct_D_from_D0_and_Bfield(D0, minus_Mz_tot_eff_diag, Bs_wavenumber[2*it+1])
    Dc = construct_D_from_D0_and_Bfield(D0, minus_Mz_tot_eff_diag, Bs_wavenumber[2*it+2])

    return (Da, Db, Dc)

def get_Dabc_reuse_Da(D0, minus_Mz_tot_eff_diag, it, deltat, Bs_wavenumber, Da, Db, Dc):
    """
    Obtain Da = D(ts[it])
           Db = D(ts[it] + deltat/2)
           Dc = D(ts[it] + deltat)

    Bs_wavenumber are the fields on the time grid with a time step of helf deltat.
    """

    Db = construct_D_from_D0_and_Bfield(D0, minus_Mz_tot_eff_diag, Bs_wavenumber[2*it+1])
    Dc = construct_D_from_D0_and_Bfield(D0, minus_Mz_tot_eff_diag, Bs_wavenumber[2*it+2])

    return (Da, Db, Dc)

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

    k1 = Da @ rho                   # np.matmul(Da, rho)
    k2 = Db @ (rho + 0.5*deltat*k1) # np.matmul(Db, rho + 0.5*deltat*k1)
    k3 = Db @ (rho + 0.5*deltat*k2) # np.matmul(Db, rho + 0.5*deltat*k2)
    k4 = Dc @ (rho +     deltat*k3) # np.matmul(Dc, rho +     deltat*k3)

    rho_new = rho + deltat * (k1 + 2*k2 + 2*k3 + k4) / 6

    return rho_new

def evolve_rho_sqme(D0, Mz_tot_diag, double_super_rho, nt, deltat, Bs):
    """
    Evolve the density matrix using the Runge-Kutta method according to the quantum master equation.

    h0 and Mv_tot are written on the basis of the eigenvectors of h0.

    Input:
      D0: double superoperator at t0 = ts[0]
      Mz_tot_diag: diagnal matrix elements of the z component of the magnetization operators
      double_super_rho: initial double super density matrix at t0.
      nt: number of time steps.
      delta: time step in unit of ps.
      Bs: list of magnetic field.

    Assumptions: 
      The magnetic field is along the z direction.
      The exchange couplings are isotropic.
      The g tensors are isotropic and identical.

    As a result, Mz_tot is real and diagonal.
    """

    # Evolve the density matrix

    dim = len(Mz_tot_diag)
    dims = dim*dim
    dimds = 2*dims

    ## It is more efficient to use minus_Mz_tot_diag and Bs_wavenumber. 
    minus_Mz_tot_diag = -1 * Mz_tot_diag
    Bs_wavenumber = Bs * Tesla2wavenumber 

    Da, Db, Dc = get_Dabc(D0, minus_Mz_tot_diag, 0, deltat, Bs_wavenumber)
    #np.savetxt("./output/Db.dat", Db, fmt="%12.4e"); exit()

    double_super_rho = evolve_deltat_sqme(Da, Db, Dc, double_super_rho, deltat)
    #np.savetxt("./output/double_super_rho.dat", double_super_rho, fmt="%12.6f")
    #print("i = 0, max(double_super_rho) = ", np.max(np.absolute(double_super_rho)))

    for i in range(1, nt):
        Da, Db, Dc = get_Dabc_reuse_Da(D0, minus_Mz_tot_diag, i, deltat, Bs_wavenumber, Dc, Da, Db)
        double_super_rho = evolve_deltat_sqme(Da, Db, Dc, double_super_rho, deltat)
        #print("i = {:d}, max(double_super_rho) = ".format(i), np.max(np.absolute(double_super_rho)))

    super_rho_re = double_super_rho[0:dims]
    super_rho_im = double_super_rho[dims:dimds]
    super_rho = super_rho_re + 1j * super_rho_im
    rho = super_rho.reshape((dim, dim))

    #print( np.max( np.absolute(rho - rho0) ) )

    ##np.savetxt("./output/rho.dat", rho, fmt="%12.6f")

    return rho


