"""
Per-stair routines of qdmag at commit 14f7d02, before the GPU refactor, kept as test
oracles for the vectorized code. The bodies are copied verbatim; only `self` now refers
to a LegacyLiouville, which is set up from the static data of a current liouville object.
"""
import numpy as np

from qdmag.core.constants import const1, Tesla2wavenumber
from qdmag.core.common import kronecker_delta, eigen_simple
from qdmag.core.quantum_master import energy2omega, omega2energy, spectral_density
from qdmag.core.liouville import liouville, get_composite_index, dec_composite_index
from qdmag.core.constants import Kelvin2wavenumber


def Phi(T, omega, I0, alpha=2):
    """
    Construct the function Phi in the quantum master equation.
    """
    beta = 1/(Kelvin2wavenumber * T)
    energy = omega2energy(omega)
    if abs( beta*energy ) < 1e-3:
        result = ( I0*const1*np.abs(omega)**(alpha-1)/beta
                   - I0*np.sign(omega)*np.abs(omega)**alpha/2 )
    elif beta * energy > 700:
        result = 0.0 * omega  # preserve scalar/array shape of omega
    else:
        numerator = spectral_density(omega, I0=I0, alpha=alpha) - spectral_density(-omega, I0=I0, alpha=alpha)
        denominator = np.expm1(beta * energy)
        result = numerator / denominator
    return result


def get_Rhbar(h, X, I0, T, alpha=2):
    eigen = eigen_simple(h)
    M = eigen.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))
    X_E = np.matmul(M_dagger, np.matmul(X, M))
    Rhbar = np.zeros(X.shape, dtype=np.complex128)
    omegas = energy2omega(eigen.eigenvalues)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            omega_ij = omegas[i] - omegas[j]
            Rhbar[i, j] = X_E[i, j] * Phi(T, omega_ij, I0, alpha=alpha)
    Rhbar = np.matmul(M, np.matmul(Rhbar, M_dagger))
    return Rhbar


def update_Rhbar(Rhbar, h, X, I0, T, alpha=2):
    eigen = eigen_simple(h)
    M = eigen.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))
    X_E = np.matmul(M_dagger, np.matmul(X, M))
    omegas = energy2omega(eigen.eigenvalues)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            omega_ij = omegas[i] - omegas[j]
            Rhbar[i, j] = X_E[i, j] * Phi(T, omega_ij, I0, alpha=alpha)
    Rhbar = np.matmul(M, np.matmul(Rhbar, M_dagger))
    return Rhbar


class LegacyLiouville:
    """
    The dense L of commit 14f7d02, built from the static data (h0, Mz, X and the
    dynamics parameters) of a current liouville object `lio`.
    """

    # Construction routines that the refactor does not change
    construct_A_vectorized = liouville.construct_A_vectorized

    def __init__(self, lio):
        for name in ("h0", "Mz", "X", "dim", "dims", "dimds", "lambdaa", "I0", "T", "alpha", "Bt", "tmin"):
            setattr(self, name, getattr(lio, name))
        self.set_up_liouville()

    def set_up_liouville(self):
        self.A0 = self.construct_A_vectorized(self.h0, dtype=np.complex128)
        self.L0 = self.construct_L(self.A0, None)
        self.h_zee = -1 * Tesla2wavenumber * self.Bt(self.tmin) * self.Mz
        self.h = self.h0 + self.h_zee
        self.A = self.construct_A_vectorized(self.h, dtype=np.complex128)
        self.Rhbar = get_Rhbar(self.h, self.X, self.I0, self.T, alpha=self.alpha)
        self.C = self.construct_C(self.X, self.Rhbar)
        self.CST = self.construct_CST(self.C)
        self.n_nzC, self.indices_nzC = self.get_indices_nzC(self.X)
        self.L = self.construct_L(self.A, self.C)
        self.LA = self.construct_L(self.A, None)
        self.LC = self.construct_L(None, self.C)

    def construct_C(self, X, Rhbar):
        XRhbar = np.matmul(X, Rhbar)
        C = np.zeros((self.dims, self.dims), dtype=np.complex128)
        for I in range(self.dims):
            i, j = dec_composite_index(I, self.dim)
            for J in range(self.dims):
                k, l = dec_composite_index(J, self.dim)
                C[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - Rhbar[i, k] * X[l, j]
        return C

    def construct_CST(self, C):
        CST = np.zeros((self.dims, self.dims), dtype=np.complex128)
        for I in range(self.dims):
            i, j = dec_composite_index(I, self.dim)
            It = get_composite_index(j, i, self.dim)
            CST[I] = C[It]
        return CST

    def get_indices_nzC(self, X):
        indices_nzX = np.nonzero(X)
        set_index_tuples_nzX = set( [(indices_nzX[0][i], indices_nzX[1][i]) for i in range(indices_nzX[0].shape[0])] )
        indices_nzC = []
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        if (l == j) or ( (l, j) in set_index_tuples_nzX ):
                            indices_nzC.append( (i, j, k, l) )
        n_nzC = len(indices_nzC)
        return (n_nzC, indices_nzC)

    def construct_L(self, A, C):
        Are = np.real(A)
        Aim = np.imag(A)
        if A is not None and C is None:
            L11 =  const1 * Aim
            L12 =  const1 * Are
            L21 = -const1 * Are
            L22 =  const1 * Aim
        elif A is None and C is not None:
            CST = self.construct_CST(C)
            Cre = np.real(C)
            Cim = np.imag(C)
            CSTre = np.real(CST)
            CSTim = np.imag(CST)
            L11 = -self.lambdaa**2 * np.pi * const1**2 * (Cre + CSTre)
            L12 =  self.lambdaa**2 * np.pi * const1**2 * (Cim + CSTim)
            L21 = -self.lambdaa**2 * np.pi * const1**2 * (Cim - CSTim)
            L22 = -self.lambdaa**2 * np.pi * const1**2 * (Cre - CSTre)
        else:
            CST = self.construct_CST(C)
            Cre = np.real(C)
            Cim = np.imag(C)
            CSTre = np.real(CST)
            CSTim = np.imag(CST)
            L11 =  const1 * Aim - self.lambdaa**2 * np.pi * const1**2 * (Cre + CSTre)
            L12 =  const1 * Are + self.lambdaa**2 * np.pi * const1**2 * (Cim + CSTim)
            L21 = -const1 * Are - self.lambdaa**2 * np.pi * const1**2 * (Cim - CSTim)
            L22 =  const1 * Aim - self.lambdaa**2 * np.pi * const1**2 * (Cre - CSTre)
        L1 = np.hstack((L11, L12))
        L2 = np.hstack((L21, L22))
        L  = np.vstack((L1 , L2 ))
        return L

    def update_C_and_CST(self):
        XRhbar = np.matmul(self.X, self.Rhbar)
        for idx in range(self.n_nzC):
            i, j, k, l = self.indices_nzC[idx]
            I = i * self.dim + j
            J = k * self.dim + l
            self.C[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - self.Rhbar[i, k] * self.X[l, j]
            It = j * self.dim + i
            self.CST[It, J] = self.C[I, J]

    def update_L_under_magnetic_field(self,  B):
        self.h_zee = -1 * Tesla2wavenumber * B * self.Mz
        Azee = self.construct_A_vectorized(self.h_zee, dtype=np.complex128)
        c1Azee = const1 * Azee
        c1Azeere = np.real(c1Azee)
        c1Azeeim = np.imag(c1Azee)
        self.h = self.h0 + self.h_zee
        self.Rhbar = update_Rhbar(self.Rhbar, self.h, self.X, self.I0, self.T, alpha=self.alpha)
        self.update_C_and_CST()
        factorC = self.lambdaa**2 * np.pi * const1**2 * self.C
        factorCST = self.lambdaa**2 * np.pi * const1**2 * self.CST
        factorCre = np.real(factorC)
        factorCim = np.imag(factorC)
        factorCSTre = np.real(factorCST)
        factorCSTim = np.imag(factorCST)
        self.LA[        0:self.dims,          0: self.dims] = self.L0[        0:self.dims,          0: self.dims] + c1Azeeim
        self.LA[        0:self.dims,  self.dims:self.dimds] = self.L0[        0:self.dims,  self.dims:self.dimds] + c1Azeere
        self.LA[self.dims:self.dimds,         0: self.dims] = self.L0[self.dims:self.dimds,         0: self.dims] - c1Azeere
        self.LA[self.dims:self.dimds, self.dims:self.dimds] = self.L0[self.dims:self.dimds, self.dims:self.dimds] + c1Azeeim
        self.LC[        0:self.dims,          0: self.dims] = - (factorCre + factorCSTre)
        self.LC[        0:self.dims,  self.dims:self.dimds] =   (factorCim + factorCSTim)
        self.LC[self.dims:self.dimds,         0: self.dims] = - (factorCim - factorCSTim)
        self.LC[self.dims:self.dimds, self.dims:self.dimds] = - (factorCre - factorCSTre)
        self.L = self.LA + self.LC
