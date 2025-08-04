import os
import copy
import numpy as np
import h5py
from filelock import FileLock
from scipy.linalg import expm
from spin_dynamics.core.constants import const1, Tesla2wavenumber
from spin_dynamics.core.common import kronecker_delta, create_outdir
from spin_dynamics.core.common import get_Mv_from_rho, get_Mz_from_rho
from spin_dynamics.core.common import eigen_simple, get_h_Zeeman_Mv_eff
from spin_dynamics.core.quantum_master import get_Rhbar, update_Rhbar
from spin_dynamics.core.pulse import get_Bt, get_pulse_RK4_double_grid

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
# Class for the Liouville superoperator.
# ============================================================================ #

class liouville:
    def __init__(self, eff, dynamics):
        """
        eff: effective_basis object
        dynamics: list of dictionaries containing parameters for the spin dynamics simulation
        """

        # Effective basis object
        self.eff = eff

        # Make aliases of h0_eff, Mz_eff, Mv_eff, and X_eff for easy access
        self.h0 = self.eff.h0_eff
        self.Mz = self.eff.Mz_eff
        self.Mv = self.eff.Mv_eff
        self.X = self.eff.X_eff
    
        # Dimensions
        self.dim = self.eff.dim            # Dimension of the effective Hamiltonian
        self.dims  = self.dim*self.dim     # Dimension of superoperators for the vectorized density matrix
        self.dimds = 2*self.dims           # Dimension of superoperators for the RI-separated vectorized density matrix
    
        print("Dimension of the effective Hamiltonian: {:6d}".format(self.dim))
        print("Dimension of superoperators for the vectorized density matrix: {:6d}".format(self.dims))
        print("Dimension of superoperators for the RI-separated vectorized density matrix: {:6d}\n".format(self.dimds))

        # Control parameters for time evolution
        self.T             = dynamics[0]['T']             # Temperature in K
        self.lambdaa       = dynamics[0]['lambdaa']       # Spin phonon coupling constant in cm-1
        self.I0            = dynamics[0]['I0']            # Prefactor for the phonon density of states in ps

        self.Bt_params     = dynamics[1]                  # Parameters for the pulsed magnetic field

        self.tmin          = dynamics[2]['tmin']          # Initial time in ps
        self.tmax          = dynamics[2]['tmax']          # Finial time in ps
        self.deltat        = dynamics[2]['deltat']        # Time step in ps
                      
        self.save_mag      = dynamics[3]['save_mag']      # Save magnetization ?
        self.nt_mag        = dynamics[3]['nt_mag']        # Calculate and save magnetization every nt_mag*deltat ps
        self.save_rho      = dynamics[3]['save_rho']      # Save rho ?
        self.nt_rho        = dynamics[3]['nt_rho']        # Save rho every nt_rho*deltat ps

        # The pulsed magnetic field
        self.Bt = get_Bt(self.Bt_params)

        # Set up the Liouville superoperator
        self.set_up_liouville()
        
    def set_up_liouville(self):
        """
        Set up the Liouville superoperator.
        """

        # h0, A0, and L0 are time independent.

        # Construct the superoperator L0 at t = 0 ps without spin-phonon coupling
        self.A0 = self.construct_A(self.h0, diagonal_H=False, dtype=np.complex128)
        self.L0 = self.construct_L(self.A0, None)
    
        # h_zee, h, A, Rhbar, C, CST, and L will be updated at each time step.
        # We will reuse them (the memory associated with them) to save time.
    
        # Zeeman term at t = tmin ps on the effective basis
        self.h_zee = -1 * Tesla2wavenumber * self.Bt(self.tmin) * self.Mz

        # Effective Hamiltonian at t = tmin ps, it will be updated at each time step
        self.h = self.h0 + self.h_zee

        # Construct the superoperator A at t = tmin ps, it will be updated at each time step
        self.A = self.construct_A(self.h, diagonal_H=False, dtype=np.complex128)
    
        # Construct the superoperator Rhbar in the S representation at t = tmin ps
        self.Rhbar = get_Rhbar(self.h, self.X, self.I0, self.T)
    
        # Construct the superoperator C which is time dependent due to Rhbar
        self.C = self.construct_C(self.X, self.Rhbar)
        self.CST = self.construct_CST(self.C)

        # Get the indices of nonzero matrix elements of C
        self.n_nzC, self.indices_nzC = self.get_indices_nzC(self.X)
    
        # Construct the superoperator L at t = tmin ps with spin-phonon coupling
        self.L = self.construct_L(self.A, self.C)

    def construct_A(self, H, diagonal_H=False, dtype=np.complex128):
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
    
        A = np.zeros((self.dims, self.dims), dtype=dtype)
        
        if diagonal_H:
            #for I in range(self.dims):
                #i, j = dec_composite_index(I, self.dim)
                #A[I, I] = H[i, i] - H[j, j]
            
            # Faster
            #for i in range(self.dim):
                #for j in range(self.dim):
                    #I = i * dim + j
                    #A[I, I] = H[i, i] - H[j, j]
            
            # Even faster
            I = 0
            for i in range(self.dim):
                for j in range(self.dim):
                    A[I, I] = H[i, i] - H[j, j]
                    I = I + 1
        else:
            for I in range(self.dims):
                i, j = dec_composite_index(I, self.dim)
                for J in range(self.dims):
                    k, l = dec_composite_index(J, self.dim)
                    A[I, J] = H[i, k] * kronecker_delta(l, j) - kronecker_delta(i, k) * H[l, j]
    
        return A

    def construct_C(self, X, Rhbar):
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
    
        C = np.zeros((self.dims, self.dims), dtype=np.complex128)
    
        for I in range(self.dims):
            i, j = dec_composite_index(I, self.dim)
            for J in range(self.dims):
                k, l = dec_composite_index(J, self.dim)
                C[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - Rhbar[i, k] * X[l, j]
    
        return C

    def construct_CST(self, C):
        """
        Get super transpose of the superoperator C.
        CST_{I J} = C_{It J}
          I -> ij -> ji -> It
        """
    
        CST = np.zeros((self.dims, self.dims), dtype=np.complex128)
    
        for I in range(self.dims):
            i, j = dec_composite_index(I, self.dim)
            It = get_composite_index(j, i, self.dim)
            CST[I] = C[It]
    
        return CST

    def get_indices_nzC(self, X):
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
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        if (l == j) or ( (l, j) in set_index_tuples_nzX ):
                            indices_nzC.append( (i, j, k, l) )
        n_nzC = len(indices_nzC)
    
        return (n_nzC, indices_nzC)

    def construct_L(self, A, C):
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
        """
        Save time by avoiding memory allocation for C.
        Rhbar is time dependent, and so is C.
        C and CST are recomputed completely at each time step.
        """
    
        XRhbar = np.matmul(self.X, self.Rhbar)
    
        for idx in range(self.n_nzC):
            i, j, k, l = self.indices_nzC[idx]
            I = i * self.dim + j
            J = k * self.dim + l
            self.C[I, J] = XRhbar[i, k] * kronecker_delta(l, j) - self.Rhbar[i, k] * self.X[l, j]
            It = j * self.dim + i
            self.CST[It, J] = self.C[I, J]

    def update_L_under_magnetic_field(self,  B):
        r"""
        L: L matrix to be updated.
        L0: L matrix at t = 0 ps.
        h: Hamiltonian at t = t ps
          h is recalculated at each time step.
          save memory by avoiding memory allocation for h.
          The off-diagonal elements of h do not change with time.
        h0: spin Hamiltonian at t = 0 ps
        Mz: the z component of the total magnetic moment operator M = (Mx, My, Mz).
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
    
        self.h_zee = -1 * Tesla2wavenumber * B * self.Mz
        Azee = self.construct_A(self.h_zee, diagonal_H=False, dtype=np.complex128)
        c1Azee = const1 * Azee
        c1Azeere = np.real(c1Azee)
        c1Azeeim = np.imag(c1Azee)
    
        # Update the diagonal elements of the Hamiltonian h
        self.h = self.h0 + self.h_zee
    
        # Update the operators C and CST
        self.Rhbar = update_Rhbar(self.Rhbar, self.h, self.X, self.I0, self.T)
        self.update_C_and_CST()
        factorC = self.lambdaa**2 * np.pi * const1**2 * self.C
        factorCST = self.lambdaa**2 * np.pi * const1**2 * self.CST
        factorCre = np.real(factorC)
        factorCim = np.imag(factorC)
        factorCSTre = np.real(factorCST)
        factorCSTim = np.imag(factorCST)
    
        # Add the Zeeman interaction and the spin-phonon coupling to the L matrix
        self.L[        0:self.dims,          0: self.dims] = self.L0[        0:self.dims,          0: self.dims] + c1Azeeim - (factorCre + factorCSTre)
        self.L[        0:self.dims,  self.dims:self.dimds] = self.L0[        0:self.dims,  self.dims:self.dimds] + c1Azeere + (factorCim + factorCSTim)
        self.L[self.dims:self.dimds,         0: self.dims] = self.L0[self.dims:self.dimds,         0: self.dims] - c1Azeere - (factorCim - factorCSTim)
        self.L[self.dims:self.dimds, self.dims:self.dimds] = self.L0[self.dims:self.dimds, self.dims:self.dimds] + c1Azeeim - (factorCre - factorCSTre)

    def get_L_max_and_expLdeltat_max(self, B, deltat):
        """
        Get the maximum of the absolute values of the elements of L and exp(L * deltat) at the given magnetic field B and time step deltat.
        """
        self.update_L_under_magnetic_field(B)
        L_max, expLdeltat_max = np.max(np.abs(self.L)), np.max(np.abs(expm(self.L*deltat)))
        print("B = {:15.6f} T, deltat = {:15.3f}  ps, max(|L|) = {:12.6f}, max(|exp(L * deltat)|) = {:12.6f}\n".format(B, deltat, L_max, expLdeltat_max))
        return (L_max, expLdeltat_max)

    def examine_L_max_and_expLdeltat_max(self, Bs, deltats, tag):
        """
        Examine the maximum of the absolute values of the elements of L and exp(L * deltat)
        at various B fields and time steps.
        Bs: list of magnetic fields in Tesla.
        deltats: list of time steps in ps.
        """
        # Initialize arrays for the maximum values
        m = len(Bs)
        n = len(deltats)
        L_max = np.zeros(m)
        expLdeltat_max = np.zeros((m, n))
        # Get the maximum of the absolute values of the elements of L and exp(L * deltat)
        for i in range(m):
            for j in range(n):
                print("Examining B = {:15.6f} T, deltat = {:15.3f} ps ...".format(Bs[i], deltats[j]))
                L_max[i], expLdeltat_max[i, j] = self.get_L_max_and_expLdeltat_max(Bs[i], deltats[j])
        # Check if the output directory exists, if not, create it.
        create_outdir()
        # Save L_max in the file L_max.dat
        with open("./output/L_max_" + tag + ".dat", "w") as f:
            f.write("# B (T)   L_max\n")
            for i in range(m):
                f.write( "{:15.6f} {:15.6f}\n".format(Bs[i], L_max[i]) )
        print("L_max saved to ./output/L_max_" + tag + ".dat")
        # Save expLdeltat_max in the file expLdeltat_max.dat
        with open("./output/expLdeltat_max_" + tag + ".dat", "w") as f:
            f.write( ("# B (T)" + n*"   dt = {:.3f} ps" + "\n").format(*deltats) )
            for i in range(m):
                f.write( ("{:15.6f}" + n*" {:15.6f}" + "\n").format(Bs[i], *expLdeltat_max[i, :]) )
        print("expLdeltat_max saved to ./output/expLdeltat_max_" + tag + ".dat")

# ============================================================================ #
# Auxilliary functions for constructing the Liouville superoperator.
# ============================================================================ #

def get_composite_index(i, j, N):
    I = i * N + j
    return I

def dec_composite_index(I, N):
    i = I // N
    j = I % N
    return (i, j)

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
# Functions for solving the quantum master equation.
# ============================================================================ #

def get_Labc(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, it):
    """
    Obtain La = L(ts[it])
           Lb = L(ts[it] + deltat/2)
           Lc = L(ts[it] + deltat)
    """
    La, _, _ = update_L_under_magnetic_field(L, L0, h, h_t0, Mz_op, Bs2[2*it  ], C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    Lb, _, _ = update_L_under_magnetic_field(L, L0, h, h_t0, Mz_op, Bs2[2*it+1], C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    Lc, _, _ = update_L_under_magnetic_field(L, L0, h, h_t0, Mz_op, Bs2[2*it+2], C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    return (La, Lb, Lc)

def get_Labc_reuse_La(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, it, La, Lb, Lc):
    """
    Obtain La = L(ts[it])
           Lb = L(ts[it] + deltat/2)
           Lc = L(ts[it] + deltat)
    """
    Lb, _, _ = update_L_under_magnetic_field(L, L0, h, h_t0, Mz_op, Bs2[2*it+1], C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    Lc, _, _ = update_L_under_magnetic_field(L, L0, h, h_t0, Mz_op, Bs2[2*it+2], C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds)
    return (La, Lb, Lc)

def evolve_deltat_liouville(La, Lb, Lc, risvrho, deltat):
    """
    Evolve rho by deltat using the Runge-Kutta method according to the quantum master equation
        d risvrho / d t = L risvrho
        risvrho = np.vstack(vrhore, vrhoim)
    Input: 
      La: L(t), the Liouville superoperator at time t. Unit: cm-1.
      Lb: L(t+deltat/2)
      Lc: L(t+deltat). ha, hb, and hc are written on the basis of the eigenvectors of h0, i.e. the zero-field spin Hamiltonian.
      risvrho: RI-separated vectorized density matrix at time t rho(t) on the basis of the eigenvectors of h0.
      deltat: time step in ps.
    """
    k1 = La @ risvrho                   # np.matmul(La, rho)
    k2 = Lb @ (risvrho + 0.5*deltat*k1) # np.matmul(Lb, rho + 0.5*deltat*k1)
    k3 = Lb @ (risvrho + 0.5*deltat*k2) # np.matmul(Lb, rho + 0.5*deltat*k2)
    k4 = Lc @ (risvrho +     deltat*k3) # np.matmul(Lc, rho +     deltat*k3)
    risvrho_new = risvrho + deltat * (k1 + 2*k2 + 2*k3 + k4) / 6
    return risvrho_new

def evolve_rho_liouville_RK4(t0, t1, deltat, Bt_params, risvrho, L, L0, h_t0, Mz_op, C, CST, X, Rhbar, lambdaa, I0, T, dim, dims, dimds, save_mag, nt_mag, save_rho, nt_rho):
    """
    Evolve the density matrix using the Runge-Kutta method according to the quantum master equation.
    """

    # Make a copy of the hamiltonian h_t0 to store the Hamiltonian at time t
    # This is to avoid repeated memory allocation for h.
    h = copy.deepcopy(h_t0)

    # Set up the pulsed magnetic field
    Bt = get_Bt(Bt_params)

    # Which elements of the C operator are non-zero ?
    n_nzC, indices_nzC = get_indices_nzC(X, dim)

    # Save the magnetization and the density matrix at specified time steps, similar to the codes in evolve_rho_liouville_stairs
    if save_rho and save_mag:
        # nt_rho should be a multiple of nt_mag
        nround_mag = int( max(nt_rho // nt_mag, 1) )
        nt_rho = nround_mag * nt_mag
        
        # nt should be a multiple of nt_rho
        nround_rho = int( max((t1 - t0)//deltat // nt_rho, 1) )
        nt = nround_rho * nt_rho
        
        # Adjust the final time
        t1 = t0 + nt*deltat

        # Get the magnetic field pulse for the Runge-Kutta method
        nt, ts, Bs2, deltat = get_pulse_RK4_double_grid(Bt, t0, t1, deltat)

        # Output directories
        outdir_rho, outdir_mag = get_outdirs(T, I0, lambdaa, Bt_params, RK4=True)
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
            f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t0, Bt(t0), Mz, chimz))

            # Loop over the rounds for saving the RI-separated vectorized density matrix
            for iround_rho in range(nround_rho):
                # Loop over the rounds for saving the magnetic moment
                for iround_mag in range(nround_mag):
                    # Loop over the nt_mag time steps
                    for it_mag in range(nt_mag):
                        it = iround_rho * nt_rho + iround_mag * nt_mag + it_mag
                        if it == 0:
                            La, Lb, Lc = get_Labc(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, 0)
                        else:
                            La, Lb, Lc = get_Labc_reuse_La(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, it, Lc, Lb, La)
                        risvrho = evolve_deltat_liouville(La, Lb, Lc, risvrho, deltat)
                    # Calculate the magnetic moment
                    Mz = get_Mz_from_risvrho(risvrho, Mz_op, dim, dims, dimds)
                    # Calculate the magnetic susceptibility
                    chimz = get_chimz_from_risvrho(h, h_t0, Bt, ts[it]+deltat, Mz_op, risvrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
                    # Save the magnetic moment and the magnetic susceptibility
                    f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(ts[it]+deltat, Bt(ts[it]+deltat), Mz, chimz))
                # Save the RI-separated vectorized density matrix
                tag = "{:.3f}".format(ts[it] + deltat)
                dset = f1.create_dataset(tag, data=risvrho)
        print("rho is saved to {}".format(fname1))
        print("mmo is saved to {}".format(fname2))
    elif save_rho and (not save_mag):
        # nt should be a multiple of nt_rho
        nround_rho = int( (t1 - t0)//deltat // nt_rho )
        nt = nround_rho * nt_rho
        
        # Adjust the final time
        t1 = t0 + nt*deltat

        # Get the magnetic field pulse for the Runge-Kutta method
        nt, ts, Bs2, deltat = get_pulse_RK4_double_grid(Bt, t0, t1, deltat)

        # Output directory
        outdir_rho, outdir_mag = get_outdirs(T, I0, lambdaa, Bt_params, RK4=True)
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
                    if it == 0:
                        La, Lb, Lc = get_Labc(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, 0)
                    else:
                        La, Lb, Lc = get_Labc_reuse_La(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, it, Lc, Lb, La)
                    risvrho = evolve_deltat_liouville(La, Lb, Lc, risvrho, deltat)
                # Save the RI-separated vectorized density matrix
                tag = "{:.3f}".format(ts[it] + deltat)
                dset = f1.create_dataset(tag, data=risvrho)
        print("rho is saved to {}".format(fname1))
    elif (not save_rho) and save_mag:
        # nt should be a multiple of nt_mag
        nround_mag = int( (t1 - t0)//deltat // nt_mag )
        nt = nround_mag * nt_mag
        
        # Adjust the final time
        t1 = t0 + nt*deltat

        # Get the magnetic field pulse for the Runge-Kutta method
        nt, ts, Bs2, deltat = get_pulse_RK4_double_grid(Bt, t0, t1, deltat)

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
                    if it == 0:
                        La, Lb, Lc = get_Labc(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, 0)
                    else:
                        La, Lb, Lc = get_Labc_reuse_La(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, it, Lc, Lb, La)
                    risvrho = evolve_deltat_liouville(La, Lb, Lc, risvrho, deltat)
                # Save magnetic moment
                Mz = get_Mz_from_risvrho(risvrho, Mz_op, dim, dims, dimds)
                chimz = get_chimz_from_risvrho(h, h_t0, Bt, ts[it]+deltat, Mz_op, risvrho, X, Rhbar, lambdaa, I0, T, dim, dims, dimds)
                f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(ts[it]+deltat, Bt(ts[it]+deltat), Mz, chimz))
        print("mmo is saved to {}".format(fname2))
    else:
        # the time period should be a multiple of deltat
        nt = int( (t1 - t0)//deltat )
        
        # Adjust the final time
        t1 = t0 + nt*deltat

        # Get the magnetic field pulse for the Runge-Kutta method
        nt, ts, Bs2, deltat = get_pulse_RK4_double_grid(Bt, t0, t1, deltat)

        # Loop over the nt time steps
        for it in range(nt):
            if it == 0:
                La, Lb, Lc = get_Labc(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, 0)
            else:
                La, Lb, Lc = get_Labc_reuse_La(L, L0, h, h_t0, Mz_op, Bs2, C, CST, X, Rhbar, n_nzC, indices_nzC, lambdaa, I0, T, dim, dims, dimds, it, Lc, Lb, La)
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

def get_outdirs(T, I0, lambdaa, Bt_params, RK4=False):
    """
    Get the output directory for the RI-separated vectorized density matrix and the magnetic moment.
    T: temperature in Kelvin.
    I0: prefactor for the phonon density of states.
    lambdaa: spin-phonon coupling constant in wavenumbers.
    Bt_params: parameters for the magnetic fields. See pulse.py for details.
    """
    if RK4:
        dir_rho = "rho_RK4"
        dir_mag = "magnetometry_RK4"
    else:
        dir_rho = "rho"
        dir_mag = "magnetometry"

    if Bt_params['Bt_type'] == 'linear':
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_linear_sweep_rate_{:.1e}/'.format(T, I0, lambdaa, Bt_params['sweep_rate'])
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_linear_sweep_rate_{:.1e}/'.format(T, I0, lambdaa, Bt_params['sweep_rate'])
    elif Bt_params['Bt_type'] == 'pwlinear':
        times = Bt_params['times']
        fields = Bt_params['fields']
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/'.format(T, I0, lambdaa)
        outdir += 'Bt_pwlinear_t{:.1e}ps-B{:.1f}T'.format(times[0], fields[0])
        for i in range(1, len(times)):
            outdir += '_t{:.1e}ps-B{:.1f}T'.format(times[i], fields[i])
    elif Bt_params['Bt_type'] == 'pwlinear_by_slope':
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_pwlinear_average_sweep_rate_{:.1e}'.format(T, I0, lambdaa, Bt_params['sweep_rate_ave'])
    elif Bt_params['Bt_type'] == 'sin':
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_sin_amplitude_{:.1f}_omega_{:.2e}'.format(T, I0, lambdaa, Bt_params['amplitude'], Bt_params['omega'])
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_sin_amplitude_{:.1f}_omega_{:.2e}'.format(T, I0, lambdaa, Bt_params['amplitude'], Bt_params['omega'])
    elif Bt_params['Bt_type'] == 'cs':
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_cs'.format(T, I0, lambdaa)
        outdir = './output/T_{:.1f}K_I0_{:.2e}_lambdaa_{:.2f}/Bt_cs'.format(T, I0, lambdaa)
    else:
        raise ValueError("Invalid Bt_type: {}".format(Bt_params['Bt_type']))

    outdir = outdir.replace('+', '')
    outdir_rho = outdir + dir_rho
    outdir_mag = outdir + dir_mag
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
            f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t0, Bt(t0), Mz, chimz))

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

                    # Save the magnetic moment and the magnetic susceptibility
                    f2.write("{:20.3f} {:20.6E} {:20.8E} {:20.8E}\n".format(t+half_deltat, Bt(t+half_deltat), Mz, chimz))
                # Save the RI-separated vectorized density matrix
                tag = "{:.3f}".format(t + half_deltat)
                dset = f1.create_dataset(tag, data=risvrho)
        print("rho is saved to {}".format(fname1))
        print("mmo is saved to {}".format(fname2))
    elif save_rho and (not save_mag):
        # nt should be a multiple of nt_rho
        nround_rho = int( (t1 - t0)//deltat // nt_rho )
        nt = nround_rho * nt_rho
        
        # Adjust the final time
        t1 = t0 + nt*deltat

        # Output directory
        create_outdir()
        outdir_rho, outdir_mag = get_outdirs(T, I0, lambdaa, Bt_params)

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
        print("rho is saved to {}".format(fname1))
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
        print("mmo is saved to {}".format(fname2))
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

