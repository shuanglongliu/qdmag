import copy
import numpy as np
from qdmag.core.Operators import Operator
from qdmag.core.common import get_kronecker_product
from qdmag.core.common import get_h_exchange, get_h_exchange_iso,  get_h_anisotropy, get_h_Zeeman, get_h_Zeeman_iso, eigen_handy
from qdmag.core.common import transform_O
from qdmag.core.constants import factor_ex

# Eigenvalues of the pool Hamiltonian (cm^-1) closer than this are taken as degenerate. The
# smallest splitting that is not a degeneracy is the Zeeman splitting of the 1e-4 T pool
# field, ~9e-5 cm^-1.
TOL_DEGENERATE = 1.0e-9


def energy_clusters(E, tol=TOL_DEGENERATE):
    """Lists of consecutive indices of the ascending values E that are degenerate within tol."""
    clusters = [[0]]
    for k in range(1, len(E)):
        if E[k] - E[k-1] > tol:
            clusters.append([k])
        else:
            clusters[-1].append(k)
    return clusters


def split_subspace(W, ops, tol=1.0e-8):
    """
    Split the subspace spanned by the orthonormal columns of W into one-dimensional pieces:
    diagonalize the projection of ops[0] on it, then that of ops[1] within each degenerate
    eigenspace, and so on. Returns the vectors in ascending order of the projected eigenvalues.
    Each vector is unique up to a phase.
    """
    if W.shape[1] == 1:
        return [W[:, 0]]
    if len(ops) == 0:
        raise RuntimeError("A degenerate subspace of dimension {} of the pool Hamiltonian "
                           "cannot be resolved.".format(W.shape[1]))
    K = W.conj().T @ ops[0] @ W
    lam, U = np.linalg.eigh((K + K.conj().T) / 2.0)
    W = W @ U
    vectors = []
    for group in energy_clusters(lam, tol):
        vectors += split_subspace(W[:, group], ops[1:], tol)
    return vectors


def fix_phase(w):
    """w with its largest component (the first one, within 1e-8, if several) real and positive."""
    a = np.abs(w)
    k = np.flatnonzero(a >= a.max() * (1.0 - 1.0e-8))[0]
    return w * (np.conj(w[k]) / a[k])


def canonical_pool_basis(h, eigen, spins):
    """
    Replace the eigenvectors of eigen = eigen_handy(h) with a basis that does not depend on the
    choices of LAPACK (phases, the basis within degenerate eigenspaces, and the mixing of
    states of different M that are split only by the small pool field), so that the
    effective basis, and through X the dynamics, are reproducible across BLAS thread counts
    and CPUs.

    h must conserve Sz and S^2, and h - c Sz must commute with S- (isotropic exchange plus a
    uniform Zeeman term along z, or Sz alone). Every pool state is then a member |S, M> of a
    multiplet, and the basis is built in the Condon-Shortley convention:
      1. Top states (M = S) for each M >= 0: the null space of S+ within the M sector of the
         product basis (Sz is diagonal there), in which h is diagonalized. Tops of equal
         energy are separated by the projections of S_12^2, S_123^2, ... in turn. Every
         vector of such a degenerate space is an eigenvector of h, so this only picks the
         basis.
      2. Each top gets its largest component real and positive.
      3. |S, M - 1> = S-|S, M> / sqrt((S + M)(S - M + 1)).
    The states are ordered by energy as by eigh, so state indices keep their meaning;
    within a degenerate cluster they are ordered by S, then by their top states.
    """
    Sz = np.real(np.diag(spins.Sv_tot[2]))
    # S- from the exact ladder matrices (Sx - i Sy has rounding errors in its elements)
    Sm = spins.zero
    for i in range(spins.nS):
        ops = copy.deepcopy(spins.IDs)
        ops[i] = Operator.Jminus(spins.Ss[i]).O
        Sm = Sm + get_kronecker_product(ops, spins.nS)
    Sp = Sm.conj().T
    chain = []
    for k in range(2, spins.nS):
        Sk = [sum(spins.global_spins[i][j] for i in range(k)) for j in range(3)]
        chain.append(Sk[0] @ Sk[0] + Sk[1] @ Sk[1] + Sk[2] @ Sk[2])
    n = len(Sz)
    sector = {}
    for i, m in enumerate(Sz):
        sector.setdefault(round(2*m), []).append(i)

    # 1-2. Top states, from the largest M down
    tops = []
    for twoM in sorted((k for k in sector if k >= 0), reverse=True):
        P = sector[twoM]
        Q = sector.get(twoM + 2, [])
        n_top = len(P) - len(Q)
        if n_top == 0:
            continue
        # Null space of S+ from sector M to M + 1: S-S+ = S^2 - Sz^2 - Sz has eigenvalues
        # S(S+1) - M(M+1) >= 2(M+1) on the rest of the sector, so it is well separated.
        A = Sp[np.ix_(Q, P)]
        _, U = np.linalg.eigh(A.conj().T @ A)
        T = np.zeros((n, n_top), dtype=np.complex128)
        T[P, :] = U[:, :n_top]
        K = T.conj().T @ h @ T
        lam, U = np.linalg.eigh((K + K.conj().T) / 2.0)
        T = T @ U
        for group in energy_clusters(lam):
            for w in split_subspace(T[:, group], [c for c in chain]):
                tops.append((0.5*twoM, fix_phase(w)))

    # 3. Multiplets
    states = []
    for rank, (S, w) in enumerate(tops):
        for step in range(int(round(2*S)) + 1):
            if step > 0:
                # ||S-|S, M>||^2 = (S + M)(S - M + 1), as in the elements of Jminus
                M = S - step + 1
                # (real and imaginary parts divided separately: NumPy's complex division
                # is not exact for a real divisor)
                w = ((Sm @ w).view(np.float64) / np.sqrt((S + M)*(S - M + 1))).view(np.complex128)
            states.append((np.real(np.vdot(w, h @ w)), S, rank, w))
    if len(states) != n:
        raise RuntimeError("The multiplets hold {} states, not {}.".format(len(states), n))

    # Order by energy; ties (degenerate clusters) by S, then by the top state
    states.sort(key=lambda x: x[0])
    E = np.array([x[0] for x in states])
    order = []
    for c in energy_clusters(E):
        order += sorted(c, key=lambda k: (states[k][1], states[k][2]))
    E = E[order]
    V = np.stack([states[k][3] for k in order], axis=1)
    scale = max(1.0, np.abs(E).max())
    if np.abs(h @ V - V * E).max() > 1.0e-8*scale or np.abs(E - eigen.eigenvalues).max() > 1.0e-8*scale:
        raise RuntimeError("The pool Hamiltonian does not conserve the total spin; "
                           "the Condon-Shortley pool basis cannot be built.")
    eigen.eigenvalues = E
    eigen.eigenvalues_offset = E - E[0]
    eigen.eigenvectors = V
    return eigen


class effective_basis:
    """
    A class to handle the effective basis for spin dynamics calculations.
    It constructs the effective Hamiltonian and operators in the reduced Hilbert space.
    """
    global factor_ex

    def __init__(self, spins, exchange, anisotropy, dynamics, states):
        """
        Initialize the effective basis.
        """
        self.spins = spins
        self.exchange = exchange
        self.anisotropy = anisotropy
        self.dynamics = dynamics
        self.get_pool_and_pick_states(states)
        self.check_degenerate_selection()
        self.construct_X_eff()
        self.get_effective_Sv()
        self.get_effective_Mv()
        self.get_effective_h0()

    def get_pool_and_pick_states(self, states):
        """
        Construct the pool of candidate states for the effective basis in the S representation, and
          pick the effective basis states from the pool.
        The pool is constructed as follows:
          1. For a single spin, the full Hilbert space is used.
             The pool is constructed based on the Zeeman interaction with a small field.
          2. For a multi-spin system, a subspace is used. 
             The pool is constructed based on the isotropic exchange interaction and a small Zeeman field.
             The selected states are the lowest energy states per Sz value.
             Assumption: the system is dominated by the isotropic exchange interaction.
        Returns:
          dim: the dimension of the effective Hilbert space.
          indices: the indices of the selected states in the pool.
          eigen_S: a pool of candidate states for the effective basis in the S representation.
            It contains the eigenvalues and eigenvectors
            The eigenvalues are sorted automatically by np.linalg.eigh.
          Sz_pool: the Sz operator on the pool basis (in the S representation).
        """
        if self.spins.nS == 1:
            # For a single spin, use the full Hilbert space.
            self.dim = self.spins.dim
            self.states = list(range(self.dim))
            # Two choices of eigen_S:
            # 1. Use the eigenstates of the isotropic Zeeman term
            # h_zee_iso = get_h_Zeeman_iso(self.spins, [0.,0.,1e-4], 'cartesian')
            # self.eigen_S = eigen_handy(h_zee_iso)
            # 2. Use the eigenstates of Sz. (Same as the first choice.)
            self.eigen_S = canonical_pool_basis(self.spins.Sv_tot[2], eigen_handy(self.spins.Sv_tot[2]), self.spins)
            self.Sz_pool = transform_O(self.spins.Sv_tot[2], self.eigen_S)
            self.Sz_pool = np.real(self.Sz_pool)
            # It dodes not solve the divergence problem to use the eigenstates of Sn = en_x*Sx + en_y*Sy + en_z*Sz as the basis for time evolution.
        elif isinstance(states, str) and (states == "all" or states == "full"):
            # For a multi-spin system, use the full Hilbert space.
            self.dim = self.spins.dim
            self.states = list(range(self.dim))
            # 1. Use the eigenstates of the isotropic Zeeman term
            # h_zee_iso = get_h_Zeeman_iso(self.spins, [0.,0.,1e-4], 'cartesian')
            # self.eigen_S = eigen_handy(h_zee_iso)
            # 2. Use the eigenstates of Sz. (Same as the first choice.)
            # self.eigen_S = eigen_handy(self.spins.Sv_tot[2])
            # 3. Use the eigenstates of h_ex_iso + h_zee_iso.
            h_ex_iso = get_h_exchange_iso(self.spins, self.exchange, factor_ex)
            h_zee_iso = get_h_Zeeman_iso(self.spins, [0.0, 0.0, 1e-4], 'cartesian')
            h = h_ex_iso + h_zee_iso
            self.eigen_S = canonical_pool_basis(h, eigen_handy(h), self.spins)
            self.Sz_pool = transform_O(self.spins.Sv_tot[2], self.eigen_S)
            self.Sz_pool = np.real(self.Sz_pool)
        else:
            # For a multi-spin system, use a subspace based on the isotropic exchange interaction and a small Zeeman field.
            h_ex_iso = get_h_exchange_iso(self.spins, self.exchange, factor_ex)
            # Check is h_ex_iso is zero
            if np.all(np.isclose(h_ex_iso, 0.0)):
                print("Warning: The isotropic exchange interaction is zero. The default effective basis may not be suitable.")
            h_zee_iso = get_h_Zeeman_iso(self.spins, [0.0, 0.0, 1e-4], 'cartesian')
            h = h_ex_iso + h_zee_iso
            # h = h_ex_iso + h_zee_iso + 1e-6 * self.spins.S2_tot  # Add a small S^2 term to lift the degeneracy of S.
            self.eigen_S = canonical_pool_basis(h, eigen_handy(h), self.spins)
            if len(states) == 0 or (isinstance(states, str) and (states == "default")):
                print("Using the default effective basis.")
                # If no indices are provided, use the minimal basis.
                self.states = []
                self.dim = int(2*self.spins.Smax + 1)
                self.Sz_pool = transform_O(self.spins.Sv_tot[2], self.eigen_S)
                self.Sz_pool = np.real(self.Sz_pool)
                for i in range(self.dim):
                    Sz_target = -self.spins.Smax + i
                    for j in range(self.spins.dim):
                        if abs(self.Sz_pool[j, j] - Sz_target) < 1e-6:
                            self.states.append(j)
                            break
            elif isinstance(states, list) and len(states) > 0:
                # The indices are specified in the the input file.
                self.states = states
                self.dim = len(self.states)
                self.Sz_pool = transform_O(self.spins.Sv_tot[2], self.eigen_S)
                self.Sz_pool = np.real(self.Sz_pool)
            else:
                print("Error: Invalid states input. It should be a list of indices, an empty list (same as 'default'), 'default', or 'all'/'full'. Stopping ...")
                exit(1)

    def check_degenerate_selection(self):
        """
        Warn if the selected states contain only part of a degenerate eigenspace of the pool
        Hamiltonian: which part is selected is then set by the convention of
        canonical_pool_basis, not by the physics.
        """
        selected = set(self.states)
        for c in energy_clusters(self.eigen_S.eigenvalues):
            n = len(selected.intersection(c))
            if 0 < n < len(c):
                print("Warning: the selected states contain {} of the {} degenerate pool states {}.".format(n, len(c), c))

    def construct_X_eff(self):
        """
        Construct the operator in the spin space, which couples to phonons.
        X_{ij} = 1 if | M_{iz} - M_{jz} | = 1. Otherwise, X_{ij} = 0. 
        """
        self.X_eff = np.zeros((self.dim, self.dim), dtype=np.float64)
        for i in range(self.dim):
            for j in range(self.dim):
                ii = self.states[i]
                jj = self.states[j]
                # When a B field of 1e-9 T is added to the isotropic qunatum Heisenberg Hamiltonian, 
                # the deviation in Sz from half integers is within 1e-8 mu_B.
                if abs(abs(self.Sz_pool[ii, ii] - self.Sz_pool[jj, jj]) - 1.0) < 1e-6:
                    self.X_eff[i, j] = 1.0

    def get_effective_O(self, O_full):
        """
        Get the effective operator in the effective Hilbert space for 
        the full operator O_full defined on the initial common basis.
        """
        # Transform the operator O_full onto the pool basis.
        O_pool = transform_O(O_full, self.eigen_S)
        # np.complex64 is not enough to capture the energy differences between the nearly degenerate states (which are indeed degenerate without perturbation).
        O_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim):
            for j in range(self.dim):
                ii = self.states[i]
                jj = self.states[j]
                O_eff[i, j] = O_pool[ii, jj]
        return O_eff

    def get_effective_Sv(self):
        """
        Get the effective spin operators in the effective Hilbert space.
        """
        # Transform the spin operators onto the pool basis.
        Sx_pool = transform_O(self.spins.Sv_tot[0], self.eigen_S)
        Sy_pool = transform_O(self.spins.Sv_tot[1], self.eigen_S)
        Sz_pool = transform_O(self.spins.Sv_tot[2], self.eigen_S)
        # Effective spin operators in the effective Hilbert space.
        # np.complex64 is not enough to capture the energy differences between the nearly degenerate states (which are indeed degenerate without perturbation).
        self.Sx_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.Sy_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.Sz_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim):
            for j in range(self.dim):
                ii = self.states[i]
                jj = self.states[j]
                self.Sx_eff[i, j] = Sx_pool[ii, jj]
                self.Sy_eff[i, j] = Sy_pool[ii, jj]
                self.Sz_eff[i, j] = Sz_pool[ii, jj]
        self.Sv_eff = [self.Sx_eff, self.Sy_eff, self.Sz_eff]

    def get_effective_Mv(self):
        """
        Get the effective magnetization operators in the effective Hilbert space.
        """
        # Transform the magnetization operators onto the pool basis.
        Mx_pool = transform_O(self.spins.Mv_tot[0], self.eigen_S)
        My_pool = transform_O(self.spins.Mv_tot[1], self.eigen_S)
        Mz_pool = transform_O(self.spins.Mv_tot[2], self.eigen_S)
        # Effective magnetization operators in the effective Hilbert space.
        # np.complex64 is not enough to capture the energy differences between the nearly degenerate states (which are indeed degenerate without perturbation).
        self.Mx_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.My_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.Mz_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim):
            for j in range(self.dim):
                ii = self.states[i]
                jj = self.states[j]
                self.Mx_eff[i, j] = Mx_pool[ii, jj]
                self.My_eff[i, j] = My_pool[ii, jj]
                self.Mz_eff[i, j] = Mz_pool[ii, jj]
        self.Mv_eff = [self.Mx_eff, self.My_eff, self.Mz_eff]

    def get_effective_h0(self):
        """
        Get the effective Hamiltonian under zero magnetic field.
        If there is only one spin, the anisotropy must be provided.
        If there are multiple spins, the exchange interaction must be provided, and the anisotropy is considered only if it is provided.
        """
        if self.spins.nS == 1:
            if len(self.anisotropy) == 0:
                print("Error: No anisotropy provided for a single spin system. Stopping ...")
                exit(1)
            else:
                h = get_h_anisotropy(self.spins, self.anisotropy)
        else:
            if len(self.exchange) == 0:
                print("Error: No exchange interaction provided for a multi-spin system. Stopping ...")
                exit(1)
            else:
                h_ex = get_h_exchange(self.spins, self.exchange, factor_ex)
            if len(self.anisotropy) == 0:
                h = h_ex
            else:
                h_ani = get_h_anisotropy(self.spins, self.anisotropy)
                h = h_ex + h_ani
        # Transform the Hamiltonian onto the pool basis.
        h_pool = transform_O(h, self.eigen_S)
        # Get the effective Hamiltonian in the effective Hilbert space.
        self.h0_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim):
            for j in range(self.dim):
                ii = self.states[i]
                jj = self.states[j]
                self.h0_eff[i, j] = h_pool[ii, jj]
        # Hermitianize the Hamitonian if it is not Hermitian.
        tolerance = 1e-12
        deviation = np.abs(np.conjugate(np.transpose(self.h0_eff)) - self.h0_eff)
        if np.isclose(deviation, tolerance).all():
            pass
        else:
            print("Warning: The Hamiltonian is not Hermitian. Hermitianizing ...")
            self.h0_eff = (self.h0_eff + np.conjugate(np.transpose(self.h0_eff))) / 2.0
            deviation = np.abs(np.conjugate(np.transpose(self.h0_eff)) - self.h0_eff)
            if np.isclose(deviation, tolerance).all():
                print("The Hamiltonian is now Hermitian.")
            else:
                print("Error: The Hamiltonian is still not Hermitian after Hermitianization. Stopping ...")
                exit(1)

