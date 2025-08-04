import numpy as np
from spin_dynamics.core.common import get_h_exchange, get_h_exchange_iso,  get_h_anisotropy, get_h_Zeeman, eigen_handy
from spin_dynamics.core.common import transform_O
from spin_dynamics.core.constants import factor_ex

class effective_basis:
    """
    A class to handle the effective basis for spin dynamics calculations.
    It constructs the effective Hamiltonian and operators in the reduced Hilbert space.
    """
    global factor_ex

    def __init__(self, spins, exchange, anisotropy, dynamics, states):
        """
        """
        self.spins = spins
        self.exchange = exchange
        self.anisotropy = anisotropy
        self.dynamics = dynamics
        self.get_pool_and_pick_states(states)
        self.construct_X_eff()
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
          eigen_pool: a pool of candidate states for the effective basis in the S representation.
            It contains the eigenvalues and eigenvectors
            The eigenvalues are sorted automatically by np.linalg.eigh.
          Sz_pool: the Sz operator on the pool basis (in the S representation).
        """
        if self.spins.nS == 1:
            # For a single spin, use the full Hilbert space.
            self.dim = self.spins.dim
            h_zee = get_h_Zeeman(self.spins, [0.,0.,1e-4], 'cartesian')
            self.eigen_pool = eigen_handy(h_zee)
            if len(states) == 0:
                # If no states are provided, use the full Hilbert space.
                self.states = list(range(self.dim))
            else:
                # The indices are specified in the input file.
                if len(states) == self.dim:
                    self.states = states
                else:
                    print("Error: The number of states provided does not match the dimension of the Hilbert space. Stopping ...")
                    exit(1)
            self.Sz_pool = transform_O(self.spins.Sv_tot[2], self.eigen_pool)
            self.Sz_pool = np.real(self.Sz_pool)
        else:
            # For a multi-spin system, use a subspace based on the isotropic exchange interaction and a small Zeeman field.
            h_ex_iso = get_h_exchange_iso(self.spins, self.exchange, factor_ex)
            h_zee = get_h_Zeeman(self.spins, [0.0, 0.0, 1e-4], 'cartesian')
            h = h_ex_iso + h_zee
            self.eigen_pool = eigen_handy(h)
            if len(states) == 0:
                # If no indices are provided, use the minimal basis.
                self.states = []
                self.dim = int(2*self.spins.Smax + 1)
                self.Sz_pool = transform_O(self.spins.Sv_tot[2], self.eigen_pool)
                self.Sz_pool = np.real(self.Sz_pool)
                for i in range(self.dim):
                    Sz_target = -self.spins.Smax + i
                    for j in range(self.spins.dim):
                        if abs(self.Sz_pool[j, j] - Sz_target) < 1e-6:
                            self.states.append(j)
                            break
            else:
                # The indices are specified in the the input file.
                self.states = states
                self.dim = len(self.states)
                self.Sz_pool = transform_O(self.spins.Sv_tot[2], self.eigen_pool)
                self.Sz_pool = np.real(self.Sz_pool)

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
        O_pool = transform_O(O_full, self.eigen_pool)
        # np.complex64 is not enough to capture the energy differences between the nearly degenerate states (which are indeed degenerate without perturbation).
        O_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim):
            for j in range(self.dim):
                ii = self.states[i]
                jj = self.states[j]
                O_eff[i, j] = O_pool[ii, jj]
        return O_eff

    def get_effective_Mv(self):
        """
        Get the effective magnetization operators in the effective Hilbert space.
        """
        # Transform the magnetization operators onto the pool basis.
        Mx_pool = transform_O(self.spins.Mv_tot[0], self.eigen_pool)
        My_pool = transform_O(self.spins.Mv_tot[1], self.eigen_pool)
        Mz_pool = transform_O(self.spins.Mv_tot[2], self.eigen_pool)
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
        h_pool = transform_O(h, self.eigen_pool)
        # Get the effective Hamiltonian in the effective Hilbert space.
        self.h0_eff = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim):
            for j in range(self.dim):
                ii = self.states[i]
                jj = self.states[j]
                self.h0_eff[i, j] = h_pool[ii, jj]

