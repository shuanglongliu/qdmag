import os
import subprocess
import copy
import math
import numpy as np
import yaml
import ray
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import N_A, mu_0
from spin_dynamics.core.constants import meV2wavenumber, Tesla2wavenumber, Kelvin2meV, Kelvin2wavenumber
from spin_dynamics.core.constants import x_mu_B, mu_B_per_Tesla_2_cm3_per_mol_phi
from spin_dynamics.external.Operators import Operator
from spin_dynamics.external.StevensOperators import StevensOpA
from spin_dynamics import __file__ as root_dir



# ========================
# Classes
# ========================

class one_local_spin:
    """ Value of spin, dimension, spin operators, the identify operators, and expectation of S^2. """
    def __init__(self, S=0.5):

        self.S = S

        self.dim = int( 2*S + 1 ) 
        
        self.Sx = Operator.Jx(S).O
        self.Sy = Operator.Jy(S).O
        self.Sz = Operator.Jz(S).O
        
        self.Sv = [self.Sx, self.Sy, self.Sz] 
        
        self.ID = np.eye(self.dim, dtype=complex)
        
        self.ss = S*(S + 1)

        return

class many_spins:
    """ Spin operators relevant to several spins """
    def __init__(self, Ss, nS, gfactor, dipole, positions):
        self.Ss = Ss
        self.nS = nS
        self.gfactor = gfactor
        self.dipole = dipole
        self.positions = positions
        self.get_local_spins()
        self.dim = np.prod([self.local_spins[i].dim for i in range(self.nS)])
        self.zero = np.zeros((self.dim, self.dim),dtype=complex)
        ## IDs: unit operators in the local spaces.
        self.IDs = [self.local_spins[i].ID for i in range(nS)]
        ## ID: unit operator in the whole space.
        self.ID = get_kronecker_product(self.IDs, self.nS)
        ## get_global_spins depends on self.ID.
        self.get_global_spins()
        self.get_total_spin()
        self.get_global_magmoms()
        self.get_total_magmom()
        if self.nS > 2:
            self.get_dipole()
        else:
            self.Pv = [self.zero, self.zero, self.zero]
        return
        
    def get_local_spins(self):
        ## Get a list of one_local_spin objects.
        ## All local operators are capsulated in the spins variable except IDs. 
        self.local_spins = []
        for i in range(self.nS):
            self.local_spins.append(one_local_spin(S=self.Ss[i]))
        return
    
    def get_global_spins(self):
        ## Local spin operators in the whole Hilbert space.
        
        ## List of vectors (as lists) of spin operators. 
        self.global_spins = []
        for i in range(self.nS):
            self.global_spins.append([])
            for j in range(3):
                ops = copy.deepcopy(self.IDs)
                ops[i] = self.local_spins[i].Sv[j]
                self.global_spins[i].append( get_kronecker_product(ops, self.nS) )
        return
    
    def get_total_spin(self):
        
        self.Sv_tot = [self.zero for i in range(3)]
        for i in range(3):
            for j in range(self.nS):
                self.Sv_tot[i] = self.Sv_tot[i] + self.global_spins[j][i]

        self.S2_tot = np.matmul(self.Sv_tot[0], self.Sv_tot[0]) + \
                      np.matmul(self.Sv_tot[1], self.Sv_tot[1]) + \
                      np.matmul(self.Sv_tot[2], self.Sv_tot[2])

        return

    def get_global_magmoms(self):
        ## Local magnetic moment operators in the whole Hilbert space.

        ## List of vectors (as lists) of magnetic moment operators
        self.global_magmoms = []
        for i in range(self.nS):
            self.global_magmoms.append([])
            i_site = self.gfactor[i]['site']
            gs = self.gfactor[i]['gs']
            gs = np.array(gs).reshape((3,3))
            A = self.gfactor[i]['reference_frame']
            A = np.array(A).reshape((3,3))
            gs = np.matmul(np.transpose(A), np.matmul(gs, A))
            for j in range(3):
                ops = copy.deepcopy(self.IDs)
                ops[i_site-1] = ops[i_site-1] * 0.0
                for k in range(3):
                    ops[i_site-1] = ops[i_site-1] - gs[j, k]*self.local_spins[i_site-1].Sv[k]
                self.global_magmoms[i].append( get_kronecker_product(ops, self.nS) )

    def get_total_magmom(self):
        
        self.Mv_tot = [self.zero for i in range(3)]
        for i in range(3):
            for j in range(self.nS):
                self.Mv_tot[i] = self.Mv_tot[i] + self.global_magmoms[j][i]

    def get_eSS(self, eij, Si, Sj, i, j, ii, jj, kk):
        ## One term of the cross product eij \cross Si \cross Sj    
        ops = copy.deepcopy(self.IDs)
        ops[i-1] = Si[jj]
        ops[j-1] = Sj[kk]
        return eij[ii]*get_kronecker_product(ops, self.nS)

    def get_dipole_one_pair(self, pair):
        i, j = pair['pair']
        Si = self.local_spins[i-1].Sv
        Sj = self.local_spins[j-1].Sv
        alpha = pair['alpha']
        pi = pair['pi']
        A = pair['reference_frame']
        A = np.array(A).reshape((3,3))
        Siprime = get_Sprime(A, Si)
        Sjprime = get_Sprime(A, Sj)
        
        ## The dot product term
        op = self.zero
        for ii in range(3):
            ops = copy.deepcopy(self.IDs)
            ops[i-1] = Siprime[ii]
            ops[j-1] = Sjprime[ii]
            op = op + get_kronecker_product(ops, self.nS)
        Px = pi[0]*op
        Py = pi[1]*op
        Pz = pi[2]*op
        
        ## The cross product term
    
        posi = np.array(self.positions[i-1])
        posj = np.array(self.positions[j-1])
        eij = posj - posi
        eij = eij / np.linalg.norm(eij)
        
        op = self.zero
        ii = 1; jj = 0; kk = 1; op = op + self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        ii = 1; jj = 1; kk = 0; op = op - self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        ii = 2; jj = 2; kk = 0; op = op - self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        ii = 2; jj = 0; kk = 2; op = op + self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        Px = Px + alpha*op                      
                                                
        op = self.zero                         
        ii = 2; jj = 1; kk = 2; op = op + self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        ii = 2; jj = 2; kk = 1; op = op - self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        ii = 0; jj = 0; kk = 1; op = op - self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        ii = 0; jj = 1; kk = 0; op = op + self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        Py = Py + alpha*op                      
                                                
        op = self.zero                         
        ii = 0; jj = 2; kk = 0; op = op + self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        ii = 0; jj = 0; kk = 2; op = op - self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        ii = 1; jj = 1; kk = 2; op = op - self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        ii = 1; jj = 2; kk = 1; op = op + self.get_eSS(eij, Siprime, Sjprime, i, j, ii, jj, kk)
        Pz = Pz + alpha*op
    
        return (Px, Py, Pz)
    
    def get_dipole(self):
        """
        \vec{P} = \alpha sum_{ij} \vec{e}_{ij} \cross (\vec{S_i} \cross \vec{S_j}) + \
                  \vec{\pi} sum_{ij} \vec{\pi} (\vec{S_i} \cdot \vec{S_j})
        """
        n_pair = len(self.dipole)
        Px = self.zero
        Py = self.zero
        Pz = self.zero
        for i_pair in range(n_pair):
            op1, op2, op3 = self.get_dipole_one_pair(self.dipole[i_pair])
            Px = Px + op1
            Py = Py + op2
            Pz = Pz + op3
        self.Pv = [Px, Py, Pz]

class eigen_spin_hamiltonian:
    def __init__(self, hamiltonian):
        ## Unit for eigenvalues: wavenumber
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(hamiltonian) 
        self.eigenvalues = np.real(self.eigenvalues)

        self.dim = self.eigenvalues.shape[0]

        # The eigenvalues by np.linalg.eigh is already in the assending order.
        self.indices = np.arange(self.dim) # np.argsort(self.eigenvalues)

        self.eigenvalues_offset = self.eigenvalues - self.eigenvalues[self.indices[0]]

        return
    

    
# ========================
# Functions
# ========================

## =================================================================
## Functions for common mathematics.
## =================================================================

def kronecker_delta(i, j):
    k = 0
    if i == j:
        k = 1
    return k

def get_kronecker_product(ops,nop):
    if nop == 1:
        prod = ops[-1]
    else:
        prod = np.kron(ops[-2],ops[-1])
        if nop >=3:
            for i in range(3,nop+1):
                ii = -i
                prod = np.kron(ops[ii], prod)
    return prod

def sample_temperature(sample):
    n = len(sample)
    m = int(n/3)
    Ts_all = []
    for i in range(m):
        nx = int((sample[3*i+1]-sample[3*i])/sample[3*i+2]) + 1
        Ts = list( np.linspace(sample[3*i], sample[3*i+1], nx, endpoint=True) )
        if i == 0:
            Ts_all = Ts_all + Ts
        elif Ts_all[-1] == Ts[0]:
            Ts_all = Ts_all + Ts[1:]
        else:
            Ts_all = Ts_all + Ts
    return Ts_all

def sph2cart_deg(sph):
    ## Convert cartesian coordinate to spherical coordinate
    sph[1:3] = np.deg2rad(sph[1:3])
    z = sph[0] * math.cos( sph[1] )
    x = sph[0] * math.sin( sph[1] ) * math.cos( sph[2] )
    y = sph[0] * math.sin( sph[1] ) * math.sin( sph[2] )
    return np.array( [x, y, z] )

def check_hermitian(op):
    maxdiff = np.max( np.absolute( np.conjugate(np.transpose(op)) - op ) )
    if maxdiff < 1.e-9:
        print("It is Hermitian.")
    else:
        print("It is not Hermitian.")
    return

def check_unitary(op):
    op_dagger = np.conjugate(np.transpose(op))
    prod = np.matmul(op_dagger, op)
    maxdiff = np.max( np.absolute( prod - np.eye(prod.shape[0]) ) )
    if maxdiff < 1.e-9:
        print("It is unitary.")
    else:
        print("It is not unitary.")
    return

def get_commutation(O1, O2):
    return np.matmul(O1, O2) - np.matmul(O2, O1) 

def check_commutation(O1, O2):
    x = np.matmul(O1, O2) 
    y = np.matmul(O2, O1) 
    maxdiff = np.max(np.absolute(x - y))
    if maxdiff < 1.e-6:
        print("Yes, they commute.")
    else:
        print("No, they don't commute.\n" + " maxdiff = {:15.10f}.".format(maxdiff))
    return

def check_zero(x, epsilon=1e-9):
    maxentry = np.max(np.absolute(x))
    if maxentry < epsilon:
        print("Yes, it is zero.")
    else:
        print("No, it is not zero." + " maxdiff = {:15.10f}.".format(maxentry))
    return

def check_eigen(O, eigen):
    """
    Check if the given vectors are the eigenvectors of an operator.
    If O |i> = sum_j c_{ij} |j>, then c_{ij} = O_{ji}.
    If |i> are the eigenvectors, then O_{ji} is diagonal.
    """

    O1 = transform_O(O, eigen)
    O1_abs = np.abs(O1)

    np.fill_diagonal(O1_abs, 0)
    is_eigen = np.all( O1_abs < 1e-8 )

    print( "Are they eigenvectors? {:s}.".format(str(is_eigen)) )

def check_real(O):
    is_real = np.all( np.imag(O) < 1e-12 )
    if is_real:
        print("It is real.")
    else:
        print("It is complex.")
    return

def get_Sprime(A, S):
    # Get spin operators in the local reference frame. 
    # e_x^prime = A[0, :]
    # e_y^prime = A[1, :]
    # e_z^prime = A[2, :]

    Sprime = []
    
    for i in range(3):
        Sprime.append(A[i, 0]*S[0])
        Sprime[i] = Sprime[i] + A[i, 1]*S[1]
        Sprime[i] = Sprime[i] + A[i, 2]*S[2]
    
    return Sprime

def convert_cmatrix_to_rmatrix(M, tag):
    """
    Return the real part of M if the imaginary part is negligibly small,
    Stop if the imaginary part is not negligible.
    """

    zero_imag = np.max( np.absolute( np.imag(M) ) ) < 1.e-12

    if zero_imag:
        print(tag, "is a real matrix. Using a matrix of real numbers to represent it.\n")
    else:
        print(tag, "is a complex matrix. Please make sure that the matrix elements are real. Stopping ...\n")
        exit()

    return np.real(M)

## =================================================================
## Functions for input and output.
## =================================================================

def read_input():

    with open(root_dir + "core/input.yaml", "r") as f:
        data = yaml.safe_load(f)
    
    Ss = data['spins']
    nS = len(Ss)
    positions = data['positions']
    exchange = data['exchange']
    anisotropy = data['anisotropy']
    gfactor = data['gfactor']
    dipole = data['dipole']
    ext_field = data['ext_field']
    BET_Bgrid = data['BET_Bgrid']
    BET_Egrid = data['BET_Egrid']
    BET_BEgrid = data['BET_BEgrid']
    BET_Tgrid = data['BET_Tgrid']
    dynamics = data['dynamics']

    return (Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, dynamics)

def save_eigenvalues(eigen, offset=True):
    
    if offset:
        eigenvalues = eigen.eigenvalues_offset
    else:
        eigenvalues = eigen.eigenvalues

    if not os.path.exists(root_dir + "output"):
        subprocess.run(["mkdir", root_dir + "output"])

    with open(root_dir + "output/eigenvalues.dat", "w") as f:
        for i in range(eigen.dim):
            f.write("{:16.12f}\n".format(eigenvalues[i]))

    return

def save_eigenvectors(eigen):

    if not os.path.exists(root_dir + "output"):
        subprocess.run(["mkdir", root_dir + "output"])

    with open(root_dir + "output/eigenvectors.dat", "w") as f:
        for i in range(eigen.dim):
            state = eigen.eigenvectors[:, i]
            f.write((eigen.dim*" {:16.12f}" + "\n").format(*state))
    
    return

def save_operator(op, base_name):

    Dim = op.shape[0]

    o_real = np.real(op)
    o_imag = np.imag(op)

    if not os.path.exists(root_dir + "output"):
        subprocess.run(["mkdir", root_dir + "output"])

    with open(root_dir + "output/{:s}.real".format(base_name), "w") as f:
        for i in range(Dim):
            for j in range(Dim):
                f.write("{:8.3f}\n".format( o_real[i, j] ))

    with open(root_dir + "output/{:s}.imag".format(base_name), "w") as f:
        for i in range(Dim):
            for j in range(Dim):
                f.write("{:8.3f}\n".format( o_imag[i, j] ))
    return

def save_spins(spins, eigen):

    """
    Calculate and save expectation of spins for all eigenvectors.
    """

    if not os.path.exists(root_dir + "output"):
        subprocess.run(["mkdir", root_dir + "output"])

    with open(root_dir + "output/spins.dat", "w") as f:
        f.write("## s1x, s1y, s1z, s2x, s2y, s2z, ..., snx, sny, snz, sx_tot, sy_tot, sz_tot, s2_tot, s_tot\n")

        for i in range(spins.dim):
            E_local_spins, E_Sv_tot, E_S2_tot, E_S_tot = get_expectation_of_spins(spins, eigen.eigenvectors[:, eigen.indices[i]])
            for i_site in range(spins.nS):
                for alpha in range(3):
                    f.write("{:8.3f} ".format(E_local_spins[i_site][alpha]))
            for alpha in range(3):
                f.write("{:12.8f} ".format(E_Sv_tot[alpha]))
            f.write("{:8.3f} ".format(E_S2_tot))
            f.write("{:8.3f}\n".format(E_S_tot))

## =================================================================
## Functions for analyzing results.
## =================================================================

def get_expectation_of_spins(spins, state):
    
    E_local_spins = []
    for i in range(spins.nS):
        E_local_spins.append([])
        for j in range(3):
            E_local_spins[i].append(np.real(np.dot(np.conjugate(state), np.matmul(spins.global_spins[i][j], state))))
    
    E_Sv_tot = []
    for i in range(3):
        E_Sv_tot.append( np.real(np.dot(np.conjugate(state), np.matmul(spins.Sv_tot[i], state))) )

    E_S2_tot = np.real(np.dot(np.conjugate(state), np.matmul(spins.S2_tot, state)))
    E_S_tot = (-1+np.sqrt(4*E_S2_tot+1))/2

    return (E_local_spins, E_Sv_tot, E_S2_tot, E_S_tot)

def get_total_spin_for_all_eigenstates(spins, eigen):

    results = []

    for i in range(eigen.dim):
        state = eigen.eigenvectors[:, eigen.indices[i]]

        E_S2_tot = np.real(np.dot(np.conjugate(state), np.matmul(spins.S2_tot, state)))
        E_S_tot = (-1+np.sqrt(4*E_S2_tot+1))/2

        E_Sv_tot = []
        for i in range(3):
            E_Sv_tot.append( np.real(np.dot(np.conjugate(state), np.matmul(spins.Sv_tot[i], state))) )
        
        results.append([E_S_tot] + E_Sv_tot)

    return results

## =================================================================
## Functions for magnetic exchange interaction.
## =================================================================

def get_h_exchange_one_pair(spins, spin_pair, factor):
    i, j = spin_pair['pair']
    Si = spins.local_spins[i-1].Sv
    Sj = spins.local_spins[j-1].Sv
    Jprime = spin_pair['coupling_matrix']
    Jprime = np.array(Jprime).reshape((3,3))
    A = spin_pair['reference_frame']
    A = np.array(A).reshape((3,3))
    Siprime = get_Sprime(A, Si)
    Sjprime = get_Sprime(A, Sj)
    h_ex = spins.zero
    for ii in range(3):
        for jj in range(3):
            ## Jprime[ii,jj] * Siprime[ii] * Sjprime[jj]
            ops = copy.deepcopy(spins.IDs)
            ops[i-1] = Siprime[ii]
            ops[j-1] = Sjprime[jj]
            h_ex = h_ex + Jprime[ii,jj] * get_kronecker_product(ops, spins.nS)
    return factor*h_ex

def get_h_exchange(spins, exchange, factor):
    n_pair = len(exchange)
    h_ex = spins.zero
    for i_pair in range(n_pair):
        h_ex = h_ex + get_h_exchange_one_pair(spins, exchange[i_pair], factor)
    return h_ex

## =================================================================
## Functions for single-ion magnetic anisotropy.
## =================================================================

def get_h_anisotropy_one_site(spins, site):
    i = site['site']
    S = spins.Ss[i-1]
    ks = site['ks']
    qs = site['qs']
    Bkqs = site['Bkqs']
    A = site['reference_frame']
    A = np.array(A).reshape((3,3))
    
    nB = len(Bkqs)
    h_ani = spins.zero
    ops = copy.deepcopy(spins.IDs)
    for ii in range(nB):
        ops[i-1] = StevensOpA(S, ks[ii], qs[ii], A)
        h_ani = h_ani + Bkqs[ii]*get_kronecker_product(ops, spins.nS)

    return h_ani

def get_h_anisotropy(spins, anisotropy):
    h_ani = spins.zero
    ## n_site = number of local spins
    for i_site in range(spins.nS):
        h_ani = h_ani + get_h_anisotropy_one_site(spins, anisotropy[i_site])
    return h_ani
    
## =================================================================
## Functions for the Zeeman term.
## =================================================================

def get_h_Zeeman(spins, Bv, coord):
    """ 
    Zeeman term H_Zee = - \vec{\mu} \cdot \vec{B} = \mu_B/\hbar \vec{B}[i] g_s[i,j] \vec{S}[j]
                      = \vec{B}[i] g_s[i,j] \vec{S}[j]

    In the last line, B takes unit of energy (cm^-1 per \mu_B), and spin takes unit of \hbar.
    
    coord: 's*' (for spherical) or else (for cartesian).

    Units: Tesla for B, deg for angles.
    """

    if coord[0] == 's' or coord[0] == 'S':
        Bv = Tesla2wavenumber*np.array(sph2cart_deg(Bv))
    else:
        Bv = Tesla2wavenumber*np.array(Bv)
    
    h_zee = spins.zero
    for i in range(3):
        h_zee = h_zee - Bv[i]*spins.Mv_tot[i]

    return h_zee

def get_h_Zeeman_Mv_tot(Mv_tot, Bv, coord):
    """ 
    Zeeman term H_Zee = - \vec{\mu} \cdot \vec{B} 

    In the last line, B takes unit of energy (cm^-1 per \mu_B), and mu takes unit of \mu_b.
    
    coord: 's*' (for spherical) or else (for cartesian).

    Units: Tesla for B, deg for angles.

    Mv_tot can be given in arbitrary basis. 
    """

    if coord[0] == 's' or coord[0] == 'S':
        Bv = Tesla2wavenumber*np.array(sph2cart_deg(Bv))
    else:
        Bv = Tesla2wavenumber*np.array(Bv)

    dim = Mv_tot[0].shape[0]
    
    h_zee = np.zeros((dim, dim), dtype=complex)
    for i in range(3):
        h_zee = h_zee - Bv[i]*Mv_tot[i]

    return h_zee



## =================================================================
## Functions for energy levels versus B field
## =================================================================

def get_energy_levels_vs_B(spins, h_ex, h_ani, Bgrid):

    Bmin, Bmax, Bstep, theta_B, phi_B = Bgrid
    nB = int((Bmax-Bmin)/Bstep) + 1

    h0 = h_ex + h_ani

    eigen = eigen_spin_hamiltonian(h0)
    energy0 = eigen.eigenvalues[eigen.indices[0]]

    eigenvalues = np.zeros((nB, eigen.dim))
    for i in range(nB):
        B = Bmin + i*Bstep
        h_zee = get_h_Zeeman(spins, [B,theta_B,phi_B], 'spherical')
        h = h0 + h_zee
        eigen = eigen_spin_hamiltonian(h)
        eigenvalues[i] = eigen.eigenvalues[eigen.indices]
    eigenvalues = eigenvalues - energy0

    with open(root_dir + "output/Zeeman.dat", "w") as f:
        for i in range(nB):
            B = Bmin + i*Bstep
            f.write((" {:12.6f}" + eigen.dim*" {:15.9f}" + "\n").format(B, *eigenvalues[i]))

def get_energy_levels_vs_B_Mv_tot(h0, Mv_tot, Bgrid):

    Bmin, Bmax, Bstep, theta_B, phi_B = Bgrid
    nB = int((Bmax-Bmin)/Bstep) + 1

    eigen = eigen_spin_hamiltonian(h0)
    energy0 = eigen.eigenvalues[eigen.indices[0]]

    eigenvalues = np.zeros((nB, eigen.dim))
    for i in range(nB):
        B = Bmin + i*Bstep
        h_zee = get_h_Zeeman_Mv_tot(Mv_tot, [B,theta_B,phi_B], 'spherical')
        h = h0 + h_zee
        eigen = eigen_spin_hamiltonian(h)
        eigenvalues[i] = eigen.eigenvalues[eigen.indices]
    eigenvalues = eigenvalues - energy0

    with open(root_dir + "output/Zeeman.dat", "w") as f:
        for i in range(nB):
            B = Bmin + i*Bstep
            f.write((" {:12.6f}" + eigen.dim*" {:15.9f}" + "\n").format(B, *eigenvalues[i]))



## ================================================================
## Functions for thermodynamic properties.
## ================================================================

### ================================================================
### Basic functions for thermodynamic properties.
### ================================================================

def get_partition_function(eigen, T):

    e_ref = eigen.eigenvalues[eigen.indices[0]]

    beta = 1/(Kelvin2wavenumber * T)

    Z = 0
    for i in range(eigen.dim):
        eigenvalue = eigen.eigenvalues[eigen.indices[i]] - e_ref
        Z += np.exp(-beta*eigenvalue)

    return Z

def get_magnetic_moment(spins, eigen, T, Z):

    e_ref = eigen.eigenvalues[eigen.indices[0]]

    beta = 1/(Kelvin2wavenumber * T)

    M = np.array([0., 0., 0.])
    for i in range(eigen.dim):
        eigenvalue = eigen.eigenvalues[eigen.indices[i]] - e_ref
        state = eigen.eigenvectors[:, eigen.indices[i]]
        for j in range(3):
            M[j] = M[j] + np.real(np.dot(np.conjugate(state), np.matmul(spins.Mv_tot[j], state))) * np.exp(-beta*eigenvalue)
    M = M/Z
    return M
    
def get_magnetic_moment_Mv_tot(Mv_tot, eigen, T, Z):

    e_ref = eigen.eigenvalues[eigen.indices[0]]

    beta = 1/(Kelvin2wavenumber * T)

    M = np.array([0., 0., 0.])
    for i in range(eigen.dim):
        eigenvalue = eigen.eigenvalues[eigen.indices[i]] - e_ref
        state = eigen.eigenvectors[:, eigen.indices[i]]
        for j in range(3):
            M[j] = M[j] + np.real(np.dot(np.conjugate(state), np.matmul(Mv_tot[j], state))) * np.exp(-beta*eigenvalue)
    M = M/Z
    return M
    
def get_chim_tensor_kernel(spins, h_ex, h_ani, B0v_sph, E0v_sph, T, dBv_sph, verbose):

    """
    chim_n = \partial M / \partial B along the e_n direction.
    B0v: B vector in spherical coordinate. Angles in deg.
    """

    E0v_cart = sph2cart_deg(E0v_sph)
    h_stark = get_h_Stark(spins, E0v_cart, 'cartesian')
  
    B0v_cart = sph2cart_deg(B0v_sph)
    dBv_cart = sph2cart_deg(dBv_sph)

    h_zee = get_h_Zeeman(spins, B0v_cart + dBv_cart, 'cartesian')
    h = h_ex + h_ani + h_zee + h_stark
    eigen_plus = eigen_spin_hamiltonian(h)
    Z_plus = get_partition_function(eigen_plus, T)
    M_plus = get_magnetic_moment(spins, eigen_plus, T, Z_plus)

    h_zee = get_h_Zeeman(spins, B0v_cart - dBv_cart, 'cartesian')
    h = h_ex + h_ani + h_zee + h_stark
    eigen_minus = eigen_spin_hamiltonian(h)
    Z_minus = get_partition_function(eigen_minus, T)
    M_minus = get_magnetic_moment(spins, eigen_minus, T, Z_minus)

    chim_n = (M_plus - M_minus) / (2*dBv_sph[0])

    if verbose:
        print("dBx, dBy, dBz = {:15.9f} {:15.9f} {:15.9f} Tesla".format(*dBv_cart))
        print("M(B0 + dB) = {:15.9f} {:15.9f} {:15.9f} mu_B".format(*M_plus))
        print("M(B0 - dB) = {:15.9f} {:15.9f} {:15.9f} mu_B".format(*M_minus))
        print(u"\u03C7_m = {:15.9f} {:15.9f} {:15.9f} ".format(*chim_n) + u"\u03BC_B/T")

    return chim_n
      
def get_chim_tensor(spins, h_ex, h_ani, E0v_sph, B0v_sph, T, dB):

    """
    chim at certain B field, E field, and temperature T
    chim_tensor_{ij} = \partial M_i / \partial B_j
    """

    chimx = get_chim_tensor_kernel(spins, h_ex, h_ani, E0v_sph, B0v_sph, T, [dB, 90.,  0.], False)
    chimy = get_chim_tensor_kernel(spins, h_ex, h_ani, E0v_sph, B0v_sph, T, [dB, 90., 90.], False)
    chimz = get_chim_tensor_kernel(spins, h_ex, h_ani, E0v_sph, B0v_sph, T, [dB,  0.,  0.], False)

    chim_tensor = np.vstack((chimx, chimy))
    chim_tensor = np.vstack((chim_tensor , chimz))

    chim_tensor = np.transpose(chim_tensor)

    return chim_tensor



### =================================================================================
### Magnetization vs B field
### at a certain E field and one or more temperatures
### =================================================================================

def get_M_at_BET_plain(args):

    """
    B: Magnitude of B field
    theta_B: polar angle of B field in deg
    phi_B: azimuthal angle of B field in deg
    E: Magnitude of E field in meV/Ang
    theta_E: polar angle of E field in deg
    phi_E: azimuthal angle of E field in deg
    T: temperature
    """

    spins, h_ex, h_ani, B, theta_B, phi_B, E, theta_E, phi_E, T = args

    h_zee   = get_h_Zeeman(spins, [B, theta_B, phi_B], "spherical")
    h_stark = get_h_Stark( spins, [E, theta_E, phi_E], "spherical")
    h = h_ex + h_ani + h_zee + h_stark
    eigen = eigen_spin_hamiltonian(h)
    Z = get_partition_function(eigen, T)
    M = get_magnetic_moment(spins, eigen, T, Z)

    return M



def get_M_at_BET_Mv_tot(args):

    """
    B: Magnitude of B field
    theta_B: polar angle of B field in deg
    phi_B: azimuthal angle of B field in deg
    E: Magnitude of E field in meV/Ang
    theta_E: polar angle of E field in deg
    phi_E: azimuthal angle of E field in deg
    T: temperature
    """

    h0, Mv_tot, B, theta_B, phi_B, E, theta_E, phi_E, T = args

    h_zee   = get_h_Zeeman_Mv_tot(Mv_tot, [B, theta_B, phi_B], "spherical")
    h = h0 + h_zee
    eigen = eigen_spin_hamiltonian(h)
    Z = get_partition_function(eigen, T)
    M = get_magnetic_moment_Mv_tot(Mv_tot, eigen, T, Z)

    return M



def get_M_vs_B_kernel(spins, h_ex, h_ani, Bs, theta_B, phi_B, Efield, T):

    """
    Bs: B fields enumerated
    theta_B: polar angle of B field in deg
    phi_B: azimuthal angle of B field in deg
    Efield[0]: Magnitude of E field in meV/Ang
    Efield[1]: theta_E in deg
    Efield[2]: phi_E in deg
    T: temperature
    """

    nB = len(Bs)

    Ms = []
    for iB in range(nB):
        print("B = {:6.2f}".format(Bs[iB]))
        h_zee   = get_h_Zeeman(spins, [Bs[iB], theta_B, phi_B], "spherical")
        h_stark = get_h_Stark( spins, Efield, "spherical")
        h = h_ex + h_ani + h_zee + h_stark
        eigen = eigen_spin_hamiltonian(h)
        Z = get_partition_function(eigen, T)
        M = get_magnetic_moment(spins, eigen, T, Z)
        Ms.append(M)

    return Ms



def get_M_vs_B(spins, h_ex, h_ani, BET_Bgrid):

    """
    Bgrid[0]: Bmin
    Bgrid[1]: Bmax
    Bgrid[2]: Bstep
    Bgrid[3]: theta_B in deg
    Bgrid[4]: phi_B in deg
    Efield[0]: Magnitude of E field in meV/Ang
    Efield[1]: theta_E in deg
    Efield[2]: phi_E in deg
    Ts: temperatures enumerated
    M-B_Exxx.dat
      rows: B field
      columns: temperatures
    """

    Bgrid = BET_Bgrid[0]
    Efield = BET_Bgrid[1]
    Ts = BET_Bgrid[2]

    nB = int((Bgrid[1]-Bgrid[0])/Bgrid[2]) + 1
    Bs = np.linspace(Bgrid[0], Bgrid[1], nB, endpoint=True)

    nT = len(Ts)

    Ms = []
    for iT in range(nT):
        print("T={:6.2f} K\n".format(Ts[iT]))
        Ms_per_T = get_M_vs_B_kernel(spins, h_ex, h_ani, Bs, Bgrid[3], Bgrid[4], Efield, Ts[iT])
        Ms.append(Ms_per_T)

    if not os.path.exists(root_dir + "output/M_vs_B"):
        subprocess.run(["mkdir", "-p", root_dir + "output/M_vs_B"])

    M1name  = root_dir + "output/M_vs_B/Mmod-B_thetaB{:.1f}_phiB{:.1f}_E{:.3f}_thetaE{:.1f}_phiE{:.1f}.dat".format(Bgrid[3], Bgrid[4], *Efield)
    M1xname = root_dir + "output/M_vs_B/Mx-B_thetaB{:.1f}_phiB{:.1f}_E{:.3f}_thetaE{:.1f}_phiE{:.1f}.dat".format(Bgrid[3], Bgrid[4], *Efield)
    M1yname = root_dir + "output/M_vs_B/My-B_thetaB{:.1f}_phiB{:.1f}_E{:.3f}_thetaE{:.1f}_phiE{:.1f}.dat".format(Bgrid[3], Bgrid[4], *Efield)
    M1zname = root_dir + "output/M_vs_B/Mz-B_thetaB{:.1f}_phiB{:.1f}_E{:.3f}_thetaE{:.1f}_phiE{:.1f}.dat".format(Bgrid[3], Bgrid[4], *Efield)

    with open(M1name,  "w") as M1 , \
         open(M1xname, "w") as M1x, \
         open(M1yname, "w") as M1y, \
         open(M1zname, "w") as M1z:
        for iB in range(nB):
            M1.write( "{:15.9f} ".format(Bs[iB]))
            M1x.write("{:15.9f} ".format(Bs[iB]))
            M1y.write("{:15.9f} ".format(Bs[iB]))
            M1z.write("{:15.9f} ".format(Bs[iB]))
            for iT in range(nT):
                M1.write( "{:15.9f} ".format(np.linalg.norm(Ms[iT][iB])))
                M1x.write("{:15.9f} ".format(Ms[iT][iB][0]))
                M1y.write("{:15.9f} ".format(Ms[iT][iB][1]))
                M1z.write("{:15.9f} ".format(Ms[iT][iB][2]))
            M1.write("\n"); M1x.write("\n"); M1y.write("\n"); M1z.write("\n")
    return



## =================================================================
## Functions for plotting
## =================================================================

def spy_sparsity(M, tag, precision=1.0e-20, figsize=(20, 20), markersize=1):
    """
    Visualize the sparsity of the matrix M
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.spy(M, precision=precision, markersize=markersize)
    plt.savefig(root_dir + "output/sparsity_of_" + tag + ".pdf")

    return


## =======================================================================
## Functions for basis transformation
## =======================================================================

def transform_O(O, eigen):
    """
    Transform an operator from the common basis to a specified basis
    """

    M = eigen.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))

    O_new = np.matmul(M_dagger, np.matmul(O, M))

    return O_new


def transform_h0(h0, eigen0):
    """
    Basis transformation for h0 on the basis of the eigenvectors of h0.
    After transformation, h0[i, j] = e_i delta_{ij} where e_i are the eigenvalues of h0.
    """

    #h0_new = transform_O(h0, eigen0)
    #print(np.all( np.absolute(h0_new.diagonal() - eigen0.eigenvalues) < 1.e-9 ) ) # True

    h0_new = np.zeros(h0.shape, dtype=np.float64)
    for i in range(h0.shape[0]):
        h0_new[i, i] = eigenp.eigenvalues[i]

    return h0_new



def transform_Sv_tot(Sv_tot, eigen_p):
    """
    Basis transformation for Sv_tot
    """

    Sv_tot_new = []

    for i in range(3):
        Si = transform_O(Sv_tot[i], eigen_p)
        Sv_tot_new.append(Si)

    return Sv_tot_new


def transform_Mv_tot(Mv_tot, eigen_p):
    """
    Basis transformation for Mv_tot
    """

    Mv_tot_new = []

    for i in range(3):
        Mi = transform_O(Mv_tot[i], eigen_p)
        Mv_tot_new.append(Mi)

    return Mv_tot_new


def get_perturbed_basis(h_ex, spins, Bfield):
    h_zee = get_h_Zeeman(spins, Bfield, 'cartesian')
    h = h_ex + h_zee
    eigen = eigen_spin_hamiltonian(h)
    return eigen

# =======================================================================
# Functions for obtaining the time-dependent Hamiltonian H(B(t))
# =======================================================================

def get_h_Mv(h0, Mv_tot, B, theta_B, phi_B):
    """
    Both h0 and Mv_tot should be on the basis of eigenvectors of h0.
    Return h under the magnetic field B.
    """

    h_zee = get_h_Zeeman_Mv_tot(Mv_tot, [B, theta_B, phi_B], 'spherical')
    h = h0 + h_zee

    return h


def get_h_Mz(h0, Mz_tot, B):
    """
    Both h0 and Mz_tot should be on the basis of eigenvectors of h0.
    Return h under the magnetic field B.

    Assumptions:
      The magnetic field is along the z direction.
    """

    h_zee = -1 * Tesla2wavenumber * B * Mz_tot
    h = h0 + h_zee

    return h


def get_habc_Mv(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B):
    """
    Obtain ha = h(ts[it])
           hb = h(ts[it] + deltat/2)
           hc = h(ts[it] + deltat)

    h0 and Mv_tot are on the basis of the eigenvectors of h0.

    ts:  A list of time with a time step of deltat
    Bs2: A list of B fields with a time step of deltat/2
         B(t = ts[it]) = Bs2[2*it]
    """

    ha = get_h_Mv(h0, Mv_tot, Bs2[2*it  ], theta_B, phi_B)
    hb = get_h_Mv(h0, Mv_tot, Bs2[2*it+1], theta_B, phi_B)
    hc = get_h_Mv(h0, Mv_tot, Bs2[2*it+2], theta_B, phi_B)

    return (ha, hb, hc)

def get_habc_Mz(h0, Mz_tot, it, deltat, Bs2):
    """
    Obtain ha = h(ts[it])
           hb = h(ts[it] + deltat/2)
           hc = h(ts[it] + deltat)

    h0 and Mv_tot are on the basis of the eigenvectors of h0.

    ts:  A list of time with a time step of deltat
    Bs2: A list of B fields with a time step of deltat/2
         B(t = ts[it]) = Bs2[2*it]

    Assumptions:
      The magnetic field is along the z direction.
    """

    ha = get_h_Mz(h0, Mz_tot, Bs2[2*it  ])
    hb = get_h_Mz(h0, Mz_tot, Bs2[2*it+1])
    hc = get_h_Mz(h0, Mz_tot, Bs2[2*it+2])

    return (ha, hb, hc)

def get_habc_reuse_ha_Mv(h0, Mv_tot, it, deltat, Bs2, theta_B, phi_B, ha, hb, hc):
    """
    Obtain ha = h(ts[it])
           hb = h(ts[it] + deltat/2)
           hc = h(ts[it] + deltat)

    h0 and Mv_tot are on the basis of the eigenvectors of h0.

    ts:  A list of time with a time step of deltat
    Bs2: A list of B fields with a time step of deltat/2
         B(t = ts[it]) = Bs2[2*it]
    """

    hb = get_h_Mv(h0, Mv_tot, Bs2[2*it+1], theta_B, phi_B)
    hc = get_h_Mv(h0, Mv_tot, Bs2[2*it+2], theta_B, phi_B)

    return (ha, hb, hc)

def get_habc_reuse_ha_Mz(h0, Mz_tot, it, deltat, Bs2, ha, hb, hc):
    """
    Obtain ha = h(ts[it])
           hb = h(ts[it] + deltat/2)
           hc = h(ts[it] + deltat)

    h0 and Mv_tot are on the basis of the eigenvectors of h0.

    ts:  A list of time with a time step of deltat
    Bs2: A list of B fields with a time step of deltat/2
         B(t = ts[it]) = Bs2[2*it]

    Assumptions:
      The magnetic field is along the z direction.
    """

    hb = get_h_Mz(h0, Mz_tot, Bs2[2*it+1])
    hc = get_h_Mz(h0, Mz_tot, Bs2[2*it+2])

    return (ha, hb, hc)



# =======================================================================
# Functions for calculating magnetization
# =======================================================================

def get_Mv_from_rho(rho, Mv_tot):
    """
    """

    Mv = []
    for i in range(3):
        Mi = np.trace( np.matmul(rho, Mv_tot[i]) )
        Mv.append(Mi)

    return Mv


def get_rho_upper(rho, indices_upper):
    rho_upper = rho[indices_upper]
    rho_upper_real = np.real(rho_upper)
    rho_upper_imag = np.imag(rho_upper)
    rho_upper = np.hstack((rho_upper_real, rho_upper_imag))
    return rho_upper


def get_indices_of_rho_upper(dim):
    indices_upper = np.triu_indices(dim)
    with open(root_dir + "output/indices_of_rho_upper.dat", "w") as f:
        for i in range(indices_upper[0].shape[0]):
            f.write("{:6d} {:6d} {:6d}\n".format(i+1, indices_upper[0][i], indices_upper[1][i]))
        for i in range(indices_upper[0].shape[0]):
            f.write("{:6d} {:6d} {:6d}\n".format(indices_upper[0].shape[0] + i + 1, indices_upper[0][i], indices_upper[1][i]))

