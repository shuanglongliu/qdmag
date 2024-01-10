import os
import subprocess
import copy
import math
import numpy as np
import yaml
from external.Operators import Operator
from external.StevensOperators import StevensOpA
from scipy.constants import N_A, mu_0
from constants import meV2wavenumber, Tesla2wavenumber, Kelvin2meV, Kelvin2wavenumber
from constants import x_mu_B, mu_B_per_Tesla_2_cm3_per_mol_phi
import ray



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
        self.indices = np.argsort(self.eigenvalues)
        self.eigenvalues_offset = self.eigenvalues - self.eigenvalues[self.indices[0]]

        self.dim = self.eigenvalues.shape[0]

        return
    

    
# ========================
# Functions
# ========================

## =================================================================
## Functions for common mathematics.
## =================================================================

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

def get_crossing_point(x1, y1, x2, y2):
    # Line 1 joins y1[0] and y2[1]
    # Line 2 joins y1[1] and y2[0]
    # Find the crossing point between line 1 and line 2. 
    dx = x2 - x1
    dy1 = y1[1] - y1[0]
    dy2 = y2[1] - y2[0]
    x = dx*dy1/(dy1 + dy2) + x1
    return x

## =================================================================
## Functions for input and output.
## =================================================================

def read_input():

    with open("input.yaml", "r") as f:
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
    fit_problem = data['fit_problem']

    return (Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, fit_problem)

def save_eigenvalues(eigen, offset):
    
    if offset:
        eigenvalues = eigen.eigenvalues_offset
    else:
        eigenvalues = eigen.eigenvalues

    if not os.path.exists("./output"):
        subprocess.run(["mkdir", "./output"])

    with open("./output/eigenvalues.dat", "w") as f:
        for i in range(len(eigenvalues)):
            f.write("{:16.12f}\n".format(eigenvalues[eigen.indices[i]]))

    return

def save_eigenvectors(spins, eigen):

    if not os.path.exists("./output"):
        subprocess.run(["mkdir", "./output"])

    with open("./output/eigenvectors.dat", "w") as f:
        for i in range(spins.dim):
            state = eigen.eigenvectors[:, eigen.indices[i]]
            f.write((spins.dim*" {:16.12f}" + "\n").format(*state))
    
    return

def save_operator(op, base_name):

    Dim = op.shape[0]

    o_real = np.real(op)
    o_imag = np.imag(op)

    if not os.path.exists("./output"):
        subprocess.run(["mkdir", "./output"])

    with open("./output/{:s}.real".format(base_name), "w") as f:
        for i in range(Dim):
            for j in range(Dim):
                f.write("{:8.3f}\n".format( o_real[i, j] ))

    with open("./output/{:s}.imag".format(base_name), "w") as f:
        for i in range(Dim):
            for j in range(Dim):
                f.write("{:8.3f}\n".format( o_imag[i, j] ))
    return

def save_spins(spins, eigen):

    """
    Calculate and save expectation of spins for all eigenvectors.
    """

    if not os.path.exists("./output"):
        subprocess.run(["mkdir", "./output"])

    with open("./output/spins.dat", "w") as f:
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

def get_total_Sz_for_all_eigenstates(spins, eigen):

    list_of_Sz_tot = []

    for i in range(eigen.dim):
        state = eigen.eigenvectors[:, eigen.indices[i]]
        list_of_Sz_tot.append( np.real(np.dot(np.conjugate(state), np.matmul(spins.Sv_tot[2], state))) )

    #print((eigen.dim * "{:12.6f}\n").format(*list_of_Sz_tot))
        
    return list_of_Sz_tot

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

## =================================================================
## Functions for the Stark term.
## =================================================================

def get_h_Stark(spins, Ev, coord):
    """ 
    H = - \vec{E} \cdot \vec{P}, where \vec{P} is the electric polarization.

    Units: mV/Angfor E, e Ang for dipole, cm^-1 for energy, deg for angles.
    """

    if coord[0] == 's' or coord[0] == 'S':
        Ev = np.array(sph2cart_deg(Ev))

    Px, Py, Pz = spins.Pv

    h_stark = -1 * meV2wavenumber * (Ev[0]*Px + Ev[1]*Py + Ev[2]*Pz)

    return h_stark

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

    with open("./output/Zeeman.dat", "w") as f:
        for i in range(nB):
            B = Bmin + i*Bstep
            f.write((" {:12.6f}" + eigen.dim*" {:15.9f}" + "\n").format(B, *eigenvalues[i]))

def check_energy_level_crossing_B1_vs_B2(spins, h_ex, h_ani, B1, B2, theta_B, phi_B, tol=0.8):

    h0 = h_ex + h_ani

    h_zee = get_h_Zeeman(spins, [B1,theta_B,phi_B], 'spherical')
    h = h0 + h_zee
    eigen1 = eigen_spin_hamiltonian(h)

    h_zee = get_h_Zeeman(spins, [B2,theta_B,phi_B], 'spherical')
    h = h0 + h_zee
    eigen2 = eigen_spin_hamiltonian(h)

    all_S_and_Sv_1 = get_total_spin_for_all_eigenstates(spins, eigen1)
    all_S_and_Sv_2 = get_total_spin_for_all_eigenstates(spins, eigen2)

    all_S_and_Sv_1 = np.array(all_S_and_Sv_1)
    all_S_and_Sv_2 = np.array(all_S_and_Sv_2)

    ds = []
    for i in range(eigen1.dim):
        dv = all_S_and_Sv_2[i] - all_S_and_Sv_1[i]
        d = np.linalg.norm(dv)
        ds.append(d)

    with open("./output/level_crossing_B1_vs_B2.log", "w") as f:
        f.write("All changes in V = |[S_tot, Sx_tot, Sy_tot, Sz_tot]| from {:8.4f} T to {:8.4f} T. dV = V_B2 - V_B1\n".format(B1, B2))
        for i in range(eigen1.dim):
            f.write("State {:3d} :: {:6.3f} = | ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) - ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) |\n".format(i+1, ds[i], *all_S_and_Sv_2[i], *all_S_and_Sv_1[i]))
        f.write("\n\n\n")

        f.write("Significant changes in V = |[S_tot, Sx_tot, Sy_tot, Sz_tot]| (bigger than {:6.3f}) from {:8.4f} T to {:8.4f} T. dV = V_B2 - V_B1\n".format(tol, B1, B2))
        for i in range(eigen1.dim):
            if ds[i] > tol:
                f.write("State {:3d} :: {:6.3f} = | ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) - ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) |\n".format(i+1, ds[i], *all_S_and_Sv_2[i], *all_S_and_Sv_1[i]))
    return

def check_energy_level_crossing(spins, h_ex, h_ani, Bmin, Bmax, Bstep, theta_B, phi_B, tol1=0.05, tol2=0.5):

    h0 = h_ex + h_ani
    eigen0 = eigen_spin_hamiltonian(h0)
    energy0 = eigen0.eigenvalues[eigen0.indices[0]]
    energy0_wavenumber = eigen0.eigenvalues_wavenumber[eigen0.indices[0]]

    nB = int((Bmax - Bmin)/Bstep) + 1

    h_zee = get_h_Zeeman(spins, [Bmin,theta_B,phi_B], 'spherical')
    h = h0 + h_zee
    eigen1 = eigen_spin_hamiltonian(h)

    with open("./output/level_crossing.log", "w") as f:
        for i in range(1, nB):
            h_zee = get_h_Zeeman(spins, [Bmin+i*Bstep,theta_B,phi_B], 'spherical')
            h = h0 + h_zee
            eigen2 = eigen_spin_hamiltonian(h)
      
            all_S_and_Sv_1 = get_total_spin_for_all_eigenstates(spins, eigen1)
            all_S_and_Sv_2 = get_total_spin_for_all_eigenstates(spins, eigen2)
        
            all_S_and_Sv_1 = np.array(all_S_and_Sv_1)
            all_S_and_Sv_2 = np.array(all_S_and_Sv_2)
        
            ds = []
            for j in range(eigen1.dim):
                dv = all_S_and_Sv_2[j] - all_S_and_Sv_1[j]
                d = np.linalg.norm(dv)
                ds.append(d)
        
            f.write("# ========================================================================\n")
            f.write("# From B1 = {:8.4f} T to B2 = {:8.4f} T.\n".format(Bmin+(i-1)*Bstep, Bmin+i*Bstep))
            f.write("# ========================================================================\n")
            f.write("\n")

            f.write("All changes in V = [S_tot, Sx_tot, Sy_tot, Sz_tot]. dV = V_B2 - V_B1. \n")
            for j in range(eigen1.dim):
                f.write("State {:3d} :: {:6.3f} = | ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) - ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) |\n".format(j+1, ds[j], *all_S_and_Sv_2[j], *all_S_and_Sv_1[j]))
            f.write("\n\n")
      
            f.write("Continuous changes in V = [S_tot, Sx_tot, Sy_tot, Sz_tot] ({:6.3f} < |dV| < {:6.3f}). dV = V_B2 - V_B1\n".format(tol1, tol2))
            f.write("State index  E(B)-E_min(B=0) / E(B)-E_min(B) meV  E(B)-E_min(B=0) / E(B)-E_min(B) cm^-1  exp(-E/kBT) T=2K 5K 10K 50K 100K 300K :: dV = V(B2) - V(B1)\n")
            for j in range(eigen1.dim):
                if ds[j] >= tol1 and ds[j] < tol2:
                    E1a = eigen1.eigenvalues[eigen1.indices[j]]
                    E1b = eigen1.eigenvalues_wavenumber[eigen1.indices[j]]
                    Emin1a = eigen1.eigenvalues[eigen1.indices[0]]
                    Emin1b = eigen1.eigenvalues_wavenumber[eigen1.indices[0]]
                    f.write("State {:3d}  {:9.3f} / {:9.3f} meV  {:10.2f} / {:10.2f} cm^-1  {:10.2E} {:10.2E} {:10.2E} {:10.2E} {:10.2E} {:10.2E} :: {:6.3f} = | ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) - ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) | :: ".format(j+1, E1a - energy0, E1a - Emin1a, E1b - energy0_wavenumber, E1b - Emin1b, np.exp(-(E1a-Emin1a)/(2*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(5*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(10*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(50*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(100*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(300*Kelvin2meV)), ds[j], *all_S_and_Sv_2[j], *all_S_and_Sv_1[j]))
                    f.write("CONCHG from B1 = {:8.4f} T to B2 = {:8.4f} T.\n".format(Bmin+(i-1)*Bstep, Bmin+i*Bstep))
            f.write("\n\n")

            f.write("Significant changes in V = [S_tot, Sx_tot, Sy_tot, Sz_tot] (|dV| > {:6.3f}). dV = V_B2 - V_B1\n".format(tol2))
            f.write("State index  E(B)-E_min(B=0) / E(B)-E_min(B) meV  E(B)-E_min(B=0) / E(B)-E_min(B) cm^-1  exp(-E/kBT) T=2K 5K 10K 50K 100K 300K :: dV = V(B2) - V(B1)\n")
            for j in range(eigen1.dim):
                if ds[j] >= tol2:
                    E1a = eigen1.eigenvalues[eigen1.indices[j]]
                    E1b = eigen1.eigenvalues_wavenumber[eigen1.indices[j]]
                    Emin1a = eigen1.eigenvalues[eigen1.indices[0]]
                    Emin1b = eigen1.eigenvalues_wavenumber[eigen1.indices[0]]

                    # Get the crossing point using linear interpolation
                    x1 = Bmin+(i-1)*Bstep
                    x2 = Bmin+i*Bstep
                    if ds[j+1] >= tol2:
                        y1 = [eigen1.eigenvalues[eigen1.indices[j]], eigen1.eigenvalues[eigen1.indices[j+1]]]
                        y2 = [eigen2.eigenvalues[eigen2.indices[j]], eigen2.eigenvalues[eigen2.indices[j+1]]]
                    else:
                        y1 = [eigen1.eigenvalues[eigen1.indices[j-1]], eigen1.eigenvalues[eigen1.indices[j]]]
                        y2 = [eigen2.eigenvalues[eigen2.indices[j-1]], eigen2.eigenvalues[eigen2.indices[j]]]
                    # If B_crossing is nan, a smaller Bstep is needed to resolve the crossing. 
                    B_crossing = get_crossing_point(x1, y1, x2, y2)

                    f.write("State {:3d}  {:9.3f} / {:9.3f} meV  {:10.2f} / {:10.2f} cm^-1  {:10.2E} {:10.2E} {:10.2E} {:10.2E} {:10.2E} {:10.2E} :: {:6.3f} = | ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) - ({:6.3f} {:6.3f} {:6.3f} {:6.3f}) | :: ".format(j+1, E1a - energy0, E1a - Emin1a, E1b - energy0_wavenumber, E1b - Emin1b, np.exp(-(E1a-Emin1a)/(2*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(5*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(10*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(50*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(100*Kelvin2meV)), np.exp(-(E1a-Emin1a)/(300*Kelvin2meV)), ds[j], *all_S_and_Sv_2[j], *all_S_and_Sv_1[j]))
                    f.write("SIGCHG from B1 = {:8.4f} T to B2 = {:8.4f} T at {:8.4f} T.\n".format(Bmin+(i-1)*Bstep, Bmin+i*Bstep, B_crossing))
            f.write("\n\n\n")

            eigen1 = eigen2
    return


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
    
def get_dipole_moment(spins, eigen, T, Z):

    e_ref = eigen.eigenvalues[eigen.indices[0]]

    beta = 1/(Kelvin2wavenumber * T)

    P = np.array( [0., 0., 0.] )
    for i in range(eigen.dim):
        eigenvalue = eigen.eigenvalues[eigen.indices[i]] - e_ref
        state = eigen.eigenvectors[:, eigen.indices[i]]
        for j in range(3):
            P[j] = P[j] + np.real(np.dot(np.conjugate(state), np.matmul(spins.Pv[j], state))) * np.exp(-beta*eigenvalue)
    P = P/Z

    return P

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

def get_dMdB(spins, h_ex, h_ani, B0v_sph, E0v_sph, T, dBv_sph, verbose):
    """
    dMdB = \partial M / \partial B along the e_n direction.
    B0v: B vector in spherical coordinate. Angles in deg.
    unit: mu_B/T.
    """
    result = get_chim_tensor_kernel(spins, h_ex, h_ani, B0v_sph, E0v_sph, T, dBv_sph, "mB/T", 1, 1000., False)
    return result

def get_d2MdB2(spins, h_ex, h_ani, B0v_sph, E0v_sph, T, dBv_sph, verbose):

    """
    d2MdB2 = \partial^2 M / \partial B^2 along the e_n direction.
    B0v: B vector in spherical coordinate. Angles in deg.
    unit: mu_B/T^2.
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

    h_zee = get_h_Zeeman(spins, B0v_cart, 'cartesian')
    h = h_ex + h_ani + h_zee + h_stark
    eigen0 = eigen_spin_hamiltonian(h)
    Z0 = get_partition_function(eigen0, T)
    M0 = get_magnetic_moment(spins, eigen0, T, Z0)

    result = (M_plus - 2*M0 + M_minus) / (dBv_sph[0]**2)

    if verbose:
        print("dBx, dBy, dBz = {:15.9f} {:15.9f} {:15.9f} Tesla".format(*dBv_cart))
        print("M(B0 + dB) = {:15.9f} {:15.9f} {:15.9f} mu_B".format(*M_plus))
        print("M(B0 - dB) = {:15.9f} {:15.9f} {:15.9f} mu_B".format(*M_minus))
        print("M(B0     ) = {:15.9f} {:15.9f} {:15.9f} mu_B".format(*M0))
        print("d2MdB2 = {:15.9f} {:15.9f} {:15.9f} mu_B/T^2".format(*result))

    return result
      
def get_chie_tensor_kernel(spins, h_ex, h_ani, B0v_sph, E0v_sph, T, dEv_sph, verbose):

    """
    chie_n = \partial P / \partial E along the e_n direction.
    E0v: B vector in spherical coordinate. Angles in deg.
    unit for chie_n: e_Ang2_per_V.
    """

    B0v_cart = sph2cart_deg(B0v_sph)
    h_zee = get_h_Zeeman(spins, B0v_cart, 'cartesian')

    E0v_cart = sph2cart_deg(E0v_sph)
    dEv_cart = sph2cart_deg(dEv_sph)
  
    h_stark = get_h_Stark(spins, E0v_cart + dEv_cart, 'cartesian')
    h = h_ex + h_ani + h_zee + h_stark
    eigen_plus = eigen_spin_hamiltonian(h)
    Z_plus = get_partition_function(eigen_plus, T)
    P_plus = get_dipole_moment(spins, eigen_plus, T, Z_plus)

    h_stark = get_h_Stark(spins, E0v_cart - dEv_cart, 'cartesian')
    h = h_ex + h_ani + h_zee + h_stark
    eigen_minus = eigen_spin_hamiltonian(h)
    Z_minus = get_partition_function(eigen_minus, T)
    P_minus = get_dipole_moment(spins, eigen_minus, T, Z_minus)

    # unit for |dEv_sph|: mV/Ang.
    chie_n = 1000*(P_plus - P_minus) / (2*dEv_sph[0])

    if verbose:
        print("dEx, dEy, dEz = {:15.9f} {:15.9f} {:15.9f} meV/Ang".format(*dEv_cart))
        print("P(E0 + dE) = {:15.9f} {:15.9f} {:15.9f} e Ang".format(*P_plus))
        print("P(E0 - dE) = {:15.9f} {:15.9f} {:15.9f} e Ang".format(*P_minus))
        print(u"\u03C7_e = {:15.9f} {:15.9f} {:15.9f} ".format(*chie_n) + "e_Ang2_per_V")

    return chie_n
      
def get_chie_tensor(spins, h_ex, h_ani, B0v_sph, E0v_sph, T, dE):

    """
    chie at certain B field, E field, and temperature T
    chie_tensor_{ij} = \partial P_i / \partial E_j
    """

    chiex = get_chie_tensor_kernel(spins, h_ex, h_ani, B0v_sph, E0v_sph, T, [dE, 90.,  0.], False)
    chiey = get_chie_tensor_kernel(spins, h_ex, h_ani, B0v_sph, E0v_sph, T, [dE, 90., 90.], False)
    chiez = get_chie_tensor_kernel(spins, h_ex, h_ani, B0v_sph, E0v_sph, T, [dE,  0.,  0.], False)

    chie_tensor = np.vstack((chiex, chiey))
    chie_tensor = np.vstack((chie_tensor , chiez))

    chie_tensor = np.transpose(chie_tensor)

    return chie_tensor



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



@ray.remote(num_cpus=1)
def get_M_at_BET_ray(args):

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



def get_M_vs_B_kernel_ray(spins, h_ex, h_ani, Bs, theta_B, phi_B, E, theta_E, phi_E, T):

    """
    Bs: B fields enumerated
    theta_B: polar angle of B field in deg
    phi_B: azimuthal angle of B field in deg
    E: Magnitude of E field in meV/Ang
    theta_E: polar angle of E field in deg
    phi_E: azimuthal angle of E field in deg
    T: temperature
    """

    nB = len(Bs)

    list_of_args = []
    for i in range(nB):
        list_of_args.append( (spins, h_ex, h_ani, Bs[i], theta_B, phi_B, E, theta_E, phi_E, T) )

    # Parallel calculations of magnetization

    #start = time.time()
    futures = [ get_M_at_BET_ray.remote(list_of_args[i]) for i in range(nB) ]
    Ms = ray.get(futures)
    #end = time.time()

    #print("# All {:5d} magnetization calculations take {:6.3f} s".format(nB, end - start))

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
        Ms_per_T = get_M_vs_B_kernel_ray(spins, h_ex, h_ani, Bs, Bgrid[3], Bgrid[4], Efield[0], Efield[1], Efield[2], Ts[iT])
        Ms.append(Ms_per_T)

    if not os.path.exists("./output/M_vs_B"):
        subprocess.run(["mkdir", "-p", "./output/M_vs_B"])

    M1name  = "./output/M_vs_B/Mmod-B_thetaB{:.1f}_phiB{:.1f}_E{:.3f}_thetaE{:.1f}_phiE{:.1f}.dat".format(Bgrid[3], Bgrid[4], *Efield)
    M1xname = "./output/M_vs_B/Mx-B_thetaB{:.1f}_phiB{:.1f}_E{:.3f}_thetaE{:.1f}_phiE{:.1f}.dat".format(Bgrid[3], Bgrid[4], *Efield)
    M1yname = "./output/M_vs_B/My-B_thetaB{:.1f}_phiB{:.1f}_E{:.3f}_thetaE{:.1f}_phiE{:.1f}.dat".format(Bgrid[3], Bgrid[4], *Efield)
    M1zname = "./output/M_vs_B/Mz-B_thetaB{:.1f}_phiB{:.1f}_E{:.3f}_thetaE{:.1f}_phiE{:.1f}.dat".format(Bgrid[3], Bgrid[4], *Efield)

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



### =================================================================================
### Magnetization and polarization vs B field and E field
### at different temperatures
### =================================================================================

def get_M_and_P_vs_B_and_E(spins, h_ex, h_ani, BET_BEgrid):

    """
    sampleB[0]: Bmin
    sampleB[1]: Bmax
    sampleB[2]: Bstep
    sampleB[3]: theta_B in deg
    sampleB[4]: phi_B in deg
    sampleE: similar to sampleB
    Ts: temperatures enumerated
    M-B_Txxx.dat / P-B_Txxx.dat:
      for temperature T=xxx Kelvin
      rows: B field
      columns: E field
    M-E_Txxx.dat / P-E_Txxx.dat:
      for temperature T=xxx Kelvin
      rows: E field
      columns: B field
      M-E_Txxx.dat is the transpose of M-B_Txxx.dat for the ease of plotting
    """

    sampleB = BET_BEgrid[0]
    sampleE = BET_BEgrid[1]
    Ts = BET_BEgrid[2]

    nT = len(Ts)

    nB = int((sampleB[1]-sampleB[0])/sampleB[2]) + 1
    Bs = np.linspace(sampleB[0], sampleB[1], nB, endpoint=True)

    nE = int((sampleE[1]-sampleE[0])/sampleE[2]) + 1
    Es = np.linspace(sampleE[0], sampleE[1], nE, endpoint=True)

    Ms = []; Ps = []
    for iT in range(nT):
        Ms.append([]); Ps.append([])
        for iB in range(nB):
            M_vs_E = []; P_vs_E = []
            for iE in range(nE):
                print("T = {:4.0f} B = {:6.2f} E = {:6.2f}".format(Ts[iT], Bs[iB], Es[iE]))
                h_zee   = get_h_Zeeman(spins, [Bs[iB], sampleB[3], sampleB[4]], "spherical")
                h_stark = get_h_Stark( spins, [Es[iE], sampleE[3], sampleE[4]], "spherical")
                h = h_ex + h_ani + h_zee + h_stark
                eigen = eigen_spin_hamiltonian(h)
                Z = get_partition_function(eigen, Ts[iT])
                M = get_magnetic_moment(spins, eigen, Ts[iT], Z)
                P = get_dipole_moment(spins, eigen, Ts[iT], Z)
                M_vs_E.append(M); P_vs_E.append(P)
            Ms[iT].append(M_vs_E); Ps[iT].append(P_vs_E)

    if not os.path.exists("./output/M_and_P_vs_B_and_E"):
        subprocess.run(["mkdir", "-p", "./output/M_and_P_vs_B_and_E"])

    for iT in range(nT):
        #print("T = {:6.3f} K".format(Ts[iT]))

        M1name  = "./output/M_and_P_vs_B_and_E/Mmod-B_T{:.3f}K.dat".format(Ts[iT])
        M1xname = "./output/M_and_P_vs_B_and_E/Mx-B_T{:.3f}K.dat".format(Ts[iT])
        M1yname = "./output/M_and_P_vs_B_and_E/My-B_T{:.3f}K.dat".format(Ts[iT])
        M1zname = "./output/M_and_P_vs_B_and_E/Mz-B_T{:.3f}K.dat".format(Ts[iT])

        M2name  = "./output/M_and_P_vs_B_and_E/Mmod-E_T{:.3f}K.dat".format(Ts[iT])
        M2xname = "./output/M_and_P_vs_B_and_E/Mx-E_T{:.3f}K.dat".format(Ts[iT])
        M2yname = "./output/M_and_P_vs_B_and_E/My-E_T{:.3f}K.dat".format(Ts[iT])
        M2zname = "./output/M_and_P_vs_B_and_E/Mz-E_T{:.3f}K.dat".format(Ts[iT])

        P1name  = "./output/M_and_P_vs_B_and_E/Pmod-B_T{:.3f}K.dat".format(Ts[iT])
        P1xname = "./output/M_and_P_vs_B_and_E/Px-B_T{:.3f}K.dat".format(Ts[iT])
        P1yname = "./output/M_and_P_vs_B_and_E/Py-B_T{:.3f}K.dat".format(Ts[iT])
        P1zname = "./output/M_and_P_vs_B_and_E/Pz-B_T{:.3f}K.dat".format(Ts[iT])

        P2name  = "./output/M_and_P_vs_B_and_E/Pmod-E_T{:.3f}K.dat".format(Ts[iT])
        P2xname = "./output/M_and_P_vs_B_and_E/Px-E_T{:.3f}K.dat".format(Ts[iT])
        P2yname = "./output/M_and_P_vs_B_and_E/Py-E_T{:.3f}K.dat".format(Ts[iT])
        P2zname = "./output/M_and_P_vs_B_and_E/Pz-E_T{:.3f}K.dat".format(Ts[iT])

        with open(M1name,  "w") as M1 , open(M2name,  "w") as M2 , \
             open(M1xname, "w") as M1x, open(M2xname, "w") as M2x, \
             open(M1yname, "w") as M1y, open(M2yname, "w") as M2y, \
             open(M1zname, "w") as M1z, open(M2zname, "w") as M2z, \
             open(P1name,  "w") as P1 , open(P2name,  "w") as P2 , \
             open(P1xname, "w") as P1x, open(P2xname, "w") as P2x, \
             open(P1yname, "w") as P1y, open(P2yname, "w") as P2y, \
             open(P1zname, "w") as P1z, open(P2zname, "w") as P2z:
            for iB in range(nB):
                M1.write( "{:15.9f} ".format(Bs[iB]))
                M1x.write("{:15.9f} ".format(Bs[iB]))
                M1y.write("{:15.9f} ".format(Bs[iB]))
                M1z.write("{:15.9f} ".format(Bs[iB]))
                P1.write( "{:15.9f} ".format(Bs[iB]))
                P1x.write("{:15.9f} ".format(Bs[iB]))
                P1y.write("{:15.9f} ".format(Bs[iB]))
                P1z.write("{:15.9f} ".format(Bs[iB]))
                for iE in range(nE):
                    M1.write( "{:15.9f} ".format(np.linalg.norm(Ms[iT][iB][iE])))
                    M1x.write("{:15.9f} ".format(Ms[iT][iB][iE][0]))
                    M1y.write("{:15.9f} ".format(Ms[iT][iB][iE][1]))
                    M1z.write("{:15.9f} ".format(Ms[iT][iB][iE][2]))
                    P1.write( "{:15.9f} ".format(np.linalg.norm(Ps[iT][iB][iE])))
                    P1x.write("{:15.9f} ".format(Ps[iT][iB][iE][0]))
                    P1y.write("{:15.9f} ".format(Ps[iT][iB][iE][1]))
                    P1z.write("{:15.9f} ".format(Ps[iT][iB][iE][2]))
                M1.write("\n"); M1x.write("\n"); M1y.write("\n"); M1z.write("\n")
                P1.write("\n"); P1x.write("\n"); P1y.write("\n"); P1z.write("\n")
            for iE in range(nE):
                M2.write( "{:15.9f} ".format(Es[iE]))
                M2x.write("{:15.9f} ".format(Es[iE]))
                M2y.write("{:15.9f} ".format(Es[iE]))
                M2z.write("{:15.9f} ".format(Es[iE]))
                P2.write( "{:15.9f} ".format(Es[iE]))
                P2x.write("{:15.9f} ".format(Es[iE]))
                P2y.write("{:15.9f} ".format(Es[iE]))
                P2z.write("{:15.9f} ".format(Es[iE]))
                for iB in range(nB):
                    M2.write( "{:15.9f} ".format(np.linalg.norm(Ms[iT][iB][iE])))
                    M2x.write("{:15.9f} ".format(Ms[iT][iB][iE][0]))
                    M2y.write("{:15.9f} ".format(Ms[iT][iB][iE][1]))
                    M2z.write("{:15.9f} ".format(Ms[iT][iB][iE][2]))
                    P2.write( "{:15.9f} ".format(np.linalg.norm(Ps[iT][iB][iE])))
                    P2x.write("{:15.9f} ".format(Ps[iT][iB][iE][0]))
                    P2y.write("{:15.9f} ".format(Ps[iT][iB][iE][1]))
                    P2z.write("{:15.9f} ".format(Ps[iT][iB][iE][2]))
                M2.write("\n"); M2x.write("\n"); M2y.write("\n"); M2z.write("\n")
                P2.write("\n"); P2x.write("\n"); P2y.write("\n"); P2z.write("\n")
    return



### =================================================================================
### Magnetic susceptibility and electric susceptibility vs B field and E field
### at different temperatures
### =================================================================================

def get_chim_and_chie_vs_B_and_E(spins, h_ex, h_ani, BET_BEgrid, dB, dE):

    """
    sampleB[0]: Bmin
    sampleB[1]: Bmax
    sampleB[2]: Bstep
    sampleB[3]: theta_B in deg
    sampleB[4]: phi_B in deg
    sampleE: similar to sampleB
    Ts: temperatures enumerated
    chim-B_Txxx.dat / chie-B_Txxx.dat:
      for temperature T=xxx Kelvin
      rows: B field
      columns: E field
    chim-E_Txxx.dat / chie-E_Txxx.dat:
      for temperature T=xxx Kelvin
      rows: E field
      columns: B field
      chim/chie-E_Txxx.dat is the transpose of chim/chie-B_Txxx.dat for the ease of plotting
    """

    sampleB = BET_BEgrid[0]
    sampleE = BET_BEgrid[1]
    Ts = BET_BEgrid[2]

    nT = len(Ts)

    nB = int((sampleB[1]-sampleB[0])/sampleB[2]) + 1
    Bs = np.linspace(sampleB[0], sampleB[1], nB, endpoint=True)

    nE = int((sampleE[1]-sampleE[0])/sampleE[2]) + 1
    Es = np.linspace(sampleE[0], sampleE[1], nE, endpoint=True)

    chims = []; chies = []
    for iT in range(nT):
        chims.append([]); chies.append([])
        for iB in range(nB):
            chim_vs_E = []; chie_vs_E = []
            for iE in range(nE):
                print("T = {:4.0f} B = {:6.2f} E = {:6.2f}".format(Ts[iT], Bs[iB], Es[iE]))
                B0v_sph = [Bs[iB], sampleB[3], sampleB[4]]
                E0v_sph = [Es[iE], sampleE[3], sampleE[4]]
                chim = get_chim_tensor(spins, h_ex, h_ani, B0v_sph, E0v_sph, Ts[iT], dB)
                chie = get_chie_tensor(spins, h_ex, h_ani, B0v_sph, E0v_sph, Ts[iT], dE)
                chim_vs_E.append(chim); chie_vs_E.append(chie)
            chims[iT].append(chim_vs_E); chies[iT].append(chie_vs_E)

    if not os.path.exists("./output/chim_and_chie_vs_B_and_E"):
        subprocess.run(["mkdir", "-p", "./output/chim_and_chie_vs_B_and_E"])

    for iT in range(nT):
        #print("T = {:6.3f} K".format(Ts[iT]))

        chimxxBname = "./output/chim_and_chie_vs_B_and_E/chimxx-B_T{:.3f}K.dat".format(Ts[iT])
        chimxyBname = "./output/chim_and_chie_vs_B_and_E/chimxy-B_T{:.3f}K.dat".format(Ts[iT])
        chimxzBname = "./output/chim_and_chie_vs_B_and_E/chimxz-B_T{:.3f}K.dat".format(Ts[iT])
        chimyxBname = "./output/chim_and_chie_vs_B_and_E/chimyx-B_T{:.3f}K.dat".format(Ts[iT])
        chimyyBname = "./output/chim_and_chie_vs_B_and_E/chimyy-B_T{:.3f}K.dat".format(Ts[iT])
        chimyzBname = "./output/chim_and_chie_vs_B_and_E/chimyz-B_T{:.3f}K.dat".format(Ts[iT])
        chimzxBname = "./output/chim_and_chie_vs_B_and_E/chimzx-B_T{:.3f}K.dat".format(Ts[iT])
        chimzyBname = "./output/chim_and_chie_vs_B_and_E/chimzy-B_T{:.3f}K.dat".format(Ts[iT])
        chimzzBname = "./output/chim_and_chie_vs_B_and_E/chimzz-B_T{:.3f}K.dat".format(Ts[iT])

        chimxxEname = "./output/chim_and_chie_vs_B_and_E/chimxx-E_T{:.3f}K.dat".format(Ts[iT])
        chimxyEname = "./output/chim_and_chie_vs_B_and_E/chimxy-E_T{:.3f}K.dat".format(Ts[iT])
        chimxzEname = "./output/chim_and_chie_vs_B_and_E/chimxz-E_T{:.3f}K.dat".format(Ts[iT])
        chimyxEname = "./output/chim_and_chie_vs_B_and_E/chimyx-E_T{:.3f}K.dat".format(Ts[iT])
        chimyyEname = "./output/chim_and_chie_vs_B_and_E/chimyy-E_T{:.3f}K.dat".format(Ts[iT])
        chimyzEname = "./output/chim_and_chie_vs_B_and_E/chimyz-E_T{:.3f}K.dat".format(Ts[iT])
        chimzxEname = "./output/chim_and_chie_vs_B_and_E/chimzx-E_T{:.3f}K.dat".format(Ts[iT])
        chimzyEname = "./output/chim_and_chie_vs_B_and_E/chimzy-E_T{:.3f}K.dat".format(Ts[iT])
        chimzzEname = "./output/chim_and_chie_vs_B_and_E/chimzz-E_T{:.3f}K.dat".format(Ts[iT])

        chiexxBname = "./output/chim_and_chie_vs_B_and_E/chiexx-B_T{:.3f}K.dat".format(Ts[iT])
        chiexyBname = "./output/chim_and_chie_vs_B_and_E/chiexy-B_T{:.3f}K.dat".format(Ts[iT])
        chiexzBname = "./output/chim_and_chie_vs_B_and_E/chiexz-B_T{:.3f}K.dat".format(Ts[iT])
        chieyxBname = "./output/chim_and_chie_vs_B_and_E/chieyx-B_T{:.3f}K.dat".format(Ts[iT])
        chieyyBname = "./output/chim_and_chie_vs_B_and_E/chieyy-B_T{:.3f}K.dat".format(Ts[iT])
        chieyzBname = "./output/chim_and_chie_vs_B_and_E/chieyz-B_T{:.3f}K.dat".format(Ts[iT])
        chiezxBname = "./output/chim_and_chie_vs_B_and_E/chiezx-B_T{:.3f}K.dat".format(Ts[iT])
        chiezyBname = "./output/chim_and_chie_vs_B_and_E/chiezy-B_T{:.3f}K.dat".format(Ts[iT])
        chiezzBname = "./output/chim_and_chie_vs_B_and_E/chiezz-B_T{:.3f}K.dat".format(Ts[iT])

        chiexxEname = "./output/chim_and_chie_vs_B_and_E/chiexx-E_T{:.3f}K.dat".format(Ts[iT])
        chiexyEname = "./output/chim_and_chie_vs_B_and_E/chiexy-E_T{:.3f}K.dat".format(Ts[iT])
        chiexzEname = "./output/chim_and_chie_vs_B_and_E/chiexz-E_T{:.3f}K.dat".format(Ts[iT])
        chieyxEname = "./output/chim_and_chie_vs_B_and_E/chieyx-E_T{:.3f}K.dat".format(Ts[iT])
        chieyyEname = "./output/chim_and_chie_vs_B_and_E/chieyy-E_T{:.3f}K.dat".format(Ts[iT])
        chieyzEname = "./output/chim_and_chie_vs_B_and_E/chieyz-E_T{:.3f}K.dat".format(Ts[iT])
        chiezxEname = "./output/chim_and_chie_vs_B_and_E/chiezx-E_T{:.3f}K.dat".format(Ts[iT])
        chiezyEname = "./output/chim_and_chie_vs_B_and_E/chiezy-E_T{:.3f}K.dat".format(Ts[iT])
        chiezzEname = "./output/chim_and_chie_vs_B_and_E/chiezz-E_T{:.3f}K.dat".format(Ts[iT])

        with open(chimxxBname, "w") as fxx, open(chimxyBname, "w") as fxy, open(chimxzBname, "w") as fxz, \
             open(chimyxBname, "w") as fyx, open(chimyyBname, "w") as fyy, open(chimyzBname, "w") as fyz, \
             open(chimzxBname, "w") as fzx, open(chimzyBname, "w") as fzy, open(chimzzBname, "w") as fzz:
            for iB in range(nB):
                fxx.write( "{:15.9f} ".format(Bs[iB]))
                fxy.write( "{:15.9f} ".format(Bs[iB]))
                fxz.write( "{:15.9f} ".format(Bs[iB]))
                fyx.write( "{:15.9f} ".format(Bs[iB]))
                fyy.write( "{:15.9f} ".format(Bs[iB]))
                fyz.write( "{:15.9f} ".format(Bs[iB]))
                fzx.write( "{:15.9f} ".format(Bs[iB]))
                fzy.write( "{:15.9f} ".format(Bs[iB]))
                fzz.write( "{:15.9f} ".format(Bs[iB]))
                for iE in range(nE):
                    fxx.write( "{:15.9f} ".format(chims[iT][iB][iE][0,0]))
                    fxy.write( "{:15.9f} ".format(chims[iT][iB][iE][0,1]))
                    fxz.write( "{:15.9f} ".format(chims[iT][iB][iE][0,2]))
                    fyx.write( "{:15.9f} ".format(chims[iT][iB][iE][1,0]))
                    fyy.write( "{:15.9f} ".format(chims[iT][iB][iE][1,1]))
                    fyz.write( "{:15.9f} ".format(chims[iT][iB][iE][1,2]))
                    fzx.write( "{:15.9f} ".format(chims[iT][iB][iE][2,0]))
                    fzy.write( "{:15.9f} ".format(chims[iT][iB][iE][2,1]))
                    fzz.write( "{:15.9f} ".format(chims[iT][iB][iE][2,2]))
                fxx.write( "\n" ); fxy.write( "\n" ); fxz.write( "\n" )
                fyx.write( "\n" ); fyy.write( "\n" ); fyz.write( "\n" )
                fzx.write( "\n" ); fzy.write( "\n" ); fzz.write( "\n" )

        with open(chimxxEname, "w") as fxx, open(chimxyEname, "w") as fxy, open(chimxzEname, "w") as fxz, \
             open(chimyxEname, "w") as fyx, open(chimyyEname, "w") as fyy, open(chimyzEname, "w") as fyz, \
             open(chimzxEname, "w") as fzx, open(chimzyEname, "w") as fzy, open(chimzzEname, "w") as fzz:
            for iE in range(nE):
                fxx.write( "{:15.9f} ".format(Es[iE]))
                fxy.write( "{:15.9f} ".format(Es[iE]))
                fxz.write( "{:15.9f} ".format(Es[iE]))
                fyx.write( "{:15.9f} ".format(Es[iE]))
                fyy.write( "{:15.9f} ".format(Es[iE]))
                fyz.write( "{:15.9f} ".format(Es[iE]))
                fzx.write( "{:15.9f} ".format(Es[iE]))
                fzy.write( "{:15.9f} ".format(Es[iE]))
                fzz.write( "{:15.9f} ".format(Es[iE]))
                for iB in range(nB):
                    fxx.write( "{:15.9f} ".format(chims[iT][iB][iE][0,0]))
                    fxy.write( "{:15.9f} ".format(chims[iT][iB][iE][0,1]))
                    fxz.write( "{:15.9f} ".format(chims[iT][iB][iE][0,2]))
                    fyx.write( "{:15.9f} ".format(chims[iT][iB][iE][1,0]))
                    fyy.write( "{:15.9f} ".format(chims[iT][iB][iE][1,1]))
                    fyz.write( "{:15.9f} ".format(chims[iT][iB][iE][1,2]))
                    fzx.write( "{:15.9f} ".format(chims[iT][iB][iE][2,0]))
                    fzy.write( "{:15.9f} ".format(chims[iT][iB][iE][2,1]))
                    fzz.write( "{:15.9f} ".format(chims[iT][iB][iE][2,2]))
                fxx.write( "\n" ); fxy.write( "\n" ); fxz.write( "\n" )
                fyx.write( "\n" ); fyy.write( "\n" ); fyz.write( "\n" )
                fzx.write( "\n" ); fzy.write( "\n" ); fzz.write( "\n" )

        with open(chiexxBname, "w") as fxx, open(chiexyBname, "w") as fxy, open(chiexzBname, "w") as fxz, \
             open(chieyxBname, "w") as fyx, open(chieyyBname, "w") as fyy, open(chieyzBname, "w") as fyz, \
             open(chiezxBname, "w") as fzx, open(chiezyBname, "w") as fzy, open(chiezzBname, "w") as fzz:
            for iB in range(nB):
                fxx.write( "{:15.9f} ".format(Bs[iB]))
                fxy.write( "{:15.9f} ".format(Bs[iB]))
                fxz.write( "{:15.9f} ".format(Bs[iB]))
                fyx.write( "{:15.9f} ".format(Bs[iB]))
                fyy.write( "{:15.9f} ".format(Bs[iB]))
                fyz.write( "{:15.9f} ".format(Bs[iB]))
                fzx.write( "{:15.9f} ".format(Bs[iB]))
                fzy.write( "{:15.9f} ".format(Bs[iB]))
                fzz.write( "{:15.9f} ".format(Bs[iB]))
                for iE in range(nE):
                    fxx.write( "{:15.9f} ".format(chies[iT][iB][iE][0,0]))
                    fxy.write( "{:15.9f} ".format(chies[iT][iB][iE][0,1]))
                    fxz.write( "{:15.9f} ".format(chies[iT][iB][iE][0,2]))
                    fyx.write( "{:15.9f} ".format(chies[iT][iB][iE][1,0]))
                    fyy.write( "{:15.9f} ".format(chies[iT][iB][iE][1,1]))
                    fyz.write( "{:15.9f} ".format(chies[iT][iB][iE][1,2]))
                    fzx.write( "{:15.9f} ".format(chies[iT][iB][iE][2,0]))
                    fzy.write( "{:15.9f} ".format(chies[iT][iB][iE][2,1]))
                    fzz.write( "{:15.9f} ".format(chies[iT][iB][iE][2,2]))
                fxx.write( "\n" ); fxy.write( "\n" ); fxz.write( "\n" )
                fyx.write( "\n" ); fyy.write( "\n" ); fyz.write( "\n" )
                fzx.write( "\n" ); fzy.write( "\n" ); fzz.write( "\n" )

        with open(chiexxEname, "w") as fxx, open(chiexyEname, "w") as fxy, open(chiexzEname, "w") as fxz, \
             open(chieyxEname, "w") as fyx, open(chieyyEname, "w") as fyy, open(chieyzEname, "w") as fyz, \
             open(chiezxEname, "w") as fzx, open(chiezyEname, "w") as fzy, open(chiezzEname, "w") as fzz:
            for iE in range(nE):
                fxx.write( "{:15.9f} ".format(Es[iE]))
                fxy.write( "{:15.9f} ".format(Es[iE]))
                fxz.write( "{:15.9f} ".format(Es[iE]))
                fyx.write( "{:15.9f} ".format(Es[iE]))
                fyy.write( "{:15.9f} ".format(Es[iE]))
                fyz.write( "{:15.9f} ".format(Es[iE]))
                fzx.write( "{:15.9f} ".format(Es[iE]))
                fzy.write( "{:15.9f} ".format(Es[iE]))
                fzz.write( "{:15.9f} ".format(Es[iE]))
                for iB in range(nB):
                    fxx.write( "{:15.9f} ".format(chies[iT][iB][iE][0,0]))
                    fxy.write( "{:15.9f} ".format(chies[iT][iB][iE][0,1]))
                    fxz.write( "{:15.9f} ".format(chies[iT][iB][iE][0,2]))
                    fyx.write( "{:15.9f} ".format(chies[iT][iB][iE][1,0]))
                    fyy.write( "{:15.9f} ".format(chies[iT][iB][iE][1,1]))
                    fyz.write( "{:15.9f} ".format(chies[iT][iB][iE][1,2]))
                    fzx.write( "{:15.9f} ".format(chies[iT][iB][iE][2,0]))
                    fzy.write( "{:15.9f} ".format(chies[iT][iB][iE][2,1]))
                    fzz.write( "{:15.9f} ".format(chies[iT][iB][iE][2,2]))
                fxx.write( "\n" ); fxy.write( "\n" ); fxz.write( "\n" )
                fyx.write( "\n" ); fyy.write( "\n" ); fyz.write( "\n" )
                fzx.write( "\n" ); fzy.write( "\n" ); fzz.write( "\n" )

    return



### =================================================================================
### Magnetic susceptibility and electric susceptibility vs temperature
### under different B fields and E fields
### =================================================================================

def get_chim_and_chie_vs_T(spins, h_ex, h_ani, BET_Tgrid, dB, dE):

    """
    Bs[1:nB]: Magnitudes of B field enumerated
    Bs[-2]: theta_B
    Bs[-1]: phi_B
    Es: similar to Bs
    sampleT[0]: Tmin1
    sampleT[1]: Tmax1
    sampleT[2]: Tstep1
    sampleT[3]: Tmin2
    sampleT[4]: Tmax2
    sampleT[5]: Tstep2
    ...
    sampleT[3n-2]: Tminn
    sampleT[3n-1]: Tmaxn
    sampleT[3n]: Tstepn
    chim-T_Bxxx_Eyyy.dat
      for B=xxx Tesla and E=yyy mV/Ang
      rows: temperature
    chie-T_Bxxx_Eyyy.dat: similar to chim-T_Bxxx_Eyyy.dat
    """

    Bs = BET_Tgrid[0]
    Es = BET_Tgrid[1]
    sampleT = BET_Tgrid[2]

    nB = len(Bs) - 2
    nE = len(Es) - 2

    Ts = sample_temperature(sampleT)
    nT = len(Ts)

    chims = []; chies = []
    for iT in range(nT):
        chims.append([]); chies.append([])
        for iB in range(nB):
            chim_vs_E = []; chie_vs_E = []
            for iE in range(nE):
                print("T = {:4.0f} B = {:6.2f} E = {:6.2f}".format(Ts[iT], Bs[iB], Es[iE]))
                B0v_sph = [Bs[iB], Bs[-2], Bs[-1]]
                E0v_sph = [Es[iE], Es[-2], Es[-1]]
                chim = get_chim_tensor(spins, h_ex, h_ani, B0v_sph, E0v_sph, Ts[iT], dB)
                chie = get_chie_tensor(spins, h_ex, h_ani, B0v_sph, E0v_sph, Ts[iT], dE)
                chim_vs_E.append(chim); chie_vs_E.append(chie)
            chims[iT].append(chim_vs_E); chies[iT].append(chie_vs_E)

    if not os.path.exists("./output/chim_and_chie_vs_T"):
        subprocess.run(["mkdir", "-p", "./output/chim_and_chie_vs_T"])

    for iB in range(nB):
        for iE in range(nE):
            #print("B = {:6.3f} T, E = {:6.3f} mV/Ang".format(Bs[iB], Es[iE]))

            f1xxname = "./output/chim_and_chie_vs_T/chimxx-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f1xyname = "./output/chim_and_chie_vs_T/chimxy-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f1xzname = "./output/chim_and_chie_vs_T/chimxz-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f1yxname = "./output/chim_and_chie_vs_T/chimyx-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f1yyname = "./output/chim_and_chie_vs_T/chimyy-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f1yzname = "./output/chim_and_chie_vs_T/chimyz-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f1zxname = "./output/chim_and_chie_vs_T/chimzx-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f1zyname = "./output/chim_and_chie_vs_T/chimzy-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f1zzname = "./output/chim_and_chie_vs_T/chimzz-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])

            f2xxname = "./output/chim_and_chie_vs_T/chimxx-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f2xyname = "./output/chim_and_chie_vs_T/chimxy-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f2xzname = "./output/chim_and_chie_vs_T/chimxz-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f2yxname = "./output/chim_and_chie_vs_T/chimyx-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f2yyname = "./output/chim_and_chie_vs_T/chimyy-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f2yzname = "./output/chim_and_chie_vs_T/chimyz-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f2zxname = "./output/chim_and_chie_vs_T/chimzx-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f2zyname = "./output/chim_and_chie_vs_T/chimzy-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])
            f2zzname = "./output/chim_and_chie_vs_T/chimzz-T_B{:.3f}T_E{:.3f}mVperAng.dat".format(Bs[iB], Es[iE])

            with open(f1xxname, "w") as f1xx, open(f1xyname, "w") as f1xy, open(f1xzname, "w") as f1xz, \
                 open(f1yxname, "w") as f1yx, open(f1yyname, "w") as f1yy, open(f1yzname, "w") as f1yz, \
                 open(f1zxname, "w") as f1zx, open(f1zyname, "w") as f1zy, open(f1zzname, "w") as f1zz:
                for iT in range(nT):
                    f1xx.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chims[iT][iB][iE][0,0]))
                    f1xy.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chims[iT][iB][iE][0,1]))
                    f1xz.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chims[iT][iB][iE][0,2]))
                    f1yx.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chims[iT][iB][iE][1,0]))
                    f1yy.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chims[iT][iB][iE][1,1]))
                    f1yz.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chims[iT][iB][iE][1,2]))
                    f1zx.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chims[iT][iB][iE][2,0]))
                    f1zy.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chims[iT][iB][iE][2,1]))
                    f1zz.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chims[iT][iB][iE][2,2]))

            with open(f2xxname, "w") as f2xx, open(f2xyname, "w") as f2xy, open(f2xzname, "w") as f2xz, \
                 open(f2yxname, "w") as f2yx, open(f2yyname, "w") as f2yy, open(f2yzname, "w") as f2yz, \
                 open(f2zxname, "w") as f2zx, open(f2zyname, "w") as f2zy, open(f2zzname, "w") as f2zz:
                for iT in range(nT):
                    f2xx.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chies[iT][iB][iE][0,0]))
                    f2xy.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chies[iT][iB][iE][0,1]))
                    f2xz.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chies[iT][iB][iE][0,2]))
                    f2yx.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chies[iT][iB][iE][1,0]))
                    f2yy.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chies[iT][iB][iE][1,1]))
                    f2yz.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chies[iT][iB][iE][1,2]))
                    f2zx.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chies[iT][iB][iE][2,0]))
                    f2zy.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chies[iT][iB][iE][2,1]))
                    f2zz.write("{:15.9f} {:15.9f}\n".format(Ts[iT], chies[iT][iB][iE][2,2]))
    return

   
   
### =================================================================================
### First and second order derivatives of magnetization against B field
### vs B field and E field at different temperatures
### =================================================================================

def get_dMdB_and_dM2dB2_vs_B_and_E(spins, h_ex, h_ani, BET_BEgrid, dB):

    """
    sampleB[0]: Bmin
    sampleB[1]: Bmax
    sampleB[2]: Bstep
    sampleB[3]: theta_B in deg
    sampleB[4]: phi_B in deg
    sampleE: similar to sampleB
    Ts: temperatures enumerated
    dB: small step of B field, 0.0001 -- 0.001 T is usually good.
    dMdB-B_Txxx.dat / d2MdB2-B_Txxx.dat:
      for temperature T=xxx Kelvin
      rows: B field
      columns: E field
    dMdB-E_Txxx.dat / d2MdB2-E_Txxx.dat:
      for temperature T=xxx Kelvin
      rows: E field
      columns: B field
      dMdB-E_Txxx.dat is the transpose of dMdB-B_Txxx.dat for the ease of plotting
    """

    sampleB = BET_BEgrid[0]
    sampleE = BET_BEgrid[1]
    Ts = BET_BEgrid[2]

    nT = len(Ts)

    nB = int((sampleB[1]-sampleB[0])/sampleB[2]) + 1
    Bs = np.linspace(sampleB[0], sampleB[1], nB, endpoint=True)

    nE = int((sampleE[1]-sampleE[0])/sampleE[2]) + 1
    Es = np.linspace(sampleE[0], sampleE[1], nE, endpoint=True)

    dBv_sph = [dB, sampleB[3], sampleB[4]]

    dMdBs = []; d2MdB2s = []
    for iT in range(nT):
        dMdBs.append([]); d2MdB2s.append([])
        for iB in range(nB):
            dMdB_vs_E = []; d2MdB2_vs_E = []
            for iE in range(nE):
                B0v_sph = [Bs[iB], sampleB[3], sampleB[4]]
                E0v_sph = [Es[iE], sampleE[3], sampleE[4]]
                print("T = {:4.0f} B = {:6.2f} E = {:6.2f}".format(Ts[iT], Bs[iB], Es[iE]))
                dMdB = get_dMdB(spins, h_ex, h_ani, B0v_sph, E0v_sph, Ts[iT], dBv_sph, False)
                d2MdB2 = get_d2MdB2(spins, h_ex, h_ani, B0v_sph, E0v_sph, Ts[iT], dBv_sph, False)
                dMdB_vs_E.append(dMdB); d2MdB2_vs_E.append(d2MdB2)
            dMdBs[iT].append(dMdB_vs_E); d2MdB2s[iT].append(d2MdB2_vs_E)

    if not os.path.exists("./output/dMdB_and_d2MdB2_vs_B_and_E"):
        subprocess.run(["mkdir", "-p", "./output/dMdB_and_d2MdB2_vs_B_and_E"])

    for iT in range(nT):
        #print("T = {:6.3f} K".format(Ts[iT]))

        F1name  = "./output/dMdB_and_d2MdB2_vs_B_and_E/dMdBmod-B_T{:.3f}K.dat".format(Ts[iT])
        F1xname = "./output/dMdB_and_d2MdB2_vs_B_and_E/dMxdB-B_T{:.3f}K.dat".format(Ts[iT])
        F1yname = "./output/dMdB_and_d2MdB2_vs_B_and_E/dMydB-B_T{:.3f}K.dat".format(Ts[iT])
        F1zname = "./output/dMdB_and_d2MdB2_vs_B_and_E/dMzdB-B_T{:.3f}K.dat".format(Ts[iT])

        F2name  = "./output/dMdB_and_d2MdB2_vs_B_and_E/dMdBmod-E_T{:.3f}K.dat".format(Ts[iT])
        F2xname = "./output/dMdB_and_d2MdB2_vs_B_and_E/dMxdB-E_T{:.3f}K.dat".format(Ts[iT])
        F2yname = "./output/dMdB_and_d2MdB2_vs_B_and_E/dMydB-E_T{:.3f}K.dat".format(Ts[iT])
        F2zname = "./output/dMdB_and_d2MdB2_vs_B_and_E/dMzdB-E_T{:.3f}K.dat".format(Ts[iT])

        G1name  = "./output/dMdB_and_d2MdB2_vs_B_and_E/d2MdB2mod-B_T{:.3f}K.dat".format(Ts[iT])
        G1xname = "./output/dMdB_and_d2MdB2_vs_B_and_E/d2MxdB2-B_T{:.3f}K.dat".format(Ts[iT])
        G1yname = "./output/dMdB_and_d2MdB2_vs_B_and_E/d2MydB2-B_T{:.3f}K.dat".format(Ts[iT])
        G1zname = "./output/dMdB_and_d2MdB2_vs_B_and_E/d2MzdB2-B_T{:.3f}K.dat".format(Ts[iT])

        G2name  = "./output/dMdB_and_d2MdB2_vs_B_and_E/d2MdB2mod-E_T{:.3f}K.dat".format(Ts[iT])
        G2xname = "./output/dMdB_and_d2MdB2_vs_B_and_E/d2MxdB2-E_T{:.3f}K.dat".format(Ts[iT])
        G2yname = "./output/dMdB_and_d2MdB2_vs_B_and_E/d2MydB2-E_T{:.3f}K.dat".format(Ts[iT])
        G2zname = "./output/dMdB_and_d2MdB2_vs_B_and_E/d2MzdB2-E_T{:.3f}K.dat".format(Ts[iT])

        with open(F1name,  "w") as F1 , open(F2name,  "w") as F2 , \
             open(F1xname, "w") as F1x, open(F2xname, "w") as F2x, \
             open(F1yname, "w") as F1y, open(F2yname, "w") as F2y, \
             open(F1zname, "w") as F1z, open(F2zname, "w") as F2z, \
             open(G1name,  "w") as G1 , open(G2name,  "w") as G2 , \
             open(G1xname, "w") as G1x, open(G2xname, "w") as G2x, \
             open(G1yname, "w") as G1y, open(G2yname, "w") as G2y, \
             open(G1zname, "w") as G1z, open(G2zname, "w") as G2z:
            for iB in range(nB):
                F1.write( "{:15.9f} ".format(Bs[iB]))
                F1x.write("{:15.9f} ".format(Bs[iB]))
                F1y.write("{:15.9f} ".format(Bs[iB]))
                F1z.write("{:15.9f} ".format(Bs[iB]))
                G1.write( "{:15.9f} ".format(Bs[iB]))
                G1x.write("{:15.9f} ".format(Bs[iB]))
                G1y.write("{:15.9f} ".format(Bs[iB]))
                G1z.write("{:15.9f} ".format(Bs[iB]))
                for iE in range(nE):
                    F1.write( "{:15.9f} ".format(np.linalg.norm(dMdBs[iT][iB][iE])))
                    F1x.write("{:15.9f} ".format(dMdBs[iT][iB][iE][0]))
                    F1y.write("{:15.9f} ".format(dMdBs[iT][iB][iE][1]))
                    F1z.write("{:15.9f} ".format(dMdBs[iT][iB][iE][2]))
                    G1.write( "{:15.9f} ".format(np.linalg.norm(d2MdB2s[iT][iB][iE])))
                    G1x.write("{:15.9f} ".format(d2MdB2s[iT][iB][iE][0]))
                    G1y.write("{:15.9f} ".format(d2MdB2s[iT][iB][iE][1]))
                    G1z.write("{:15.9f} ".format(d2MdB2s[iT][iB][iE][2]))
                F1.write("\n"); F1x.write("\n"); F1y.write("\n"); F1z.write("\n")
                G1.write("\n"); G1x.write("\n"); G1y.write("\n"); G1z.write("\n")
            for iE in range(nE):
                F2.write( "{:15.9f} ".format(Es[iE]))
                F2x.write("{:15.9f} ".format(Es[iE]))
                F2y.write("{:15.9f} ".format(Es[iE]))
                F2z.write("{:15.9f} ".format(Es[iE]))
                G2.write( "{:15.9f} ".format(Es[iE]))
                G2x.write("{:15.9f} ".format(Es[iE]))
                G2y.write("{:15.9f} ".format(Es[iE]))
                G2z.write("{:15.9f} ".format(Es[iE]))
                for iB in range(nB):
                    F2.write( "{:15.9f} ".format(np.linalg.norm(dMdBs[iT][iB][iE])))
                    F2x.write("{:15.9f} ".format(dMdBs[iT][iB][iE][0]))
                    F2y.write("{:15.9f} ".format(dMdBs[iT][iB][iE][1]))
                    F2z.write("{:15.9f} ".format(dMdBs[iT][iB][iE][2]))
                    G2.write( "{:15.9f} ".format(np.linalg.norm(d2MdB2s[iT][iB][iE])))
                    G2x.write("{:15.9f} ".format(d2MdB2s[iT][iB][iE][0]))
                    G2y.write("{:15.9f} ".format(d2MdB2s[iT][iB][iE][1]))
                    G2z.write("{:15.9f} ".format(d2MdB2s[iT][iB][iE][2]))
                F2.write("\n"); F2x.write("\n"); F2y.write("\n"); F2z.write("\n")
                G2.write("\n"); G2x.write("\n"); G2y.write("\n"); G2z.write("\n")
    return



### =================================================================================
### Functions for fitting experimental magnetization
### =================================================================================

def read_exp_magnetization(path, fname):

    Ms = np.loadtxt(path + fname)
    Bs = Ms[:, 0]
    Ms = Ms[:, 1]
    nB = Ms.shape[0]

    return (nB, Bs, Ms)
    
def set_exchange(exchange, fit_problem):

    for i_par in range(fit_problem["n_par"]):
        i_pair = fit_problem["elements_to_vary"][i_par]["pair"] - 1
        element = fit_problem["elements_to_vary"][i_par]["element"]
        index = element[0]*3 + element[1]
        exchange[i_pair]['coupling_matrix'][index] = fit_problem["current_values"][i_par]

    for i_constrain in range(len(fit_problem["constrain_elements"])):
        i_pair  = fit_problem["constrain_elements"][i_constrain]["pair"] - 1
        element = fit_problem["constrain_elements"][i_constrain]["element1"]
        index1  = element[0]*3 + element[1]
        element = fit_problem["constrain_elements"][i_constrain]["element2"]
        index2  = element[0]*3 + element[1]
        factor  = fit_problem["constrain_elements"][i_constrain]["factor"]
        exchange[i_pair]['coupling_matrix'][index2] = factor * exchange[i_pair]['coupling_matrix'][index1]

    for i_constrain in range(len(fit_problem["constrain_pairs"])):
        i_pair = fit_problem["constrain_pairs"][i_constrain][0] - 1
        for i_j_pair in range(1, len(fit_problem["constrain_pairs"][i_constrain])):
            j_pair = fit_problem["constrain_pairs"][i_constrain][i_j_pair] - 1
            exchange[j_pair]['coupling_matrix'] = exchange[i_pair]['coupling_matrix']

    return exchange

def set_anisotropy(anisotropy, fit_problem):

    for i_par in range(fit_problem["n_par"]):
        i_site = fit_problem["elements_to_vary"][i_par]["site"] - 1
        i_Bkq = fit_problem["elements_to_vary"][i_par]["i_Bkq"]
        if fit_problem["elements_to_vary"][i_par]["is_B22"]:
            # B20 is the entry right before the entry of B20/B22
            anisotropy[i_site]['Bkqs'][i_Bkq] = anisotropy[i_site]['Bkqs'][i_Bkq-1] * fit_problem["current_values"][i_par]
        else:
            anisotropy[i_site]['Bkqs'][i_Bkq] = fit_problem["current_values"][i_par]

    for i_constrain in range(len(fit_problem["constrain_sites"])):
        i_site = fit_problem["constrain_sites"][i_constrain][0] - 1
        for i_j_site in range(1, len(fit_problem["constrain_sites"][i_constrain])):
            j_site = fit_problem["constrain_sites"][i_constrain][i_j_site] - 1
            anisotropy[j_site]['Bkqs'] = anisotropy[i_site]['Bkqs']

    return anisotropy



## =================================================================
## Functions transition matrix elements
## =================================================================

def get_transition_strength_for_one_pair(spins, eigen, theta_B, phi_B, i=0, j=1):
    # Get transition strength | <j| H_zee @ B_MW_x/y/z = 1 T |i> |^2, where MW stands for microwave.
    # For external static B field (B, theta_B, phi_B). The magnitude B is implicitly in eigenvectors.

    ge = 2.0023 # g factor

    oneTeslamuB2wavenumber = 0.46686449369202904

    # Parallel microwave B field
    Bx_para, By_para, Bz_para = sph2cart_deg([oneTeslamuB2wavenumber, theta_B, phi_B])
    h_zee_para = -spins.Mv_tot[0]*Bx_para + -spins.Mv_tot[1]*By_para + -spins.Mv_tot[2]*Bz_para

    # Perpendicular microwave B field
    if theta_B >= 90: 
        theta_B = theta_B - 90
    else:
        theta_B = theta_B + 90
    Bx_perp_1, By_perp_1, Bz_perp_1 = sph2cart_deg([oneTeslamuB2wavenumber, theta_B, phi_B])
    h_zee_perp_1 = -spins.Mv_tot[0]*Bx_perp_1 + -spins.Mv_tot[1]*By_perp_1 + -spins.Mv_tot[2]*Bz_perp_1

    Bx_perp_2, By_perp_2, Bz_perp_2 = np.cross([Bx_para, By_para, Bz_para], [Bx_perp_1, By_perp_1, Bz_perp_1])/oneTeslamuB2wavenumber
    h_zee_perp_2 = -spins.Mv_tot[0]*Bx_perp_2 + -spins.Mv_tot[1]*By_perp_2 + -spins.Mv_tot[2]*Bz_perp_2

    state_i = eigen.eigenvectors[:, eigen.indices[i]]
    state_j = eigen.eigenvectors[:, eigen.indices[j]]
    h_zee_para_ij = np.dot(np.conjugate(state_j), np.matmul(h_zee_para, state_i))
    h_zee_perp_1_ij = np.dot(np.conjugate(state_j), np.matmul(h_zee_perp_1, state_i))
    h_zee_perp_2_ij = np.dot(np.conjugate(state_j), np.matmul(h_zee_perp_2, state_i))

    ts_para = np.real( np.conjugate(h_zee_para_ij) * h_zee_para_ij  )
    ts_perp_1 = np.real( np.conjugate(h_zee_perp_1_ij) * h_zee_perp_1_ij )
    ts_perp_2 = np.real( np.conjugate(h_zee_perp_2_ij) * h_zee_perp_2_ij )
    ts_perp = ts_perp_1 + ts_perp_2

    #print("{:6.2f} {:6.2f} {:15.9f} {:15.9f} {:15.9f}".format(theta_B, phi_B, ts_perp_1, ts_perp_2, ts_para))

    return (ts_para, ts_perp)



def get_transition_strength_vs_B_for_one_pair(spins, h0, BET_Bgrid, i=0, j=1):

    """  
    Bgrid[0]: Bmin
    Bgrid[1]: Bmax
    Bgrid[2]: Bstep
    Bgrid[3]: theta_B in deg
    Bgrid[4]: phi_B in deg
    Bs: Magnitudes of B field
    theta_B: polar angle of B field in deg
    phi_B: azimuthal angle of B field in deg
    E: Magnitude of E field in meV/Ang
    theta_E: polar angle of E field in deg
    phi_E: azimuthal angle of E field in deg
    h0: h_ex + h_ani, or h_ani
    """

    Bgrid = BET_Bgrid[0]
    nB = int((Bgrid[1]-Bgrid[0])/Bgrid[2]) + 1
    Bs = np.linspace(Bgrid[0], Bgrid[1], nB, endpoint=True)
    theta_B = Bgrid[3]
    phi_B = Bgrid[4]

    ts_para = []
    ts_perp = []

    h_stark = get_h_Stark( spins, BET_Bgrid[1], "spherical")
    h0 = h0 + h_stark

    for iB in range(nB):
        B = Bs[iB]
        h_zee   = get_h_Zeeman(spins, [B, theta_B, phi_B], "spherical")
        h = h0 + h_zee
        eigen = eigen_spin_hamiltonian(h)
        t_para, t_perp = get_transition_strength_for_one_pair(spins, eigen, theta_B, phi_B, i=i, j=j)
        ts_para.append(t_para)
        ts_perp.append(t_perp)

    if not os.path.exists("./output"):
        subprocess.run(["mkdir", "./output"])

    with open("./output/transition_strength_{:d}-{:d}_vs_B.dat".format(i, j), "w") as f:
        f.write("# B (T)   para. transition strength   perp. transition strength\n")
        for iB in range(nB):
            f.write("{:8.4f} {:12.6E} {:12.6E}\n".format(Bs[iB], ts_para[iB], ts_perp[iB]))

