import os
import subprocess
import copy
import math
import yaml
import numpy as np
import pandas as pd
from qdmag.core.constants import Tesla2wavenumber, Kelvin2wavenumber
from qdmag.core.Operators import Operator
from qdmag.core.StevensOperators import StevensOpA

# ========================
# Classes
# ========================

class one_local_spin:
    """ Spin operators relevant to one local spin. """
    def __init__(self, S=0.5):
        # Spin value
        self.S = S
        # Multiplicity of the spin
        self.dim = int( 2*S + 1 ) 
        # Sx , Sy, Sz: spin operators in the local space.
        self.Sx = Operator.Jx(S).O
        self.Sy = Operator.Jy(S).O
        self.Sz = Operator.Jz(S).O
        self.Sv = [self.Sx, self.Sy, self.Sz] 
        # Zero operator in the local space.
        self.zero = np.zeros((self.dim, self.dim), dtype=complex)
        # Identity operator in the local space.
        self.ID = np.eye(self.dim, dtype=complex)
        # A commonly used constant
        self.ss = S*(S + 1)

class many_spins:
    """ Spin operators relevant to several spins """
    def __init__(self, Ss, nS, gfactor):
        self.Ss = Ss
        self.nS = nS
        self.gfactor = gfactor
        self.get_Smax()
        self.get_local_spins()
        self.dim = np.prod([self.local_spins[i].dim for i in range(self.nS)])
        # Zero operator in the whole Hilbert space.
        self.zero = np.zeros((self.dim, self.dim),dtype=complex)
        # IDs: unit operators in the local spaces.
        self.IDs = [self.local_spins[i].ID for i in range(nS)]
        # ID: unit operator in the whole space.
        self.ID = get_kronecker_product(self.IDs, self.nS)
        # get_global_spins depends on self.ID.
        self.get_global_spins()
        self.get_total_spin()
        self.get_global_magmoms()
        self.get_total_magmom()

    def get_Smax(self):
        """
        Get the total spin when all local spins are aligned.
        """
        self.Smax = sum(self.Ss)
        
    def get_local_spins(self):
        """
        Get a list of one_local_spin objects.
        All local operators are capsulated in the spins variable except IDs. 
        """
        self.local_spins = []
        for i in range(self.nS):
            self.local_spins.append(one_local_spin(S=self.Ss[i]))
    
    def get_global_spins(self):
        """
        Local spin operators in the whole Hilbert space.
        """
        ## List of vectors (as lists) of spin operators. 
        self.global_spins = []
        for i in range(self.nS):
            self.global_spins.append([])
            for j in range(3):
                ops = copy.deepcopy(self.IDs)
                ops[i] = self.local_spins[i].Sv[j]
                self.global_spins[i].append( get_kronecker_product(ops, self.nS) )
    
    def get_total_spin(self):
        """
        Total spin operators in the whole Hilbert space.
        """
        self.Sv_tot = [self.zero for i in range(3)]
        for i in range(3):
            for j in range(self.nS):
                self.Sv_tot[i] = self.Sv_tot[i] + self.global_spins[j][i]
        self.S2_tot = np.matmul(self.Sv_tot[0], self.Sv_tot[0]) + \
                      np.matmul(self.Sv_tot[1], self.Sv_tot[1]) + \
                      np.matmul(self.Sv_tot[2], self.Sv_tot[2])

    def get_global_magmoms(self):
        """
        Local magnetic moment operators in the whole Hilbert space.
        """
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
        r"""
        Total magnetic moment operator in the whole Hilbert space.
        \vec{M} = \sum_{i=1}^{nS} \vec{\mu}_i = \sum_{i=1}^{nS} matmul(g_s[i], \vec{S}_i)
        In general, each component of \vec{M} is a matrix of complex numbers.
        """
        self.Mv_tot = [self.zero for i in range(3)]
        for i in range(3):
            for j in range(self.nS):
                self.Mv_tot[i] = self.Mv_tot[i] + self.global_magmoms[j][i]

class eigen_handy:
    """
    A handy class for bookkeeping the eigenvalues and eigenvectors of a (spin) Hamiltonian.
    """
    def __init__(self, hamiltonian):
        ## Unit for eigenvalues: wavenumber
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(hamiltonian) 
        self.eigenvalues = np.real(self.eigenvalues)
        self.dim = self.eigenvalues.shape[0]
        # The eigenvalues by np.linalg.eigh is already in the assending order.
        self.indices = np.arange(self.dim) # np.argsort(self.eigenvalues)
        self.eigenvalues_offset = self.eigenvalues - self.eigenvalues[self.indices[0]]

class eigen_simple:
    """
    A simple class for bookkeeping the eigenvalues and eigenvectors of a (spin) Hamiltonian.
    """
    def __init__(self, hamiltonian):
        ## Unit for eigenvalues: wavenumber
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(hamiltonian) 
        self.eigenvalues = np.real(self.eigenvalues)
        self.dim = self.eigenvalues.shape[0]
    
    

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

## =================================================================
## Functions for input and output.
## =================================================================

def read_input():
    """
    Read input options from input.yaml file.
    """
    with open("./input.yaml", "r") as f:
        data = yaml.safe_load(f)
    Ss = data['spins']
    nS = len(Ss)
    if 'exchange' in data:
        exchange = data['exchange']
    else:
        exchange = []
    if 'anisotropy' in data:
        anisotropy = data['anisotropy']
    else:
        anisotropy = []
    gfactor = data['gfactor']
    BT_Bgrid = data['BT_Bgrid']
    BT_Tgrid = data['BT_Tgrid']
    dynamics = data['dynamics']
    if 'states' in data:
        states = data['states']
    else:
        states = []
    n_threads = data['n_threads']
    return (Ss, nS, exchange, anisotropy, gfactor, BT_Bgrid, BT_Tgrid, dynamics, states, n_threads)

def create_outdir():
    """
    Create the output directory if it does not exist.
    """
    if not os.path.exists("./output"):
        subprocess.run(["mkdir", "-p", "./output"])

def save_eigenvalues(eigen, offset=True):
    """
    Save eigenvalues to a file.
    """
    if offset:
        eigenvalues = eigen.eigenvalues_offset
    else:
        eigenvalues = eigen.eigenvalues
    create_outdir()
    with open("./output/eigenvalues.dat", "w") as f:
        for i in range(eigen.dim):
            f.write("{:16.12f}\n".format(eigenvalues[i]))
    print("The eigenvalues are saved in ./output/eigenvalues.dat")

def save_eigenvectors(eigen):
    """
    Save eigenvectors to a file.
    """
    create_outdir()
    with open("./output/eigenvectors.dat", "w") as f:
        for i in range(eigen.dim):
            state = eigen.eigenvectors[:, i]
            f.write((eigen.dim*" {:16.12f}" + "\n").format(*state))
    print("The eigenvectors are saved in ./output/eigenvectors.dat")

def save_operator(op, base_name):
    """
    Save the given operator to a file.
    """
    n = op.shape[0]
    o_real = np.real(op)
    o_imag = np.imag(op)
    o_mod  = np.sqrt(o_real**2 + o_imag**2)
    create_outdir()
    with open("./output/{:s}.txt".format(base_name), "w") as f:
        f.write("# Matrix elements of {:s} operator\n".format(base_name))
        f.write("# i, j, real, imag, modulus\n")
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.3e} {:12.3e {:12.3e}}\n".format(i, j, o_real[i, j], o_imag[i, j], o_mod[i, j]))
    print("The {:s} operator is saved in ./output/{:s}.txt".format(base_name, base_name))

def save_spins(spins, eigen):
    """
    Calculate and save expectation of spins for all eigenvectors.
    """
    create_outdir()
    with open("./output/spins.dat", "w") as f:
        f.write("## s1x, s1y, s1z, s2x, s2y, s2z, ..., snx, sny, snz, sx_tot, sy_tot, sz_tot, s2_tot, s_tot # eigenvalue (cm^-1)\n")

        for i in range(spins.dim):
            # E_local_spins, E_Sv_tot, E_S2_tot, E_S_tot = get_expectation_of_spins(spins, eigen.eigenvectors[:, eigen.indices[i]])
            E_local_spins, E_Sv_tot, E_S2_tot, E_S_tot = get_expectation_of_spins(spins, eigen.eigenvectors[:, i])
            for i_site in range(spins.nS):
                for alpha in range(3):
                    f.write("{:8.3f} ".format(E_local_spins[i_site][alpha]))
            for alpha in range(3):
                f.write("{:12.8f} ".format(E_Sv_tot[alpha]))
            f.write("{:8.3f} ".format(E_S2_tot))
            f.write("{:8.3f} ".format(E_S_tot))
            # f.write("# {:8.2f}\n".format(eigen.eigenvalues_offset[eigen.indices[i]]))
            f.write("# {:8.2f}\n".format(eigen.eigenvalues_offset[i]))
    print("The expectation values of spins are saved in ./output/spins.dat")

def print_emat_array(emat):
    """
    Print the emat array in a format suitable for use in a Python script.
    """
    print("emats_file[0] = [ \\")
    print("        [ {:15.8f}, {:15.8f}, {:15.8f} ], \\".format(*emat[0]))
    print("        [ {:15.8f}, {:15.8f}, {:15.8f} ], \\".format(*emat[1]))
    print("        [ {:15.8f}, {:15.8f}, {:15.8f} ], \\".format(*emat[2]))
    print("        ]")

## =================================================================
## Functions for magnetic exchange interaction.
## =================================================================

def get_h_exchange_one_pair(spins, spin_pair, factor):
    """
    Get the exchange Hamiltonian for one pair of spins.
    """
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
    """
    Get the exchange Hamiltonian for all pairs of spins.
    """
    n_pair = len(exchange)
    if n_pair == 0:
        return spins.zero
    h_ex = spins.zero
    for i_pair in range(n_pair):
        h_ex = h_ex + get_h_exchange_one_pair(spins, exchange[i_pair], factor)
    return h_ex

def get_h_exchange_iso_one_pair(spins, spin_pair, factor):
    """
    Get the isotropic exchange Hamiltonian for one pair of spins.
    """
    i, j = spin_pair['pair']
    Si = spins.local_spins[i-1].Sv
    Sj = spins.local_spins[j-1].Sv
    Jprime = spin_pair['coupling_matrix']
    Jprime = np.array(Jprime).reshape((3,3))
    J_iso = np.mean(np.diagonal(Jprime))
    Jprime = J_iso * np.eye(3)
    A = spin_pair['reference_frame']
    A = np.array(A).reshape((3,3))
    Siprime = get_Sprime(A, Si)
    Sjprime = get_Sprime(A, Sj)
    h_ex = spins.zero
    for ii in range(3):
        jj = ii
        ops = copy.deepcopy(spins.IDs)
        ops[i-1] = Siprime[ii]
        ops[j-1] = Sjprime[jj]
        ## Jprime[ii,jj] * Siprime[ii] * Sjprime[jj]
        h_ex = h_ex + Jprime[ii,jj] * get_kronecker_product(ops, spins.nS)
    return factor*h_ex

def get_h_exchange_iso(spins, exchange, factor):
    """
    Get the isotropic exchange Hamiltonian for all pairs of spins.
    """
    n_pair = len(exchange)
    if n_pair == 0:
        return spins.zero
    h_ex = spins.zero
    for i_pair in range(n_pair):
        h_ex = h_ex + get_h_exchange_iso_one_pair(spins, exchange[i_pair], factor)
    return h_ex

## =================================================================
## Functions for single-ion magnetic anisotropy.
## =================================================================

def get_h_anisotropy_one_site(spins, site):
    """
    Get the zero-field splitting (ZFS) Hamiltonian for one site.
    """
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
    """
    Get the zero-field splitting (ZFS) Hamiltonian for all sites.
    """
    h_ani = spins.zero
    if len(anisotropy) == 0:
        return h_ani
    for i_site in range(spins.nS):
        h_ani = h_ani + get_h_anisotropy_one_site(spins, anisotropy[i_site])
    return h_ani
    
def get_h_anisotropy_one_site_ikq(spins, site, ikq):
    """
    Get the zero-field splitting (ZFS) Hamiltonian for one site due to 
    a specific Stevens' operator or crystal field parameter (CFP).
    """
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
    if ikq < 0:
        print("ikq should be non-negative. Stopping ...")
        exit()
    if ikq >= nB:
        print("ikq is out of range. Stopping ...")
        exit()
    ops[i-1] = StevensOpA(S, ks[ikq], qs[ikq], A)
    h_ani = h_ani + Bkqs[ikq]*get_kronecker_product(ops, spins.nS)
    return h_ani

def get_h_anisotropy_ikq(spins, anisotropy, ikq):
    """
    Get the zero-field splitting (ZFS) Hamiltonian for all sites due to
    a specific Stevens' operator or crystal field parameter (CFP).
    """
    h_ani = spins.zero
    if len(anisotropy) == 0:
        return h_ani
    for i_site in range(spins.nS):
        h_ani = h_ani + get_h_anisotropy_one_site_ikq(spins, anisotropy[i_site], ikq)
    return h_ani
    
## =================================================================
## Functions for the Zeeman term.
## =================================================================

def get_h_Zeeman(spins, Bv, coord):
    r""" 
    Zeeman term H_Zee = - \vec{\mu} \cdot \vec{B} = \mu_B/\hbar \vec{B}[i] g_s[i,j] \vec{S}[j]
                      = \vec{B}[i] g_s[i,j] \vec{S}[j]
    B takes unit of energy (cm^-1 per \mu_B), and spin takes unit of \hbar.
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

def get_h_Zeeman_iso(spins, Bv, coord):
    r""" 
    Zeeman term H_Zee = - \vec{\mu} \cdot \vec{B} = \mu_B/\hbar \vec{B}[i] g_s[i,j] \vec{S}[j]
                      = \vec{B}[i] g_s[i,j] \vec{S}[j]
                      = -2 \vec{B}[i] \vec{S}[i] # Iostropic g-factor
    B takes unit of energy (cm^-1 per \mu_B), and spin takes unit of \hbar.
    coord: 's*' (for spherical) or else (for cartesian).
    Units: Tesla for B, deg for angles.
    """

    if coord[0] == 's' or coord[0] == 'S':
        Bv = Tesla2wavenumber*np.array(sph2cart_deg(Bv))
    else:
        Bv = Tesla2wavenumber*np.array(Bv)
    
    h_zee = spins.zero
    for i in range(3):
        h_zee = h_zee  + 2.0*Bv[i]*spins.Sv_tot[i]

    return h_zee

def get_h_Zeeman_Mv_eff(Mv_tot, Bv, coord):
    r""" 
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

## =======================================================================
## Functions for basis transformation
## =======================================================================

def transform_O(O, eigen):
    """
    Transform an operator from the common basis to a basis defined by the eigen object.
    O and the eigenvectors (of the eigen object) are on the common basis.
    O_new is on the basis of the eigenvectors of the eigen object.
    """
    M = eigen.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))
    O_new = np.matmul(M_dagger, np.matmul(O, M))
    return O_new

def back_transform_O(O, eigen):
    """
    Transform an operator from the basis defined by the eigen object to the common basis
    O is on the basis of the eigenvectors of the eigen object.
    The eigenvectors (of the eigen object) written on the common basis.
    O_new is on the common basis after transformation.
    """
    M = eigen.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))
    O_new = np.matmul(M, np.matmul(O, M_dagger))
    return O_new

## =================================================================
## Functions for energy levels versus B field
## =================================================================

def get_Zeeman_energy_levels(spins, h_ex, h_ani, BT_Bgrid):
    """
    Calculate the Zeeman energy levels in the full spin space (relative to the space for the effective Hamiltonian).
    """
    Bmin, Bmax, Bstep, theta_B, phi_B = BT_Bgrid[0]
    nB = int((Bmax-Bmin)/Bstep) + 1
    h0 = h_ex + h_ani
    eigen = eigen_handy(h0)
    energy0 = eigen.eigenvalues[eigen.indices[0]]
    eigenvalues = np.zeros((nB, eigen.dim))
    for i in range(nB):
        B = Bmin + i*Bstep
        h_zee = get_h_Zeeman(spins, [B,theta_B,phi_B], 'spherical')
        h = h0 + h_zee
        eigen = eigen_handy(h)
        eigenvalues[i] = eigen.eigenvalues[eigen.indices]
    eigenvalues = eigenvalues - energy0
    create_outdir()
    with open("./output/Zeeman.dat", "w") as f:
        for i in range(nB):
            B = Bmin + i*Bstep
            f.write((" {:12.6f}" + eigen.dim*" {:15.9f}" + "\n").format(B, *eigenvalues[i]))
    print("The Zeeman energy levels are saved in ./output/Zeeman.dat")

def get_Zeeman_energy_levels_Mv_tot(h0, Mv_tot, BT_Bgrid):
    """
    Calculate the Zeeman energy levels in the effective Hilbert space.
    Assumption: Mv_tot is the total magnetic moment operator in the effective Hilbert space.
    Note: The effective space can be as big as the full space. 
    """
    Bmin, Bmax, Bstep, theta_B, phi_B = BT_Bgrid[0]
    nB = int((Bmax-Bmin)/Bstep) + 1
    eigen = eigen_handy(h0)
    energy0 = eigen.eigenvalues[eigen.indices[0]]
    eigenvalues = np.zeros((nB, eigen.dim))
    for i in range(nB):
        B = Bmin + i*Bstep
        h_zee = get_h_Zeeman_Mv_eff(Mv_tot, [B,theta_B,phi_B], 'spherical')
        h = h0 + h_zee
        eigen = eigen_handy(h)
        eigenvalues[i] = eigen.eigenvalues[eigen.indices]
    eigenvalues = eigenvalues - energy0
    create_outdir()
    with open("./output/Zeeman_eff.dat", "w") as f:
        for i in range(nB):
            B = Bmin + i*Bstep
            f.write((" {:12.6f}" + eigen.dim*" {:15.9f}" + "\n").format(B, *eigenvalues[i]))
    print("The Zeeman energy levels are saved in ./output/Zeeman_eff.dat")

## =================================================================
## Functions for analyzing results.
## =================================================================

def get_expectation_of_spins(spins, state):
    """
    Driver function to calculate the expectation values of local spins and total spin.
    """
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

def get_projections(eff, Bv_cart):
    """
    A function to get the projections onto the |Ms> basis.
    h: hamiltonian on the effeictive basis which are the eigenstates of Sz.
    Sz: the Sz operator on the effective basis.
    """
    h = eff.h0_eff + get_h_Zeeman_Mv_eff(eff.Mv_eff, Bv_cart, 'cartesian') 
    Sz = np.real(eff.Sz_eff)
    eigen = eigen_handy(h)
    state_indices = [i for i in range(1, eigen.dim+1)]
    projections = 100*np.abs( eigen.eigenvectors )**2
    create_outdir()
    with open("./output/projections.dat", "w") as f:
        f.write(( "{:>6s}" + eigen.dim*"{:>6d}" + "\n").format("ms", *state_indices))
        for i_basis in range(eigen.dim):
            f.write("{:>6.1f}".format(Sz[i_basis, i_basis]))
            for i_state in range(eigen.dim):
                f.write("{:>6.1f}".format(projections[i_basis, i_state]))
            f.write("\n")
    print("The projections onto the |Ms> basis are saved in ./output/projections.dat")

## =======================================================================
## Functions for obtaining the time-dependent Hamiltonian H(B(t))
## =======================================================================

def get_h_Mv(h0, Mv_tot, B, theta_B, phi_B):
    """
    Both h0 and Mv_tot should be on the basis of eigenvectors of h0.
    Return h under the magnetic field B.
    """
    h_zee = get_h_Zeeman_Mv_eff(Mv_tot, [B, theta_B, phi_B], 'spherical')
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

## =======================================================================
## Functions related to the density matrix
## =======================================================================

def get_rho0(eigen0, T):
    """
    Get the equilibrium density matrix rho0 based on the eigenalues stored in eigen0.
    rho0_kk = p_k which is the probability of occupying the state |k>
    T: Temperature in Kelvin
    """
    rho0 = np.zeros((eigen0.dim, eigen0.dim), dtype=np.complex128)
    e_ref = eigen0.eigenvalues[eigen0.indices[0]]
    beta = 1/(Kelvin2wavenumber * T)
    for i in range(eigen0.dim):
        eigenvalue = eigen0.eigenvalues[i] - e_ref
        rho0[i, i] = np.exp(-beta*eigenvalue)
    rho0 = rho0 / np.trace(rho0)
    return rho0

def get_rhoe(energies, T):
    """
    Get the equilibrium density matrix based on the given energies.
    rho0_kk = p_k which is the probability of occupying the state |k>
    T: Temperature in Kelvin
    """
    dim = energies.shape[0]
    rhoe = np.zeros((dim, dim), dtype=np.complex128)
    e_ref = np.min(energies)
    beta = 1/(Kelvin2wavenumber * T)
    for i in range(dim):
        eigenvalue = energies[i] - e_ref
        rhoe[i, i] = np.exp(-beta*eigenvalue)
    rhoe = rhoe / np.trace(rhoe)
    return rhoe

def get_Mv_from_rho(rho, Mv_tot):
    """
    Calculate the magnetization <M> from the density matrix rho.
    """
    Mv = []
    for i in range(3):
        Mi = np.real( np.trace( np.matmul(rho, Mv_tot[i]) ) )
        Mv.append(Mi)
    return Mv

def get_Mz_from_rho(rho, Mz_tot):
    """
    Calculate the magnetization <M_z> from the density matrix rho.
    """
    return np.real( np.trace( np.matmul(rho, Mz_tot) ) )

## ================================================================
## Functions for thermodynamic properties.
## ================================================================

### ================================================================
### Basic functions for thermodynamic properties.
### ================================================================

def get_partition_function(eigen, T):
    """
    Calculate the partition function Z at temperature T.
    """
    e_ref = eigen.eigenvalues[eigen.indices[0]]
    beta = 1/(Kelvin2wavenumber * T)
    Z = 0
    P = np.zeros(eigen.dim) # Distribution probabilities
    for i in range(eigen.dim):
        eigenvalue = eigen.eigenvalues[eigen.indices[i]] - e_ref
        P[i] = np.exp(-beta*eigenvalue)
        Z += P[i]
    P = P/Z  # Normalize the probabilities
    return (Z, P)

def get_magnetic_moment(spins, eigen, T, Z):
    """
    Calculate the magnetic moment <M> at temperature T.
    """
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
    """
    Calculate the magnetic moment <M> at temperature T using Mv_tot.
    This function works for the effective Hilbert space.
    """
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
    
def get_chim_tensor_kernel(spins, h_ex, h_ani, B0v_sph, T, dBv_sph, verbose):
    r"""
    chim_n = \partial M / \partial B along the e_n direction.
    B0v: B vector in spherical coordinate. Angles in deg.
    """
    B0v_cart = sph2cart_deg(B0v_sph)
    dBv_cart = sph2cart_deg(dBv_sph)
    h_zee = get_h_Zeeman(spins, B0v_cart + dBv_cart, 'cartesian')
    h = h_ex + h_ani + h_zee
    eigen_plus = eigen_handy(h)
    Z_plus, _ = get_partition_function(eigen_plus, T)
    M_plus = get_magnetic_moment(spins, eigen_plus, T, Z_plus)
    h_zee = get_h_Zeeman(spins, B0v_cart - dBv_cart, 'cartesian')
    h = h_ex + h_ani + h_zee
    eigen_minus = eigen_handy(h)
    Z_minus, _ = get_partition_function(eigen_minus, T)
    M_minus = get_magnetic_moment(spins, eigen_minus, T, Z_minus)
    chim_n = (M_plus - M_minus) / (2*dBv_sph[0])
    if verbose:
        print("dBx, dBy, dBz = {:15.9f} {:15.9f} {:15.9f} Tesla".format(*dBv_cart))
        print("M(B0 + dB) = {:15.9f} {:15.9f} {:15.9f} mu_B".format(*M_plus))
        print("M(B0 - dB) = {:15.9f} {:15.9f} {:15.9f} mu_B".format(*M_minus))
        print(u"\u03C7_m = {:15.9f} {:15.9f} {:15.9f} ".format(*chim_n) + u"\u03BC_B/T")
    return chim_n
      
def get_chim_tensor(spins, h_ex, h_ani, B0v_sph, T, dB):
    r"""
    chim at certain B field, E field, and temperature T
    chim_tensor_{ij} = \partial M_i / \partial B_j
    """
    chimx = get_chim_tensor_kernel(spins, h_ex, h_ani, B0v_sph, T, [dB, 90.,  0.], False)
    chimy = get_chim_tensor_kernel(spins, h_ex, h_ani, B0v_sph, T, [dB, 90., 90.], False)
    chimz = get_chim_tensor_kernel(spins, h_ex, h_ani, B0v_sph, T, [dB,  0.,  0.], False)
    chim_tensor = np.vstack((chimx, chimy))
    chim_tensor = np.vstack((chim_tensor , chimz))
    chim_tensor = np.transpose(chim_tensor)
    return chim_tensor

def get_equilibrium_occupations(h0, Mv_tot, BT_Bgrid, T):
    """
    Calculate the equilibrium occupations (probability distribution) of the eigenstates 
    of the system under the different magnetic fields.
    """
    # Sampled magnetic fields
    Bmin, Bmax, Bstep, theta_B, phi_B = BT_Bgrid[0]
    nB = int((Bmax-Bmin)/Bstep) + 1
    Bs = np.zeros(nB, dtype=np.float64)
    Ps = np.zeros((nB, h0.shape[0]), dtype=np.float64)
    # Loop over the magnetic fields
    for i in range(nB):
        Bs[i] = Bmin + i*Bstep
        h_zee = get_h_Zeeman_Mv_eff(Mv_tot, [0,0,Bs[i]], 'cartesian')
        eigen = eigen_handy(h0 + h_zee)
        Z, Ps[i, :] = get_partition_function(eigen, T)
    # Save the results to a file
    create_outdir()
    ostring = "{:8.3f}" + h0.shape[0]*" {:12.3e}" + "\n"
    with open('./output/Ps.dat'.format(T), 'w') as f:
        f.write('# T = {:.3f} K\n'.format(T))
        f.write('# B (Tesla) rho_ii, i=1,...,{:d}\n'.format(h0.shape[0]))
        for i in range(nB):
            f.write(ostring.format(Bs[i], *Ps[i]))
    print("The equilibrium occupations (probability distribution) are saved in ./output/Ps.dat")

def get_equilibrium_occupations_light(h0, Mv_tot, B_cart, T):
    """
    Calculate the equilibrium occupations (probability distribution) of the eigenstates 
    of the system under the a magnetic field along z direction.
    """
    h_zee = get_h_Zeeman_Mv_eff(Mv_tot, B_cart, 'cartesian')
    eigen = eigen_handy(h0 + h_zee)
    _, P = get_partition_function(eigen, T)
    # Save the results to a file
    create_outdir()
    ostring = "{:>6d} {:>12.3e}\n"
    with open('./output/P.dat', 'w') as f:
        f.write('# Equilibrium occupations at Bx, By, Bz = {:.3f}, {:.3f}, {:.3f} Tesla and T = {:.3f} K\n'.format(*B_cart, T))
        f.write('#    i       rho_ii\n')
        for i in range(eigen.dim):
            f.write(ostring.format(i+1, P[i]))
    print("The equilibrium occupations (probability distribution) are saved in ./output/P.dat")

### =================================================================================
### Magnetization vs B field
### =================================================================================

def get_M_vs_B_kernel(spins, h_ex, h_ani, Bs, theta_B, phi_B, T):
    """
    Calculate the magnetization <M> versus B field at a certain temperature T in the full Hilbert space.
    Bs: B fields enumerated
    theta_B: polar angle of B field in deg
    phi_B: azimuthal angle of B field in deg
    T: temperature
    """
    nB = len(Bs)
    Ms = []
    for iB in range(nB):
        print("B = {:6.2f}".format(Bs[iB]))
        h_zee   = get_h_Zeeman(spins, [Bs[iB], theta_B, phi_B], "spherical")
        h = h_ex + h_ani + h_zee
        eigen = eigen_handy(h)
        Z, _ = get_partition_function(eigen, T)
        M = get_magnetic_moment(spins, eigen, T, Z)
        Ms.append(M)
    return Ms

def get_M_vs_B_Mv_tot_kernel(h0, Mv_tot, Bs, theta_B, phi_B, T):
    """
    Calculate the magnetization <M> versus B field at a certain temperature T in the effective Hilbert space.
    Bs: B fields enumerated
    theta_B: polar angle of B field in deg
    phi_B: azimuthal angle of B field in deg
    T: temperature
    """
    nB = len(Bs)
    Ms = []
    for iB in range(nB):
        print("B = {:6.2f}".format(Bs[iB]))
        h_zee   = get_h_Zeeman_Mv_eff(Mv_tot, [Bs[iB], theta_B, phi_B], "spherical")
        h = h0 + h_zee
        eigen = eigen_handy(h)
        Z, _ = get_partition_function(eigen, T)
        M = get_magnetic_moment_Mv_tot(Mv_tot, eigen, T, Z)
        Ms.append(M)
    return Ms

def get_M_vs_B(spins, h_ex, h_ani, BT_Bgrid):
    """
    Bgrid[0]: Bmin
    Bgrid[1]: Bmax
    Bgrid[2]: Bstep
    Bgrid[3]: theta_B in deg
    Bgrid[4]: phi_B in deg
    Ts: temperatures enumerated
    """
    # Set up the B grid and temperatures
    Bgrid = BT_Bgrid[0]
    Ts = BT_Bgrid[1]
    nB = int((Bgrid[1]-Bgrid[0])/Bgrid[2]) + 1
    Bs = np.linspace(Bgrid[0], Bgrid[1], nB, endpoint=True)
    nT = len(Ts)
    # Store results using a data frame with the columns: B field, Mx, My, Mz, and T
    for iT in range(nT):
        print("T={:6.2f} K\n".format(Ts[iT]))
        Ms = get_M_vs_B_kernel(spins, h_ex, h_ani, Bs, Bgrid[3], Bgrid[4], Ts[iT])
        for iB in range(nB):
            if iT == 0 and iB == 0:
                # Create the data frame for the first entry
                df = pd.DataFrame({"B": [Bs[iB]], "Mx": [Ms[iB][0]], "My": [Ms[iB][1]], "Mz": [Ms[iB][2]], "T": [Ts[iT]]})
            else:
                # Create a DataFrame from the dictionary 
                new_row = pd.DataFrame([{"B": Bs[iB], "Mx": Ms[iB][0], "My": Ms[iB][1], "Mz": Ms[iB][2], "T": Ts[iT]}])
                # Append the new data to the existing data frame
                df = pd.concat([df, new_row], ignore_index=True)
    create_outdir()
    # Save the data frame to a CSV file
    df.to_csv("./output/M-B.csv", index=False)
    print("The magnetization versus B field is saved in ./output/M-B.csv")
    return

def get_M_vs_B_Mv_tot(h0, Mv_tot, BT_Bgrid):
    """
    Bgrid[0]: Bmin
    Bgrid[1]: Bmax
    Bgrid[2]: Bstep
    Bgrid[3]: theta_B in deg
    Bgrid[4]: phi_B in deg
    Ts: temperatures enumerated
    """
    # Set up the B grid and temperatures
    Bgrid = BT_Bgrid[0]
    Ts = BT_Bgrid[1]
    nB = int((Bgrid[1]-Bgrid[0])/Bgrid[2]) + 1
    Bs = np.linspace(Bgrid[0], Bgrid[1], nB, endpoint=True)
    nT = len(Ts)
    # Store results using a data frame with the columns: B field, Mx, My, Mz, and T
    for iT in range(nT):
        print("T={:6.2f} K\n".format(Ts[iT]))
        Ms = get_M_vs_B_Mv_tot_kernel(h0, Mv_tot, Bs, Bgrid[3], Bgrid[4], Ts[iT])
        for iB in range(nB):
            if iT == 0 and iB == 0:
                # Create the data frame for the first entry
                df = pd.DataFrame({"B": [Bs[iB]], "Mx": [Ms[iB][0]], "My": [Ms[iB][1]], "Mz": [Ms[iB][2]], "T": [Ts[iT]]})
            else:
                # Create a DataFrame from the dictionary 
                new_row = pd.DataFrame([{"B": Bs[iB], "Mx": Ms[iB][0], "My": Ms[iB][1], "Mz": Ms[iB][2], "T": Ts[iT]}])
                # Append the new data to the existing data frame
                df = pd.concat([df, new_row], ignore_index=True)
    create_outdir()
    # Save the data frame to a CSV file
    df.to_csv("./output/M-B_eff.csv", index=False)
    print("The magnetization versus B field is saved in ./output/M-B_eff.csv.")



