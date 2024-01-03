import numpy as np
from scipy.linalg import expm
from common import sph2cart_deg
from constants import Kelvin2wavenumber, Tesla2wavenumber
from constants import const1
import pickle

# =======================================================================
# Functions for basis transformation
# =======================================================================

def transform_O(O, eigen0):
    """
    Transform an operator from the basis of O to the basis of eigenvectors of h0
    """

    M = eigen0.eigenvectors
    M_dagger = np.conjugate(np.transpose(M))

    O_new = np.matmul(M_dagger, np.matmul(O, M))

    return O_new



def transform_h0(h0, eigen0):
    """
    Basis transformation for h0
    """

    h0_new = transform_O(h0, eigen0)

    return h0_new



def transform_Mv_tot(Mv_tot, eigen0):
    """
    Basis transformation for Mv_tot
    """

    Mv_tot_new = []

    for i in range(3):
        Mi = transform_O(Mv_tot[i], eigen0)
        Mv_tot_new.append(Mi)

    return Mv_tot_new


# =======================================================================
# Functions for obtaining the time-dependent magnetic field
# =======================================================================

def load_cs():
    """
    Load the monotone cubic spline object for the pulse field
    Units: ps for time and T for magnetic field
    """

    with open("cs_pulse.pickle", "rb") as f:
        cs = pickle.load(f)

    return cs


def get_pulse(cs, tmin, tmax, deltat):
    """
    Magnetic pulse field
    """

    nt = int( (tmax - tmin)/deltat ) + 1
    ts = np.linspace(tmin, tmax, nt, endpoint=True)

    Bs = cs(ts)

    deltat = (tmax - tmin) / (nt - 1)

    return (nt, ts, Bs, deltat)


# =======================================================================
# Functions for obtaining the time-dependent Hamiltonian H(B(t))
# =======================================================================

def get_h_Zeeman_h0basis(Mv_tot, Bv, coord):
    """ 
    This function works on the basis of eigenvectors of h0.
    Assumption: Mv_tot is written on the basis of eigenvectors of h0.

    The function get_h_Zeeman in common.py works on the Sz basis.

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

    dim = Mv_tot[0].shape[0]
    
    h_zee = np.zeros((dim, dim), dtype=complex)
    for i in range(3):
        h_zee = h_zee - Bv[i]*Mv_tot[i]

    return h_zee

def get_h_h0basis(h0, Mv_tot, Bs, theta_B, phi_B, iB):
    """
    Both h0 and Mv_tot should be on the basis of eigenvectors of h0.
    Return hs = [h(t1), h(t2), ..., h(tn)]
    """

    h_zee = get_h_Zeeman_h0basis(Mv_tot, [Bs[iB], theta_B, phi_B], 'spherical')
    h = h0 + h_zee

    return h


# =======================================================================
# Functions for time evolution operators
# =======================================================================

def get_deltaU(h, deltat):

    """
    Get the time evolution operator during time deltat on the basis of eigenvectors of h0
    Assumption: h is on the basis of of eigenvectors of h0
    const1 takes care of unit conversion and hbar
    """

    deltaU = expm(-1j*const1*h*deltat)

    return deltaU


def get_deltaU_v2(h, deltat, eigen0):

    """
    Get the time evolution operator during time deltat on the basis of eigenvectors of h0
    Assumption: h is on the Sz basis
    This is equivalent to get_deltaU, verified numerically.
    const1 takes care of unit conversion and hbar
    """

    deltaU = expm(-1j*const1*h*deltat)

    deltaU = transform_O(deltaU, eigen0)

    return deltaU



def get_DeltaU(h0, Mv_tot, Bs, theta_B, phi_B, iB, jB, deltat):
    """
    Input: 
      h0: Spin Hamiltonian under zero field on the basis of of eigenvectors of h0. 
      Mv_tot: Magnetization operators on the basis of of eigenvectors of h0.
      Bs: A time sequence of magnetic fields. Unit: Tesla.
      theta_B: polar angle of Bs. Unit: deg.
      phi_B: azimuthal angle of Bs. Unit: deg.
      iB, jB: iB = it, jB = jt. Bs[iB] is the magnetic field at time t_i = ts[it]. Similarly for Bs[jB]. iB < jB.
      deltat: time step, i.e. deltat = ts[i+1] - ts[i]
    Output: DeltaU = deltaU(ts[jt]) \cdot deltaU(ts[jt-1]) \codt ... \cdot deltaU(ts[it+1]) \cdot deltaU(ts[it])

    Note: The first letter "D" in DeltaU here is a capital letter, indicating time evolution over a period which may be long.
    """

    h = get_h_h0basis(h0, Mv_tot, Bs, theta_B, phi_B, iB)
    DeltaU = get_deltaU(h, deltat)

    for i in range(iB+1, jB+1):
        h = get_h_h0basis(h0, Mv_tot, Bs, theta_B, phi_B, i)
        deltaU = get_deltaU(h, deltat)
        DeltaU = np.matmul(deltaU, DeltaU)

    return DeltaU



@ray.remote(num_cpus=1)
def get_DeltaU_ray(args):
    """
    Input: 
      h0: Spin Hamiltonian under zero field on the basis of of eigenvectors of h0. 
      Mv_tot: Magnetization operators on the basis of of eigenvectors of h0.
      nB: nB = len(Bs), the number of magnet fields.
      Bs: A time sequence of magnetic fields. Unit: Tesla.
      theta_B: polar angle of Bs. Unit: deg.
      phi_B: azimuthal angle of Bs. Unit: deg.
      deltat: time step
    Output: DeltaU = deltaU(ts[nt]) \cdot deltaU(ts[nt-1]) \codt ... \cdot deltaU(ts[1]) \cdot deltaU(ts[0]). ts correspond to Bs.

    Note: The first letter "D" in DeltaU here is a capital letter, indicating time evolution over a period which may be long.
    """

    h0, Mv_tot, nB, Bs, theta_B, phi_B, deltat = args

    h = get_h_h0basis(h0, Mv_tot, Bs, theta_B, phi_B, 0)
    DeltaU = get_deltaU(h, deltat)

    for i in range(1, nB):
        h = get_h_h0basis(h0, Mv_tot, Bs, theta_B, phi_B, i)
        deltaU = get_deltaU(h, deltat)
        DeltaU = np.matmul(deltaU, DeltaU)

    return DeltaU



def get_DeltaUs_ray(h0, Mv_tot, theta_B, phi_B, tmin, tmax, deltat, nperiod):

    cs = load_cs()

    nt, ts, Bs, deltat = get_pulse(cs, tmin, tmax, deltat)

    nDeltaU = nperiod

    if nDeltaU > nt:
        print("nperiod should be smaller than the number of time steps. Stopping ...")
        exit()

    # Resume here.



# =======================================================================
# Functions for initializing and evovling the density matrix
# =======================================================================

# rho = deltaU(tn) deltaU(tn-1) ... deltaU(t1) rho0 deltaU_dagger(t1) deltaU_dagger(t2) ... deltaU_dagger(tn)

def get_rho0(eigen0, T):
    """
    Get initial density matrix rho0 on the basis of eigenvectors of h0
    rho0_kk = p_k which is the probability of occupying the state |k>
    T: Temperature in Kelvin
    """

    rho0 = np.zeros((eigen0.dim, eigen0.dim))

    e_ref = eigen0.eigenvalues[eigen0.indices[0]]

    beta = 1/(Kelvin2wavenumber * T)

    for i in range(eigen0.dim):
        eigenvalue = eigen0.eigenvalues[i] - e_ref
        rho0[i, i] = np.exp(-beta*eigenvalue)

    rho0 = rho0 / np.trace(rho0)

    return rho0


def evolve_Deltat(rho, DeltaU):
    """
    Evolve the density matrix rho by Deltat: rho_new = DeltaU rho DeltaU_dagger.
    Both rho and DeltaU should be on the basis of eigenvectors of h0
    """

    DeltaU_dagger = np.conjugate(np.transpose(DeltaU))

    rho_new = np.matmul(DeltaU, np.matmul(rho, DeltaU_dagger))

    return rho_new



# =======================================================================
# Functions for calculating magnetization
# =======================================================================

def get_M(rho, Mv_tot):
    """
    Both rho and Mv_tot are on the basis of eigenvectors of h0
    """

    M = []
    for i in range(3):
        Mi = np.trace( np.matmul(rho, Mv_tot[i]) )
        M.append(Mi)

    return M


