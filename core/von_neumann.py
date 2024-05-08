import numpy as np
import ray
from scipy.linalg import expm
from spin_dynamics.core.common import sph2cart_deg
from spin_dynamics.core.common import get_h_Zeeman_Mv_tot
from spin_dynamics.core.constants import Kelvin2wavenumber, Tesla2wavenumber
from spin_dynamics.core.constants import const1
import pickle

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

    h = get_h_Mv(h0, Mv_tot, Bs[iB], theta_B, phi_B)
    DeltaU = get_deltaU(h, deltat)

    for i in range(iB+1, jB+1):
        h = get_h_Mv(h0, Mv_tot, Bs[i], theta_B, phi_B)
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

    h = get_h_Mv(h0, Mv_tot, Bs[0], theta_B, phi_B)
    DeltaU = get_deltaU(h, deltat)

    for i in range(1, nB):
        h = get_h_Mv(h0, Mv_tot, Bs[i], theta_B, phi_B)
        deltaU = get_deltaU(h, deltat)
        DeltaU = np.matmul(deltaU, DeltaU)

    return DeltaU



def get_DeltaUs_ray(h0, Mv_tot, nt, ts, deltat, Bs, theta_B, phi_B, nperiod):
    """
    Get TIME EVOLUTION OPERATORS DeltaUs.
    DeltaU (not deltaU) is defined as deltaU(ts[nt]) \cdot deltaU(ts[nt-1]) \codt ... \cdot deltaU(ts[1]) \cdot deltaU(ts[0]).

    Input: 
      h0: Spin Hamiltonian under zero field on the basis of of eigenvectors of h0. 
      Mv_tot: Magnetization operators on the basis of of eigenvectors of h0.
      nt: number of time points
      ts: time grid with a time step deltat
      deltat: time step
      Bs: magnetic fields on the time grid ts
      theta_B: polar angle of Bs. Unit: deg.
      phi_B: azimuthal angle of Bs. Unit: deg.
      nperiod: number of time periods, one time period can have many deltats.

    Output: DeltaUs = [DeltaU_n, ..., DeltaU_2, DeltaU_1]
    """

    nDeltaU = nperiod

    if nDeltaU > nt:
        print("nperiod should be smaller than the number of time steps. Stopping ...")
        exit()

    # Divide ts into nperiod time periods

    nt_per_period = nt // nperiod
    nt_left_over = nt % nperiod

    # get a list of arguments for get_DeltaU_ray

    list_of_args = []

    for iperiod in range(nperiod):
        iB = iperiod * nt_per_period
        jB = iB + nt_per_period
        list_of_args.append( (h0, Mv_tot, nt_per_period, Bs[iB: jB], theta_B, phi_B, deltat) )

    if nt_left_over != 0:
        iB = nperiod * nt_per_period
        list_of_args.append( (h0, Mv_tot, nt_left_over, Bs[iB: nt], theta_B, phi_B, deltat) )

        nperiod = nperiod + 1

    # Parallel calculations of DeltaUs

    futures = [ get_DeltaU_ray.remote(list_of_args[i]) for i in range(nperiod) ]
    DeltaUs = ray.get(futures)

    return DeltaUs



# =======================================================================
# Functions for initializing and evovling the density matrix
# =======================================================================

# rho = deltaU(tn) deltaU(tn-1) ... deltaU(t1) rho0 deltaU_dagger(t1) deltaU_dagger(t2) ... deltaU_dagger(tn)

def get_rho0(eigen0, T):
    """
    Get initial density matrix rho0 on the perturbed basis
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
    Get the density matrix on the perturbed basis for the thermal equilibrium
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


def evolve_Deltat(rho, DeltaU):
    """
    Evolve the density matrix rho by Deltat: rho_new = DeltaU rho DeltaU_dagger.
    Both rho and DeltaU should be on the basis of eigenvectors of h0
    """

    DeltaU_dagger = np.conjugate(np.transpose(DeltaU))

    rho_new = np.matmul(DeltaU, np.matmul(rho, DeltaU_dagger))

    return rho_new



def evolve_Deltats(rho, DeltaUs):
    """
    Evolve the density matrix rho by Deltats: rho_new = DeltaU_n ... DeltaU_2 DeltaU_1 rho DeltaU_1_dagger DeltaU_2_dagger ... DeltaU_n_dagger.
    Both rho and DeltaU should be on the basis of eigenvectors of h0
    """

    nDeltaU = len(DeltaUs)

    rho_new = evolve_Deltat(rho, DeltaUs[0])

    for i in range(1, nDeltaU):
        rho_new = evolve_Deltat(rho_new, DeltaUs[i])

    return rho_new



def save_rho(rho, t):
    """
    Save the density matrix at time t. Unit for time: ps.
    """

    fname = "rho_t{:.6f}ps.pickle".format(t)

    with open(fname, "wb") as f:
        pickle.dump(rho, f)

    return



