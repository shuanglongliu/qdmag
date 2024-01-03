import numpy as np
from scipy.optimize import basinhopping
from multiprocessing import Process
from common import get_h_exchange, get_h_anisotropy
from common import get_M_at_BET_ray
from common import read_exp_magnetization
from common import sph2cart_deg
from common import set_exchange, set_anisotropy
import time
import ray



def dice(nx, xmin, xmax):
    """
    random initial parameters
    """
    p0 = np.zeros(nx)
    for i in range(nx):
        p0[i] = np.random.uniform(xmin[i], xmax[i])
        #print(xmin[i], xmax[i], p0[i])
    return p0

class TakeStep:
    def __init__(self, nx, xmin, xmax):
        self.rng = np.random.default_rng()
        self.nx = nx
        self.xmin = xmin
        self.xmax = xmax
        self.xlen = []
        for i in range(nx):
            self.xlen.append(xmax[i] - xmin[i])

    def __call__(self, x):
        for i in range(self.nx):
            x[i] += self.rng.uniform(0, self.xlen[i])
            if x[i] > self.xmax[i]:
                x[i] -= self.xlen[i]
        return x

class SetBounds:
    def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin



def print_fx(x, f, accepted):
    nx = len(x)
    ostring = "{:15.9E}  " + " {:8.3f}"*(nx-1) + " {:12.6f} {:3d}"
    print(ostring.format(f, *x, accepted))



def get_list_of_args(spins, h_ex, h_ani, nB_T, Bs_T, theta_B, phi_B, E, theta_E, phi_E, nT, Ts_exp, Ms_exp_T):

    nB_tot = 0
    Ms_exp = []
    list_of_args = []
    for iT in range(nT):
        nB_tot = nB_tot + nB_T[iT]
        for iB in range(nB_T[iT]):
            Ms_exp.append(Ms_exp_T[iT][iB])
            list_of_args.append( (spins, h_ex, h_ani, Bs_T[iT][iB], theta_B, phi_B, E, theta_E, phi_E, Ts_exp[iT]) )
    Ms_exp = np.array(Ms_exp)

    return (nB_tot, Ms_exp, list_of_args)



def get_distance(x, *args):
    """
    Get the difference between the experimental magnetization and the calculated magnetization.
    """

    # Unpack the input options

    spins, exchange, anisotropy, nB_T, Bs_T, theta_B, phi_B, E, theta_E, phi_E, nT, Ts_exp, Ms_exp_T, fit_problem = args

    if fit_problem[0]["l_fit"]:
        fit_problem[0]["current_values"] = [x[i] for i in range(fit_problem[0]["n_par"])]

    if fit_problem[1]["l_fit"]:
        if fit_problem[0]["l_fit"]:
            fit_problem[1]["current_values"] = [x[i] for i in range(fit_problem[0]["n_par"], fit_problem[0]["n_par"] + fit_problem[1]["n_par"])]
        else:
            fit_problem[1]["current_values"] = [x[i] for i in range(fit_problem[1]["n_par"])]

    if fit_problem[2]["l_fit"]:
        if fit_problem[0]["l_fit"] and fit_problem[1]["l_fit"]:
            fit_problem[2]["current_values"] = [x[i] for i in range(fit_problem[0]["n_par"] + fit_problem[1]["n_par"], len(x))]
        elif fit_problem[0]["l_fit"]:
            fit_problem[2]["current_values"] = [x[i] for i in range(fit_problem[0]["n_par"], len(x))]
        elif fit_problem[1]["l_fit"]:
            fit_problem[2]["current_values"] = [x[i] for i in range(fit_problem[1]["n_par"], len(x))]
        else:
            fit_problem[2]["current_values"] = [x[i] for i in range(len(x))]

    # Set spin Hamiltonian parameters

    if fit_problem[0]["l_fit"]:
         exchange = set_exchange(exchange, fit_problem[0])

    if fit_problem[1]["l_fit"]:
         anisotropy = set_anisotropy(anisotropy, fit_problem[1])

    # Set up the exchange and the ZFS/anisotropy Hamiltonians

    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)

    # Convert the nested list Ms_exp_T into a flat list Ms_exp
    # Construct the list of arguments for the function get_M_at_BET_ray
    # The varying parts of the argument are B and T

    nB_tot, Ms_exp, list_of_args = get_list_of_args(spins, h_ex, h_ani, nB_T, Bs_T, theta_B, phi_B, E, theta_E, phi_E, nT, Ts_exp, Ms_exp_T)


    # Parallel calculations of magnetization

    #start = time.time()
    futures = [ get_M_at_BET_ray.remote(list_of_args[i]) for i in range(nB_tot) ]
    Ms_cal = ray.get(futures)
    #end = time.time()

    #print("# All {:5d} magnetization calculations take {:6.3f} s".format(nB_tot, end - start))
    
    # In experiments, the measured magnetization is along the magnetic field direction

    e_B = sph2cart_deg([1, theta_B, phi_B])
    Ms_cal_along_B = np.array( [np.dot(Ms_cal[iB], e_B) for iB in range(nB_tot)] )

    # Calculate the distance between the experimental and the calculated magnetization curves.

    if fit_problem[2]["l_fit"]:
        if fit_problem[2]["l_shift"]:
            dist = np.sum(np.abs(fit_problem[2]["current_values"][0]*Ms_exp + fit_problem[2]["current_values"][1] - Ms_cal_along_B)) / nB_tot
        else:
            dist = np.sum(np.abs(fit_problem[2]["current_values"][0]*Ms_exp - Ms_cal_along_B)) / nB_tot

    print(("# dist, *x = {:18.10E}" + len(x)*", {:18.10E}").format(dist, *x))

    return dist




def fit_magnetization(exp_technique, Ts_exp, spins, exchange, anisotropy, ext_field, fit_problem):

    """
    ext_field: constant external B and E fields
    B, theta_B, phi_B, E, theta_E, phi_E = ext_field
    The magnitude of B field is not used here since B is varied for magnetization measurement
    """

    # Read experimental magnetization

    nB_T = []
    Bs_T = []
    Ms_exp_T = []

    nT = len(Ts_exp)
    for iT in range(nT):
        fname = "{:s}_{:.1f}K.dat".format(exp_technique, Ts_exp[iT])
        nB, Bs, Ms_exp = read_exp_magnetization("./interpolate/", fname)
        nB_T.append(nB)
        Bs_T.append(Bs)
        Ms_exp_T.append(Ms_exp)

    # Prepare inital values, lower limits, and upper limits

    x0 = []
    xmin = []
    xmax = []

    for i in range(3):
        if fit_problem[i]["l_fit"] == 1:
            x0 = x0 + fit_problem[i]["initial_values"]
            xmin = xmin + fit_problem[i]["lower_limits"]
            xmax = xmax + fit_problem[i]["upper_limits"]

    nx = len(x0)

    # Randomize initial values if any of them is out of the boundary

    x0 = np.array(x0)
    if np.all(x0 > xmin) and np.all(x0 < xmax):
        pass
    else:
        x0 = dice(nx, xmin, xmax)

    # Set lower and upper limits for the global and local minimizers

    bounds_basinhopping = SetBounds(xmax=xmax, xmin=xmin)
    bounds_local_minimizer = [[xmin[i], xmax[i]] for i in range(len(xmin))]

    # A callable object for varying the variables

    take_step = TakeStep(nx, xmin, xmax)

    # Gather args for get_distance

    _, theta_B, phi_B, E, theta_E, phi_E = ext_field
    args = (spins, exchange, anisotropy, nB_T, Bs_T, theta_B, phi_B, E, theta_E, phi_E, nT, Ts_exp, Ms_exp_T, fit_problem)

    # Parameters for the local minimizer and the distance function to be minimized.

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds_local_minimizer, 
        "tol":1.e-2, 
        "args": args
    }

    #print("# Local miminum, J1, J2, D1, D2, EoD1, EoD2, factor, accepted")
    ret = basinhopping(get_distance, x0, minimizer_kwargs=minimizer_kwargs, niter=10, take_step=take_step, callback=print_fx, accept_test=bounds_basinhopping, seed=np.random.default_rng(), T=0.5)

    #print("# Global miminum, J1, J2, D1, D2, EoD1, EoD2, factor, accepted")
    print_fx(ret.x, ret.fun, 1)


    return



