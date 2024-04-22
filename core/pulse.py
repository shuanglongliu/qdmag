# =======================================================================
# Functions for obtaining the time-dependent magnetic field
# =======================================================================

import numpy as np
import pickle
from spin_dynamics import __file__ as root_dir

def get_B_sin(t):
    """
    t: time in ps
    B = a*sin(b*t): magnetic field in Tesla

    Test: 
        t = np.linspace(0, 1e10, 101)
        B = get_B_sin(t)
        plt.plot(t, B)
    """

    return 65.3473*np.sin(0.187486*t/1e9)

def load_cs():
    """
    Load the monotone cubic spline object for the pulse field
    Units: ps for time and T for magnetic field
    """

    with open(root_dir + "pulse/cs_pulse.pickle", "rb") as f:
        cs = pickle.load(f)

    return cs


def get_pulse_for_TEO(cs, tmin, tmax, deltat):
    """
    Magnetic pulse field for time evolution operator (TEO).
    """

    nt = int( (tmax - tmin)/deltat )
    ts = np.linspace(tmin, tmax, nt, endpoint=False)

    Bs = cs(ts)

    deltat = (tmax - tmin) / nt

    return (nt, ts, Bs, deltat)


def get_pulse_for_Runge_Kutta(cs, tmin, tmax, deltat):
    """
    Magnetic pulse field
    B(t = ts[it]) = Bs[it]
    """

    nt = int( (tmax - tmin)/deltat )
    deltat = (tmax - tmin) / nt
    ts = np.linspace(tmin, tmax, nt+1, endpoint=True)

    Bs = cs(ts)


    return (nt, ts, Bs, deltat)

def get_pulse_for_Runge_Kutta_double_grid(cs, tmin, tmax, deltat):
    """
    Magnetic pulse field

    ts:  step = deltat
    Bs2: step = deltat/2
    B(t = ts[it]) = Bs2[2*it]
    """

    nt = int( (tmax - tmin)/deltat )
    deltat = (tmax - tmin) / nt
    ts = np.linspace(tmin, tmax, nt+1, endpoint=True)

    ts2 = np.linspace(tmin, tmax, 2*nt+1, endpoint=True)
    Bs2 = cs(ts2)


    return (nt, ts, Bs2, deltat)

def get_partial_double_grid(nt, ts, Bs2, nt_part, i_part):

    n_part = nt // nt_part

    #print("i_part = {:6d} / n_part = {:6d}".format(i_part, n_part))

    if nt_part > nt:
        print("The partial grid has more time points in the whole grid. Stopping ...")
        exit()
    elif i_part >= n_part:
        print("The partial grid is beyond the whole grid. Stopping ...")
        exit()

    it_min = nt_part*i_part
    it_max = nt_part*(i_part + 1)

    ts_part = ts[it_min: it_max + 1]

    iB_min = 2*nt_part*i_part
    iB_max = 2*nt_part*(i_part + 1)

    Bs2_part = Bs2[iB_min: iB_max + 1]

    return (ts_part, Bs2_part)

def get_partial_double_grid_left(nt, ts, Bs2, nt_left):

    it_min = nt - nt_left
    it_max = nt

    ts_part = ts[it_min: it_max + 1]

    iB_min = 2*nt - 2*nt_left
    iB_max = 2*nt

    Bs2_part = Bs2[iB_min: iB_max + 1]

    return (ts_part, Bs2_part)

