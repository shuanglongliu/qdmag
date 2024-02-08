# =======================================================================
# Functions for obtaining the time-dependent magnetic field
# =======================================================================

import numpy as np
import pickle

def load_cs():
    """
    Load the monotone cubic spline object for the pulse field
    Units: ps for time and T for magnetic field
    """

    with open("cs_pulse.pickle", "rb") as f:
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

