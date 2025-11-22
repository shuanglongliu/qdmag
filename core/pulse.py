# =======================================================================
# Functions for obtaining the time-dependent magnetic field
# =======================================================================

import numpy as np
import pickle
from qdmag import root_dir
from scipy.interpolate import interp1d

def get_Bt(Bt_params):
    """
    Bt_params: dictionary with the following keys and values:
        'Bt_type': 'linear' or 'pwlinear' or 'sin' or 'cspline'
        'sweep_rate': sweep rate in T per ps
        'times': turning time points in ps
        'fields': magnetic fields at the turning points in Tesla
        'sweep_rates': slope of the linear segments in T per ps before normalization
        'sweep_rate_ave': average slope in T per ps
        'omega': angular frequency of the sine wave in rad ps^-1
        'amplitude': amplitude of the sine wave in T
    """

    Bt_type = Bt_params['Bt_type']

    if Bt_type == 'linear':
        sweep_rate = Bt_params['sweep_rate']
        Bt = get_B_linear(sweep_rate)
    elif Bt_type == 'pwlinear':
        times = Bt_params['times']
        fields = Bt_params['fields']
        Bt = get_B_pwlinear(times, fields)
    elif Bt_type == 'sin':
        omega = Bt_params['omega']
        amplitude = Bt_params['amplitude']
        Bt = get_B_sin(omega, amplitude)
    elif Bt_type == 'cspline':
        Bt = get_B_cspline()
    else:
        print("Invalid Bt_type. Stopping ...")
        exit()

    return Bt

def get_B_sin(omega, amplitude):
    """
    t: time in ps
    B = amplitude*sin(omega*t): magnetic field in Tesla

    An example: 65.3473*np.sin(0.187486e-09*t) as fitted to a pulsed field
    """
    def B_sin(t):
        return amplitude*np.sin(omega*t)
    return B_sin

def get_B_linear(sweep_rate):
    """
    sweep_rate: rate in T per ps
    t: time in ps
    B = a*t: magnetic field in Tesla
    """
    def B_linear(t):
        return sweep_rate*t
    return B_linear

def get_B_pwlinear(times=[0, 1e6, 1e7, 1e8, 1e9], fields=[0, 3, 5, 8, 10]):
    """
    times: turning time points in ps
    fields: magnetic fields at the turning points in Tesla
    Return a piecewise linear magnetic field function
    """
    # Create the piecewise linear function
    B_pwlinear = interp1d(times, fields, kind='linear', fill_value="extrapolate")
    return B_pwlinear

def get_B_cspline():
    """
    Load the monotone cubic spline object for the pulse field
    Units: ps for time and T for magnetic field
    """
    with open("./cspline.pickle", "rb") as f:
        B_cs = pickle.load(f)
    return B_cs

def get_pulse_RK4_double_grid(Bt, tmin, tmax, deltat):
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
    Bs2 = Bt(ts2)
    return (nt, ts, Bs2, deltat)

if __name__ == "__main__":

    pass

