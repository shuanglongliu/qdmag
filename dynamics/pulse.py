# =======================================================================
# Functions for obtaining the time-dependent magnetic field
# =======================================================================

import numpy as np
import pickle
from spin_dynamics import root_dir
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

def get_Bt(Bt_params):
    """
    Bt_params: dictionary with the following keys and values:
        'Bt_type': 'linear' or 'pwlinear' or 'sin' or 'cs'
        'sweep_rate': sweep rate in T per 0s
        'times': turning time points in ps
        'fields': magnetic fields at the turning points in Tesla
        'omega': angular frequency of the sine wave in rad ms^-1
        'amplitude': amplitude of the sine wave in T
        'theta_B': polar angle of magnetic field in deg
        'phi_B': azimuthal angle of magnetic field in deg
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
    elif Bt_type == 'cs':
        Bt = get_B_cs()
    else:
        print("Invalid Bt_type. Stopping ...")
        exit()

    return Bt

def get_B_sin_offset(t, *argv):
    B0 = argv[0]
    return get_B_sin(t) - B0

def get_B_sin(omega, amplitude):
    """
    t: time in ps
    B = amplitude*sin(omega*t/1e9): magnetic field in Tesla

    An example: 65.3473*np.sin(0.187486*t/1e9) as fitted to a pulsed field
    """

    def B_sin(t):
        return amplitude*np.sin(omega*t/1e9)

    return B_sin

def get_B_linear(sweep_rate):
    """
    sweep_rate: rate in T per ms
    t: time in ps
    B = a*t/1e9: magnetic field in Tesla
    """
    def B_linear(t):
        return sweep_rate*t/1e9

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

def get_B_cs():
    """
    Load the monotone cubic spline object for the pulse field
    Units: ps for time and T for magnetic field
    """

    with open(root_dir + "pulse/cs_pulse.pickle", "rb") as f:
        B_cs = pickle.load(f)

    return B_cs


def get_pulse_for_TEO(Bt, tmin, tmax, deltat):
    """
    Magnetic pulse field for time evolution operator (TEO).
    """

    nt = int( (tmax - tmin)/deltat )
    ts = np.linspace(tmin, tmax, nt, endpoint=False)

    Bs = Bt(ts)

    deltat = (tmax - tmin) / nt

    return (nt, ts, Bs, deltat)


def get_pulse_for_Runge_Kutta(Bt, tmin, tmax, deltat):
    """
    Magnetic pulse field
    B(t = ts[it]) = Bs[it]
    """

    nt = int( (tmax - tmin)/deltat )
    deltat = (tmax - tmin) / nt
    ts = np.linspace(tmin, tmax, nt+1, endpoint=True)

    Bs = Bt(ts)


    return (nt, ts, Bs, deltat)

def get_pulse_for_Runge_Kutta_double_grid(Bt, tmin, tmax, deltat):
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

if __name__ == "__main__":

    #B_offset = get_B_sin_offset(0, 0.1); print(B_offset)

    # find the time in ps at which B = arrgs[0], the second argument of fsolve is the initial guess
    # solution = fsolve(get_B_sin_offset, 1e8, args=[3.491377]); print(solution)
    # solution = fsolve(get_B_sin_offset, 10e+06, args=[0.2]); print(solution)

    #B = get_B_sin(2.84993929e+08); print(B) # 3.49000000560594
    #B = get_B_sin(2.85811311e+08); print(B) # 3.49999999467692
    # B = get_B_sin(16324285.0); print(B) # 0.19999999
    # B = get_B_sin(16324000.0); print(B) # 0.19999650
    # B = get_B_sin(17000000.0); print(B) # 0.20827861
    B = get_B_sin(1e+09); print(B) # 0.20827861

    #B = get_B_sin(2.85e+08); print(B) # 3.490074279544974
    #B = get_B_sin(2.86e+08); print(B) # 3.502308436017445

    # B = get_B_sin(0.5e+09); print(B) # 6.116883809339946

