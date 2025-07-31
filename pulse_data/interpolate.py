import os
import pickle
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_cs():

    # Load data from pulse.dat, which is experimentally measured pulse data
    
    data = np.loadtxt("pulse.dat")



    # Convert ms to ps

    data[:, 0] = 1e9 * data[:, 0]
    
    
    
    # Interpolate
    
    #ys = np.interp(xs, data[:, 0], data[:, 1]) # linear
    #cs = CubicSpline(data[:, 0], data[:, 1]) # cubic spline
    cs = PchipInterpolator(data[:, 0], data[:, 1]) # monotone cubic spline

    with open("cs_pulse.pickle", "wb") as f:
        pickle.dump(cs, f, pickle.HIGHEST_PROTOCOL)
    
    return

def load_cs_and_save_file():
    with open("cs_pulse.pickle", "rb") as f:
        cs = pickle.load(f)

    xs = np.linspace(0, 1e10, 101, endpoint=True)
    ys = cs(xs)

    with open("./pulse_interpolate.dat", "w") as f:
        for i in range(len(xs)):
            f.write("{:18.6f} {:12.6f}\n".format(xs[i], ys[i]))

    return

if __name__ == "__main__":

    get_cs()

    load_cs_and_save_file()

