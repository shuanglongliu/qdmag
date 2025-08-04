"""
A code for interpolating pulse data from a file and saving the interpolated function.
"""

import os
import pickle
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

def get_cs(basename="pulse"):

    # File name
    fname = os.path.join(os.path.dirname(__file__), basename + ".dat")

    # Load data from pulse.dat, which is experimentally measured pulse data
    data = np.loadtxt(fname)

    # Convert ms to ps
    data[:, 0] = 1e9 * data[:, 0]
    
    # Interpolate
    #cs = CubicSpline(data[:, 0], data[:, 1]) # cubic spline
    cs = PchipInterpolator(data[:, 0], data[:, 1]) # monotone cubic spline

    # Save the interpolated function
    fname = os.path.join(os.path.dirname(__file__), "cs_" + basename + ".pickle")
    with open(fname, "wb") as f:
        pickle.dump(cs, f, pickle.HIGHEST_PROTOCOL)

def load_cs_and_save_file(basename="pulse"):
    fname = os.path.join(os.path.dirname(__file__), "cs_" + basename + ".pickle")
    with open(fname, "rb") as f:
        cs = pickle.load(f)

    nx = 101
    xs = np.linspace(0, 1e10, nx, endpoint=True)
    ys = cs(xs)

    fname = os.path.join(os.path.dirname(__file__), basename + "_interpolate.dat")
    with open(fname, "w") as f:
        for i in range(nx):
            f.write("{:18.6f} {:12.6f}\n".format(xs[i], ys[i]))

if __name__ == "__main__":

    # get_cs("pulse1")

    # load_cs_and_save_file("pulse1")

    pass

