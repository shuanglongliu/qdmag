import numpy as np
from scipy.integrate import quad
from spin_dynamics.dynamics.constants import Tesla2wavenumber
from spin_dynamics.dynamics.super_quantum_master import construct_D_using_Bfield

def get_two_norm_of_D(t, D0, Mz_tot_diag, Bt, dim, dims):

    B = Bt(t)
    B_wavenumber = Tesla2wavenumber * B
    minus_Mz_tot_diag = -1 * Mz_tot_diag

    D = construct_D_using_Bfield(D0, minus_Mz_tot_diag, B_wavenumber, dim, dims)

    norm = np.linalg.norm(D, 2)

    return norm


