import numpy as np
from spin_dynamics.dynamics.common import spy_sparsity

def construct_X_eff(Sz_full, selected_states, multiphonon=False, imbalance=False):
    """
    Construct the operator in the spin space, which couples to phonons.
    X_{ij} = 1 if | M_{iz} - M_{jz} | = 1. Otherwise, X_{ij} = 0. 
    """

    n = len(selected_states)
    X_eff = np.zeros((n, n), dtype=np.float64)
    # X_eff = np.zeros((n, n), dtype=np.complex128) # Complex numbers are needed for numba functions when Rhbar is complex (when h != h_iso_ex).

    if multiphonon:
        for i in range(n):
            for j in range(n):
                ii = selected_states[i]
                jj = selected_states[j]
                diff = abs(Sz_full[ii, ii] - Sz_full[jj, jj])
                # When a B field of 1e-9 T is added to the isotropic qunatum Heisenberg Hamiltonian,
                # the deviation in Sz from half integers is within 1e-8 mu_B.
                if abs(diff - 1.0) < 1e-6:
                    X_eff[i, j] = 1.0
                elif abs(diff - 2.0) < 1e-6:
                    # X_eff[i, j] = 0.1
                    X_eff[i, j] = 1.0
                elif abs(diff - 3.0) < 1e-6:
                    # X_eff[i, j] = 0.05
                    X_eff[i, j] = 1.0
                elif abs(diff - 4.0) < 1e-6:
                    # X_eff[i, j] = 0.02
                    X_eff[i, j] = 1.0
                elif abs(diff - 5.0) < 1e-6:
                    # X_eff[i, j] = 0.01
                    X_eff[i, j] = 1.0
    else:
        if imbalance:
            for i in range(n):
                for j in range(n):
                    ii = selected_states[i]
                    jj = selected_states[j]
                    diff = Sz_full[ii, ii] - Sz_full[jj, jj]
                    # When a B field of 1e-9 T is added to the isotropic qunatum Heisenberg Hamiltonian, 
                    # the deviation in Sz from half integers is within 1e-8 mu_B.
                    if abs(diff - 1.0) < 1e-6:
                        X_eff[i, j] = 1.0
                    elif abs(diff + 1.0) < 1e-6:
                        X_eff[i, j] = 0.1
        else:
            for i in range(n):
                for j in range(n):
                    ii = selected_states[i]
                    jj = selected_states[j]
                    diff = abs(Sz_full[ii, ii] - Sz_full[jj, jj])
                    # When a B field of 1e-9 T is added to the isotropic qunatum Heisenberg Hamiltonian, 
                    # the deviation in Sz from half integers is within 1e-8 mu_B.
                    if abs(diff - 1.0) < 1e-6:
                        X_eff[i, j] = 1.0

    return X_eff

def get_effective_O(O_full, selected_states):
    """
    Input: 
        O_full: the operator in the whole Hilbert space. 
            it should be on the perturbed basis, i.e. the eigenstates of h_ex + h_zee(B = 1e-4 T).
        selected_states: a list of indices of the selected states counting from zero.
    """

    n = len(selected_states)

    # np.complex64 is not enough to capture the energy differences between the nearly degenerate states (which are indeed degenerate without perturbation).
    O_eff = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            O_eff[i, j]     = O_full[ii, jj]

    return O_eff

def get_effective_Mv(Mv_full, selected_states):
    """
    Input: 
        Mv_full: the magnetic moment operator in the whole Hilbert space. 
            it should be on the perturbed basis, i.e. the eigenstates of h_ex + h_zee(B = 1e-4 T).
        selected_states: a list of indices of the selected states counting from zero.
    """

    n = len(selected_states)

    # np.complex64 is not enough to capture the energy differences between the nearly degenerate states (which are indeed degenerate without perturbation).
    Mx_eff     = np.zeros((n, n), dtype=np.complex128)
    My_eff     = np.zeros((n, n), dtype=np.complex128)
    Mz_eff     = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            Mx_eff[i, j]     = Mv_full[0][ii, jj]
            My_eff[i, j]     = Mv_full[1][ii, jj]
            Mz_eff[i, j]     = Mv_full[2][ii, jj]

    Mv_eff = [Mx_eff, My_eff, Mz_eff]

    return (Mx_eff, My_eff, Mz_eff, Mv_eff)

def set_up_the_effective_system(h_t0_full, h_tmin_full, S2_full, Sz_full, Mv_full, selected_states, multiphonon=False, imbalance=False):
    """
    Obtain h_t0_eff and Mv_eff on the basis of selected states. 
    All the input operators should be on the perturbed basis, i.e. the eigenstates of h_ex + h_zee(B = 1e-4 T).

    Input: 
        h_t0_full:       The full initial hamiltonian in the whole Hilbert space. 
                         It is diagonal on the perturbed basis since [h_ex, Mz_tot] = 0.
        h_tmin_full:     The hamiltonian at t_min, in the whole Hilbert space. 
        Sz_full:         Sz_tot operator on the perturbed basis.
        Mv_full:         the magnetization operators in the whole Hilbert space. 
                         Mz_tot is diagonal on the perturbed basis since [h_ex, Mz_tot] = 0 and
                         the perturbative B field of 1e-4 T is big enough to lift degeneracies.
        selected_states: a list of indices of the selected states counting from zero.
        n:               the number of selected states, 
                         i.e. the dimension of the effective Hilbert space.
    """

    n = len(selected_states)

    # h_t0_eff is actually real. 
    # Mz is also real when the magnetic field is along the z axis.
    # We use complex matrices here for general cases.
    # We need to code a real verion later.

    # np.complex64 is not enough to capture the energy differences between the nearly degenerate states (which are indeed degenerate without perturbation).
    h_t0_eff   = np.zeros((n, n), dtype=np.complex128)
    h_tmin_eff = np.zeros((n, n), dtype=np.complex128)
    S2_eff     = np.zeros((n, n), dtype=np.complex128)
    Sz_eff     = np.zeros((n, n), dtype=np.complex128)
    Mx_eff     = np.zeros((n, n), dtype=np.complex128)
    My_eff     = np.zeros((n, n), dtype=np.complex128)
    Mz_eff     = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            h_t0_eff[i, j]     = h_t0_full[ii, jj]
            h_tmin_eff[i, j] = h_tmin_full[ii, jj]
            S2_eff[i, j]     = S2_full[ii, jj]
            Sz_eff[i, j]     = Sz_full[ii, jj]
            Mx_eff[i, j]     = Mv_full[0][ii, jj]
            My_eff[i, j]     = Mv_full[1][ii, jj]
            Mz_eff[i, j]     = Mv_full[2][ii, jj]

    Mv_eff = [Mx_eff, My_eff, Mz_eff]

    X_eff = construct_X_eff(Sz_full, selected_states, multiphonon=multiphonon, imbalance=imbalance)

    return (h_t0_eff, h_tmin_eff, S2_eff, Sz_eff, Mx_eff, My_eff, Mz_eff, Mv_eff, X_eff, n)

def spy_the_effective_system(h_eff, S2_eff, Sz_eff, Mv_eff, X_eff):
    """
    When the perturbative magnetic field is 1e-4 T, the matrix elements of Mx,y,z_eff smaller than 1e-8 are numerically insigficant.
    These small numbers are numerical errors for Mz_eff which should be exactly diagonal.
    The numerical results do not change if we set them to zero. 
    If the perturbative magnetic field is smaller than 1e-4 T, there could be bigger matrix elements in Mx,y,z_eff that cannot be ignored numerically.
    """

    n = h_eff.shape[0]

    h_eff_abs = np.abs(h_eff)
    S2_eff_abs = np.abs(S2_eff)
    Sz_eff_abs = np.abs(Sz_eff)
    Mx_eff_abs = np.abs(Mv_eff[0])
    My_eff_abs = np.abs(Mv_eff[1])
    Mz_eff_abs = np.abs(Mv_eff[2])

    with open ("./output/h_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, h_eff_abs[i, j]))

    with open ("./output/S2_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, S2_eff_abs[i, j]))

    with open ("./output/Sz_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, Sz_eff_abs[i, j]))

    with open ("./output/Mx_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, Mx_eff_abs[i, j]))

    with open ("./output/My_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, My_eff_abs[i, j]))

    with open ("./output/Mz_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, Mz_eff_abs[i, j]))

    with open("./output/X_eff.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("i   j   Sz_i   Sz_j   X_ij   = {:5d}   {:5d}   {:8.3f}   {:8.3f}   {:5.1f}\n".format( \
                     i, j, np.real(Sz_eff[i, i]), np.real(Sz_eff[j, j]), X_eff[i, j]))

    spy_sparsity(h_eff_abs,    "h_eff_abs",    precision=1.0e-8, figsize=(10, 10), markersize=5) 
    spy_sparsity(S2_eff_abs,   "S2_eff_abs",   precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(Sz_eff_abs,   "Sz_eff_abs",   precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(Mx_eff_abs,   "Mx_eff_abs",   precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(My_eff_abs,   "My_eff_abs",   precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(Mz_eff_abs,   "Mz_eff_abs",   precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(X_eff,        "X_eff",        precision=1.0e-8, figsize=(10, 10), markersize=5)

    return


