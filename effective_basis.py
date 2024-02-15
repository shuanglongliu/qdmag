import numpy as np
from common import spy_sparsity

def set_up_the_effective_system(h0_full, Mv_full, selected_states, save_to_file=False):
    """
    Obtain h0_eff and Mv_eff on the basis of selected states. 

    Input: 
        h0_full: the full initial hamiltonian on the basis of its eigenvectors. So, it is diagonal, and the diagonal matrix elements are the eigenvalues.
        Mv_full: the magnetization vectors on the basis of the eigenvectors of the initial Hamitonian.
        selected_states: a list of indices of the selected states counting from zero.
    """

    n = len(selected_states)

    # h0_eff is actually real. 
    # Mz is also real when the magnetic field is along the z axis.
    # We use complex matrices here for general cases.
    # We need to code a real verion later.

    # np.complex64 is not enough to capture the energy differences between the nearly degenerate states (which are indeed degenerate without perturbation).
    h0_eff = np.zeros((n, n), dtype=np.complex128)
    Mx_eff = np.zeros((n, n), dtype=np.complex128)
    My_eff = np.zeros((n, n), dtype=np.complex128)
    Mz_eff = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            h0_eff[i, j] = h0_full[ii, jj]
            Mx_eff[i, j] = Mv_full[0][ii, jj]
            My_eff[i, j] = Mv_full[1][ii, jj]
            Mz_eff[i, j] = Mv_full[2][ii, jj]

    Mv_eff = [Mx_eff, My_eff, Mz_eff]

    # When the perturbative magnetic field is 1e-4 T, the matrix elements of Mx,y,z_eff smaller than 1e-8 are numerically insigficant.
    # They are actually numerical errors for Mz_eff which should be exactly diagonal.
    # The numerical results do not change within numerical errors if we set them to zero. 
    # If the perturbative magnetic field is smaller than 1e-4 T, there could be bigger matrix elements in Mx,y,z_eff that cannot be ignored numerically.

    h0_eff_abs = np.abs(h0_eff)
    Mx_eff_abs = np.abs(Mx_eff)
    My_eff_abs = np.abs(My_eff)
    Mz_eff_abs = np.abs(Mz_eff)

    for i in range(n):
        for j in range(n):
            if h0_eff_abs[i, j] < 1e-8:
                h0_eff[i, j] = 0.0
            if Mx_eff_abs[i, j] < 1e-8:
                Mx_eff[i, j] = 0.0
            if My_eff_abs[i, j] < 1e-8:
                My_eff[i, j] = 0.0
            if Mz_eff_abs[i, j] < 1e-8:
                Mz_eff[i, j] = 0.0

    h0_eff_abs = np.abs(h0_eff)
    Mx_eff_abs = np.abs(Mx_eff)
    My_eff_abs = np.abs(My_eff)
    Mz_eff_abs = np.abs(Mz_eff)

    if save_to_file:
        with open ("./output/h0_eff_abs.dat", "w") as f:
            for i in range(n):
                for j in range(n):
                    f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, h0_eff_abs[i, j]))

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

        spy_sparsity(h0_eff_abs, "h0_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5) 
        spy_sparsity(Mx_eff_abs, "Mx_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5)
        spy_sparsity(My_eff_abs, "My_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5)
        spy_sparsity(Mz_eff_abs, "Mz_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5)

    return (h0_eff, Mv_eff)

def get_effective_Mv(Mv_full, selected_states):
    """
    Obtain Mv_eff on the basis of selected states. 

    Input: 
        Mv_full: the magnetization vectors on the basis of the eigenvectors of the initial Hamitonian.
        selected_states: a list of indices of the selected states counting from zero.
    """

    n = len(selected_states)

    Mx_eff = np.zeros((n, n), dtype=np.complex128)
    My_eff = np.zeros((n, n), dtype=np.complex128)
    Mz_eff = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            Mx_eff[i, j] = Mv_full[0][ii, jj]
            My_eff[i, j] = Mv_full[1][ii, jj]
            Mz_eff[i, j] = Mv_full[2][ii, jj]

    Mv_eff = [Mx_eff, My_eff, Mz_eff]

    return Mv_eff

def get_effective_operator(O_full, selected_states, is_real=False):
    """
    Obtain the effective operator of O on the basis of selected states. 

    Input: 
        O_full: the O operator in the whole Hilbert space.
        selected_states: a list of indices of the selected states counting from zero.
    """

    n = len(selected_states)

    if is_real:
        O_eff = np.zeros((n, n), dtype=np.float64)
    else:
        O_eff = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            O_eff[i, j] = O_full[ii, jj]

    return O_eff

def get_effective_operator_diag(O_full, selected_states, is_real=False):
    """
    Obtain the effective operator of O on the basis of selected states. 

    Input: 
        O_full: the O operator in the whole Hilbert space.
        selected_states: a list of indices of the selected states counting from zero.

    Assumption:
        O_full is diagonal.
    """

    n = len(selected_states)

    if is_real:
        O_eff_vec = np.zeros(n, dtype=np.float64)
    else:
        O_eff_vec = np.zeros(n, dtype=np.complex128)

    for i in range(n):
        ii = selected_states[i]
        O_eff_vec[i] = O_full[ii, ii]

    return O_eff_vec

def construct_X_eff(total_Sz_for_all_eigenstates, selected_states, save_to_file=False):
    """
    Construct the operator in the spin space, which couples to phonons.
    X_{ij} = 1 if | M_{iz} - M_{jz} | = 1. Otherwise, X_{ij} = 0. 
    """

    n = len(selected_states)
    X = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            diff = abs(total_Sz_for_all_eigenstates[selected_states[i]] - total_Sz_for_all_eigenstates[selected_states[j]])
            # When a B field of 1e-9 T is added to the isotropic qunatum Heisenberg Hamiltonian, 
            # the deviation in Sz from half integers is within 1e-8 mu_B.
            if abs(diff - 1.0) < 1e-6:
                X[i, j] = 1.0

    if save_to_file:
        with open("./output/X.dat", "w") as f:
            for i in range(n):
                for j in range(n):
                    f.write("i   j   Sz_i   Sz_j   X_ij   = {:5d}   {:5d}   {:8.3f}   {:8.3f}   {:5.1f}\n".format( \
                         i, j, total_Sz_for_all_eigenstates[selected_states[i]], total_Sz_for_all_eigenstates[selected_states[j]], X[i, j]))
        spy_sparsity(X, "X", precision=1.0e-20, figsize=(10, 10), markersize=5)

    return X

