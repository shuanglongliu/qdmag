import numpy as np

def set_up_the_effective_system(h0_full, Mv_tot_full, selected_states):
    """
    Obtain h0_eff and Mv_tot_eff on the basis of selected states. 

    Input: 
        h0_full: the full initial hamiltonian on the basis of its eigenvectors. So, it is diagonal, and the diagonal matrix elements are the eigenvalues.
        Mv_tot_full: the magnetization vectors on the basis of the eigenvectors of the initial Hamitonian.
        selected_states: a list of indices of the selected states counting from zero.
    """

    n = len(selected_states)

    # h0_eff is actually real. 
    # Mz is also real when the magnetic field is along the z axis.
    # We use complex matrices here for general cases.
    # We need to code a real verion later.

    h0_eff = np.zeros((n, n), dtype=np.complex64)
    Mx_tot = np.zeros((n, n), dtype=np.complex64)
    My_tot = np.zeros((n, n), dtype=np.complex64)
    Mz_tot = np.zeros((n, n), dtype=np.complex64)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            h0_eff[i, j] = h0_full[ii, jj]
            Mx_tot[i, j] = Mv_tot_full[0][ii, jj]
            My_tot[i, j] = Mv_tot_full[1][ii, jj]
            Mz_tot[i, j] = Mv_tot_full[2][ii, jj]

    Mv_tot_eff = [Mx_tot, My_tot, Mz_tot]

    return (h0_eff, Mv_tot_eff)

def get_effective_Mv_tot(Mv_tot_full, selected_states):
    """
    Obtain Mv_tot_eff on the basis of selected states. 

    Input: 
        Mv_tot_full: the magnetization vectors on the basis of the eigenvectors of the initial Hamitonian.
        selected_states: a list of indices of the selected states counting from zero.
    """

    n = len(selected_states)

    Mx_tot = np.zeros((n, n), dtype=np.complex64)
    My_tot = np.zeros((n, n), dtype=np.complex64)
    Mz_tot = np.zeros((n, n), dtype=np.complex64)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            Mx_tot[i, j] = Mv_tot_full[0][ii, jj]
            My_tot[i, j] = Mv_tot_full[1][ii, jj]
            Mz_tot[i, j] = Mv_tot_full[2][ii, jj]

    Mv_tot_eff = [Mx_tot, My_tot, Mz_tot]

    return Mv_tot_eff

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
        O_eff = np.zeros((n, n), dtype=np.complex64)

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
        O_eff_vec = np.zeros(n, dtype=np.complex64)

    for i in range(n):
        ii = selected_states[i]
        O_eff_vec[i] = O_full[ii, ii]

    return O_eff_vec

def construct_X_eff(total_Sz_for_all_eigenstates, selected_states):
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
            #print("{:5d}::{:8.3f},   {:5d}::{:8.3f},   {:5.1f}".format(i, total_Sz_for_all_eigenstates[selected_states[i]], j, total_Sz_for_all_eigenstates[selected_states[j]], X[i, j]))
    return X

