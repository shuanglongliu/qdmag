import numpy as np
from spin_dynamics.core.common import spy_sparsity
from spin_dynamics.core.quantum_master import construct_Rhbar
from spin_dynamics import __file__ as root_dir

def get_effective_operator(O_full, selected_states, dtype=np.complex128):
    """
    Obtain the effective operator of O on the basis of selected states. 

    Input: 
        O_full: the O operator in the whole Hilbert space.
        selected_states: a list of indices of the selected states counting from zero.

    dtype = np.float64 or np.complex128
    """

    n = len(selected_states)

    O_eff = np.zeros((n, n), dtype=dtype)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            O_eff[i, j] = O_full[ii, jj]

    return O_eff

def get_effective_operator_diag(O_full, selected_states, dtype=np.complex128):
    """
    Obtain the effective operator of O on the basis of selected states. 

    Input: 
        O_full: the O operator in the whole Hilbert space.
        selected_states: a list of indices of the selected states counting from zero.

    Assumption:
        O_full is diagonal.

    dtype = np.float64 or np.complex128
    """

    n = len(selected_states)

    O_eff_vec = np.zeros(n, dtype=dtype)

    for i in range(n):
        ii = selected_states[i]
        O_eff_vec[i] = O_full[ii, ii]

    return O_eff_vec

def get_effective_Mv(Mv_full, selected_states):
    """
    Obtain Mv_eff on the basis of selected states. 

    Input: 
        Mv_full: the magnetization vectors in the whole Hilbert space. 
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

def construct_X_eff(Sz_full, selected_states):
    """
    Construct the operator in the spin space, which couples to phonons.
    X_{ij} = 1 if | M_{iz} - M_{jz} | = 1. Otherwise, X_{ij} = 0. 
    """

    n = len(selected_states)
    X_eff = np.zeros((n, n), dtype=np.float64)

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

def set_up_the_effective_system(h0_full, S2_full, Sz_full, Mv_full, selected_states):
    """
    Obtain h0_eff and Mv_eff on the basis of selected states. 
    All the operators are on the perturbed basis, i.e. the eigenstates of h_ex + h_zee(B = 1e-4 T).

    Input: 
        h0_full: the full initial hamiltonian h_ex in the whole Hilbert space. 
                 It is diagonal on the perturbed basis since [h_ex, Mz_tot] = 0.
        Sz_full: Sz_tot operator on the perturbed basis.
        Mv_full: the magnetization operators in the whole Hilbert space. 
                 Mz_tot is diagonal on the perturbed basis since [h_ex, Mz_tot] = 0 and
                 the perturbative B field of 1e-4 T is big enough to lift degeneracies.
        selected_states: a list of indices of the selected states counting from zero.
    """

    n = len(selected_states)

    # h0_eff is actually real. 
    # Mz is also real when the magnetic field is along the z axis.
    # We use complex matrices here for general cases.
    # We need to code a real verion later.

    # np.complex64 is not enough to capture the energy differences between the nearly degenerate states (which are indeed degenerate without perturbation).
    h0_eff = np.zeros((n, n), dtype=np.complex128)
    S2_eff = np.zeros((n, n), dtype=np.complex128)
    Sz_eff = np.zeros((n, n), dtype=np.complex128)
    Mx_eff = np.zeros((n, n), dtype=np.complex128)
    My_eff = np.zeros((n, n), dtype=np.complex128)
    Mz_eff = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            ii = selected_states[i]
            jj = selected_states[j]
            h0_eff[i, j] = h0_full[ii, jj]
            S2_eff[i, j] = S2_full[ii, jj]
            Sz_eff[i, j] = Sz_full[ii, jj]
            Mx_eff[i, j] = Mv_full[0][ii, jj]
            My_eff[i, j] = Mv_full[1][ii, jj]
            Mz_eff[i, j] = Mv_full[2][ii, jj]

    Mv_eff = [Mx_eff, My_eff, Mz_eff]

    X_eff = construct_X_eff(Sz_full, selected_states)

    return (h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff)

def spy_the_effective_system(h0_eff, S2_eff, Sz_eff, Mv_eff, X_eff):
    """
    When the perturbative magnetic field is 1e-4 T, the matrix elements of Mx,y,z_eff smaller than 1e-8 are numerically insigficant.
    They are actually numerical errors for Mz_eff which should be exactly diagonal.
    The numerical results do not change within numerical errors if we set them to zero. 
    If the perturbative magnetic field is smaller than 1e-4 T, there could be bigger matrix elements in Mx,y,z_eff that cannot be ignored numerically.
    """

    n = h0_eff.shape[0]

    h0_eff_abs = np.abs(h0_eff)
    S2_eff_abs = np.abs(S2_eff)
    Sz_eff_abs = np.abs(Sz_eff)
    Mx_eff_abs = np.abs(Mv_eff[0])
    My_eff_abs = np.abs(Mv_eff[1])
    Mz_eff_abs = np.abs(Mv_eff[2])

    with open (root_dir + "output/h0_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, h0_eff_abs[i, j]))

    with open (root_dir + "output/S2_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, S2_eff_abs[i, j]))

    with open (root_dir + "output/Sz_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, Sz_eff_abs[i, j]))

    with open (root_dir + "output/Mx_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, Mx_eff_abs[i, j]))

    with open (root_dir + "output/My_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, My_eff_abs[i, j]))

    with open (root_dir + "output/Mz_eff_abs.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("{:5d} {:5d} {:12.4e}\n".format(i, j, Mz_eff_abs[i, j]))

    with open(root_dir + "output/X_eff.dat", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write("i   j   Sz_i   Sz_j   X_ij   = {:5d}   {:5d}   {:8.3f}   {:8.3f}   {:5.1f}\n".format( \
                     i, j, np.real(Sz_eff[i, i]), np.real(Sz_eff[j, j]), X_eff[i, j]))

    spy_sparsity(h0_eff_abs, "h0_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5) 
    spy_sparsity(S2_eff_abs, "S2_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(Sz_eff_abs, "Sz_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(Mx_eff_abs, "Mx_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(My_eff_abs, "My_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(Mz_eff_abs, "Mz_eff_abs", precision=1.0e-8, figsize=(10, 10), markersize=5)
    spy_sparsity(X_eff,      "X_eff",      precision=1.0e-8, figsize=(10, 10), markersize=5)

    return


