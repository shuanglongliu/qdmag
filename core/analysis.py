import numpy as np
import matplotlib.pyplot as plt
from qdmag.core.common import create_outdir

def check_hermitian(op):
    maxdiff = np.max( np.absolute( np.conjugate(np.transpose(op)) - op ) )
    if maxdiff < 1.e-9:
        print("It is Hermitian.")
    else:
        print("It is not Hermitian.")

def check_unitary(op):
    op_dagger = np.conjugate(np.transpose(op))
    prod = np.matmul(op_dagger, op)
    maxdiff = np.max( np.absolute( prod - np.eye(prod.shape[0]) ) )
    if maxdiff < 1.e-9:
        print("It is unitary.")
    else:
        print("It is not unitary.")

def get_commutation(O1, O2):
    return np.matmul(O1, O2) - np.matmul(O2, O1) 

def check_commutation(O1, O2):
    x = np.matmul(O1, O2) 
    y = np.matmul(O2, O1) 
    maxdiff = np.max(np.absolute(x - y))
    if maxdiff < 1.e-6:
        print("Yes, they commute.")
    else:
        print("No, they don't commute.\n" + " maxdiff = {:15.10f}.".format(maxdiff))

def check_eigen(O, eigen):
    """
    Check if the given vectors are the eigenvectors of an operator.
    If O |i> = sum_j c_{ij} |j>, then c_{ij} = O_{ji}.
    If |i> are the eigenvectors, then O_{ji} is diagonal.
    """
    O1 = transform_O(O, eigen)
    O1_abs = np.abs(O1)
    np.fill_diagonal(O1_abs, 0)
    is_eigen = np.all( O1_abs < 1e-8 )
    print( "Are they eigenvectors? {:s}.".format(str(is_eigen)) )

def check_real(O):
    is_real = np.all( np.imag(O) < 1e-12 )
    if is_real:
        print("It is real.")
    else:
        print("It is complex.")

def spy_sparsity(M, tag, precision=1.0e-20, figsize=(20, 20), markersize=1):
    """
    Visualize the sparsity of the matrix M
    """
    create_outdir()
    fig, ax = plt.subplots(figsize=figsize)
    ax.spy(M, precision=precision, markersize=markersize)
    plt.savefig("./output/sparsity_of_" + tag + ".pdf")


def spy_M(M, tag, width=10, markersize=2, threshold=1e-9):
    """
    Visualize the sparsity of the matrix M and save it to a file.
    """
    create_outdir()
    with open("./output/" + tag + ".dat", "w") as f:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.abs(M[i, j]) >= threshold:
                    f.write("{:6d} {:6d} {:12.6f} {:12.6f} {:12.6f}\n".format(i+1, j+1, np.real(M[i, j]), np.imag(M[i, j]), np.abs(M[i, j])))
    spy_sparsity(M, tag, precision=1.0e-20, figsize=(width, width), markersize=markersize)

