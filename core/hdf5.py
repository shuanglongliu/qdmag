import h5py
import numpy as np

def get_risvrho_from_hdf5(fname, t, dimds):
    with h5py.File(fname, "r") as f1:
        risvrho = np.array( f1["{:.3f}".format(t)][0:dimds] )
    return risvrho

def check_conditions_of_rho(fname, t, lio, tolerance=1e-6, verbose=False):
    with h5py.File(fname, "r") as f1:
        risvrho = f1["{:.3f}".format(t)][0:lio.dimds]
    rho_re = risvrho[0:lio.dims]
    rho_im = risvrho[lio.dims:lio.dimds]
    rho = np.reshape(rho_re + 1j * rho_im, (lio.dim, lio.dim))
    is_normalized = 0
    is_hermitian = 0
    is_positive = 0
    is_cschineq = 0
    # Check if the trace is 1
    trace_rho = np.trace(rho)
    if np.abs(trace_rho - 1) < tolerance:
        is_normalized = 1
    # Check if the density matrix is Hermitian
    if np.allclose(rho, rho.conj().T, atol=tolerance):
        is_hermitian = 1
    # Check if the density matrix is positive semi-definite
    eigenvalues = np.linalg.eigvalsh(rho)
    if np.all(eigenvalues >= -tolerance):
        is_positive = 1
    # Check if the Cauchy-Schwarz inequality holds, |\rho_ij|^2 <= \rho_ii * \rho_jj
    rhosq = np.abs(rho)**2
    np.fill_diagonal(rhosq, 0)
    rhoou = np.outer(np.diag(rho), np.diag(rho))
    np.fill_diagonal(rhoou, 0)
    diff = rhoou - rhosq
    diff = diff + np.eye(lio.dim)
    if np.all( diff >= -tolerance):
        is_cschineq = 1
    if verbose:
        # Print the results
        print("Conditions for the density matrix:")
        print("Normalized: ", is_normalized)
        print("Hermitian: ", is_hermitian)
        print("Positive semi-definite: ", is_positive)
        print("Cauchy-Schwarz inequality: ", is_cschineq)
    return is_normalized, is_hermitian, is_positive, is_cschineq

def get_hdf5_file_size(t, dt, dimds):
    """
    Calculate the file size of a double precision HDF5 file.
    Parameters:
        t (int): Total time in picoseconds.
        dt (float): Time step in picoseconds.
        dimds (int): Dimension of the RI-separated vectorized density matrix.
    Returns:
        float: Estimated file size in gigabytes.
    """
    bitsperfloat64 = 64 # bits per double precision float
    bit2byte = 8 # factor for converting bits to bytes
    byte2Kb = 1024 # factor for converting bytes to Kb
    byte2Mb = 1024 * 1024 # factor for converting bytes to Mb
    byte2Gb = 1024 * 1024 * 1024 # factor for converting bytes to Gb
    fsize = ( t / dt + 1) * dimds * bitsperfloat64 / bit2byte
    print("File size of hdf5 file: {:.2f} bytes".format(fsize))
    fsize = ( t / dt + 1) * dimds * bitsperfloat64 / bit2byte / byte2Kb
    print("File size of hdf5 file: {:.2f} Kb".format(fsize))
    fsize = ( t / dt + 1) * dimds * bitsperfloat64 / bit2byte / byte2Mb
    print("File size of hdf5 file: {:.2f} Mb".format(fsize))
    fsize = ( t / dt + 1) * dimds * bitsperfloat64 / bit2byte / byte2Gb
    print("File size of hdf5 file: {:.2f} Gb".format(fsize))

def combine_hdf5_files(file1, file2, output_file):
    with h5py.File(output_file, 'w') as outfile:
        # Open the first file and copy its contents to the output file
        with h5py.File(file1, 'r') as f1:
            def copy_items(group, target):
                for name, item in group.items():
                    if isinstance(item, h5py.Dataset):
                        target.create_dataset(name, data=item[...])
                    elif isinstance(item, h5py.Group):
                        subgroup = target.create_group(name)
                        copy_items(item, subgroup)
            
            copy_items(f1, outfile)
        # Open the second file and copy its contents to the output file
        with h5py.File(file2, 'r') as f2:
            def merge_items(group, target):
                for name, item in group.items():
                    if name in target:
                        print(f"Warning: Dataset or group '{name}' already exists. Skipping...")
                        continue
                    if isinstance(item, h5py.Dataset):
                        target.create_dataset(name, data=item[...])
                    elif isinstance(item, h5py.Group):
                        subgroup = target.create_group(name)
                        copy_items(item, subgroup)
            
            merge_items(f2, outfile)
    print(f"Combined file saved as {output_file}")

def check_hdf5(fname):
    with h5py.File(fname, "r") as f1:
        h5keys = list ( f1.keys() )
        nkeys = len(h5keys)
        print("Number of keys in the file:", nkeys)
        print("First 5 Keys in the file:", h5keys[0:5])
        print("Last 5 Keys in the file:", h5keys[-5:])
        print("Data shape of each entry:", f1[h5keys[0]].shape)

