import h5py
import numpy as np

def get_rho_from_hdf5(fname, t, dimds):
    with h5py.File(fname, "r") as f1:
        rho = np.array( f1["{:.3f}".format(t)][0:dimds] )
    return rho

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

