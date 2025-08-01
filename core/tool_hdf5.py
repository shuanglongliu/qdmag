import numpy as np
import h5py

def get_hdf5_file_size():
    t = 17000000 # Total time
    span = 100000 # Time span for each data set of U
    dimds = 2048
    # The factor 8 converts bits to bytes
    # The factor 1024 / 1024 /1024 converts bytes to Gb
    fsize = ( t / span ) * 2048 * 2048 * 8 / 1024 / 1024 /1024 
    print("File size of double_super_U.hdf5: {:.2f} Gb".format(fsize))

def get_rho_from_hdf5(fname, t, dimds):
    with h5py.File(fname, "r") as f1:
        rho = np.array( f1["{:.3f}".format(t)][0:dimds] )
    return rho

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

def test_hdf5():
    fname = "/blue/m2qm-efrc/shuan.liu.neu/projects/spin_dynamics/output/risvrho_0.000-30.000_step0.001ps.h5"
    with h5py.File(fname, "r") as f1:
        h5keys = list ( f1.keys() )
        print("Keys in the file:", h5keys)

        # print(f1[h5keys[0]].shape)
        # print(f1[h5keys[0]][0:10, 0:10])

        # print(list( f1.attrs.items() ))
        # print("File-level attributes:")
        # for key, value in f1.attrs.items():
             # print(f"{key}: {value}")

    return

if __name__ == "__main__":

    # file1 = 'risvrho_0.000-10.000_step0.001ps.hdf5'
    # file2 = 'risvrho_10.000-30.000_step0.001ps.hdf5'
    # output_file = 'risvrho_0.000-30.000_step0.001ps.hdf5'
    # combine_hdf5_files(file1, file2, output_file)

    test_hdf5()

