from hdf5 import get_hdf5_file_size, combine_hdf5_files, check_hdf5

if __name__ == "__main__":
    
    t = 1000  # Total time in ps
    dt = 0.001  # Time step in ps
    dimds = 1024  # Dimension of the RI-separated vectorized density matrix
    get_hdf5_file_size(t, dt, dimds)

    # file1 = 'risvrho_0.000-10.000_step0.001ps.hdf5'
    # file2 = 'risvrho_10.000-30.000_step0.001ps.hdf5'
    # output_file = 'risvrho_0.000-30.000_step0.001ps.hdf5'
    # combine_hdf5_files(file1, file2, output_file)

    # file1 = 'risvrho_0.000-10.000_step0.001ps.hdf5'
    # check_hdf5(file1)

