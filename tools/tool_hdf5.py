from spin_dynamics.core.hdf5 import get_hdf5_file_size, combine_hdf5_files, check_hdf5

if __name__ == "__main__":
    
    # t = 1000  # Total time in ps
    # dt = 0.001  # Time step in ps
    # dim = 16 # Dimension of the effective Hamiltonian
    # dimds = 2*dim**2 # Dimension of the RI-separated vectorized density matrix
    # get_hdf5_file_size(t, dt, dimds)

    # file1 = 'risvrho_0.000-10.000_step0.001ps.hdf5'
    # file2 = 'risvrho_10.000-30.000_step0.001ps.hdf5'
    # output_file = 'risvrho_0.000-30.000_step0.001ps.hdf5'
    # combine_hdf5_files(file1, file2, output_file)

    file1 = './output/T_0.6K_I0_1.00e-14_lambdaa_10.00/Bt_linear_sweep_rate_50.0/rho/0.000-0.010ps_dt0.001ps.h5'
    check_hdf5(file1)

