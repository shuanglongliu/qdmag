import h5py

def test_hdf5():
    fname = "/home/shuan.liu.neu/git/spin_dynamics/output/risvrho_0.000-10.000.hdf5"
    fname = "/home/shuan.liu.neu/git/spin_dynamics/output/risvrho_0.000-10000.000.hdf5"
    fname = "/home/shuan.liu.neu/git/spin_dynamics/output/double_super_U.hdf5"
    with h5py.File(fname, "r") as f1:
        h5keys = list ( f1.keys() )
        print("Keys in the file:", h5keys)

        print(f1[h5keys[0]].shape)
        print(f1[h5keys[0]][0:10, 0:10])

        # print(list( f1.attrs.items() ))
        # print("File-level attributes:")
        # for key, value in f1.attrs.items():
             # print(f"{key}: {value}")

    return


if __name__ == "__main__":

    test_hdf5()

