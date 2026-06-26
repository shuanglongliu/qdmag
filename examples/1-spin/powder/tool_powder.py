import os
import subprocess
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from qdmag.core.common import print_emat_array

root_dir = os.path.dirname(os.path.abspath(__file__))

input = """
spins:
  - 8.0

anisotropy:
  - site: 1
    ks: [   2,   2,   2,   2,   2,   4,   4,   4,   4,   4,   4,   4,   4,   4,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12 ]
    qs: [  -2,  -1,   0,   1,   2,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,   7,   8, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12 ]
    Bkqs: [   3.4994750592E-02,  -2.7086096290E-06,   2.7144135449E-02,  -3.8607263323E-06,   8.5970155653E-02,   4.9292781766E-03,   3.7420773069E-07,  -6.3647621451E-04,  -5.8482618912E-07,   2.5841103918E-03,  -7.7752312986E-07,  -1.5636414673E-03,   8.1060758891E-07,   5.0515480805E-03,  -3.6232763990E-05,  -4.0403705378E-08,   1.3397089227E-04,   1.4505877737E-09,  -7.4211810050E-06,   1.4044754324E-08,  -2.7399050125E-05,   1.3071993690E-08,  -1.8231027282E-05,   1.6135737389E-08,   1.3729465845E-04,  -7.6496350770E-09,  -1.5794354771E-05,   1.4485020379E-08,   1.4738915304E-12,   3.2913487895E-09,  -6.0575611584E-12,   3.2967609246E-09,  -1.2676070811E-13,  -1.3725365053E-10,  -5.8313174807E-14,   9.0399850984E-11,  -4.6479436245E-14,  -3.3713870741E-10,   8.9090318133E-13,   3.3785241673E-09,  -2.4254533148E-12,   1.4347234348E-09,   1.2258449233E-12,   3.5498244158E-10,  -8.6486845220E-12,  -1.3223211813E-13,   2.1295410042E-10,   5.8644230868E-14,  -3.8016711952E-11,   2.8512537705E-16,  -3.7975151723E-12,  -1.1011816128E-14,   4.4556788958E-12,   3.8483425640E-17,  -2.7807784120E-12,   2.1578480178E-15,   1.0946226472E-11,  -7.7951728095E-15,  -3.8917957747E-12,  -3.6446147447E-15,  -1.6571837454E-11,   1.7798064405E-14,   5.2192898827E-12,   5.6895406482E-14,   3.2761852347E-12,   2.2136397908E-13,   8.1544513779E-16,  -1.5276986114E-12,  -1.2854124465E-15,   4.8629684574E-12,   8.7750621765E-16,  -3.8663894329E-13,   3.2650282248E-16,  -2.7144005718E-13,   7.4410060826E-18,  -2.2596178830E-14,  -2.5002105866E-17,   1.3681893596E-14,  -2.7848527210E-17,  -5.5511396757E-14,  -5.3588680407E-17,  -2.7817397251E-13,  -2.8852743990E-16,  -1.6854123596E-13,   3.7028383618E-16,   1.1918827338E-13,   7.1765297823E-16,   5.7864007571E-13,  -6.8329337066E-16,  -2.0566624637E-13 ]
    reference_frame: [{exx:10.6f}, {exy:10.6f}, {exz:10.6f}, {eyx:10.6f}, {eyy:10.6f}, {eyz:10.6f}, {ezx:10.6f}, {ezy:10.6f}, {ezz:10.6f}]

gfactor:
  - site: 1
    gs: [1.24, 0.0, 0.0, 0.0, 1.24, 0.0, 0.0, 0.0, 1.24]
    reference_frame: [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000 ]

BT_Bgrid:
  - [0., {staticB_max:.3f}, {staticB_step:.3f}, 0., 0.] # Bmin, Bmax, Bstep, thetaB, phiB 
  - [2.0] # [2.0, 10.0, 20.0, 40.0] # T1, T2, ..., Tn

BT_Tgrid:
  - [0., 0.5, 1., 0., 0.] # B1, B2, ..., Bn, thetaB, phiB
  - [0.2, 1, 0.2, 2, 10, 2, 20, 300, 20] # Tmin1, Tmax1, Tstep1, Tmin2, Tmax2, Tstep2, ..., Tminn, Tmaxn, Tstepn

dynamics:
    - T: {T:.3f}          # Temperature in K
      lambdaa: {lambdaa:.2f}   # Spin phonon coupling constant in cm-1
      I0: {I0:.2e}     # Prefactor for phonon density of states. 1e-4

    - Bt_type: '{Bt_type:s}'  # Type of the magnetic pulse as a function of time. Options: 'linear', 'pwlinear', 'sin', 'pulse'
      sweep_rate: {sweep_rate:.1e}  # Slope of the magnetic field vs time. Unit: T per ps. Used only when Bt = 'linear'.

    - tmin: {tmin:.1e}             # Initial time in ps
      tmax: {tmax:.1e}             # Finial time in ps
      deltat: {deltat:.1e}      # Time step in ps

    - save_mag: {save_mag:s} # Calculate and save magnetization during the dynamics ?
      nt_mag: {nt_mag:d}  # Calculate and save magnetization every nt_mag*deltat ps
      save_rho: {save_rho:s} # Save the density matrix ?
      nt_rho: {nt_rho:d}  # Save the density matrix every nt_rho*deltat ps
      save_drdt: false # Save the time derivative of the density matrix ?
      nt_drdt: 1 # Save the time derivative of the density matrix every nt_drdt*deltat ps

n_threads: {n_threads:d} # Number of threads used in the calculation

"""

job_array = """#!/bin/bash -l

#SBATCH --account=m2qm-efrc
#SBATCH --qos=m2qm-efrc
#SBATCH --job-name=qmag_Ho
#SBATCH --mail-type=None
#SBATCH --mail-user=shuan.liu@northeastern.edu
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH -t 4-00:00:00
#SBATCH --error=/dev/null
#SBATCH --output=/dev/null
#SBATCH --array=1-{n_jobs:d}

cd ./$SLURM_ARRAY_TASK_ID || exit 1

# Thermal equilibrium
python "{root_dir:s}/tool_magnetization.py"

# Dynamics
python "{root_dir:s}/tool_staircase.py"
"""

class powder:
    def __init__(self):
        self.load_points_and_weights()

        # Parameters for determining the BT_Egrid in the input file
        self.staticB_max = 50.0 # T
        self.staticB_step = 0.1 # T

        # Parameters for automatically determining input parameters for spin dynamics
        # Height of each stair step in T. deltat will be set to dynamicB_step/sweep_rate.
        self.dynamicB_step = 0.001
        # Maximum magnetic field in T. tmax will be set to dynamicB_max/sweep_rate. 
        # Assuming that the magnetic field is linear in time.
        self.dynamicB_max = 50.0 

        # Magnetic field step in T for saving the magnetization during the dynamics.
        self.dynamicB_step_saveMag = 0.01
        # Magnetic field step in T for saving the density matrix during the dynamics.
        self.dynamicB_step_saveRho = 0.01

        # Input parameters for spin dynamics
        self.T = 2.0 # Kelvin
        self.lambdaa = 10.0 # cm-1
        self.I0 = 1e-14 # Prefactor for phonon density of states
        self.Bt_type = 'linear' # Type of the magnetic pulse as a function of time
        self.sweep_rate = 50.0e-09 # T per ps
        self.tmin = 0.0 # Initial time in ps
        self.tmax = self.dynamicB_max / self.sweep_rate # Final time in ps
        self.deltat = self.dynamicB_step / self.sweep_rate # Time step in ps
        self.save_mag = 'true' # Calculate and save magnetization during the dynamics ?
        self.set_nt_mag() # self.nt_mag: Calculate and save magnetization every nt_mag*deltat ps
        self.save_rho = 'false' # Save the density matrix ?
        self.set_nt_rho() # self.nt_rho: Save the density matrix every nt_rho*deltat ps
        self.n_threads = 16

    def load_points_and_weights(self):
        """
        Load points and weights from a file.
        """
        points_and_weights = np.loadtxt('points_and_weights.txt', comments='#', ndmin=2)
        self.n_points = 3 # points_and_weights.shape[0]
        self.alphas = points_and_weights[:, 0]
        self.betas = points_and_weights[:, 1]
        self.gammas = points_and_weights[:, 2]
        self.weights = points_and_weights[:, 3]

    def set_nt_mag(self):
        """
        Get the number of time steps for saving magnetization.
        """
        if self.dynamicB_step_saveMag > self.dynamicB_step:
            self.nt_mag = int(self.dynamicB_step_saveMag / self.dynamicB_step)
        else:
            self.nt_mag = 1
        # print(f"nt_mag = {self.nt_mag} for dynamicB_step_saveMag = {self.dynamicB_step_saveMag} and dynamicB_step = {self.dynamicB_step}")
        return 

    def set_nt_rho(self):
        """
        Get the number of time steps for saving the density matrix.
        """
        if self.dynamicB_step_saveRho > self.dynamicB_step:
            self.nt_rho = int(self.dynamicB_step_saveRho / self.dynamicB_step)
        else:
            self.nt_rho = 1
        # print(f"nt_rho = {self.nt_rho} for dynamicB_step_saveRho = {self.dynamicB_step_saveRho} and dynamicB_step = {self.dynamicB_step}")
        return

    def set_directory_name(self, i):
        self.directory = "{:d}".format(i+1)

    def get_emat(self, i):
        """
        Get the basis vectors for the i-th orientation.
        emat = [ex, ey, ez] with ex, ey, ez being row-vectors
        """
        rot = R.from_euler('ZYZ', [self.alphas[i], self.betas[i], self.gammas[i]], degrees=True)
        rotmat = rot.as_matrix()
        # emat = np.transpose( rotmat * np.eye(3) ) = np.transpose( rotmat )
        self.emat = rotmat.T
        # print_emat_array(self.emat)

    def get_input(self, i):
        self.get_emat(i)
        self.set_directory_name(i)
        os.chdir(self.directory)
        os.system('pwd')
        with open("input.yaml", "w") as f:
            f.write(input.format( \
                exx=self.emat[0, 0], exy=self.emat[0, 1], exz=self.emat[0, 2],\
                eyx=self.emat[1, 0], eyy=self.emat[1, 1], eyz=self.emat[1, 2],\
                ezx=self.emat[2, 0], ezy=self.emat[2, 1], ezz=self.emat[2, 2],\
                staticB_max=self.staticB_max, staticB_step=self.staticB_step,\
                T=self.T, lambdaa=self.lambdaa, I0=self.I0,\
                Bt_type=self.Bt_type, sweep_rate=self.sweep_rate,\
                tmin=self.tmin, tmax=self.tmax, deltat=self.deltat,\
                save_mag=self.save_mag, nt_mag=self.nt_mag,\
                save_rho=self.save_rho, nt_rho=self.nt_rho,\
                n_threads=self.n_threads))
        os.chdir(root_dir)

    def create_directories(self):
        """
        Create directories for output files.
        """
        for i in range(self.n_points):
            self.set_directory_name(i)
            if not os.path.exists(self.directory):
                subprocess.run(['mkdir', '-p', self.directory])

    def get_inputs(self):
        """
        Get the input files for each orientation.
        """
        for i in range(self.n_points):
            self.get_input(i)

        with open("spin.job", "w") as f:
            f.write(job_array.format(n_jobs=self.n_points, root_dir=root_dir))
        print("Job array script saved as spin.job in the root directory.")

    def submit_job_array(self):
        """
        Submit the job array to the queue.
        """
        subprocess.run(['sbatch', 'spin.job'])

    def check_job_status(self, i, verbose=True):
        """
        Is this job done?
        """
        self.status_eq = 'unknown'
        Bs_eq = self.read_M_eq(i, take_B=True)
        if (Bs_eq is None):
            self.status_eq = 'no output'
            if verbose:
                print(f"{i+1:5d}: No output file for the equilibrium job.")
        elif isinstance(Bs_eq, str) and Bs_eq == 'empty':
            self.status_eq = 'empty'
            if verbose:
                print(f"{i+1:5d}: The output file for the equilibrium job is empty.")
        elif (not np.isclose(Bs_eq["B"][Bs_eq.shape[0]-1], self.staticB_max) ):
            self.status_eq = 'incomplete'
            if verbose:
                print(f"{i+1:5d}: Equilibrium job is incomplete.")
        else:
            self.status_eq = 'done'
        #     if verbose:
        #         print(f"{i+1:5d}: Equilibrium job is done.")

        self.status_dy = 'unknown'
        Bs_dy = self.read_M_dy(i, take_B=True)
        if (Bs_dy is None):
            self.status_dy = 'no output'
            if verbose:
                print(f"{i+1:5d}: No output file for the dynamical job.")
        elif isinstance(Bs_dy, str) and Bs_dy == 'empty':
            self.status_dy = 'empty'
            if verbose:
                print(f"{i+1:5d}: The output file for the dynamical job is empty.")
        elif (not np.isclose(Bs_dy["B"][Bs_dy.shape[0]-1], self.dynamicB_max) ):
            self.status_dy = 'incomplete'
            if verbose:
                print(f"{i+1:5d}: Dynamical job is incomplete.")
        else:
            self.status_dy = 'done'
        #     if verbose:
        #         print(f"{i+1:5d}: Dynamical job is done.")

    def check_all_job_status(self, skip=[]):
        """
        Check the status of all jobs.
        """
        for i in range(self.n_points):
            if i+1 in skip:
                continue
            self.check_job_status(i)

    def read_M_eq(self, i, take_B=False):
        self.set_directory_name(i)
        # print(self.directory)
        # read csv file
        fname = os.path.join(self.directory, "output/M-B.csv")
        # Check if the file exists
        if not os.path.exists(fname):
            return None
        # Check if the file is empty
        if os.path.getsize(fname) == 0:
            return 'empty'
        df = pd.read_csv(fname)
        if take_B:
            # Take the column "B" and save it to a new data frame
            df = df[["B"]]
        else:
            # Take the column "Mz" and save it to a new data frame
            df = df[["Mz"]]
        # Rename the column "Mz" to self.directory which is a string
        df.rename(columns={"Mz": self.directory}, inplace=True)
        return df

    def get_M_eq_avg(self):
        """
        Get the average equilibrium magnetization for all orientations.
        """
        print(1)
        df = self.read_M_eq(0, take_B=True)
        Ms = self.read_M_eq(0)
        for i in range(1, self.n_points):
            print(i+1)
            column = self.read_M_eq(i)
            # Horizontally stack the dataframes
            Ms = pd.concat([Ms, column], axis=1)
        df['avg'] = Ms.dot(self.weights)
        df = pd.concat([df, Ms], axis=1)

        # Save the data to a csv file
        df.to_csv("M-B_eq.csv", index=False)
        print("Average equilibrium magnetization saved to ./M-B_eq.csv")

    def read_M_dy(self, i, take_B=False):
        self.set_directory_name(i)
        # print(self.directory)
        # read csv file
        fname = f"output/T_{self.T:.1f}K_I0_{self.I0:.2e}_lambdaa_{self.lambdaa:.2f}/Bt_linear_sweep_rate_{self.sweep_rate:.1f}/magnetometry/0.000-{self.tmax:.3f}ps_dt{self.deltat:.3f}ps.dat"
        fname = os.path.join(self.directory, fname)
        # Check if the file exists
        if not os.path.exists(fname):
            return None
        df = pd.read_csv(fname, sep=r'\s+', header=None)
        if take_B:
            # Take the second column and save it to a new data frame with a column name "B"
            column = df.iloc[:, 1]
            df = pd.DataFrame({"B": column})
        else:
            # Take the third column and save it to a new data frame with a column name self.directory
            column = df.iloc[:, 2]
            df = pd.DataFrame({self.directory: column})
        return df

    def get_M_dy_avg(self):
        """
        Get the average dynaical magnetization for all orientations.
        """
        print(1)
        df = self.read_M_dy(0, take_B=True)
        Ms = self.read_M_dy(0)
        for i in range(1, self.n_points):
            print(i+1)
            column = self.read_M_dy(i)
            # Horizontally stack the dataframes
            Ms = pd.concat([Ms, column], axis=1)
        df['avg'] = Ms.dot(self.weights)
        df = pd.concat([df, Ms], axis=1)

        # Save the data to a csv file
        df.to_csv("M-B_dy.csv", index=False)
        print("Average dynamical magnetization saved to ./M-B_dy.csv")

    def remove_directories(self):
        """
        Remove all directories.
        """
        for i in range(self.n_points):
            self.set_directory_name(i)
            subprocess.run(['rm', '-rf', self.directory])

if __name__ == "__main__":

    # print("Root directory: ", root_dir)

    pow = powder()
    # pow.create_directories()
    pow.get_inputs()
    # pow.submit_job_array()
    # pow.check_all_job_status()
    # pow.get_M_eq_avg()
    # pow.get_M_dy_avg()


    # =============================================
    # Caution!!! It will remove all directories.
    # =============================================
    # pow.remove_directories() 



