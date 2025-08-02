# This is a standalone utility for sampling arbitrary molecular orientations

import os
import subprocess
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from spin_dynamics.core.common import print_emat_array

root_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

input = """
spins:
  - 2.5
  - 2.5
  - 2.5

exchange:
  - pair: [1, 2]
    coupling_matrix: [-2.42,  0.00,  0.00, 0.00,  -2.42,  0.00, 0.00,  0.00,  -2.42]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - pair: [2, 3]
    coupling_matrix: [-2.42,  0.00,  0.00, 0.00,  -2.42,  0.00, 0.00,  0.00,  -2.42]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - pair: [1, 3]
    coupling_matrix: [0.00,  0.00,  0.00, 0.00,  0.00,  0.00, 0.00,  0.00,  0.00]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

anisotropy:
  - site: 1
    ks: [2, 2]
    qs: [0, 2]
    Bkqs: [0.0556, 0.0400]
    reference_frame: [{exx:10.6f}, {exy:10.6f}, {exz:10.6f}, {eyx:10.6f}, {eyy:10.6f}, {eyz:10.6f}, {ezx:10.6f}, {ezy:10.6f}, {ezz:10.6f}]

  - site: 2
    ks: [2, 2]
    qs: [0, 2]
    Bkqs: [0.0556, 0.0400]
    reference_frame: [{exx:10.6f}, {exy:10.6f}, {exz:10.6f}, {eyx:10.6f}, {eyy:10.6f}, {eyz:10.6f}, {ezx:10.6f}, {ezy:10.6f}, {ezz:10.6f}]

  - site: 3
    ks: [2, 2]
    qs: [0, 2]
    Bkqs: [0.0556, 0.0400]
    reference_frame: [{exx:10.6f}, {exy:10.6f}, {exz:10.6f}, {eyx:10.6f}, {eyy:10.6f}, {eyz:10.6f}, {ezx:10.6f}, {ezy:10.6f}, {ezz:10.6f}]
  
gfactors:
  - site: 1
    gs: [2.000,  0.000,  0.000,  0.000,   2.000,  0.000,  0.000,  0.000,   2.000]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - site: 2
    gs: [2.000,  0.000,  0.000,  0.000,   2.000,  0.000,  0.000,  0.000,   2.000]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - site: 3
    gs: [2.000,  0.000,  0.000,  0.000,   2.000,  0.000,  0.000,  0.000,   2.000]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

BT_Bgrid:
  - [0., {staticB_max:.3f}, {staticB_step:.3f}, 0., 0.] # Bmin, Bmax, Bstep, thetaB, phiB 
  - [0.6] # [2.0, 10.0, 20.0, 40.0] # T1, T2, ..., Tn

BT_Tgrid:
  - [0., 0.5, 1., 0., 0.] # B1, B2, ..., Bn, thetaB, phiB
  - [0.2, 1, 0.2, 2, 10, 2, 20, 300, 20] # Tmin1, Tmax1, Tstep1, Tmin2, Tmax2, Tstep2, ..., Tminn, Tmaxn, Tstepn

dynamics:
    - T: {T:.3f}          # Temperature in K
      lambdaa: {lambdaa:.2f}   # Spin phonon coupling constant in cm-1
      I0: {I0:.1e}     # Prefactor for phonon density of states. 1e-4

    - Bt_type: '{Bt_type:s}'  # Type of the magnetic pulse as a function of time. Options: 'linear', 'pwlinear', 'sin', 'pulse'
      sweep_rate: {sweep_rate:.1f}  # Slope of the magnetic field vs time. Unit: T per ms. Used only when Bt = 'linear'.
      times: {times:s} # Turning points of the magnetic field in ps. Used only when Bt = 'pwlinear'.
      fields: {fields:s} # Magnetic field at the turning points in T. Used only when Bt = 'pwlinear'.
      omega: {omega:.2f} #  Angular frequency of the sine wave in rad ms^-1. The period is 2 pi / omega ms. Used only when Bt = 'sin'.
      amplitude: {amplitude:.1f} # Amplitude of the sine wave in T. Used only when Bt = 'sin'.

    - tmin: {tmin:.1e}             # Initial time in ps
      tmax: {tmax:.1e}             # Finial time in ps
      deltat: {deltat:.1e}      # Time step in ps

    - save_mag: {save_mag:s} # Calculate and save magnetization during the dynamics ?
      nt_mag: {nt_mag:d}  # Calculate and save magnetization every nt_mag*deltat ps
      save_rho: {save_rho:s} # Save the density matrix ?
      nt_rho: {nt_rho:d}  # Save the density matrix every nt_rho*deltat ps, nt_rho will be adjusted to be a multiple of nt_mag

    - multiphonon: {multiphonon:s} # Include multiphonon processes in the dynamics ?
      imbalance: {imbalance:s} # Make X unsymmetrical for single phonon processes ?

    - states: {states:s} # List of spin states to be included in the dynamics.
    
n_thread: {n_thread:d} # Number of threads used in the calculation

"""

job_array = """#!/bin/bash -l

#SBATCH --account=m2qm-efrc
#SBATCH --qos=m2qm-efrc
#SBATCH --job-name=dynamics
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
python /home/shuan.liu.neu/git/spin_dynamics/tools/tool_magnetization.py

# Dynamics
python /home/shuan.liu.neu/git/spin_dynamics/tools/tool_staircase.py
"""

class orientation_sampling:
    def __init__(self):
        # Euler angles for determing the molecular orientations
        self.load_euler_angles()

        # Parameters for determining the BT_Egrid in the input file
        self.staticB_max = 50.0 # T
        self.staticB_step = 0.1 # T

        # Parameters for automatically determining input parameters for spin dynamics
        # Maximum magnetic field in T. tmax will be set to dynamicB_max/sweep_rate. 
        # Assuming that the magnetic field is linear in time.
        self.dynamicB_max = 50.0 
        # Height of each stair step in T. deltat will be set to dynamicB_step/sweep_rate.
        self.dynamicB_step = 0.001 

        # Input parameters for spin dynamics
        self.T = 0.6 # Kelvin
        self.lambdaa = 10.0 # cm-1
        self.I0 = 1e-14 # Prefactor for phonon density of states
        self.Bt_type = 'linear' # Type of the magnetic pulse as a function of time
        self.sweep_rate = 50.0 # T per ms
        self.times = '[0.0, 1.0e+9, 10.0e+9]' # Turning points of the magnetic field in ps
        self.fields = '[0.0, 10.0, 100.0]' # Magnetic field at the turning points in T
        self.omega = 0.2 # Angular frequency of the sine wave in rad ms^-1
        self.amplitude = 65.0 # Amplitude of the sine wave in T
        self.theta_B = 0.0 # Polar angle of the pulsed magnetic field
        self.phi_B = 0.0 # Azimuthal angle of the pulsed magnetic field
        self.tmin = 0e+9 # Initial time in ps
        self.tmax = self.dynamicB_max / self.sweep_rate * 1e+9 # Final time in ps
        self.deltat = self.dynamicB_step / self.sweep_rate * 1e+9 # Time step in ps
        self.save_mag = 'true' # Calculate and save magnetization during the dynamics ?
        self.nt_mag = 10 # Calculate and save magnetization every nt_mag*deltat ps
        self.save_rho = 'true' # Save the density matrix ?
        self.nt_rho = 10 # Save the density matrix every nt_rho*deltat ps
        self.multiphonon = 'false'
        self.imbalance = 'false'
        self.states = '[200,150,88,30,10,0,1,2,3,4,5,17,41,99,173,215]'
        self.n_thread = 16

    def load_euler_angles(self):
        """
        Load points and weights from a file.
        """
        euler_angles = np.loadtxt('euler_angles.txt', comments='#', ndmin=2)
        self.n_points = euler_angles.shape[0]
        self.alphas = euler_angles[:, 0]
        self.betas = euler_angles[:, 1]
        self.gammas = euler_angles[:, 2]

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
                times=self.times, fields=self.fields,\
                omega=self.omega, amplitude=self.amplitude,\
                theta_B=self.theta_B, phi_B=self.phi_B,\
                tmin=self.tmin, tmax=self.tmax, deltat=self.deltat,\
                save_mag=self.save_mag, nt_mag=self.nt_mag,\
                save_rho=self.save_rho, nt_rho=self.nt_rho,\
                multiphonon=self.multiphonon, imbalance=self.imbalance,\
                states=self.states, n_thread=self.n_thread))
        os.chdir(root_dir)

    def submit_a_job(self, i):
        """
        Submit a job to the queue.
        """
        self.set_directory_name(i)
        os.chdir(self.directory)
        subprocess.run(['sbatch', 'spin.job'])
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
            f.write(job_array.format(n_jobs=self.n_points))
        print("Job array script saved as spin.job in the root directory.")

    def submit_job_array(self):
        """
        Submit the job array to the queue.
        """
        subprocess.run(['sbatch', 'spin.job'])

    def check_job_status(self, i):
        """
        Is this job done?
        """
        Bs_eq = self.read_M_eq(i, take_B=True)
        Bs_dy = self.read_M_dy(i, take_B=True)

        if (Bs_eq is None):
            return
        if (Bs_dy is None):
            return

        if (not np.isclose(Bs_eq["B"][Bs_eq.shape[0]-1], self.staticB_max) ):
            print(f"{i+1:5d} Equilibrium job is incomplete.")
        # else:
        #     print(f"{i+1:5d} Equilibrium job is complete.")

        if ( not np.isclose(Bs_dy["B"][Bs_dy.shape[0]-1], self.dynamicB_max) ):
            print(f"{i+1:5d} Dynamics job is incomplete.")
        # else:
        #     print(f"{i+1:5d} Dynamics job is complete.")

    def check_all_job_status(self):
        """
        Check the status of all jobs.
        """
        for i in range(self.n_points):
            self.check_job_status(i)

    def read_M_eq(self, i, take_B=False):
        self.set_directory_name(i)
        # print(self.directory)
        # read csv file
        fname = os.path.join(self.directory, "output/M-B.csv")
        # Check if the file exists
        if not os.path.exists(fname):
            print(f"{i+1:5d} No output file for the equilibrium job.")
            return None
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

    def get_M_eq(self):
        """
        Get the average equilibrium magnetization for all orientations.
        """
        df = self.read_M_eq(0, take_B=True)
        print(1)
        Ms = self.read_M_eq(0)
        for i in range(1, self.n_points):
            print(i+1)
            column = self.read_M_eq(i)
            # Horizontally stack the dataframes
            Ms = pd.concat([Ms, column], axis=1)
        df['avg'] = Ms.mean(axis=1)
        df = pd.concat([df, Ms], axis=1)

        # Save the data to a csv file
        df.to_csv("M-B_eq.csv", index=False)

    def read_M_dy(self, i, take_B=False):
        self.set_directory_name(i)
        # print(self.directory)
        # read csv file
        fname = os.path.join(self.directory, f"output/T_{self.T:.1f}K_I0_{self.I0:.2e}_lambdaa_{self.lambdaa:.2f}/Bt_linear_sweep_rate_{self.sweep_rate:.1f}/magnetometry/0.000-{self.tmax:.3f}ps_dt{self.deltat:.3f}ps.dat")
        # Check if the file exists
        if not os.path.exists(fname):
            print(f"{i+1:5d} No output file for the dynamical job.")
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

    def get_M_dy(self):
        """
        Get the average dynaical magnetization for all orientations.
        """
        df = self.read_M_dy(0, take_B=True)
        print(1)
        Ms = self.read_M_dy(0)
        for i in range(1, self.n_points):
            print(i+1)
            column = self.read_M_dy(i)
            # Horizontally stack the dataframes
            Ms = pd.concat([Ms, column], axis=1)
        df['avg'] = Ms.mean(axis=1)
        df = pd.concat([df, Ms], axis=1)

        # Save the data to a csv file
        df.to_csv("M-B_dy.csv", index=False)

    def remove_directories(self):
        """
        Remove all directories.
        """
        for i in range(self.n_points):
            self.set_directory_name(i)
            subprocess.run(['rm', '-rf', self.directory])

if __name__ == "__main__":

    # print("Root directory: ", root_dir)

    sam = orientation_sampling()
    # sam.create_directories()
    # sam.get_inputs()
    # sam.submit_job_array()
    # sam.check_all_job_status()
    # sam.get_M_eq()
    # sam.get_M_dy()


    # =============================================
    # Caution!!! It will remove all directories.
    # =============================================
    # sam.remove_directories() 



