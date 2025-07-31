# This is a standalone utility for scanning I0 and sweep_rate

import os
import subprocess
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from spin_dynamics.core.common import print_emat_array
from spin_dynamics.core.data import input_powder
from spin_dynamics.core.data import job_script
from spin_dynamics.core.data import job_array_scan

source_dir = "/home/shuan.liu.neu/git/spin_dynamics/"
root_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

class scanning:
    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0

        # Parameters for determining the BT_Egrid in the input file
        self.staticB_max = 50.0 # T
        self.staticB_step = 0.1 # T

        # Parameters for automatically determining input parameters for spin dynamics
        self.dynamicB_step = 0.001 # Heights of each stair step in T. deltat will be set to dynamicB_step/sweep_rate.
        self.dynamicB_max = 50.0 # Maximum magnetic field in T. tmax will be set to dynamicB_max/sweep_rate. Assuming that the magnetic field is linear in time.

        # Prefactor for the phonon density of states in cm^-3 K^-1
        self.I0s = [1e-14, 1e-13, 1e-12] 
        self.n_I0s = len(self.I0s)

        # Sweep rates in T per ms
        self.sweep_rates = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        self.n_sweep_rates = len(self.sweep_rates)

        # Total number calculations
        self.n_scanning = self.n_I0s * self.n_sweep_rates

        # Input parameters for spin dynamics
        self.T = 0.6 # Kelvin
        self.lambdaa = 10.0 # cm-1
        self.I0 = None # Prefactor for phonon density of states. To be set in get_input() based on I0s
        self.Bt_type = 'linear' # Type of the magnetic pulse as a function of time
        self.sweep_rate = None # T per ms. To be set in get_input() based on sweep_rates
        self.times = '[0.0, 1.0e+9, 10.0e+9]' # Turning points of the magnetic field in ps
        self.fields = '[0.0, 10.0, 100.0]' # Magnetic field at the turning points in T
        self.omega = 0.2 # Angular frequency of the sine wave in rad ms^-1
        self.amplitude = 65.0 # Amplitude of the sine wave in T
        self.theta_B = 0.0 # Polar angle of the pulsed magnetic field
        self.phi_B = 0.0 # Azimuthal angle of the pulsed magnetic field
        self.tmin = 0e+9 # Initial time in ps
        self.tmax = None # Final time in ps, to be set in get_input() based on dynamicB_max and sweep_rate
        self.deltat = None # To be set in get_input() based on dynamicB_step and sweep_rate
        self.save_mag = 'true' # Calculate and save magnetization during the dynamics ?
        self.nt_mag = 100 # Calculate and save magnetization every nt_mag*deltat ps
        self.save_rho = 'false' # Save the density matrix ?
        self.nt_rho = 10 # Save the density matrix every nt_rho*deltat ps
        self.multiphonon = 'false'
        self.imbalance = 'false'
        self.states = '[200,150,88,30,10,0,1,2,3,4,5,17,41,99,173,215]'
        self.n_thread = 16

    def get_i_I0_and_i_sweep_rate(self, i_scanning):
        """
        i_scanning = i_I0 * self.n_I0s + i_sweep_rate
        Get the index of I0s and sweep_rates based on the scanning index.
        """
        i_I0 = i_scanning // self.n_sweep_rates
        i_sweep_rate = i_scanning % self.n_sweep_rates
        return (i_I0, i_sweep_rate)

    def set_directory_name(self, i_scanning):
        self.directory = f"{i_scanning+1:d}"

    def get_emat(self):
        """
        Get the basis vectors for the chosen orientation.
        emat = [ex, ey, ez] with ex, ey, ez being row-vectors
        """
        rot = R.from_euler('ZYZ', [self.alpha, self.beta, self.gamma], degrees=True)
        rotmat = rot.as_matrix()
        # emat = np.transpose( rotmat * np.eye(3) ) = np.transpose( rotmat )
        self.emat = rotmat.T
        # print_emat_array(self.emat)

    def set_loose_parameters(self, i_scanning):
        self.set_directory_name(i_scanning)
        i, j = self.get_i_I0_and_i_sweep_rate(i_scanning)
        self.I0 = self.I0s[i]
        self.sweep_rate = self.sweep_rates[j]
        self.tmax = self.dynamicB_max / self.sweep_rate * 1e+9 
        self.deltat = self.dynamicB_step / self.sweep_rate * 1e+9 # Time step in ps

    def get_input(self, i_scanning):
        self.set_directory_name(i_scanning)
        os.chdir(self.directory)
        os.system('pwd')
        self.get_emat()
        self.set_loose_parameters(i_scanning)
        with open("input.yaml", "w") as f:
            f.write(input_powder.format( \
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
        # subprocess.run(['ln', '-sf', source_dir + 'tools/tool_staircase.py', '.'])
        # subprocess.run(['ln', '-sf', source_dir + 'tools/tool_magnetization.py', '.'])
        # Skip the job script creation. Use job array instead.
        # with open("spin.job", "w") as f:
        #     f.write(job_script)
        os.chdir(root_dir)

    def submit_a_job(self, i, j):
        """
        Submit a job to the queue.
        """
        self.set_directory_name(i, j)
        os.chdir(self.directory)
        subprocess.run(['sbatch', 'spin.job'])
        os.chdir(root_dir)

    def create_directories(self):
        """
        Create directories for output files.
        """
        for i in range(self.n_scanning):
            self.set_directory_name(i)
            if not os.path.exists(self.directory):
                subprocess.run(['mkdir', '-p', self.directory])

    def get_inputs(self):
        """
        Get the input files for each orientation.
        """
        for i in range(self.n_scanning):
            self.get_input(i)

        with open("spin.job", "w") as f:
            f.write(job_array_scan.format(n_jobs=self.n_scanning))
        print("Job array script saved as spin.job in the root directory (for this set of calculations).")

    def submit_job_array(self):
        """
        Submit the job array to the queue.
        """
        subprocess.run(['sbatch', 'spin.job'])

    def check_job_status(self, i):
        """
        Is this job done?
        """
        if i == 0:
            Bs_eq = self.read_M_eq(i, take_B=True)
            if (Bs_eq is None):
                return
            if (not np.isclose(Bs_eq["B"][Bs_eq.shape[0]-1], self.staticB_max) ):
                print(f"{i+1:5d} Equilibrium job is incomplete.")
            # else:
            #     print(f"{i+1:5d} Equilibrium job is complete.")

        Bs_dy = self.read_M_dy(i, take_B=True)
        if (Bs_dy is None):
            return
        if ( not np.isclose(Bs_dy["B"][Bs_dy.shape[0]-1], self.dynamicB_max) ):
            print(f"{i+1:5d} Dynamics job is incomplete.")
        # else:
        #     print(f"{i+1:5d} Dynamics job is complete.")

    def check_all_job_status(self):
        """
        Check the status of all jobs.
        """
        for i in range(self.n_scanning):
            self.check_job_status(i)

    def read_M_eq(self, i, take_B=False):
        self.set_directory_name(i)
        # print(self.directory)
        self.set_loose_parameters(i)
        # read csv file
        fname = os.path.join(self.directory, "output/M-B.csv")
        # Check if the file exists
        if not os.path.exists(fname):
            print(f"{i+1:5d} No output file for the equilibrium job.")
            return None
        try:
            df = pd.read_csv(fname)
        except pd.errors.EmptyDataError:
            print(f"{i+1:5d} The output file for the equilibrium job is empty.")
            return None
        if take_B:
            # Take the column "B" and save it to a new data frame
            df = df[["B"]]
        else:
            # Take the column "Mz" and save it to a new data frame
            df = df[["Mz"]]
        return df

    def get_M_eq(self):
        # Just copy 1/output/M-B.csv to M-B_eq.csv using subprocess
        subprocess.run(['cp', '1/output/M-B.csv', 'M-B_eq.csv'])

    def read_M_dy(self, i, take_B=False):
        self.set_directory_name(i)
        # print(self.directory)
        self.set_loose_parameters(i)
        # read csv file
        fname = os.path.join(self.directory, f"output/T_{self.T:.1f}K_I0_{self.I0:.2e}_lambdaa_{self.lambdaa:.2f}/Bt_linear_sweep_rate_{self.sweep_rate:.1f}/magnetometry/0.000-{self.tmax:.3f}ps_dt{self.deltat:.3f}ps.dat")
        # Check if the file exists
        if not os.path.exists(fname):
            print(f"{i+1:5d} No output file for the dynamical job.")
            return None
        try:
            df = pd.read_csv(fname, sep=r'\s+', header=None)
        except pd.errors.EmptyDataError:
            print(f"{i+1:5d} The output file for the dynamical job is empty.")
            return None
        if take_B:
            # Take the second column and save it to a new data frame with a column name "B"
            column = df.iloc[:, 1]
            df = pd.DataFrame({"B": column})
        else:
            # Create a column that contains self.I0 with the name "I0"
            column1 = pd.Series([self.I0] * df.shape[0], name="I0")
            # Create a column that contains self.sweep_rate with the name "sweep_rate"
            column2 = pd.Series([self.sweep_rate] * df.shape[0], name="sweep_rate")
            # Take the second column and save it to a new data frame with a column name "B"
            column3 = df.iloc[:, 1]
            # Take the third column and save it to a new data frame with a column name "Mz"
            column4 = df.iloc[:, 2]
            # Create a new dataframe with the columns "I0", "sweep_rate", "B", and "Mz"
            df = pd.DataFrame({"I0": column1, "sweep_rate": column2, "B": column3, "Mz": column4})

        return df

    def get_M_dy(self):
        """
        Get the dynaical magnetization for all calculations.
        """
        print(0+1)
        df = self.read_M_dy(0)
        for i in range(1, self.n_scanning):
            print(i+1)
            dfi = self.read_M_dy(i)
            df = pd.concat([df, dfi], axis=0)

        # Save the data to a csv file
        df.to_csv("M-B_dy.csv", index=False)

    def remove_directories(self):
        """
        Remove all directories.
        """
        for i in range(self.n_dynamicB_steps):
            self.set_directory_name(i)
            subprocess.run(['rm', '-rf', self.directory])

if __name__ == "__main__":

    # print("Root directory: ", root_dir)

    scan = scanning()
    # scan.create_directories()
    # scan.get_inputs()
    # scan.submit_job_array()
    # scan.check_all_job_status()
    scan.get_M_eq()
    scan.get_M_dy()


    # =============================================
    # Caution!!! It will remove all directories.
    # =============================================
    # scan.remove_directories() 



