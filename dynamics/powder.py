# This is a standalong utility for calculating the powder avarage of magnetization

import os
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R
from spin_dynamics.dynamics.common import print_emat_array
from spin_dynamics.dynamics.data import input_powder
from spin_dynamics.dynamics.data import job_script
from spin_dynamics.dynamics.data import job_array

source_dir = "/home/shuan.liu.neu/git/spin_dynamics/"
root_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

class powder:
    def __init__(self):
        self.load_points_and_weights()

        # Parameters for automatically determining input parameters for spin dynamics
        dB = 0.01 # Height of each stair step in T. deltat will be set to dB/sweep_rate.
        B_max = 10.0 # Maximum magnetic field in T. tmax will be set to B_max/sweep_rate. Assuming that the magnetic field is linear in time.

        # Input parameters for spin dynamics
        self.T = 0.6 # Kelvin
        self.lambdaa = 10.0 # cm-1
        self.I0 = 1e-12 # Prefactor for phonon density of states
        self.Bt_type = 'linear' # Type of the magnetic pulse as a function of time
        self.sweep_rate = 1000.0 # T per ms
        self.times = '[0.0, 1.0e+9, 10.0e+9]' # Turning points of the magnetic field in ps
        self.fields = '[0.0, 10.0, 100.0]' # Magnetic field at the turning points in T
        self.omega = 0.2 # Angular frequency of the sine wave in rad ms^-1
        self.amplitude = 65.0 # Amplitude of the sine wave in T
        self.theta_B = 0.0 # Polar angle of the pulsed magnetic field
        self.phi_B = 0.0 # Azimuthal angle of the pulsed magnetic field
        self.tmin = 0e+9 # Initial time in ps
        self.tmax = B_max / self.sweep_rate * 1e+9 # Final time in ps
        self.deltat = dB / self.sweep_rate * 1e+9 # Time step in ps
        self.save_mag = 'true' # Calculate and save magnetization during the dynamics ?
        self.nt_mag = 1 # Calculate and save magnetization every nt_mag*deltat ps
        self.save_rho = 'false' # Save the density matrix ?
        self.nt_rho = 10 # Save the density matrix every nt_rho*deltat ps
        self.multiphonon = 'false'
        self.imbalance = 'false'
        self.states = '[200,150,88,30,10,0,1,2,3,4,5,17,41,99,173,215]'
        self.n_thread = 16

    def load_points_and_weights(self):
        """
        Load points and weights from a file.
        """
        points_and_weights = np.loadtxt('points_and_weights.txt', comments='#')
        self.n_points = points_and_weights.shape[0]
        self.alphas = points_and_weights[:, 0]
        self.betas = points_and_weights[:, 1]
        self.gammas = points_and_weights[:, 2]
        self.weights = points_and_weights[:, 3]

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
        subprocess.run(['ln', '-sf', source_dir + 'tools/tool_staircase.py', '.'])
        subprocess.run(['ln', '-sf', source_dir + 'tools/tool_magnetization.py', '.'])
        with open("input.yaml", "w") as f:
            f.write(input_powder.format( \
                exx=self.emat[0, 0], exy=self.emat[0, 1], exz=self.emat[0, 2],\
                eyx=self.emat[1, 0], eyy=self.emat[1, 1], eyz=self.emat[1, 2],\
                ezx=self.emat[2, 0], ezy=self.emat[2, 1], ezz=self.emat[2, 2],\
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
        # Skip the job script creation. Use job array instead.
        # with open("spin.job", "w") as f:
        #     f.write(job_script)
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


if __name__ == "__main__":

    print("Root directory: ", root_dir)

    pow = powder()
    # pow.create_directories()
    pow.get_inputs()
    # pow.submit_job_array()

