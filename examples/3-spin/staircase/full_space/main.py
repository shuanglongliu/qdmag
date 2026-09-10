import os
import subprocess
import pandas as pd
from data import input, job_script, job_array, code_1, code_2

root_dir = os.path.dirname(os.path.abspath(__file__))

class time_slices:
    def __init__(self, n_runs, t_span, deltat, first_run=1):
        # Number of time slides, one for each run
        self.n_runs = n_runs
        # Time span for each run in ps
        self.tspan = t_span
        # Number of time steps in each run
        self.n_steps = int(t_span / deltat)
        # Time step size in ps
        self.deltat = deltat
        # First time slice to run
        self.first_run = first_run

        # Save magnetization every dt_mag ps
        dt_mag = 0.1  # in ps
        # Save density matrix every dt_rho ps
        dt_rho = 10.0  # in ps

        self.nt_mag = int(dt_mag / deltat)
        self.nt_rho = int(dt_rho / deltat)

    def get_a_slice(self, i):
        i = i + self.first_run - 1
        self.tmin = i * self.tspan
        self.tmax = (i+1) * self.tspan
        self.tmin_old = (i-1) * self.tspan
        self.tmax_old = i * self.tspan
        self.dir_name = f"run_{i + self.first_run}"

    def create_directories(self):
        # Create directories for each time slice
        for i in range(self.n_runs):
            dir_name = f"run_{i + self.first_run}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

    def get_input_files(self):
        # Get input files for each time slice
        for i in range(self.n_runs):
            self.get_a_slice(i)
            os.chdir(self.dir_name)
            subprocess.run(["pwd"])
            with open("input.yaml", "w") as f:
                f.write(input.format(tmin=self.tmin, tmax=self.tmax, deltat=self.deltat,
                                     nt_mag=self.nt_mag, nt_rho=self.nt_rho))
            with open("spin.job", "w") as f:
                f.write(job_script)
            with open("tool_staircase.py", "w") as f:
                if i + self.first_run == 1:
                    f.write(code_1)
                else:
                    f.write(code_2.format(i_old=i+self.first_run-1, tmin_old=self.tmin_old, tmax_old=self.tmax_old, deltat=self.deltat))
            os.chdir(root_dir)
        with open("spin.job", "w") as f:
            f.write(job_array.format(i_start=self.first_run, i_end=self.first_run+self.n_runs-1))

    def submit_jobs(self):
        subprocess.run(["sbatch", "spin.job"])

    def gather_results(self):
        # Get input files for each time slice
        for i in range(self.n_runs):
            self.get_a_slice(i)
            os.chdir(self.dir_name)
            subprocess.run(["pwd"])
            # The magnetometry file is a csv with the columns t, B, Mx, My and Mz.
            # See set_up_outdirs in core/liouville.py for the path.
            fname = f"output/T_0.6K_I0_1.00e-14_lambdaa_10.00/Bt_linear_sweep_rate_5.0e-08/magnetometry/t{self.tmin:.3f}-{self.tmax:.3f}ps_dt{self.deltat:.3f}ps.csv"
            if i == 0:
                data = pd.read_csv(fname)
            else:
                tada = pd.read_csv(fname)
                data = pd.concat([data, tada], ignore_index=True)
            os.chdir(root_dir)
        with open("M-B.dat", "w") as f:
            for t, B, Mx, My, Mz in data[["t", "B", "Mx", "My", "Mz"]].itertuples(index=False):
                f.write(f"{t:15.3f} {B:15.6e} {Mx:15.6e} {My:15.6e} {Mz:15.6e}\n")

if __name__ == "__main__":
    # Define the number of runs, steps, and time step size
    n_runs = 1
    t_span = 1.0e5 # in ps
    deltat = 2.0e4  # in ps

    # Create an instance of time_slices
    ts = time_slices(n_runs, t_span, deltat, first_run=1)
    ts.create_directories()
    ts.get_input_files()
    # ts.submit_jobs()
    # ts.gather_results()

