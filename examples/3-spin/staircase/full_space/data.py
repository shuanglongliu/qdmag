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
    reference_frame: [  1.000000,   0.000000,   0.000000,   0.000000,   1.000000,   0.000000,   0.000000,   0.000000,   1.000000]

  - site: 2
    ks: [2, 2]
    qs: [0, 2]
    Bkqs: [0.0556, 0.0400]
    reference_frame: [  1.000000,   0.000000,   0.000000,   0.000000,   1.000000,   0.000000,   0.000000,   0.000000,   1.000000]

  - site: 3
    ks: [2, 2]
    qs: [0, 2]
    Bkqs: [0.0556, 0.0400]
    reference_frame: [  1.000000,   0.000000,   0.000000,   0.000000,   1.000000,   0.000000,   0.000000,   0.000000,   1.000000]
  
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
  - [0., 10.000, 1.000, 0., 0.] # Bmin, Bmax, Bstep, thetaB, phiB 
  - [0.6] # [2.0, 10.0, 20.0, 40.0] # T1, T2, ..., Tn

BT_Tgrid:
  - [0., 0.5, 1., 0., 0.] # B1, B2, ..., Bn, thetaB, phiB
  - [0.2, 1, 0.2, 2, 10, 2, 20, 300, 20] # Tmin1, Tmax1, Tstep1, Tmin2, Tmax2, Tstep2, ..., Tminn, Tmaxn, Tstepn

dynamics:
    - T: 0.600          # Temperature in K
      lambdaa: 10.00   # Spin phonon coupling constant in cm-1
      I0: 1.0e-14     # Prefactor for phonon density of states. 1e-4

    - Bt_type: 'linear'  # Type of the magnetic pulse as a function of time. Options: 'linear', 'pwlinear', 'sin', 'pulse'
      sweep_rate: 50.0e-09  # Slope of the magnetic field vs time. Unit: T per ps. Used only when Bt = 'linear'.

    - tmin: {tmin:.1e}             # Initial time in ps
      tmax: {tmax:.1e}             # Finial time in ps
      deltat: {deltat:.1e}           # Time step in ps

    - save_mag: true # Calculate and save magnetization during the dynamics ?
      nt_mag: 1  # Calculate and save magnetization every nt_mag*deltat ps
      save_rho: false # Save the density matrix ?
      nt_rho: 100  # Save the density matrix every nt_rho*deltat ps, nt_rho will be adjusted to be a multiple of nt_mag
      save_drdt: false # Save the time derivative of the density matrix ?
      nt_drdt: 100 # Save the time derivative of the density matrix every nt_drdt*deltat ps

# List of spin states to be included in the dynamics. 
# This is the same as the default basis set, as determined automatically by the code.
# states: [200,150,88,30,10,0,1,2,3,4,5,17,41,99,173,215] 
states: 'all'
    
n_thread: 32 # Number of threads used in the calculation
"""

job_script = """#!/bin/bash -l

#SBATCH --account=m2qm-efrc
#SBATCH --qos=m2qm-efrc
#SBATCH --job-name=qmag_Mn3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shuan.liu@northeastern.edu
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH -t 4-00:00:00
#SBATCH --error=stderr
#SBATCH --output=stdout

. ~/.bash_aliases; keep_log

python tool_staircase.py
"""

job_array = """#!/bin/bash -l

#SBATCH --account=m2qm-efrc
#SBATCH --qos=m2qm-efrc
#SBATCH --job-name=qmag_Mn3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shuan.liu@northeastern.edu
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH -t 4-00:00:00
#SBATCH --error=stderr
#SBATCH --output=stdout
#SBATCH --array={i_start:d}-{i_end:d}%1

cd ./run_$SLURM_ARRAY_TASK_ID || exit 1

. ~/.bash_aliases; keep_log

python tool_staircase.py
"""

code_1 = """import os
from qdmag.core.common import read_input, many_spins
from qdmag.core.effective_basis import effective_basis
from qdmag.core.liouville import liouville

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, states, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)

    # Set up the effective basis
    eff = effective_basis(spins, exchange, anisotropy, dynamics, states)

    # Set up the quantum master equation
    lio = liouville(eff, dynamics)
    lio.get_initial_rho(from_file=False)
    # lio.get_initial_rho(from_file=True, 
    #     fname="./output/T_0.6K_I0_1.00e-14_lambdaa_10.00/Bt_linear_sweep_rate_5.0e-08/rho/0.000-1.000ps_dt0.001ps.h5",
    #     t_init=lio.tmin)
    lio.evolve_rho(method="staircase")
"""

code_2 = """import os
from qdmag.core.common import read_input, many_spins
from qdmag.core.effective_basis import effective_basis
from qdmag.core.liouville import liouville

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, states, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)

    # Set up the effective basis
    eff = effective_basis(spins, exchange, anisotropy, dynamics, states)

    # Set up the quantum master equation
    lio = liouville(eff, dynamics)
    # lio.get_initial_rho(from_file=False)
    lio.get_initial_rho(from_file=True, 
        fname="../run_{i_old:d}/output/T_0.6K_I0_1.00e-14_lambdaa_10.00/Bt_linear_sweep_rate_5.0e-08/rho/{tmin_old:.3f}-{tmax_old:.3f}ps_dt{deltat:.3f}ps.h5",
        t_init=lio.tmin)
    lio.evolve_rho(method="staircase")
"""

