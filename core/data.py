input = """
spins:
  - 2.5
  - 2.5
  - 2.5

positions:
  - [0.00000000,     0.00000000,     0.00000000] 
  - [0.00000000,     0.00000000,     3.00000000] 
  - [0.00000000,     0.00000000,     6.00000000] 

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
    Bkqs: [0.0, 0.0]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - site: 2
    ks: [2, 2]
    qs: [0, 2]
    Bkqs: [0.0, 0.0]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - site: 3
    ks: [2, 2]
    qs: [0, 2]
    Bkqs: [0.0, 0.0]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]
  
gfactor:
  - site: 1
    gs: [2.000,  0.000,  0.000,  0.000,   2.000,  0.000,  0.000,  0.000,   2.000]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - site: 2
    gs: [2.000,  0.000,  0.000,  0.000,   2.000,  0.000,  0.000,  0.000,   2.000]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - site: 3
    gs: [2.000,  0.000,  0.000,  0.000,   2.000,  0.000,  0.000,  0.000,   2.000]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]


dipole:
  - pair: [1, 2]
    alpha: 0.0
    pi: [0.0, 0.0, 0.0]
    reference_frame: [1., 0., 0.,  0., 1., 0.,  0., 0., 1.]

  - pair: [2, 3]
    alpha: 0.0
    pi: [0.0, 0.0, 0.0]
    reference_frame: [1., 0., 0.,  0., 1., 0.,  0., 0., 1.]

  - pair: [3, 1]
    alpha: 0.0
    pi: [0.0, 0.0, 0.0]
    reference_frame: [1., 0., 0.,  0., 1., 0.,  0., 0., 1.]

ext_field: [0., 0., 0., 0., 0., 0.] # B, thetaB, phiB, E, thetaE, phiE

BET_Bgrid:
  - [0., 50., 0.01, 0., 0.] # Bmin, Bmax, Bstep, thetaB, phiB 
  - [0., 0., 0.] # E, thetaE, phiE
  - [0.6] # [2.0, 10.0, 20.0, 40.0] # T1, T2, ..., Tn

BET_Egrid:
  - [0., 0., 0.] # B, thetaB, phiB 
  - [0., 10., 11., 0., 0.] # Emin, Emax, Estep, thetaE, phiE
  - [2.0] # T1, T2, ..., Tn

BET_BEgrid:
  - [0., 200., 0.1, 0., 0.] # Bmin, Bmax, Bstep, thetaB, phiB 
  - [0., 10., 11., 0., 0.] # Emin, Emax, Estep, thetaE, phiE
  - [2.0] # T1, T2, ..., Tn

BET_Tgrid:
  - [0., 0.5, 1., 0., 0.] # B1, B2, ..., Bn, thetaB, phiB
  - [0., 1.0, 5., 10., 50., 100., 0., 0.] # E1, E2, ..., En, thetaE, phiE
  - [0.2, 1, 0.2, 2, 10, 2, 20, 300, 20] # Tmin1, Tmax1, Tstep1, Tmin2, Tmax2, Tstep2, ..., Tminn, Tmaxn, Tstepn

dynamics:
    - T: {T:.3f}          # Temperature in K
      lambdaa: {lambdaa:.2f}   # Spin phonon coupling constant in cm-1, 0.1
      I0: {I0:.1e}     # Prefactor for phonon density of states. 1e-4

    - Bt_type: '{Bt_type:s}'  # Type of the magnetic pulse as a function of time. Options: 'linear', 'pwlinear', 'sin', 'pulse'
      sweep_rate: {sweep_rate:.1f}  # Slope of the magnetic field vs time. Unit: T per ms. Used only when Bt = 'linear'.
      times: [{times:s}] # Turning points of the magnetic field in ps. Used only when Bt = 'pwlinear'.
      fields: [{fields:s}] # Magnetic field at the turning points in T. Used only when Bt = 'pwlinear'.
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

input_powder = """
spins:
  - 2.5
  - 2.5
  - 2.5

positions:
  - [0.00000000,     0.00000000,     0.00000000] 
  - [0.00000000,     0.00000000,     3.00000000] 
  - [0.00000000,     0.00000000,     6.00000000] 

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
  
gfactor:
  - site: 1
    gs: [2.000,  0.000,  0.000,  0.000,   2.000,  0.000,  0.000,  0.000,   2.000]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - site: 2
    gs: [2.000,  0.000,  0.000,  0.000,   2.000,  0.000,  0.000,  0.000,   2.000]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]

  - site: 3
    gs: [2.000,  0.000,  0.000,  0.000,   2.000,  0.000,  0.000,  0.000,   2.000]
    reference_frame: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]


dipole:
  - pair: [1, 2]
    alpha: 0.0
    pi: [0.0, 0.0, 0.0]
    reference_frame: [1., 0., 0.,  0., 1., 0.,  0., 0., 1.]

  - pair: [2, 3]
    alpha: 0.0
    pi: [0.0, 0.0, 0.0]
    reference_frame: [1., 0., 0.,  0., 1., 0.,  0., 0., 1.]

  - pair: [3, 1]
    alpha: 0.0
    pi: [0.0, 0.0, 0.0]
    reference_frame: [1., 0., 0.,  0., 1., 0.,  0., 0., 1.]

ext_field: [0., 0., 0., 0., 0., 0.] # B, thetaB, phiB, E, thetaE, phiE

BET_Bgrid:
  - [0., {staticB_max:.3f}, {staticB_step:.3f}, 0., 0.] # Bmin, Bmax, Bstep, thetaB, phiB 
  - [0., 0., 0.] # E, thetaE, phiE
  - [0.6] # [2.0, 10.0, 20.0, 40.0] # T1, T2, ..., Tn

BET_Egrid:
  - [0., 0., 0.] # B, thetaB, phiB 
  - [0., 10., 11., 0., 0.] # Emin, Emax, Estep, thetaE, phiE
  - [2.0] # T1, T2, ..., Tn

BET_BEgrid:
  - [0., 200., 0.1, 0., 0.] # Bmin, Bmax, Bstep, thetaB, phiB 
  - [0., 10., 11., 0., 0.] # Emin, Emax, Estep, thetaE, phiE
  - [2.0] # T1, T2, ..., Tn

BET_Tgrid:
  - [0., 0.5, 1., 0., 0.] # B1, B2, ..., Bn, thetaB, phiB
  - [0., 1.0, 5., 10., 50., 100., 0., 0.] # E1, E2, ..., En, thetaE, phiE
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

job_script = """#!/bin/bash -l

#SBATCH --account=m2qm-efrc
#SBATCH --qos=m2qm-efrc-b
#SBATCH --job-name=dynamics
#SBATCH --mail-type=All
#SBATCH --mail-user=shuan.liu@northeastern.edu
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH -t 1-00:00:00
#SBATCH --error=/dev/null
#SBATCH --output=/dev/null

source ~/.bash_aliases; keep_log

# Thermal equilibrium
python tool_magnetization.py

# Dynamics
python tool_staircase.py
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

job_array_scan = """#!/bin/bash -l

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
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    python /home/shuan.liu.neu/git/spin_dynamics/tools/tool_magnetization.py
fi

# Dynamics
python /home/shuan.liu.neu/git/spin_dynamics/tools/tool_staircase.py
"""

