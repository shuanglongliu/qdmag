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
  - [0.2] # [2.0, 10.0, 20.0, 40.0] # T1, T2, ..., Tn

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
      theta_B: {theta_B:.1f}    # Polar angle of the pulsed magnetic field
      phi_B: {phi_B:.1f}      # Azimuthal angle of the pulsed magnetic field

    - tmin: {tmin:.1e}             # Initial time in ps
      tmax: {tmax:.1e}             # Finial time in ps
      deltat: {deltat:.1e}      # Time step in ps

    - save_mag: {save_mag:s} # Calculate and save magnetization during the dynamics ?
      nt_mag: {nt_mag:d}  # Calculate and save magnetization every nt_mag*deltat ps
      save_rho: {save_rho:s} # Save the density matrix ?
      nt_rho: {nt_rho:d}  # Save the density matrix every nt_rho*deltat ps, nt_rho will be adjusted to be a multiple of nt_mag

n_thread: {n_thread:d} # Number of threads used in the calculation

"""

