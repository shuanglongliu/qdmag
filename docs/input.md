# The input file `input.yaml`

Every tool reads its parameters from `input.yaml` in the working directory. Complete examples are in `examples/`.

## Conventions

- **Units:** energies and coupling constants in cm⁻¹, magnetic fields in T, times in ps, temperatures in K, and angular frequencies in rad ps⁻¹.
- **Reference frames:** each `reference_frame` is a 3×3 matrix, given row by row. Its rows are the local x, y, and z axes expressed in the lab frame.
- **Indices:** site indices (`pair`, `site`) start from 1. State indices (`states`) start from 0.

## Keys

| Key | Meaning |
|---|---|
| `spins` | List of the local spins Sᵢ (or J for lanthanide ions). |
| `exchange` | Optional list of spin pairs. `pair`: indices i, j of the two spins, starting from 1. `coupling_matrix`: the nine elements of **J**ᵢⱼ in cm⁻¹, row by row, with H_ex = −2 Σ **S**ᵢᵀ **J**ᵢⱼ **S**ⱼ. `reference_frame`: the frame of **J**ᵢⱼ. |
| `anisotropy` | Optional list of sites. `site`: index of the spin, starting from 1. `ks`, `qs`, `Bkqs`: lists of k, q, and B_k^q (in cm⁻¹) of the extended Stevens operators. `reference_frame`: the frame of the Stevens operators. |
| `gfactor` | List of sites. `site`. `gs`: the nine elements of the g-tensor, row by row. `reference_frame`: the frame of the g-tensor. |
| `BT_Bgrid` | Field grid for equilibrium properties and Zeeman diagrams: `[Bmin, Bmax, Bstep, thetaB, phiB]` (T and degrees), followed by a list of temperatures. |
| `BT_Tgrid` | Temperature grid for equilibrium properties: a list of fields followed by `thetaB` and `phiB`, and a list of `[Tmin, Tmax, Tstep]` triples. |
| `dynamics`, block 1 | `T`: temperature of the phonon bath. `lambdaa`: spin-phonon coupling λ in cm⁻¹. `I0`: prefactor of the spectral density I(ω) = I0 ω^α θ(ω), in ps^(α+1) rad^(−α) (ps³ rad⁻² for α = 2). `alpha`: α (default 2). |
| `dynamics`, block 2 | `Bt_type`: `linear` (key `sweep_rate` in T/ps), `pwlinear` (keys `times` in ps and `fields` in T at the turning points), `sin` (keys `omega` in rad ps⁻¹ and `amplitude` in T), or `cspline` (reads `cspline.pickle` from the working directory). |
| `dynamics`, block 3 | `tmin`, `tmax`, `deltat`: initial time, final time, and time step. `method`: `staircase` (default) or `RK4`. `exp_propagator`: `Pade` (default), `Taylor`, `Krylov`, `Taylor_sparse`, or `Krylov_sparse`. Optional `krylov_m`, `krylov_tol`, `device` (`cpu` or `gpu`), and `gpu_id`. |
| `dynamics`, block 4 | Optional output control: `save_mag`, `save_rho`, `save_drdt` (true or false), and `nt_mag`, `nt_rho`, `nt_drdt`, the saving intervals in units of `deltat`. |
| `states` | Optional basis for the time evolution of a multispin system: `default` or an empty list (the minimal basis, one state per m_s), `all` (the full Hilbert space), or a list of pool indices starting from 0. |
| `n_threads` | Number of threads. |
