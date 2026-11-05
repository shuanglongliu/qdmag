# Input reference

Every tool reads its parameters from `input.yaml` in the working directory. Complete examples are in `examples/`.

## Conventions

- **Units:** energies and coupling constants in cm⁻¹, magnetic fields in T, times in ps, temperatures in K, and angular frequencies in rad ps⁻¹.
- **Reference frames:** each `reference_frame` is a 3×3 matrix, given row by row. Its rows are the local x, y, and z axes expressed in the lab frame.
- **Indices:** site indices (`pair`, `site`) start from 1. State indices (`states`) start from 0.

## Keys

| Key | Meaning |
|---|---|
| `spins` | List of the local spins $S_i$ (or $J$ for lanthanide ions). |
| `exchange` | Optional list of spin pairs. `pair`: indices i, j of the two spins, starting from 1. `coupling_matrix`: the nine elements of $\mathbf{J}_{ij}$ in cm⁻¹, row by row, with $H_\mathrm{ex} = -2 \sum \mathbf{S}_i^\mathrm{T} \mathbf{J}_{ij} \mathbf{S}_j$. `reference_frame`: the frame of $\mathbf{J}_{ij}$. |
| `anisotropy` | Optional list of sites. `site`: index of the spin, starting from 1. `ks`, `qs`, `Bkqs`: lists of $k$, $q$, and $B_k^q$ (in cm⁻¹) of the extended Stevens operators. `reference_frame`: the frame of the Stevens operators. |
| `gfactor` | List of sites. `site`. `gs`: the nine elements of the g-tensor, row by row. `reference_frame`: the frame of the g-tensor. |
| `BT_Bgrid` | Field grid for equilibrium properties and Zeeman diagrams: `[Bmin, Bmax, Bstep, thetaB, phiB]` (T and degrees), followed by a list of temperatures. |
| `BT_Tgrid` | Temperature grid for equilibrium properties: a list of fields followed by `thetaB` and `phiB`, and a list of `[Tmin, Tmax, Tstep]` triples. |
| `dynamics`, block 1 | `T`: temperature of the phonon bath. `lambdaa`: spin-phonon coupling $\lambda$ in cm⁻¹. `I0`: prefactor of the spectral density $I(\omega) = I_0 \omega^\alpha \theta(\omega)$, in $\mathrm{ps}^{\alpha+1}\,\mathrm{rad}^{-\alpha}$ (ps³ rad⁻² for $\alpha = 2$). `alpha`: $\alpha$ (default 2). |
| `dynamics`, block 2 | `Bt_type`: `linear` (key `sweep_rate` in T/ps), `pwlinear` (keys `times` in ps and `fields` in T at the turning points), `sin` (keys `omega` in rad ps⁻¹ and `amplitude` in T), or `cspline` (reads `cspline.pickle` from the working directory). |
| `dynamics`, block 3 | `tmin`, `tmax`, `deltat`: initial time, final time, and time step. `method`: `staircase` (default) or `RK4`. `exp_propagator`: `Pade` (default), `Taylor`, `Krylov`, or `Strang` (see [tools](tools.md)). `L_storage`: `dense` (default) or `sparse`, for `Taylor` and `Krylov`. Optional `krylov_m`, `krylov_tol`, `strang_order`, `strang_nsub`, `device` (`cpu` or `gpu`), and `gpu_id`. |
| `dynamics`, block 4 | Optional output control: `save_mag`, `save_rho`, `save_drdt` (true or false), and `nt_mag`, `nt_rho`, `nt_drdt`, the saving intervals in units of `deltat`. |
| `states` | Optional basis for the time evolution of a multispin system: `default` or an empty list (the minimal basis, one state per m_s), `all` (the full Hilbert space), or a list of pool indices starting from 0. |
| `n_threads` | Number of threads. |

## Options in detail

This section states, for each option, its type, whether it is required, its default, the allowed values, and what the code does with it. "Required" means that `read_input()` fails with a `KeyError` when the key is missing. Symbols follow the [Theory](theory.md) page.

### `spins` (required)

A list of numbers, one per spin center: the spin quantum number $S_i$ (or $J_i$ for a lanthanide ion), integer or half-integer, e.g. `[2.5, 2.5, 2.5]`. The order fixes the site indices used elsewhere: the first entry is site 1. The full Hilbert space has dimension $\prod_i (2S_i + 1)$, and $S_\mathrm{max} = \sum_i S_i$.

### `exchange` (optional, default: no exchange)

A list of pairs, each listed once. Each entry has three keys:

- `pair`: `[i, j]`, the site indices of the two spins (from 1).
- `coupling_matrix`: the nine elements of the $3 \times 3$ matrix $\mathbf{J}'_{ij}$ in cm⁻¹, row by row, in the frame `reference_frame`.
- `reference_frame`: the $3 \times 3$ matrix $F$ of that frame, row by row (see Conventions above).

The spin components in the local frame are $S'_{i,a} = \sum_c F_{ac} S_{i,c}$, and the exchange Hamiltonian is

$$
H_\mathrm{ex} = -2 \sum_{\text{pairs}} \sum_{a,b} J'_{ij,ab}\, S'_{i,a} S'_{j,b} = -2 \sum_{\text{pairs}} \mathbf{S}_i^\mathrm{T} F^\mathrm{T} \mathbf{J}'_{ij} F\, \mathbf{S}_j .
$$

With this sign, $J > 0$ is ferromagnetic and $J < 0$ is antiferromagnetic. A pair with a zero matrix may be listed; it contributes nothing. The isotropic part of each pair, $\tfrac{1}{3}\operatorname{tr}\mathbf{J}'_{ij}$, also defines the pool basis of a multispin system (see `states` below).

### `anisotropy` (optional, default: no anisotropy)

A list of sites, with at most one entry per site. Each entry has these keys:

- `site`: the site index (from 1).
- `ks`, `qs`, `Bkqs`: three lists of equal length. Term $n$ is $B_{k_n}^{q_n} O_{k_n}^{q_n}$, where $O_k^q$ is the extended Stevens operator built from the local spin components $S'_{i,a}$ and $B_k^q$ is in cm⁻¹. The ranks $k = 1, 2, 3, 4, 6, 8, 10, 12$ with $-k \le q \le k$ are implemented, and $k = q = 0$ gives zero. For example, $O_2^0 = 3S_z'^2 - S(S+1)$ and $O_2^2 = (S_+'^2 + S_-'^2)/2$, so the usual $D$ and $E$ correspond to $B_2^0 = D/3$ and $B_2^2 = E$.
- `reference_frame`: the frame $F$ in which the operators are defined.

The anisotropy Hamiltonian is $H_\mathrm{ani} = \sum_\text{sites} \sum_n B_{k_n}^{q_n} O_{k_n}^{q_n}(\mathbf{S}'_i)$.

### `gfactor` (required)

A list with exactly one entry for each spin. Each entry has these keys:

- `site`: the site index (from 1).
- `gs`: the nine elements of $\mathbf{g}'_i$, row by row, in the frame `reference_frame`.
- `reference_frame`: that frame, $F$.

In the lab frame the tensor is $\mathbf{g}_i = F^\mathrm{T} \mathbf{g}'_i F$. The magnetic moment, in $\mu_B$, is $\hat{\mathbf{M}} = -\sum_i \mathbf{g}_i \hat{\mathbf{S}}_i$, and the Zeeman term is $H_\mathrm{Z} = -\mu_B \mathbf{B} \cdot \hat{\mathbf{M}}$. In the time evolution the field is along the lab $z$ axis: $H(t) = H_0 - \mu_B B(t) \hat{M}_z$, with $H_0 = H_\mathrm{ex} + H_\mathrm{ani}$.

### `BT_Bgrid` (required)

A list of two items:

1. `[Bmin, Bmax, Bstep, thetaB, phiB]`: the field grid in T, and the field direction in degrees as spherical angles in the lab frame, so that $\mathbf{B} = B(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$. The grid has $\lfloor (B_\mathrm{max} - B_\mathrm{min})/B_\mathrm{step} \rfloor + 1$ points from `Bmin`. Choose `Bmax - Bmin` as a multiple of `Bstep`, so that the grid ends exactly at `Bmax`.
2. A list of temperatures in K.

`tool_magnetization.py` computes $M(B)$ on this grid for every temperature. `tool_zeeman.py` and `tool_occupation.py` use the field grid only. `tool_basis.py` takes its default field range from `Bmin` and `Bmax`. The direction `thetaB`, `phiB` applies to these equilibrium tools only; the time evolution always has the field along $z$.

### `BT_Tgrid` (required)

A list of two items:

1. The fields `[B1, ..., Bn, thetaB, phiB]`.
2. The temperature ranges `[Tmin1, Tmax1, Tstep1, Tmin2, ...]`.

`read_input()` requires this key, but no tool in the current release uses it, so any well-formed value works.

### `dynamics` (required by the time-evolution tools)

A list of three or four blocks, in this order.

**Block 1: phonon bath.** The model is described in [Theory](theory.md).

| Key | Required | Meaning |
|---|---|---|
| `T` | yes | Bath temperature in K. The initial density matrix is the thermal state at this temperature. |
| `lambdaa` | yes | Spin–phonon coupling $\lambda$ in cm⁻¹. The dissipator scales as $\lambda^2$. |
| `I0` | yes | Prefactor $I_0$ of the spectral density $I(\omega) = I_0\,\omega^\alpha\,\theta(\omega)$, in $\mathrm{ps}^{\alpha+1}\,\mathrm{rad}^{-\alpha}$. The dissipator scales as $I_0$. |
| `alpha` | no, default 2 | Exponent $\alpha$: $< 1$ is sub-Ohmic, $= 1$ Ohmic, and $> 1$ super-Ohmic. |

**Block 2: field sweep $B(t)$.** $t$ is in ps and $B$ in T. `Bt_type` selects the shape, and each shape takes its own keys:

| `Bt_type` | Keys | $B(t)$ |
|---|---|---|
| `linear` | `sweep_rate` (T/ps) | `sweep_rate` $\cdot\, t$. $B = 0$ at $t = 0$, not at `tmin`. 50 T/ms is `5.0e-08`. |
| `pwlinear` | `times` (ps), `fields` (T), lists of equal length | Linear interpolation between the points (`times[n]`, `fields[n]`), with `times` increasing. Outside the range, the first or last segment is extrapolated linearly. |
| `sin` | `omega` (rad/ps), `amplitude` (T) | `amplitude` $\cdot \sin($`omega` $\cdot\, t)$. |
| `cspline` | none | The callable pickled in `cspline.pickle` in the working directory, evaluated at $t$ in ps. It is made by `tools/cspline/tool_cspline.py` from measured pulse data. |

Any other `Bt_type` stops the run with an error message.

**Block 3: time integration.**

| Key | Required | Meaning |
|---|---|---|
| `tmin`, `tmax` | yes | Initial and final times in ps. The run starts from the thermal state of $H(t_\mathrm{min})$ at temperature `T`. |
| `deltat` | yes | Time step $\Delta t$ in ps: the width of one stair (staircase) or one step (RK4). |
| `method` | no, default `staircase` | `staircase`: $B(t)$ is replaced by a constant field on each stair, taken at the stair midpoint $t_\mathrm{min} + (n + \tfrac12)\Delta t$, and $\rho \leftarrow e^{\mathcal{L}\Delta t}\rho$ is applied once per stair, so large $\Delta t$ is stable. `RK4`: classical fourth-order Runge–Kutta with the field at $t$, $t + \Delta t/2$ and $t + \Delta t$. It needs a dense $\mathcal{L}$, and $\Delta t$ is reset to $(t_\mathrm{max} - t_\mathrm{min})/\lfloor (t_\mathrm{max} - t_\mathrm{min})/\Delta t \rfloor$. RK4 is for short, small-step runs. |
| `exp_propagator` | no, default `Pade` | How the staircase evaluates $e^{\mathcal{L}\Delta t}\rho$ (see [Tools](tools.md)). `Pade`: forms the full matrix exponential (scaling and squaring), which is exact. `Taylor`: the action of the exponential, by `scipy.sparse.linalg.expm_multiply`. `Krylov`: Arnoldi projection, with `krylov_m` and `krylov_tol`. `Strang`: symmetric splitting of the coherent and dissipative parts on the $d \times d$ density matrix, which never forms $\mathcal{L}$ and is approximate. It is restricted to `device: cpu` and `method: staircase`, and to `save_drdt: false`. `Taylor_sparse` and `Krylov_sparse` (before 1.3.0) are still accepted and set `L_storage: sparse`. The RK4 method does not use this option. |
| `L_storage` | no, default `dense` | `dense` or `sparse` (CSR) storage of $\mathcal{L}$, a $2d^2 \times 2d^2$ real matrix. `sparse` applies to `Taylor` and `Krylov` only; `Pade` and `Strang` raise an error with it. |
| `krylov_m` | no, default 30 | `Krylov` only: the maximum dimension of the Krylov subspace, capped at $2d^2$. The step is split into the fewest substeps of width $\tau$ with $\tau\lVert\mathcal{L} - \mu I\rVert_1 \le \max(m/2, 1)$, where $\mu$ is the mean of the diagonal of $\mathcal{L}$. |
| `krylov_tol` | no, default `1e-10` | `Krylov` only: the tolerance of the a posteriori error estimate of each substep, relative to $\lVert\rho\rVert$. A substep that misses it is halved, up to 10 times, after which the run stops with an error. |
| `strang_order` | no, default 2 | `Strang` only: the order $K \ge 1$ of the Taylor series of $e^{\mathcal{L}_\mathrm{d}\tau}$. $K = 2$ (as in Spinach) keeps the scheme second order in $\Delta t$; $K = 1$ makes it first order. |
| `strang_nsub` | no, default 1 | `Strang` only: the number $N_s \ge 0$ of substeps of the dissipative factor, $\tau = \Delta t/N_s$. With 0, $N_s = \max(1, \lceil 4\,\mathrm{lam}\,\lVert\tilde X\rVert_F \lVert\tilde R\rVert_F\,\Delta t / 0.5 \rceil)$ is chosen on every stair, so that $\tau\lVert\mathcal{L}_\mathrm{d}\rVert \le 0.5$. |
| `device` | no, default `cpu` | `cpu` (NumPy/SciPy) or `gpu` (CuPy, CUDA). On a GPU only `Pade` and `Krylov` with a dense $\mathcal{L}$ are available. |
| `gpu_id` | no, default 0 | The index of the GPU when `device: gpu`. |

**Block 4: output (optional).** The whole block, and each key in it, may be omitted.

| Key | Default | Meaning |
|---|---|---|
| `save_mag` | `true` | Write $t$, $B$, $M_x$, $M_y$, $M_z$ (in $\mu_B$) to a CSV file under `output/.../magnetometry/`. |
| `nt_mag` | about 1/100 of the stairs | Save the magnetization every `nt_mag` stairs: $\max(\mathrm{round}(N/100), 1)$, with $N = \mathrm{round}((t_\mathrm{max} - t_\mathrm{min})/\Delta t)$. |
| `save_rho` | `false` | Write the density matrix to an HDF5 file (for restarts and analysis). |
| `nt_rho` | 1 | The interval in stairs for `save_rho`. |
| `save_drdt` | `false` | Write the diagonal of $d\rho/dt$ (the coherent, dissipative and total population changes) to a CSV file. It needs $\mathcal{L}$, so `Strang` rejects it. |
| `nt_drdt` | 1 | The interval in stairs for `save_drdt`. |

The intervals are nested from fine to coarse, `nt_mag` ≤ `nt_rho` ≤ `nt_drdt`, among the outputs that are switched on. Each coarser interval is rounded down to a multiple of the finer one, and at least one multiple is kept. The number of stairs is then rounded to a multiple of the coarsest interval. If that does not reproduce `tmax` to within 1e-3 ps, the run stops with an error that names the interval to change. To avoid this, choose `tmax - tmin` as a multiple of `deltat` times the coarsest active interval.

The output directory encodes `T`, `I0`, `lambdaa` and the sweep, e.g. `output/T_0.6K_I0_1.00e-14_lambdaa_10.00/Bt_linear_sweep_rate_5.0e-08/`. Concurrent runs must therefore use separate working directories.

### `states` (optional, default `default`)

The basis for the time evolution of a multispin system. For a single spin it is ignored, and the full space of $2S + 1$ eigenstates of $S_z$ is used.

For a multispin system, the candidates (the *pool*) are the eigenstates of the isotropic exchange plus a Zeeman term of $10^{-4}$ T along $z$ with $g = 2$. They are sorted by energy, numbered from 0, and phased by the Condon–Shortley convention, so each one has a definite total $S$ and $M$. `tool_basis.py` prints them with their energy, $S$ and $M$. The allowed values are:

- `default` or `[]`: the minimal basis, the lowest pool state for each $M = -S_\mathrm{max}, \dots, S_\mathrm{max}$, so $d = 2S_\mathrm{max} + 1$. If the isotropic exchange is zero, a warning says that this basis may not be suitable.
- `all` (or `full`): all $\prod_i (2S_i + 1)$ pool states.
- A list of pool indices, e.g. `[0, 1, 2, 3, 4, 5]`: exactly these states, so $d$ is the length of the list. A warning is printed if the list contains only part of a degenerate set of pool states, which should be kept together.

In every case, the phonons couple states $i$ and $j$ of the basis if and only if $|M_i - M_j| = 1$ ($X_{ij} = 1$, and 0 otherwise). Section 4.3 of the paper and `tool_basis.py` describe how to choose and check a basis.

### `n_threads` (required)

A positive integer. Each tool sets the environment variable `OMP_NUM_THREADS` to this value at start-up. NumPy has already been imported by then, and most BLAS libraries read the variable only when they are loaded. To be sure of the thread count, also set `OMP_NUM_THREADS` in the shell (or the job script) before starting Python.
