# Tools

The `tools/` directory contains standalone utility scripts for common tasks. Each script reads input parameters from an `input.yaml` file in the working directory via `read_input()`.

## `tool_staircase.py`

Runs the quantum master equation using the **staircase approximation**. This is the primary solver for long-time (millisecond-scale) dynamics. The initial density matrix can be set from thermal equilibrium or loaded from an existing HDF5 file to continue a prior run.

```python
lio.evolve_rho()   # scheme taken from the 'method' key of input.yaml
```

The time-integration scheme is chosen by the optional `method` key in the third `dynamics` block of `input.yaml`, either `staircase` (default) or `RK4`.

On each stair the field is constant, and the density matrix is advanced as $\rho \leftarrow \exp(\mathcal{L}\Delta t)\,\rho$. How this is evaluated is chosen by the optional `exp_propagator` key in the same block, and how `L` is stored by the optional `L_storage` key (`dense`, the default, or `sparse` in CSR format). The propagators fall into three classes by accuracy:

| Class | `exp_propagator` | `L_storage` | Method |
| --- | --- | --- | --- |
| Exact | `Pade` (default) | `dense` | Forms the full matrix exponential (scaling-and-squaring Padé). Cost independent of `deltat`, but `L` must fit in memory |
| Exact to a tolerance | `Taylor` | `dense` or `sparse` | Action of the exponential by a truncated Taylor series (`expm_multiply`). Cost grows with `\|\|L*deltat\|\|` |
| | `Krylov` | `dense` or `sparse` | Arnoldi projection onto a Krylov subspace. Cost grows with `\|\|L*deltat\|\|`; uses `L` only through matrix-vector products. Options `krylov_m` (subspace dimension, default 30) and `krylov_tol` (default 1e-10) |
| Approximate | `Strang` | — (forms no `L`) | Symmetric (Strang) splitting of the coherent and dissipative parts, $\exp(\mathcal{L}\Delta t) \approx \exp(\mathcal{L}_\mathrm{c}\Delta t/2)\,\exp(\mathcal{L}_\mathrm{d}\Delta t)\,\exp(\mathcal{L}_\mathrm{c}\Delta t/2)$, on the $d \times d$ density matrix. Handles bases far too large for the others (e.g. the full 216-state Mn3 trimer). Options `strang_order` and `strang_nsub` |

The first two classes act on the vectorized density matrix ($2d^2$ real numbers, for a $d$-dimensional basis) with `L` as a $2d^2 \times 2d^2$ matrix. `Strang` applies the same $\mathcal{L} = \mathcal{L}_\mathrm{c} + \mathcal{L}_\mathrm{d}$ without storing it: in the eigenbasis of the stair Hamiltonian, $\exp(\mathcal{L}_\mathrm{c} t)$ multiplies each element $\rho_{kn}$ by the phase $e^{-i(E_k - E_n)t/\hbar}$, and $\mathcal{L}_\mathrm{d}\rho$ takes a few $d \times d$ matrix products.

The names `Taylor_sparse` and `Krylov_sparse` of earlier versions are still accepted and mean `Taylor` or `Krylov` with `L_storage: sparse`. An unrecognized name or a combination that does not exist (`Pade` or `Strang` with `L_storage: sparse`) raises at construction, and `method: RK4` requires a dense `L`.

**`Strang`.** On each stair, in the eigenbasis of the stair Hamiltonian, the density matrix is propagated as $\rho \leftarrow \exp(\mathcal{L}_\mathrm{c}\Delta t/2)\,\exp(\mathcal{L}_\mathrm{d}\Delta t)\,\exp(\mathcal{L}_\mathrm{c}\Delta t/2)\,\rho$, where $\exp(\mathcal{L}_\mathrm{c} t)$ is the exact coherent evolution and $\mathcal{L}_\mathrm{d}$ the dissipator. This is the scheme of `experiments/pulsed_field.m` in [Spinach](https://github.com/IlyaKuprov/Spinach) (commit cc7bfb78), after G. Strang, *SIAM J. Numer. Anal.* **5**, 506 (1968). $\exp(\mathcal{L}_\mathrm{d}\Delta t)$ is a Taylor series of order `strang_order` (default 2, as in Spinach) over `strang_nsub` substeps of width $\tau$ (default 1; 0 picks them so that $\tau\lVert\mathcal{L}_\mathrm{d}\rVert \le 0.5$). The model (basis, X, Φ, field at the stair midpoint) is the same as for `Pade`; only the propagator is approximate. Each stair costs a few $d \times d$ matrix products, $O(d^3)$, instead of $O(d^6)$ for `Pade`:

| $d$ | `Pade`, per stair | `Strang`, per stair (2 threads) |
| --- | --- | --- |
| 17–40 | 0.05–4.5 s (4 cores) | 0.2–0.7 ms |
| 216 | infeasible (dense `L` of 70 GB) | 25–45 ms |

The error comes from the non-commutation of the coherent and dissipative parts. For the Mn3 trimer at 0.6 K and 50 T/ms, `Strang` agrees with `Pade` to within 1e-6 μB over 0–10 T for bases of 16, 26, 34 and 42 states at `deltat` 2e4 and 1e3 ps. Check it the same way, against `Pade` on a smaller basis or by halving `deltat`, before relying on it for a new system. `Strang` runs on the CPU with `method: staircase` only, and does not support `save_drdt`.

Keep the default `Pade` for production runs that fit in memory: the alternatives only overtake it for `||L*deltat||_1` below roughly 1e4, and a typical `deltat = 1e4 ps` is far above that. 

## `tool_RK4.py`

Runs the quantum master equation using the **fourth-order Runge–Kutta (RK4)** method. Suitable for short-time, high-accuracy propagation. Shares the same interface as `tool_staircase.py`, including the option to restart from a saved density matrix. Passing `method` explicitly overrides the `method` key of `input.yaml`, so this script always runs RK4.

```python
lio.evolve_rho(method="RK4")
```

Both solvers read their output controls from the optional fourth `dynamics` block in `input.yaml` (`save_mag`, `save_rho`, `save_drdt`, and the corresponding `nt_mag`/`nt_rho`/`nt_drdt` save intervals). The block, and every key within it, may be omitted, falling back to defaults that save only the magnetization at roughly 100 points over `[tmin, tmax]`.

## `tool_magnetization.py`

Computes the **equilibrium magnetization M(B)** as a function of applied field. The `do_full_space` and `do_effective_space` flags at the top of the script choose the space, and each writes its own file: the full Hilbert space goes to `output/M-B.csv` and the reduced effective basis to `output/M-B_eff.csv`. The script as shipped has `do_effective_space = True` and `do_full_space = False`, so it writes `output/M-B_eff.csv` unless the flags are changed.

## `tool_zeeman.py`

Computes and saves the **Zeeman energy level diagram** — eigenvalues of the spin Hamiltonian as a function of applied field. Supports both the full and effective Hilbert spaces.

## `tool_eigen.py`

Diagonalizes the spin Hamiltonian at a fixed field (default: $B_z = 10^{-4}$ T) and saves the **eigenvalues** and **eigenvectors** to files. Useful for inspecting the spin levels.

## `tool_spins.py`

Computes and saves the **expectation values of individual spin operators** $\langle S_{i,\alpha} \rangle$ for each eigenstate. Useful for characterizing the spin composition of the certain states.

## `tool_projection.py`

Computes the **state composition** of the eigenstates in terms of the effective basis states at a specified field. 

## `tool_occupation.py`

Computes the **thermal equilibrium occupation probabilities** of the effective basis states over a range of magnetic field values at a given temperature.

## `tool_basis.py`

Suggests an **effective basis** for a multispin system: the pool states whose energy $E_i + g\mu_B B m_{s,i}$ comes within a cutoff of the instantaneous ground state for some field of the sweep (step 1 of the procedure for choosing an effective basis in the paper). The energies are those of the pool Hamiltonian, i.e., the isotropic exchange plus the Zeeman term of the actual $g$ tensors along $z$; the zero-field splitting is not included. Degenerate pool states are kept together.

```bash
python -m qdmag.tools.tool_basis --ecut 25            # cutoff in cm-1; field range from BT_Bgrid
python -m qdmag.tools.tool_basis --ecut 25 --bmin 0 --bmax 10
```

It prints the selected states with their energy, $S$, $m_s$, and smallest gap to the ground state, followed by a `states:` list to paste into `input.yaml`. For `examples/3-spin/staircase/26`, a cutoff of 25 cm⁻¹ gives the 26 states of that example plus the $S = 9/2$, $m_s = -9/2$ state. Check the suggestion against the full Hamiltonian (Zeeman diagram and equilibrium magnetization), and enlarge the basis until the nonequilibrium magnetization no longer changes.

## `tool_conditions_rho.py`

Validates the **physical conditions of the density matrix** $\rho(t)$ stored in an HDF5 file: normalization (Tr ρ = 1), Hermiticity, positive semidefiniteness, and the Cauchy–Schwarz inequality. Reads every `skip`-th time point to reduce I/O overhead.

## `tool_transform_rho.py`

Transforms the density matrix between representations. Converts $\rho(t)$ from the **spin (S) representation** (eigenstates of $\hat{S}_{\text{tot},z}$) to the **energy (E) representation** (eigenstates of the full Hamiltonian). Reads from one HDF5 file and writes to another.

## `tool_quadrature.py`

Generates and saves **Lebedev quadrature points and weights** for powder averaging. The average needs only two angles: with the field along the global $z$ axis and an isotropic $g$ tensor, the only geometric variable is the direction $\mathbf{n}$ of the field in the ZFS reference frame, and the first Euler angle $\alpha$ merely rotates the Hamiltonian about the field axis. Each Lebedev point is therefore read as one $\mathbf{n}$ and mapped back onto Euler angles $(\alpha, \beta, \gamma) = (0, \arccos n_z, \mathrm{atan2}(n_y, -n_x))$, which are written in degrees together with the weights to `points_and_weights.txt`. The angles are in degrees both in the file and in the returned array. Optional visualization of the sampled directions is available via matplotlib.

## `tool_powder.py`

Drives a **powder average** over the orientations generated by `tool_quadrature.py`. It reads `points_and_weights.txt` from the working directory, creates one subdirectory per orientation, and writes an `input.yaml` into each with the molecule rotated by that orientation's Euler angles: every reference frame $F$ of the template (anisotropy, $g$ tensors, exchange) becomes $F R^T$. Earlier versions rotated the ZFS frame only, which is the same thing only when $g$ and the exchange are isotropic. The `input` template and the parameters set in `powder.__init__` are meant to be edited for the system at hand.

| Method | Purpose |
| --- | --- |
| `create_directories()` | Make the per-orientation subdirectories. Run before `get_inputs()` |
| `get_inputs()` | Write each `input.yaml`, plus a SLURM job array `spin.job` that runs `tool_magnetization.py` and `tool_staircase.py` in every subdirectory |
| `submit_job_array()` | `sbatch spin.job` |
| `check_all_job_status()` | Report which orientations are missing, empty, or incomplete |
| `get_M_eq_avg()`, `get_M_dy_avg()` | Combine the per-orientation equilibrium and dynamical magnetizations into `M-B_eq.csv` and `M-B_dy.csv`, weighting each orientation by its Lebedev weight |
| `convergence_test()` | Powder average the equilibrium magnetization over a series of Lebedev degrees, to `M-B_convergence.csv` |
| `remove_directories()` | Delete every per-orientation subdirectory |

The steps in `__main__` are commented out; uncomment them one at a time, in the order above. 

## `tool_powder_dynamics.py`

Runs the **powder averaged dynamics in one job**: all orientations of `points_and_weights.txt` are evolved together (`core/powder_dynamics.py`), with the orientation as a leading batch axis of L and ρ. The molecule is rotated as in `tool_powder.py` (every reference frame $F$ becomes $F R^T$). On a GPU this replaces the one-job-per-orientation workflow of `tool_powder.py`. Run it in a directory with the `input.yaml` of a single run (molecule in its reference orientation; `device: gpu` in the third `dynamics` block for a GPU) and `points_and_weights.txt`:

```bash
python tool_powder_dynamics.py [--batch-size N]
```

Supported: `method: staircase` with `exp_propagator: Pade`, saving the magnetization only. The result goes to `output/.../magnetometry_powder/`, with t, B, the powder averaged Mx, My, Mz, and the Mz of each orientation (`Mz_1`, ..., `Mz_n`, numbered as the rows of `points_and_weights.txt`). `--batch-size` sets how many orientations are evolved together; by default all of them on the CPU (set it there if memory is short), and on a GPU as many as fit into 70% of its free memory (about 57 MB per orientation at dim 17, 0.32 GB at dim 26, 1.8 GB at dim 40). Larger batches are processed in chunks.

On the CPU each orientation gives bit for bit the numbers of its single run. Measured time per orientation and stair (single spin, Pade, ||L Δt||₁ ≈ 3.5e5), with the gain from batching over one orientation at a time:

| dim | 16 Zen 5 cores | RTX PRO 6000, batched | B200, batched |
| --- | --- | --- | --- |
| 17 | 38 ms | 12.4 ms (1.6x) | 3.0 ms (1.7x) |
| 26 | 180 ms | 96 ms (1.2x) | 9.4 ms (1.2x) |
| 40 | 1.50 s | 1.10 s (1.0x) | 61 ms (1.05x) |

Batching helps only at small dim, where one exponential does not fill the GPU, and saturates at about 32-64 orientations. For example, 350 orientations (Lebedev degree 31) at dim 17 over 10⁴ stairs take about 3 h on one B200, 12 h on an RTX PRO 6000 and 37 h on 16 Zen 5 cores.

## `tool_cspline.py`

Lives in `tools/cspline/`. Fits a monotone cubic spline (`PchipInterpolator`) to the experimental pulse data in `pulse.dat`, converting the time axis from ms to ps, and pickles the interpolant to `cspline.pickle`. That pickle is what `Bt_type: 'cspline'` loads at run time, from the working directory of the calculation.

## `tool_benchmark.py`

Times the staircase solver per stair (`update_L` and one application of exp(L Δt), separately) on the CPU or a GPU, for single spins of a chosen dimension, with the zero-field splitting scaled so that ||L Δt|| stays comparable across dimensions. Also times the host and device diagonalization (`eigh`) and prints the hardware and library versions (`info`). The number of BLAS threads must be set before NumPy is imported, so use one process per thread count:

```bash
OMP_NUM_THREADS=16 python tool_benchmark.py stairs --device cpu --dims 17 26 40 --reps 3
python tool_benchmark.py stairs --device gpu --dims 17 26 40 64 100 --reps 3
```

`powder --dim D --nbs 1 8 32` times the batched powder dynamics (`tool_powder_dynamics.py`) per orientation and stair for batches of the given sizes.

Results go to stdout, as CSV (`stairs`, `powder`, `eigh`) or JSON (`info`).

## `tool_hdf5.py`

HDF5 file utilities: estimates the **file size** for a given simulation (time duration, time step, and density matrix dimension), **combines** two sequential HDF5 output files into one, and **inspects** the contents of an existing HDF5 file.

## `tool_divergence.py`

Diagnostic tool for examining the **magnitude of the Liouville superoperator** $\mathcal{L}$ and the time-evolution operator $\exp(\mathcal{L} \Delta t)$ over a grid of magnetic field values and time steps. Useful for identifying parameter regimes where the staircase propagation may become numerically unstable.

## `tool_commutation.py`

Diagnostic tool for checking **whether the Hamiltonian commutes with itself over time**. Checks (1) if the time-independent part $\hat{H}_\text{ex} + \hat{H}_\text{ZFS}$ commutes with the instantaneous Zeeman term $\hat{H}_\text{Zee}(t)$, and (2) if the full Hamiltonian $\hat{H}(t)$ commutes with its value at $t = 0$, at several sampled times over `[tmin, tmax]`. Useful for checking whether the eigenbasis stays fixed during the pulse (e.g., for an isotropic **g**-tensor with the field along a fixed axis).

## `tool_rate.py`

Evolves the spin state using a **classical rate equation** (transition rate matrix approach) instead of the full quantum master equation. Uses the staircase approximation for time propagation and computes the magnetization from the resulting level populations. Useful as a computationally cheap reference or sanity check.

## `tool_time_step.py`

Utility for **convergence testing of the staircase time step**. Automatically generates `input.yaml` files and SLURM job scripts for a series of calculations with systematically varied step sizes $\Delta B$. Also provides methods for collecting and comparing equilibrium and dynamical magnetization results across step sizes.
