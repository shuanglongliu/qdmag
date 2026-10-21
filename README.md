# qdmag

A Python package for simulating nonequilibrium magnetization of magnetic molecules under a time-varying external magnetic field, based on a generalized Lindblad quantum master equation.

## Overview

Magnetic molecules — including transition metal complexes and lanthanide-based complexes — are open quantum systems due to their coupling to lattice vibrations (phonons). This spin-phonon coupling governs magnetization relaxation and limits quantum coherence times. Under a time-varying external magnetic field, as in pulsed-field magnetometry, the spin system is driven out of thermal equilibrium and its dynamics cannot be captured by equilibrium methods alone.

`qdmag` implements the generalized Lindblad quantum master equation of Saito *et al.* [1] to simulate this nonequilibrium magnetization. It supports long-time evolution up to a few milliseconds through a staircase approximation and an effective Hamiltonian scheme for large spin systems.

## Features

- **Spin Hamiltonian formalism** including:
  - Magnetic exchange interaction (isotropic, anisotropic, antisymmetric, and symmetric parts of the full exchange coupling tensor **J**)
  - Zero-field splitting (ZFS) via extended Stevens operators up to 12th order
  - Zeeman interaction with an external magnetic field
- **Four supported magnetic field profiles** B(t), selected by the `Bt_type` key:
  - `linear` — linear sweep: B(t) = a·t
  - `pwlinear` — piecewise linear
  - `cspline` — monotone cubic spline fit to experimental pulse data, built by `tool_cspline.py`
  - `sin` — sinusoidal: B(t) = B₀ sin(ωt)
- **Two time-propagation methods**:
  - Staircase approximation for numerically stable long-time propagation (millisecond timescales)
  - Fourth-order Runge–Kutta (RK4) for high-accuracy short-time propagation
- **Effective Hamiltonian** construction for large spin systems, reducing the full Hilbert space dimension N = Πᵢ(2Sᵢ+1) to a computationally feasible subspace of thermally relevant states
- **Powder averaging** over molecular orientations using a Lebedev quadrature over the field direction, which needs two Euler angles rather than three (350 orientations at degree 31); all orientations can be evolved together in one job (`tool_powder_dynamics.py`), batched on a GPU
- **Liouville form** of the quantum master equation for efficient matrix-based time propagation
- **HDF5-based I/O** for density matrix storage, retrieval, and file management
- **Optional GPU backend** (CuPy) for the time evolution on CUDA GPUs, selected by the `device` key; the CPU stays the default and does not need CuPy

## Installation

```bash
git clone https://github.com/shuanglongliu/qdmag.git
```

Add the directory that contains qdmag to the environment variable PYTHONPATH. 

### Dependencies

- Python 3.x
- NumPy
- SciPy ≥ 1.15 (quadrature weights, spline fitting, matrix exponential, and Euler angle transformations). The powder averaging calls `scipy.integrate.lebedev_rule`, which was added in 1.15.
- PyYAML (parsing of `input.yaml`)
- h5py (HDF5 I/O for density matrix storage and retrieval)
- pandas (tabular output of magnetization and level population time series)
- matplotlib (optional; sparsity visualization and quadrature point plots)
- CuPy (optional; only for `device: gpu`, see **Running on a GPU**)
- pytest (optional; for the test suite in `tests/`)

### GPU support (optional)

Install a CuPy build that matches the CUDA version of the driver, for example for CUDA 12:

```bash
pip install cupy-cuda12x
```

(`cupy-cuda13x` for CUDA 13.) On a cluster, load a matching CUDA module before a GPU run, e.g. `module load cuda/12.9.1` on HiPerGator. CuPy is imported only when `device: gpu` is requested, so CPU runs work without it. CuPy compiles its kernels on first use and caches them in `~/.cupy`, so the first run on a new GPU model is slower.

## Theory

### Spin Hamiltonian

The spin Hamiltonian has three terms:

$$\hat{H} = \hat{H}_\text{ex} + \hat{H}_\text{ZFS} + \hat{H}_\text{Zee}$$

**Exchange interaction:**

$$\hat{H}_\text{ex} = -2 \sum_{ij} \hat{\boldsymbol{S}}_i^\mathrm{T} \boldsymbol{J}_{ij} \hat{\boldsymbol{S}}_j$$

The 3×3 exchange coupling matrix **J** can be decomposed into isotropic, traceless anisotropic, antisymmetric, and symmetric off-diagonal parts.

**Zero-field splitting:**

$$\hat{H}_\text{ZFS} = \sum_{i,kq} B_{i,k}^q \hat{O}_{i,k}^q(S_i)$$

where $\hat{O}_{i,k}^q$ are the extended Stevens operators. The order $k$ takes values $2, 4, \ldots, 2S$ and the component $q$ runs from $-k$ to $k$.

**Zeeman term:**

$$\hat{H}_\text{Zee} = \sum_i \mu_B \boldsymbol{B}^\mathrm{T} \mathbf{g}_i \hat{\boldsymbol{S}}_i$$

The magnetic field is assumed to be along the $z$-axis in the numerical implementation.

### Quantum Master Equation

The generalized Lindblad equation reads:

$$\frac{d\rho(t)}{dt} = \frac{1}{i\hbar}[\hat{H}, \rho(t)] - \Gamma\rho(t)$$

where the dissipator $\Gamma\rho(t) = ([X, R\rho(t)] + [X, R\rho(t)]^\dagger) \lambda^2\pi/\hbar$ accounts for spin-phonon coupling. The phonon spectral density is taken as $I(\omega) = I_0 \omega^\alpha \theta(\omega)$. The adjustable parameters are the prefactor $I_0$, the spin-phonon coupling constant $\lambda$ (typically a few $\textrm{cm}^{-1}$), and the spectral-density exponent $\alpha$ (default 2, giving a super-Ohmic bath; $\alpha < 1$ is sub-Ohmic and $\alpha = 1$ is Ohmic).

### Staircase Approximation

The continuous B(t) profile is replaced by a staircase function. Within each step the Hamiltonian is constant and the propagator has the exact closed-form solution:

$$\rho(t + \Delta t) = \exp(\mathcal{L} \Delta t) \rho(t)$$

This maintains numerical stability for time steps orders of magnitude larger than those required by Runge–Kutta methods, making millisecond simulations feasible.

### Effective Hamiltonian

For large multinuclear systems where the full Hilbert space dimension N = Πᵢ(2Sᵢ+1) is too large, an effective Hamiltonian of dimension n ≪ N is constructed by selecting thermally relevant basis states. The basis states are chosen as the lowest-energy eigenstates of isotropic exchange interaction with a perturbative Zeeman term for each value of the total $S_z$ projection by default. The small perturbative Zeeman field ensures the eigenstates are also eigenstates of the total $\hat{S}_z$ operator.

### Powder Averaging

For powder samples, the net magnetization is averaged over all molecular orientations. Two angles are enough for this, rather than three. The external field lies along the global $z$ axis, so the magnetization depends on the orientation only through the direction $\mathbf{n}$ of the field in the ZFS reference frame,

$$\mathbf{n}(\beta, \gamma) = (-\sin\beta\cos\gamma, \, \sin\beta\sin\gamma, \, \cos\beta)$$

which does not contain the first Euler angle $\alpha$. Turning the molecule by $\alpha$ leaves the magnetization unchanged. The average is then a single Lebedev quadrature over the unit sphere,

$$\overline{M} = \frac{1}{4\pi} \int M(\mathbf{n}) \, d\Omega = \sum_i w_i M(\mathbf{n}_i), \qquad \sum_i w_i = 1$$

and each Lebedev point is mapped back onto the Euler angles that the drivers consume. Quadrature points and weights are saved to a text file for inspection and record.

Convergence with the Lebedev degree can be checked with `powder.convergence_test` in `tool_powder.py`, which powder averages the equilibrium magnetization over a series of degrees. 

## Case Studies

Three case studies are included to demonstrate the package:

| System | Description | Hilbert space |
|--------|-------------|---------------|
| Ho(pzdo)₄ | Mononuclear Ho³⁺ complex, J = 8, ZFS up to 12th order | Full (17 states) |
| Spin-1/2 dimer | Two coupled S = 1/2 spins, four exchange coupling types | Full (4 states) |
| (CH₆N₃)₂MnCl₄ | Mn trimer, S = 5/2 local spins | Effective (16 or 26 states from full 216) |

## Usage


### Input file

All input parameters are loaded from an `input.yaml` file in the working directory. Example input files can be found in the `examples/` folder. 

### Run the code

```bash
python tool_staircase.py 
```

The `tool_staircase.py` script must be run in the same directory as the input file. This script and other useful scripts can be found in the `tools/` directory. See the **Tools** section below for brief descriptions of each script. 

### Running on a GPU

The time evolution runs on a CUDA GPU when the third `dynamics` block of `input.yaml` sets `device: gpu`:

```yaml
    - tmin: 0.0e+00
      tmax: 2.0e+08
      deltat: 2.0e+04
      device: gpu   # cpu (default) or gpu
      gpu_id: 0     # which CUDA device, default 0
```

Both keys are optional; without them the run is on the CPU, unchanged. On the GPU live the Liouville superoperator L, the matrix exponential, and the vectorized density matrix for the whole run. The effective basis, the diagonalization of the Hamiltonian at each stair, and the output stay on the host, so `n_threads` still matters. The output files are the same as for a CPU run.

| Supported with `device: gpu` | Not supported (raises an error) |
| --- | --- |
| `method: staircase` with `exp_propagator: Pade` (recommended) or `Krylov`; `method: RK4` | `exp_propagator: Taylor`, `Taylor_sparse`, `Krylov_sparse` |

`Krylov` works on the GPU but is slower there than on the CPU, since each substep is a chain of small matrix-vector products; use `Pade`.

**Accuracy.** GPU and CPU results are not bitwise identical (different BLAS), but agree to about 1e-10 μB per stair. Over a production-length run (`examples/3-spin/staircase/26`, 10⁴ stairs, B = 0 → 10 T) the magnetizations differ by at most 3e-8 μB, and the trace of ρ drifts no more than on the CPU. The GPU uses its own Padé matrix exponential (`core/expm.py`), which follows SciPy's choice of scaling; it drifts 3-6 times less in the trace of ρ than `cupyx.scipy.linalg.expm`.

**Memory.** A GPU run holds about 18 arrays of dimds × dimds float64 numbers, where dimds = 2 dim² is the size of L and dim is the dimension of the (effective) Hilbert space: about 0.26 GB at dim 26, 1.5 GB at dim 40, 58 GB at dim 100, 121 GB at dim 120. dim is limited to about 110 on a 96 GB GPU (RTX PRO 6000) and to about 134 on a B200 (180 GB).

**Which device to use.** The time per stair is set by double-precision (FP64) matrix products, so what counts is a GPU's FP64 rate. Measured time per stair (Pade, ||L Δt||₁ ≈ 3.5e5), compared with AMD EPYC 9655P (Zen 5) cores with OpenBLAS:

| dim | 4 cores | 16 cores | NVIDIA L4 | NVIDIA RTX PRO 6000 | NVIDIA B200 |
| --- | --- | --- | --- | --- | --- |
| 17 | 0.046 s | 0.038 s | 0.038 s | 0.020 s | 0.0044 s |
| 26 | 0.41 s | 0.18 s | 0.32 s | 0.11 s | 0.011 s |
| 40 | 4.5 s | 1.50 s | 4.0 s | 1.10 s | 0.064 s |
| 64 | 72 s | 22 s | - | 18 s | 0.82 s |
| 100 | - | 277 s | - | 259 s | 11 s |
| 120 | - | - | - | out of memory | 33 s |

- Consumer and inference GPUs with low FP64 rates (e.g., the L4) are worth about 4 CPU cores for one run: fine for development and testing, no gain for production.
- An RTX PRO 6000 is worth about 16 Zen 5 cores at large dim, and up to 2 times that at small dim, where the CPU does not use its threads well.
- For a single run these two GPUs therefore pay off only if they are free while the CPU cores are not.
- A data-center GPU with a full FP64 rate (B200, about 20 times the RTX PRO 6000) is worth about 9 times 16 Zen 5 cores at dim 17, 17 times at dim 26 and about 25 times from dim 40 on. Use it for production runs from dim ~26 on, and for dim above ~110, where only it has the memory.

`tools/tool_benchmark.py` measures the time per stair on a given machine.

### Convergence

The time step `dt` should be chosen to converge the magnetization. A practical starting point is `dt = 0.0001 T / sweep_rate`. Decreasing `dt` until the result no longer changes is recommended. 

## Tools

The `tools/` directory contains standalone utility scripts for common tasks. Each script reads input parameters from an `input.yaml` file in the working directory via `read_input()`.

### `tool_staircase.py`

Runs the quantum master equation using the **staircase approximation**. This is the primary solver for long-time (millisecond-scale) dynamics. The initial density matrix can be set from thermal equilibrium or loaded from an existing HDF5 file to continue a prior run.

```python
lio.evolve_rho()   # scheme taken from the 'method' key of input.yaml
```

The time-integration scheme is chosen by the optional `method` key in the third `dynamics` block of `input.yaml`, either `staircase` (default) or `RK4`.

The propagator that applies `exp(L*deltat)` at each stair is chosen by the optional `exp_propagator` key in the same block:

| `exp_propagator` | Method |
| --- | --- |
| `Pade` (default) | Full matrix exponential, scaling-and-squaring Padé. Cost independent of `deltat`, but `L` must fit in memory |
| `Taylor` | Action of the exponential by truncated Taylor series (`expm_multiply`). Cost grows with `\|\|L*deltat\|\|` |
| `Krylov` | Arnoldi projection onto a Krylov subspace. Cost grows with `\|\|L*deltat\|\|`; uses `L` only through matrix-vector products |
| `Taylor_sparse`, `Krylov_sparse` | As above, but `L` is built and held in sparse CSR format |

`Krylov` also accepts `krylov_m` (subspace dimension, default 30) and `krylov_tol` (default 1e-10). An unrecognized name raises at construction, and `method: RK4` requires a dense `L` so it rejects the `_sparse` variants.

Keep the default `Pade` for production runs: the alternatives only overtake it for `||L*deltat||_1` below roughly 1e4, and a typical `deltat = 1e4 ps` is far above that. 

### `tool_RK4.py`

Runs the quantum master equation using the **fourth-order Runge–Kutta (RK4)** method. Suitable for short-time, high-accuracy propagation. Shares the same interface as `tool_staircase.py`, including the option to restart from a saved density matrix. Passing `method` explicitly overrides the `method` key of `input.yaml`, so this script always runs RK4.

```python
lio.evolve_rho(method="RK4")
```

Both solvers read their output controls from the optional fourth `dynamics` block in `input.yaml` (`save_mag`, `save_rho`, `save_drdt`, and the corresponding `nt_mag`/`nt_rho`/`nt_drdt` save intervals). The block, and every key within it, may be omitted, falling back to defaults that save only the magnetization at roughly 100 points over `[tmin, tmax]`.

### `tool_magnetization.py`

Computes the **equilibrium magnetization M(B)** as a function of applied field. The `do_full_space` and `do_effective_space` flags at the top of the script choose the space, and each writes its own file: the full Hilbert space goes to `output/M-B.csv` and the reduced effective basis to `output/M-B_eff.csv`. The script as shipped has `do_effective_space = True` and `do_full_space = False`, so it writes `output/M-B_eff.csv` unless the flags are changed.

### `tool_zeeman.py`

Computes and saves the **Zeeman energy level diagram** — eigenvalues of the spin Hamiltonian as a function of applied field. Supports both the full and effective Hilbert spaces.

### `tool_eigen.py`

Diagonalizes the spin Hamiltonian at a fixed field (default: $B_z = 10^{-4}$ T) and saves the **eigenvalues** and **eigenvectors** to files. Useful for inspecting the spin levels.

### `tool_spins.py`

Computes and saves the **expectation values of individual spin operators** $\langle S_{i,\alpha} \rangle$ for each eigenstate. Useful for characterizing the spin composition of the certain states.

### `tool_projection.py`

Computes the **state composition** of the eigenstates in terms of the effective basis states at a specified field. 

### `tool_occupation.py`

Computes the **thermal equilibrium occupation probabilities** of the effective basis states over a range of magnetic field values at a given temperature.

### `tool_conditions_rho.py`

Validates the **physical conditions of the density matrix** $\rho(t)$ stored in an HDF5 file: normalization (Tr ρ = 1), Hermiticity, positive semidefiniteness, and the Cauchy–Schwarz inequality. Reads every `skip`-th time point to reduce I/O overhead.

### `tool_transform_rho.py`

Transforms the density matrix between representations. Converts $\rho(t)$ from the **spin (S) representation** (eigenstates of $\hat{S}_{\text{tot},z}$) to the **energy (E) representation** (eigenstates of the full Hamiltonian). Reads from one HDF5 file and writes to another.

### `tool_quadrature.py`

Generates and saves **Lebedev quadrature points and weights** for powder averaging. The average needs only two angles: with the field along the global $z$ axis and an isotropic $g$ tensor, the only geometric variable is the direction $\mathbf{n}$ of the field in the ZFS reference frame, and the first Euler angle $\alpha$ merely rotates the Hamiltonian about the field axis. Each Lebedev point is therefore read as one $\mathbf{n}$ and mapped back onto Euler angles $(\alpha, \beta, \gamma) = (0, \arccos n_z, \mathrm{atan2}(n_y, -n_x))$, which are written in degrees together with the weights to `points_and_weights.txt`. The angles are in degrees both in the file and in the returned array. Optional visualization of the sampled directions is available via matplotlib.

### `tool_powder.py`

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

### `tool_powder_dynamics.py`

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

### `tool_cspline.py`

Lives in `tools/cspline/`. Fits a monotone cubic spline (`PchipInterpolator`) to the experimental pulse data in `pulse.dat`, converting the time axis from ms to ps, and pickles the interpolant to `cspline.pickle`. That pickle is what `Bt_type: 'cspline'` loads at run time, from the working directory of the calculation.

### `tool_benchmark.py`

Times the staircase solver per stair (`update_L` and one application of exp(L Δt), separately) on the CPU or a GPU, for single spins of a chosen dimension, with the zero-field splitting scaled so that ||L Δt|| stays comparable across dimensions. Also times the host and device diagonalization (`eigh`) and prints the hardware and library versions (`info`). The number of BLAS threads must be set before NumPy is imported, so use one process per thread count:

```bash
OMP_NUM_THREADS=16 python tool_benchmark.py stairs --device cpu --dims 17 26 40 --reps 3
python tool_benchmark.py stairs --device gpu --dims 17 26 40 64 100 --reps 3
```

`powder --dim D --nbs 1 8 32` times the batched powder dynamics (`tool_powder_dynamics.py`) per orientation and stair for batches of the given sizes.

Results go to stdout, as CSV (`stairs`, `powder`, `eigh`) or JSON (`info`).

### `tool_hdf5.py`

HDF5 file utilities: estimates the **file size** for a given simulation (time duration, time step, and density matrix dimension), **combines** two sequential HDF5 output files into one, and **inspects** the contents of an existing HDF5 file.

### `tool_divergence.py`

Diagnostic tool for examining the **magnitude of the Liouville superoperator** $\mathcal{L}$ and the time-evolution operator $\exp(\mathcal{L} \Delta t)$ over a grid of magnetic field values and time steps. Useful for identifying parameter regimes where the staircase propagation may become numerically unstable.

### `tool_commutation.py`

Diagnostic tool for checking **whether the Hamiltonian commutes with itself over time**. Checks (1) if the time-independent part $\hat{H}_\text{ex} + \hat{H}_\text{ZFS}$ commutes with the instantaneous Zeeman term $\hat{H}_\text{Zee}(t)$, and (2) if the full Hamiltonian $\hat{H}(t)$ commutes with its value at $t = 0$, at several sampled times over `[tmin, tmax]`. Useful for checking whether the eigenbasis stays fixed during the pulse (e.g., for an isotropic **g**-tensor with the field along a fixed axis).

### `tool_rate.py`

Evolves the spin state using a **classical rate equation** (transition rate matrix approach) instead of the full quantum master equation. Uses the staircase approximation for time propagation and computes the magnetization from the resulting level populations. Useful as a computationally cheap reference or sanity check.

### `tool_time_step.py`

Utility for **convergence testing of the staircase time step**. Automatically generates `input.yaml` files and SLURM job scripts for a series of calculations with systematically varied step sizes $\Delta B$. Also provides methods for collecting and comparing equilibrium and dynamical magnetization results across step sizes.

## Tests

The `tests/` directory has a pytest suite: unit tests of the Liouville superoperator, the propagators and the matrix exponential, and regression tests that compare short runs of the examples against reference outputs in `tests/reference/` (made from the CPU code at commit `14f7d02` by `tests/make_reference.py`). Run it from the repository, pointing `QDMAG_TEST_DIR` at a directory for the files the runs write (the default is a temporary directory):

```bash
QDMAG_TEST_DIR=/path/to/scratch python -m pytest tests
```

- The suite sets the number of BLAS threads to 4, the value the references were made with, before NumPy is imported.
- Tests marked `gpu` (`tests/test_gpu.py`, and the GPU variants of the regression tests) run only if CuPy imports and a CUDA device is visible; otherwise they are skipped. GPU results are compared with tolerances, not bitwise.
- The effective basis is not unique inside degenerate blocks, and LAPACK returns different bases on different CPUs, thread counts and even nodes. Regression cases whose effective basis differs from the reference are skipped with a message rather than failed.

## Acknowledgments

This work is supported by the Center for Molecular Magnetic Quantum Materials (M2QM), an Energy Frontier Research Center (EFRC) funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award DE-SC0019330. 

## References

[1] K. Saito, S. Takesue, and S. Miyashita, Energy transport in the integrable system in contact with various types of phonon reservoirs, *Phys. Rev. E* **61**, 2397 (2000).

[2] H. Nakano and S. Miyashita, Magnetization Process of Nanoscale Iron Cluster, *J. Phys. Soc. Jpn.* **70**, 2151–2157 (2001).

[3] S. Liu, X. Chen, A. Cupo, J. N. Fry, and H.-P. Cheng, *qdmag*: A Python package for simulating nonequilibrium magnetization using quantum master equations, to be submitted to *Computer Physics Communications* (2026). 
