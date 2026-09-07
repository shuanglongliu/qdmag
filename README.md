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
- **Four supported magnetic field profiles** B(t):
  - Linear sweep: B(t) = a·t
  - Piecewise linear
  - Cubic spline fit to experimental pulse data
  - Sinusoidal: B(t) = B₀ sin(ωt)
- **Two time-propagation methods**:
  - Staircase approximation for numerically stable long-time propagation (millisecond timescales)
  - Fourth-order Runge–Kutta (RK4) for high-accuracy short-time propagation
- **Effective Hamiltonian** construction for large spin systems, reducing the full Hilbert space dimension N = Πᵢ(2Sᵢ+1) to a computationally feasible subspace of thermally relevant states
- **Powder averaging** over random molecular orientations using Lebedev + Gauss-Legendre quadrature (up to 1118 orientations)
- **Liouville form** of the quantum master equation for efficient matrix-based time propagation
- **HDF5-based I/O** for density matrix storage, retrieval, and file management

## Installation

```bash
git clone https://github.com/shuanglongliu/qdmag.git
```

Add the directory that contains qdmag to the evironment variable PYTHONPATH. 

### Dependencies

- Python 3.x
- NumPy
- SciPy (quadrature weights, spline fitting, matrix exponential, and Euler angle transformations)
- h5py (HDF5 I/O for density matrix storage and retrieval)
- pandas (tabular output of magnetization and level population time series)
- matplotlib (optional; sparsity visualization and quadrature point plots)

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

For powder samples, the net magnetization is averaged over all molecular orientations:

$$\overline{M} = \frac{1}{8\pi^2}\sum_{i,j,k} w^l_{ij} w^g_k M(\alpha_{ij}, \beta_{ij}, \gamma_k)$$

using Lebedev quadrature for the first two Euler angles (α, β) and Gauss-Legendre quadrature for the third (γ). Quadrature points and weights are saved to a text file for inspection and record.

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

### Convergence

The time step `dt` should be chosen to converge the magnetization. A practical starting point is `dt = 0.0001 T / sweep_rate`. Decreasing `dt` until the result no longer changes is recommended. 

## Tools

The `tools/` directory contains standalone utility scripts for common tasks. Each script reads input parameters from an `input.yaml` file in the working directory via `read_input()`.

### `tool_staircase.py`

Runs the quantum master equation using the **staircase approximation**. This is the primary solver for long-time (millisecond-scale) dynamics. The initial density matrix can be set from thermal equilibrium or loaded from an existing HDF5 file to continue a prior run.

```python
lio.evolve_rho(method="staircase")
```

The staircase propagator is selected by the optional `propagator` key in the third `dynamics` block of `input.yaml`, which takes one of five values:

| `propagator` | Method | Cost |
| --- | --- | --- |
| `Pade` (default) | Forms the full matrix exponential `expm(L*deltat)` with the scaling-and-squaring Padé approximant, then applies it | Fixed O(n^3), independent of `deltat`, but the matrix must fit in memory |
| `Taylor` | Evaluates the action `exp(L*deltat) @ risvrho` with a truncated Taylor series (`scipy.sparse.linalg.expm_multiply`) | Proportional to `||L*deltat||` |
| `Krylov` | Projects onto a Krylov subspace built by Arnoldi and exponentiates the small Hessenberg matrix | Proportional to `||L*deltat||`; uses `L` only through matrix-vector products, so it does not need `L` in dense form |
| `Taylor_sparse` | As `Taylor`, but `L` is built and held in sparse CSR format | As `Taylor`, with a much cheaper per-stair rebuild of `L` |
| `Krylov_sparse` | As `Krylov`, but `L` is built and held in sparse CSR format | As `Krylov`, with a much cheaper per-stair rebuild of `L` |

`Krylov` accepts two further optional keys: `krylov_m` (subspace dimension, default 30) and `krylov_tol` (per-substep error tolerance, default 1e-10, kept tight because the error accumulates over as many as 1e5 stairs). The choice is resolved once when the `liouville` object is constructed, so it costs nothing inside the time loop, and an unrecognized name raises immediately.

Which to use is governed by `||L*deltat||_1`, since only `Pade` is insensitive to it. Measured on the 1-spin example (`dimds = 578`, `krylov_m = 30`), one propagation costs (median of 5):

| `deltat` (ps) | `\|\|L*deltat\|\|_1` | `Pade` [s] | `Taylor` [s] | `Krylov` [s] | `max\|Krylov - Pade\|` |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 5.4e-01 | 0.0223 | 0.0019 | **0.0004** | 1.1e-16 |
| 0.1 | 5.4e+00 | 0.0643 | 0.0029 | **0.0006** | 6.1e-16 |
| 1 | 5.4e+01 | 0.0552 | 0.0118 | **0.0009** | 1.6e-15 |
| 10 | 5.4e+02 | 0.0746 | 0.1029 | **0.0048** | 1.7e-14 |
| 100 | 5.4e+03 | 0.1127 | 0.7232 | **0.0360** | 2.9e-13 |

All three agree to machine precision. `Krylov` is the fastest of the three over this whole range and, extrapolating its linear growth in `||L*deltat||`, overtakes `Pade` only above `||L*deltat||_1 ~ 1e4`. For long-time pulsed-field runs the time step is large — `deltat = 1e4 ps` gives `||L*deltat||_1 = 5.4e5` — and there `Pade` wins by well over an order of magnitude, which is why it remains the default. Check with `np.linalg.norm(lio.L * lio.deltat, 1)` before committing to a long run.

Timings on a laptop vary by 20-30% run to run; the orderings above are stable but the individual numbers should be read as indicative.

### Sparse superoperators

The `_sparse` propagators change how `L` is stored and rebuilt, not the mathematics. Selecting `Taylor_sparse` or `Krylov_sparse` makes `liouville` build `L0`, `LA`, `LC` and `L` with `scipy.sparse` and skip the dense superoperator entirely, which is what makes large effective bases reachable — a dense `L` for the Mn trimer full space (`dim = 216`, `dimds = 93312`) would need 65 GiB.

The rebuild of `L` at every stair is also much cheaper, because the sparse path assembles `A` and `C` from Kronecker products instead of looping in Python, and skips `get_indices_nzC` entirely. Measured in a real stair loop on the 1-spin example (`dimds = 578`, `deltat = 1 ps`, median of 15 interleaved rounds):

| `propagator` | rebuild `L` [s] | propagate [s] | total per stair [s] |
| --- | --- | --- | --- |
| `Pade` | 0.0257 | 0.0766 | 0.1023 |
| `Taylor` | 0.0314 | 0.0153 | 0.0467 |
| `Krylov` | 0.0248 | 0.0133 | 0.0381 |
| `Taylor_sparse` | **0.0064** | 0.0283 | 0.0347 |
| `Krylov_sparse` | **0.0066** | 0.0223 | **0.0288** |

All five agree to ~1e-13 after 100 stairs. The sparse rebuild is about 4x faster than the dense one here, and the advantage grows with the basis size — 12.8x on `examples/3-spin/staircase/26` (`dim = 26`), since it is the `dims**2` Python loops that are eliminated.

The sparse *propagation*, however, is a loss at these densities: both `_sparse` variants propagate more slowly than their dense twins, because sparse matrix-vector products do not beat dense BLAS until the density falls well below ~5%. And it does not fall with `dim`. Although `A` is sparse (`2*dim-1` nonzeros per row, so `~2/dim`), the dissipator is not: since `C = kron(X Rhbar, Id) - kron(Rhbar, X.T)` and `Rhbar` is essentially dense, the density of `C` — and hence of `L` — tracks the density of the spin-transition operator `X`, not `dim`. Measured across the four staircase examples (`dim` = 4, 16, 17, 26), `L` stays at 22-34% dense with no downward trend. Use the `_sparse` propagators when `L` cannot be stored densely at all, not as a general speed-up.

`method="RK4"` requires a dense `L` and raises if a `_sparse` propagator is selected.

Independently of the propagator, `construct_A` now has a vectorized twin, `construct_A_vectorized`, which computes the same superoperator as `kron(H, Id) - kron(Id, H.T)` rather than by a Python double loop over `dims**2`. It is bitwise identical and about 80x faster (0.0636 s to 0.0008 s at `dimds = 578`), and it is what the dense path uses; the original loop is kept for reference. This cuts the per-stair rebuild of a dense `L` from about 0.086 s to 0.023 s, so existing `Pade` runs at the production `deltat` get roughly 1.6x faster end to end.

Note: `propagator` replaces the earlier boolean `exp_Taylor` key. Input files that still set `exp_Taylor: true` fall back to the default `Pade` without warning.

### `tool_RK4.py`

Runs the quantum master equation using the **fourth-order Runge–Kutta (RK4)** method. Suitable for short-time, high-accuracy propagation. Shares the same interface as `tool_staircase.py`, including the option to restart from a saved density matrix.

```python
lio.evolve_rho(method="RK4")
```

Both solvers read their output controls from the optional fourth `dynamics` block in `input.yaml` (`save_mag`, `save_rho`, `save_drdt`, and the corresponding `nt_mag`/`nt_rho`/`nt_drdt` save intervals). The block, and every key within it, may be omitted, falling back to defaults that save only the magnetization at roughly 100 points over `[tmin, tmax]`.

### `tool_magnetization.py`

Computes the **equilibrium magnetization M(B)** as a function of applied field. Supports both the full Hilbert space and the reduced effective basis. Output is written to `output/M-B.csv`.

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

Generates and saves **Lebedev + Gauss-Legendre quadrature points and weights** for powder averaging. Euler angles (α, β, γ) in degrees and combined weights are written to `points_and_weights.txt`. Optional visualization of the Lebedev sphere points and Gauss-Legendre γ points is available via matplotlib.

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

## Acknowledgments

This work is supported by the Center for Molecular Magnetic Quantum Materials (M2QM), an Energy Frontier Research Center (EFRC) funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award DE-SC0019330. 

## References

[1] K. Saito, S. Takesue, and S. Miyashita, Energy transport in the integrable system in contact with various types of phonon reservoirs, *Phys. Rev. E* **61**, 2397 (2000).

[2] H. Nakano and S. Miyashita, Magnetization Process of Nanoscale Iron Cluster, *J. Phys. Soc. Jpn.* **70**, 2151–2157 (2001).

[3] S. Liu, X. Chen, A. Cupo, J. N. Fry, and H.-P. Cheng, *qdmag*: A Python package for simulating nonequilibrium magnetization using quantum master equations, to be submitted to *J. Comput. Phys.* (2026). 
