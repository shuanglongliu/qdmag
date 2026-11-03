# Usage

## Input file

All input parameters are loaded from an `input.yaml` file in the working directory. Example input files can be found in the `examples/` folder. All keys, their units, and the conventions for reference frames and indices are listed in [The input file](input.md).

## Run the code

```bash
python -m qdmag.tools.tool_staircase
```

or, without installation, `python /path/to/qdmag/tools/tool_staircase.py`. The script must be run in the same directory as the input file. This script and other useful scripts can be found in the `tools/` directory. See [Tools](tools.md) for brief descriptions of each script. 

## Running on a GPU

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
| `method: staircase` with `exp_propagator: Pade` (recommended) or `Krylov`; `method: RK4` | `exp_propagator: Taylor`, `Taylor_sparse`, `Krylov_sparse`, `Strang` |

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

## Convergence

The staircase propagator is exact on each stair, so the time-step error comes mostly from avoided level crossings, where the eigenstates change within a narrow field window. For two levels whose energy difference changes with the field at a rate $\Delta k$ (in cm⁻¹/T) and whose minimal gap is $\Delta$, the window has a width $\delta B = \Delta / \Delta k$.

- Choose `deltat` so that the field step per stair, `sweep_rate * deltat`, is no larger than about $2\,\delta B$ of the narrowest avoided crossing between the populated levels. $\delta B$ can be estimated in advance from the Zeeman diagram (`tool_zeeman.py`). For a pulse that is not linear, use the local sweep rate at each crossing.
- `deltat` should also be much shorter than the period of an oscillating field (`Bt_type: sin`) and of any Rabi oscillation of interest.
- Always confirm convergence by repeating the calculation with `deltat` halved. `tool_time_step.py` sets up such a series.

Examples from the paper:

- Ho(pzdo)₄: a field step of $10^{-4}$ T per stair reproduces the Landau–Zener probabilities of the avoided crossings at 2.71 T and 6.09 T at 10 and 50 T/ms, whereas $10^{-3}$ T per stair makes the passages artificially diabatic. At 50 T/ms this is `deltat = 2e3` ps, as in `examples/1-spin/linear`.
- (CH₆N₃)₂MnCl₄, 16 states, 50 T/ms: a field step of $5 \times 10^{-5}$ T per stair (`deltat = 1e3` ps) agrees with half that step to 0.015 μB, whereas $10^{-3}$ T per stair is off by up to about 0.14 μB.

## Tests

The `tests/` directory has a pytest suite: unit tests of the Liouville superoperator, the propagators and the matrix exponential, and regression tests that compare short runs of the examples against reference outputs in `tests/reference/` (made by `tests/make_reference.py` from the CPU code at commit `14f7d02`, and for the 3-spin cases at the commit that introduced the Condon–Shortley pool basis; `tests/reference/meta.json` records the commit of each case). Run it from the repository, pointing `QDMAG_TEST_DIR` at a directory for the files the runs write (the default is a temporary directory):

```bash
QDMAG_TEST_DIR=/path/to/scratch python -m pytest tests
```

- The suite sets the number of BLAS threads to 4, the value the references were made with, before NumPy is imported. The effective basis does not depend on it (`tests/test_effective_basis.py` checks 1 against 4 threads); only the rounding does.
- Tests marked `gpu` (`tests/test_gpu.py`, and the GPU variants of the regression tests) run only if CuPy imports and a CUDA device is visible; otherwise they are skipped. GPU results are compared with tolerances, not bitwise.
