# Examples

Each example directory in [`examples/`](https://github.com/shuanglongliu/qdmag/tree/main/examples) holds an `input.yaml` and a copy of the tool it runs. Run the tool in that directory, for example

```bash
cd examples/1-spin/linear
python -m qdmag.tools.tool_staircase
```

The results are written to `output/` in the same directory.

## One spin: Ho(pzdo)₄, J = 8

[`examples/1-spin/`](https://github.com/shuanglongliu/qdmag/tree/main/examples/1-spin) uses the full 17-state Hilbert space with zero-field splitting up to 12th order.

| Directory | Tool | What it computes |
|---|---|---|
| `eigen` | `tool_eigen.py` | Eigenvalues and eigenvectors at a fixed field |
| `zeeman` | `tool_zeeman.py` | Zeeman diagram |
| `equilibrium` | `tool_magnetization.py`, `tool_occupation.py` | Equilibrium magnetization and level occupations |
| `spins` | `tool_spins.py` | Expectation values of the spin operators |
| `projection` | `tool_projection.py` | Composition of the eigenstates |
| `linear` | `tool_staircase.py` | Dynamics under a linear field sweep (`Bt_type: linear`) |
| `pwlinear` | `tool_staircase.py` | Piecewise-linear pulse (`pwlinear`) |
| `sinusoidal` | `tool_staircase.py` | Sinusoidal field (`sin`) |
| `cspline` | `tool_staircase.py` | Measured pulse fitted by a monotone cubic spline (`cspline`) |
| `time_step` | `tool_time_step.py` | Convergence with the stair width |
| `powder` | `tool_quadrature.py`, `tool_powder.py` | Powder average over Lebedev orientations |

## Two spins: S = 1/2 dimer

[`examples/2-spin/`](https://github.com/shuanglongliu/qdmag/tree/main/examples/2-spin) covers four types of exchange coupling: isotropic (`isoJ`), $J_{zz}$ only (`Jzz`), $J_{xx}$ only (`Jxx`), and antisymmetric (`antiJ`). Each has a `magnetization` (equilibrium) and a `staircase` (dynamics) subdirectory.

## Three spins: (CH₆N₃)₂MnCl₄ Mn trimer, S = 5/2

[`examples/3-spin/`](https://github.com/shuanglongliu/qdmag/tree/main/examples/3-spin) uses the effective Hamiltonian with 16 or 26 of the 216 states.

| Directory | What it computes |
|---|---|
| `basis_states` | The pool of candidate basis states (`main.py`) |
| `eigen`, `magnetization` | Eigenstates and equilibrium magnetization |
| `zeeman/16`, `zeeman/26`, `zeeman/full_space` | Zeeman diagrams in the effective and full spaces |
| `staircase/16`, `staircase/26`, `staircase/full_space` | Dynamics in the effective and full spaces |
