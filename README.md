# qdmag

[![docs](https://github.com/shuanglongliu/qdmag/actions/workflows/docs.yml/badge.svg)](https://shuanglongliu.github.io/qdmag/)

A Python package for simulating nonequilibrium magnetization of magnetic molecules under a time-varying external magnetic field, based on a generalized Lindblad quantum master equation.

**Documentation: https://shuanglongliu.github.io/qdmag**

<!-- start-overview -->
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
<!-- end-overview -->

## Installation

```bash
git clone https://github.com/shuanglongliu/qdmag.git
cd qdmag
pip install .
```

This also installs the required dependencies (NumPy, SciPy ≥ 1.15, PyYAML, h5py, pandas). Optional extras: `pip install ".[plot]"` (matplotlib), `".[gpu]"` (CuPy for CUDA 12), and `".[test]"` (pytest). See [Installation](https://shuanglongliu.github.io/qdmag/installation.html) for the GPU setup.

## Quick start

Every tool reads `input.yaml` from the working directory and writes its results to `output/` there. For example, the time evolution of a Ho³⁺ ion under a linear field sweep:

```bash
cd examples/1-spin/linear
python -m qdmag.tools.tool_staircase
```

As shipped, this run sweeps the field from 0 to 50 T in 5 × 10⁵ stairs, which takes hours on one core. For a test of about a minute, set `tmax: 2.0e+06` in `input.yaml`.

## Documentation

- [Theory](https://shuanglongliu.github.io/qdmag/theory.html): spin Hamiltonian, quantum master equation, staircase approximation, effective Hamiltonian, powder averaging
- [Usage](https://shuanglongliu.github.io/qdmag/usage.html): running the code, running on a GPU, convergence, tests
- [The input file](https://shuanglongliu.github.io/qdmag/input.html): all keys of `input.yaml`, their units and conventions (also in [docs/input.md](https://github.com/shuanglongliu/qdmag/blob/main/docs/input.md))
- [Tools](https://shuanglongliu.github.io/qdmag/tools.html): the scripts in `tools/`
- [Examples](https://shuanglongliu.github.io/qdmag/examples.html): the example inputs in `examples/`
- [API reference](https://shuanglongliu.github.io/qdmag/api.html): the modules of `qdmag.core`

## Tests

```bash
QDMAG_TEST_DIR=/path/to/scratch python -m pytest tests
```

See [Usage](https://shuanglongliu.github.io/qdmag/usage.html#tests) for details.

## Citing qdmag

See [Citing](https://shuanglongliu.github.io/qdmag/citing.html).

## Acknowledgments

This work is supported by the Center for Molecular Magnetic Quantum Materials (M2QM), an Energy Frontier Research Center (EFRC) funded by the U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award DE-SC0019330. 

## License

qdmag is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](https://github.com/shuanglongliu/qdmag/blob/main/LICENSE).

## References

[1] K. Saito, S. Takesue, and S. Miyashita, Energy transport in the integrable system in contact with various types of phonon reservoirs, *Phys. Rev. E* **61**, 2397 (2000).

[2] H. Nakano and S. Miyashita, Magnetization Process of Nanoscale Iron Cluster, *J. Phys. Soc. Jpn.* **70**, 2151–2157 (2001).

[3] S. Liu, X. Chen, A. Cupo, J. N. Fry, and H.-P. Cheng, *qdmag*: A Python package for simulating nonequilibrium magnetization of molecules using quantum master equations, submitted to *Computer Physics Communications* (2026). 
