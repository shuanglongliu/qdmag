# Installation

```bash
git clone https://github.com/shuanglongliu/qdmag.git
cd qdmag
pip install .
```

This also installs the required dependencies. Optional extras: `pip install ".[plot]"` (matplotlib), `".[gpu]"` (CuPy for CUDA 12), and `".[test]"` (pytest). Alternatively, add the directory that contains qdmag to the environment variable PYTHONPATH. 

## Dependencies

- Python 3.x
- NumPy
- SciPy ≥ 1.15 (quadrature weights, spline fitting, matrix exponential, and Euler angle transformations). The powder averaging calls `scipy.integrate.lebedev_rule`, which was added in 1.15.
- PyYAML (parsing of `input.yaml`)
- h5py (HDF5 I/O for density matrix storage and retrieval)
- pandas (tabular output of magnetization and level population time series)
- matplotlib (optional; sparsity visualization and quadrature point plots)
- CuPy (optional; only for `device: gpu`, see [Running on a GPU](usage.md#running-on-a-gpu))
- pytest (optional; for the test suite in `tests/`)

## GPU support (optional)

Install a CuPy build that matches the CUDA version of the driver, for example for CUDA 12:

```bash
pip install cupy-cuda12x
```

(`cupy-cuda13x` for CUDA 13.) On a cluster, load a matching CUDA module before a GPU run, e.g. `module load cuda/12.9.1` on HiPerGator. CuPy is imported only when `device: gpu` is requested, so CPU runs work without it. CuPy compiles its kernels on first use and caches them in `~/.cupy`, so the first run on a new GPU model is slower.
