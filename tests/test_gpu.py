"""
Tests of the GPU backend (core/backend.py) against the CPU path. Marked gpu, so they are
skipped without CuPy or a CUDA device (see conftest.py). End-to-end GPU runs against the
references are in test_regression.py.
"""
import numpy as np
import pytest

from cases import set_up_case
from qdmag.core.backend import Backend

SYSTEMS = ["1spin_linear_Pade", "1spin_sin_RK4", "2spin_Jzz_Pade", "3spin_16_Pade"]
FIELDS = [0.0, 1.0e-3, 0.37, 2.5, 7.0]


def test_invalid_device():
    with pytest.raises(ValueError, match="Invalid device"):
        Backend("tpu")


def test_cpu_backend_is_numpy():
    backend = Backend("cpu")
    a = np.arange(3.0)
    assert backend.xp is np
    assert backend.to_device(a) is a and backend.to_host(a) is a


@pytest.mark.gpu
@pytest.mark.parametrize("name", SYSTEMS)
def test_L_gpu_matches_cpu(name, workdir):
    """L, LA and LC on the GPU agree with the CPU to round-off of the largest element."""
    cpu = set_up_case(name, workdir / "cpu", "cpu")
    gpu = set_up_case(name, workdir / "gpu", "gpu")
    to_host = gpu.backend.to_host
    for B in FIELDS:
        cpu.update_L(B)
        gpu.update_L(B)
        atol = 4 * np.finfo(float).eps * np.abs(cpu.L).max()
        np.testing.assert_allclose(to_host(gpu.L), cpu.L, rtol=0, atol=atol)
        for part_gpu, part_cpu in zip(gpu.get_LA_LC(), cpu.get_LA_LC()):
            atol = 4 * np.finfo(float).eps * np.abs(part_cpu).max()
            np.testing.assert_allclose(to_host(part_gpu), part_cpu, rtol=0, atol=atol)


@pytest.mark.gpu
def test_gpu_expm_step_matches_cpu(workdir):
    """
    One Pade stair on the GPU agrees with the CPU, for the largest regression system, to the
    accuracy of either expm against extended precision (~1e-10 on rho; journal Section 13).
    """
    cpu = set_up_case("3spin_26_Pade", workdir / "cpu", "cpu")
    gpu = set_up_case("3spin_26_Pade", workdir / "gpu", "gpu")
    for lio in (cpu, gpu):
        lio.update_L(0.37)
        lio.propagate_risvrho()
    np.testing.assert_allclose(gpu.backend.to_host(gpu.risvrho), cpu.risvrho, rtol=0, atol=1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("propagator", ["Taylor", "Taylor_sparse", "Krylov_sparse"])
def test_gpu_rejects_unsupported_propagator(propagator, workdir):
    import yaml
    from cases import write_input, working_directory
    from qdmag.core.common import read_input, many_spins
    from qdmag.core.effective_basis import effective_basis
    from qdmag.core.liouville import liouville

    write_input("1spin_linear_Pade", workdir, "gpu")
    with open(workdir / "input.yaml") as f:
        data = yaml.safe_load(f)
    data["dynamics"][2]["exp_propagator"] = propagator
    with open(workdir / "input.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    with working_directory(workdir):
        Ss, nS, exchange, anisotropy, gfactor, BT_Bgrid, BT_Tgrid, dynamics, states, n_threads = read_input()
        eff = effective_basis(many_spins(Ss, nS, gfactor), exchange, anisotropy, dynamics, states)
        with pytest.raises(ValueError, match="not available with device = 'gpu'"):
            liouville(eff, dynamics)


@pytest.mark.gpu
def test_expm_stack_gpu_matches_cpu(workdir):
    """expm_stack on the GPU agrees with expm on the CPU, member by member, on rho."""
    import cupy
    from qdmag.core import expm as qe
    lio = set_up_case("3spin_16_Pade", workdir, "cpu")
    As = []
    for B in [0.0, 2.5, 5.0]:
        lio.update_L(B)
        As.append(lio.L * lio.deltat)
    E = cupy.asnumpy(qe.expm_stack(cupy.asarray(np.stack(As)), cupy))
    for b in range(len(As)):
        np.testing.assert_allclose(E[b] @ lio.risvrho, qe.expm(As[b]) @ lio.risvrho, rtol=0, atol=1e-10)
