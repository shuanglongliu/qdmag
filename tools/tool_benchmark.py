"""
Per-stair timing of the staircase solver on the CPU or a GPU (journals/2026-10-19_gpu_plan.md,
Section 6).

The systems are single spins S with dim = 2S + 1, e.g. --dims 17 26 40 64 100, with an
axial plus rhombic zero-field splitting scaled with S so that the spectrum width, and so
||L deltat|| and the number of squarings in expm, stay comparable across dim. Each row
times update_L (host eigh + Rhbar, C and L on the device) and one application of
exp(L deltat), separately, synchronized, after one warm-up stair.

The number of BLAS threads must be set before NumPy is imported, so run one process per
thread count:

    OMP_NUM_THREADS=4 python tool_benchmark.py stairs --device cpu --dims 17 26 40 --reps 3
    python tool_benchmark.py stairs --device gpu --dims 17 26 40 64 100 --reps 3
    python tool_benchmark.py eigh --dims 17 26 64 128 256 512
    python tool_benchmark.py info

Results go to stdout as CSV rows (stairs, eigh) or JSON (info).
"""
import os
import sys
import io
import json
import time
import argparse
import platform
import resource
import contextlib
from pathlib import Path

import numpy as np

# Import qdmag from the parent of the repository
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def get_lio(dim, device, exp_propagator, deltat=2.0e4):
    """liouville object for a single spin S = (dim - 1)/2 at t = 0."""
    from qdmag.core.common import many_spins
    from qdmag.core.effective_basis import effective_basis
    from qdmag.core.liouville import liouville

    S = (dim - 1) / 2
    B20 = -30.0 / S**2  # spectrum width ~ 3 |B20| S^2 ~ 100 cm-1 for every S
    anisotropy = [{"site": 1, "ks": [2, 2], "qs": [0, 2], "Bkqs": [B20, 0.1 * abs(B20)],
                   "reference_frame": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}]
    gfactor = [{"site": 1, "gs": [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
                "reference_frame": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}]
    dynamics = [{"T": 0.6, "lambdaa": 10.0, "I0": 1.0e-14, "alpha": 2.0},
                {"Bt_type": "linear", "sweep_rate": 5.0e-8},
                {"tmin": 0.0, "tmax": deltat, "deltat": deltat,
                 "device": device, "exp_propagator": exp_propagator},
                {"save_mag": False}]
    with contextlib.redirect_stdout(io.StringIO()):
        eff = effective_basis(many_spins([S], 1, gfactor), [], anisotropy, dynamics, [])
        lio = liouville(eff, dynamics)
        lio.get_initial_rho(from_file=False)
    return lio


def gpu_name():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)["name"].decode()
    except Exception:
        return ""


def gpu_pool_gb():
    try:
        import cupy
        return cupy.get_default_memory_pool().total_bytes() / 1e9
    except Exception:
        return float("nan")


def host_maxrss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def stairs(args):
    from qdmag.core import expm as qe

    print("device,gpu,threads,dim,dimds,exp_propagator,reps,update_L_ms,propagate_ms,stair_ms,"
          "norm1_Ldt,s_bound,gpu_pool_GB,host_maxrss_GB", flush=True)
    threads = os.environ.get("OMP_NUM_THREADS", "")
    for i, dim in enumerate(args.dims):
        reps = args.reps[min(i, len(args.reps) - 1)]
        lio = get_lio(dim, args.device, args.exp_propagator)
        sync = lio.backend.synchronize
        xp = lio.xp
        lio.update_L(0.01)
        lio.propagate_risvrho()
        sync()
        A = lio.L * lio.deltat
        norm1 = float(xp.abs(A).sum(axis=0).max())
        s_bound = qe.scaling_exponent(A, xp)  # from ||A||_1 only: an upper bound for the s used
        del A
        tL = tP = 0.0
        for r in range(reps):
            t0 = time.perf_counter()
            lio.update_L(0.02 + 0.01 * r)
            sync()
            t1 = time.perf_counter()
            lio.propagate_risvrho()
            sync()
            t2 = time.perf_counter()
            tL += t1 - t0
            tP += t2 - t1
        tL, tP = 1e3 * tL / reps, 1e3 * tP / reps
        print("{},{},{},{},{},{},{},{:.2f},{:.2f},{:.2f},{:.3e},{},{:.2f},{:.2f}".format(
            args.device, gpu_name() if args.device == "gpu" else "", threads, dim, lio.dimds,
            args.exp_propagator, reps, tL, tP, tL + tP, norm1, s_bound, gpu_pool_gb(), host_maxrss_gb()), flush=True)
        del lio
        if args.device == "gpu":
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()


def eigh(args):
    """numpy vs cupy eigh of a complex Hermitian dim x dim matrix, as in get_Rhbar."""
    import cupy
    print("dim,threads,numpy_eigh_ms,cupy_eigh_ms,cupy_eigh_with_transfers_ms", flush=True)
    rng = np.random.default_rng(0)
    sync = cupy.cuda.Device().synchronize
    for dim in args.dims:
        h = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        h = h + h.conj().T
        hg = cupy.asarray(h)
        np.linalg.eigh(h)
        cupy.linalg.eigh(hg)
        sync()
        n = max(3, min(200, int(2e5 / dim**2)))
        t = time.perf_counter()
        for _ in range(n):
            np.linalg.eigh(h)
        tc = (time.perf_counter() - t) / n
        t = time.perf_counter()
        for _ in range(n):
            cupy.linalg.eigh(hg)
        sync()
        tg = (time.perf_counter() - t) / n
        t = time.perf_counter()
        for _ in range(n):
            w, v = cupy.linalg.eigh(cupy.asarray(h))
            w, v = cupy.asnumpy(w), cupy.asnumpy(v)
        tgt = (time.perf_counter() - t) / n
        print("{},{},{:.3f},{:.3f},{:.3f}".format(dim, os.environ.get("OMP_NUM_THREADS", ""),
                                                  1e3 * tc, 1e3 * tg, 1e3 * tgt), flush=True)


def info(args):
    import scipy
    out = {"node": platform.node(), "python": platform.python_version(), "numpy": np.__version__,
           "scipy": scipy.__version__, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    try:
        cfg = np.show_config(mode="dicts")
        out["blas"] = cfg["Build Dependencies"]["blas"]
    except Exception:
        pass
    with open("/proc/cpuinfo") as f:
        out["cpu"] = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), "")
    try:
        import cupy
        rt = cupy.cuda.runtime
        p = rt.getDeviceProperties(cupy.cuda.Device().id)
        out.update({"cupy": cupy.__version__, "cuda_runtime": rt.runtimeGetVersion(),
                    "cuda_driver": rt.driverGetVersion(), "gpu": p["name"].decode(),
                    "gpu_cc": "{}.{}".format(p["major"], p["minor"]), "gpu_mem_GB": p["totalGlobalMem"] / 1e9})
    except Exception as err:
        out["cupy"] = "unavailable: {}".format(err)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("stairs", help="time update_L and the propagator per stair")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--dims", type=int, nargs="+", default=[17, 26, 40])
    p.add_argument("--reps", type=int, nargs="+", default=[3],
                   help="stairs to time per dim; the last value applies to the remaining dims")
    p.add_argument("--exp-propagator", default="Pade")
    p.set_defaults(func=stairs)
    p = sub.add_parser("eigh", help="host vs device eigh of a dim x dim Hermitian matrix")
    p.add_argument("--dims", type=int, nargs="+", default=[17, 26, 40, 64, 128, 256, 512])
    p.set_defaults(func=eigh)
    p = sub.add_parser("info", help="node, CPU, BLAS, GPU, CUDA and CuPy versions")
    p.set_defaults(func=info)
    args = parser.parse_args()
    args.func(args)
