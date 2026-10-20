"""
Array backend for the time evolution: NumPy/SciPy on the CPU, or CuPy on a CUDA GPU.

The device is resolved once, when a liouville object is created, from the 'device'
key of the third dynamics block (see liouville.__init__). Code in the time loop then
calls backend.xp, backend.expm, ... and never branches on the device.

CuPy is imported only when device = 'gpu' is requested, so the CPU path does not
need it. On the CPU, to_device and to_host return their argument unchanged, so the
CPU path performs exactly the same operations as code written directly in NumPy.
"""
import functools

import numpy as np
import scipy.linalg

from qdmag.core import expm as _expm

DEVICES = ('cpu', 'gpu')


class Backend:
    """
    name       : 'cpu' or 'gpu'
    xp         : the array module, numpy or cupy
    expm       : dense matrix exponential, scipy.linalg.expm on the CPU, core/expm.py on
                 the GPU (more accurate there than cupyx.scipy.linalg.expm; see that file)
    to_device  : host array -> array on the device
    to_host    : array on the device -> NumPy array
    synchronize: wait for the device to finish (for timing); no-op on the CPU
    """

    def __init__(self, device='cpu', gpu_id=0):
        if device not in DEVICES:
            raise ValueError("Invalid device: {}. Choose from {}.".format(device, list(DEVICES)))
        self.name = device
        if device == 'cpu':
            self.xp = np
            self.expm = scipy.linalg.expm
            self.to_device = _identity
            self.to_host = _identity
            self.synchronize = _no_op
            self.description = 'CPU (NumPy {}, SciPy {})'.format(np.__version__, scipy.__version__)
        else:
            self._set_up_gpu(gpu_id)

    def _set_up_gpu(self, gpu_id):
        try:
            import cupy
        except ImportError as err:
            raise RuntimeError("device = 'gpu' requires CuPy, which could not be imported: {}. "
                               "Install a CuPy build that matches the CUDA version, "
                               "e.g. pip install cupy-cuda12x.".format(err)) from err
        try:
            ndevices = cupy.cuda.runtime.getDeviceCount()
        except cupy.cuda.runtime.CUDARuntimeError as err:
            raise RuntimeError("device = 'gpu' requested, but no CUDA device is usable: {}".format(err)) from err
        if not 0 <= gpu_id < ndevices:
            raise RuntimeError("gpu_id = {} requested, but {} CUDA device(s) are visible.".format(gpu_id, ndevices))
        cupy.cuda.Device(gpu_id).use()
        self.gpu_id = gpu_id
        self.xp = cupy
        self.expm = functools.partial(_expm.expm, xp=cupy)
        self.to_device = cupy.asarray
        self.to_host = cupy.asnumpy
        self.synchronize = cupy.cuda.Device(gpu_id).synchronize
        props = cupy.cuda.runtime.getDeviceProperties(gpu_id)
        self.description = 'GPU {} ({}, CuPy {})'.format(gpu_id, props['name'].decode(), cupy.__version__)


def _identity(a):
    return a


def _no_op():
    pass
