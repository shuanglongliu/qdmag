"""
Powder average of the staircase dynamics with all orientations evolved together
(journals/2026-10-19_gpu_plan.md, Section 4.5).

tools/tool_powder.py runs one job per orientation. Here the orientations are the leading
(batch) axis of L and risvrho instead, so one GPU evolves many orientations in lockstep:
they share the field B(t), the stairs, the save points, and everything but the
orientation of the molecule.

What changes with the orientation: rotating the molecule rotates its tensors, so h0 and,
for an anisotropic g, Mz and Mv. What does not: the pool basis of the effective basis
comes from the isotropic exchange and an isotropic Zeeman term, which are rotation
invariant, so dim, the selected states, and X are the same for all orientations.
powder_liouville checks this.

On the CPU each orientation is propagated with scipy.linalg.expm as in a single run, so a
powder run gives the same numbers as one run per orientation, bit for bit. On the GPU the
exponentials of all orientations are computed together (core/expm.py, expm_stack).
Orientations that do not fit into GPU memory at once are processed in chunks, one after
the other (run_powder).
"""
import copy
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from qdmag.core.constants import const1, Tesla2wavenumber
from qdmag.core.common import many_spins, eigen_handy, get_rhoe, back_transform_O, get_Mv_from_rho, rotate_frames
from qdmag.core.effective_basis import effective_basis
from qdmag.core.liouville import liouville
from qdmag.core import expm as _expm


def rotate_molecule(data, euler_angles):
    """
    The input (a dict as read from input.yaml) for the molecule rotated by the intrinsic ZYZ
    Euler angles (alpha, beta, gamma), in degrees, as in points_and_weights.txt.

    Every tensor of the molecule rotates with it (common.rotate_frames with emat = R^T, R
    the rotation matrix): each reference_frame F becomes F R^T, for the anisotropy, the g
    tensors and the exchange, as in tools/tool_powder.py. data itself is left unchanged.
    """
    R = Rotation.from_euler('ZYZ', euler_angles, degrees=True).as_matrix()
    return rotate_frames(copy.deepcopy(data), R.T)


def get_effective_basis(data):
    """effective_basis for an input dict, with the defaults of common.read_input."""
    spins = many_spins(data['spins'], len(data['spins']), data['gfactor'])
    return effective_basis(spins, data.get('exchange', []), data.get('anisotropy', []),
                           data['dynamics'], data.get('states', []))


class powder_liouville(liouville):
    """
    liouville for a batch of orientations of one molecule: h0, Mz, Mv, L0, the Zeeman blocks,
    C, CST, L and risvrho carry a leading axis of length nb, one entry per orientation. The
    per-stair update is the one of liouville, which is written to accept that axis.

    effs   : effective_basis of each orientation
    weights: quadrature weight of each orientation. The magnetization written is the
             weighted sum over the batch, so it is the powder average when the weights of
             all orientations sum to 1 and they are all in one batch (see run_powder).
    labels : names of the per-orientation columns Mz_<label> (default 1, ..., nb)

    Supported: method = staircase, exp_propagator = Pade, save_mag only.
    """

    # Written next to the magnetometry of a single run, not over it
    mag_dir_suffix = '_powder'

    def __init__(self, effs, weights, dynamics, labels=None):
        self.effs = effs
        self.nb = len(effs)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.labels = list(labels) if labels is not None else list(range(1, self.nb + 1))
        if self.weights.shape != (self.nb,) or len(self.labels) != self.nb:
            raise ValueError("Need one weight and one label per orientation.")
        eff0 = effs[0]
        for eff in effs[1:]:
            if (eff.dim != eff0.dim or list(eff.states) != list(eff0.states)
                    or not np.array_equal(eff.X_eff, eff0.X_eff)):
                raise ValueError("The orientations do not share the effective basis (dim, states, X); "
                                 "they cannot be evolved as one batch.")
        super().__init__(eff0, dynamics)

    def set_up_exp_propagator(self):
        """Called by liouville.__init__: set up the batched dense L and bind the propagator."""
        if self.method != 'staircase' or self.exp_propagator != 'Pade':
            raise ValueError("The powder driver supports method = staircase with exp_propagator = Pade only, "
                             "not {} / {}.".format(self.method, self.exp_propagator))
        if self.save_rho or self.save_drdt:
            raise ValueError("The powder driver saves the magnetization only; set save_rho and save_drdt to false.")
        self.sparse_L = False
        if self.device == 'cpu':
            self.propagate_risvrho = self.propagate_risvrho_Pade_each
        else:
            self.propagate_risvrho = self.propagate_risvrho_Pade_stack
        # The orientation dependent operators, stacked, on the host
        self.h0 = np.stack([eff.h0_eff for eff in self.effs])
        self.Mz = np.stack([eff.Mz_eff for eff in self.effs])
        self.Mvs = [eff.Mv_eff for eff in self.effs]
        self.set_up_liouville()
        self.update_L = self.update_L_under_magnetic_field

    def set_up_liouville(self):
        """
        Batched counterpart of liouville.set_up_liouville. L0 and the Zeeman blocks are built
        on the host one orientation at a time, by the same routines as for a single run.
        """
        to_device = self.backend.to_device
        nb, dims, dimds = self.nb, self.dims, self.dimds

        self.index_transpose = to_device(np.arange(dims).reshape(self.dim, self.dim).T.ravel())
        self.X_dev = to_device(self.X)
        self.lam = self.lambdaa**2 * np.pi * const1**2

        L0 = np.empty((nb, dimds, dimds))
        c1Azee_unit_re = np.empty((nb, dims, dims))
        c1Azee_unit_im = np.empty((nb, dims, dims))
        for b in range(nb):
            L0[b] = self.construct_L(self.construct_A_vectorized(self.h0[b], dtype=np.complex128), None)
            c1Azee_unit = const1 * self.construct_A_vectorized(-1 * Tesla2wavenumber * self.Mz[b], dtype=np.complex128)
            c1Azee_unit_re[b] = np.real(c1Azee_unit)
            c1Azee_unit_im[b] = np.imag(c1Azee_unit)
        self.L0 = to_device(L0)
        self.c1Azee_unit_re = to_device(c1Azee_unit_re)
        self.c1Azee_unit_im = to_device(c1Azee_unit_im)
        del L0, c1Azee_unit_re, c1Azee_unit_im

        self.L = self.xp.empty((nb, dimds, dimds), dtype=np.float64)
        self.update_L_under_magnetic_field(self.Bt(self.tmin))

    def get_initial_rho(self, from_file=False, fname=None, t_init=None):
        """Thermal rho at t = tmin for each orientation, as liouville.get_initial_rho does for one."""
        if from_file:
            raise NotImplementedError("The powder driver starts from the thermal rho only.")
        risvrho = np.empty((self.nb, self.dimds))
        for b in range(self.nb):
            eigen = eigen_handy(self.h[b])
            rho = back_transform_O(get_rhoe(eigen.eigenvalues, self.T), eigen)
            risvrho[b] = self.convert_rho_to_risvrho(rho)
        self.risvrho = self.backend.to_device(risvrho)

    def propagate_risvrho_Pade_each(self):
        """CPU: scipy.linalg.expm one orientation at a time, as in a single run."""
        for b in range(self.nb):
            self.risvrho[b] = self.backend.expm(self.L[b] * self.deltat) @ self.risvrho[b]

    def propagate_risvrho_Pade_stack(self):
        """GPU: the exponentials of all orientations together."""
        E = _expm.expm_stack(self.L * self.deltat, self.xp)
        self.risvrho = self.xp.matmul(E, self.risvrho[..., None])[..., 0]

    def get_Mvs(self):
        """Magnetization of each orientation, shape (nb, 3)."""
        risvrho = self.backend.to_host(self.risvrho)
        return np.array([get_Mv_from_rho(self.convert_risvrho_to_rho(risvrho[b]), self.Mvs[b])
                         for b in range(self.nb)])

    def unit_save_mag(self, fobj, t, header=False):
        """t, B, the weighted sums of Mx, My, Mz over the batch, and Mz of each orientation."""
        Mvs = self.get_Mvs()
        wM = self.weights @ Mvs
        row = {'t': t, 'B': self.Bt(t), 'Mx': wM[0], 'My': wM[1], 'Mz': wM[2]}
        row.update({'Mz_{}'.format(label): Mvs[b, 2] for b, label in enumerate(self.labels)})
        pd.DataFrame([row]).to_csv(fobj, header=header, index=False)


def memory_per_orientation(dim):
    """Rough device memory in bytes per orientation in a batch: ~20 dimds x dimds float64 arrays
    (L0, L, Zeeman blocks, C, CST, and the temporaries of expm_stack)."""
    dimds = 2 * dim * dim
    return 20 * 8 * dimds**2


def default_batch_size(dim, device, n_orientations, fraction=0.7):
    """All orientations on the CPU; on the GPU as many as fit into `fraction` of the free memory."""
    if device != 'gpu':
        return n_orientations
    import cupy
    free, total = cupy.cuda.runtime.memGetInfo()
    return max(1, min(n_orientations, int(fraction * free // memory_per_orientation(dim))))


def run_powder(data, euler_angles, weights, batch_size=None):
    """
    Powder averaged staircase dynamics, run in the current directory.

    data        : the input (a dict as read from input.yaml) in the reference orientation
    euler_angles: (n, 3) intrinsic ZYZ Euler angles in degrees, one row per orientation
    weights     : (n,) quadrature weights, summing to 1
    batch_size  : orientations per batch (default: all on the CPU, as many as fit on a GPU)

    Returns the DataFrame written to the magnetometry_powder file: t, B, the powder averaged
    Mx, My, Mz, and Mz_1, ..., Mz_n of each orientation (numbered as the rows of euler_angles).
    """
    euler_angles = np.asarray(euler_angles, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    if not np.isclose(weights.sum(), 1.0):
        print("Warning: the weights sum to {:.15f}, not 1.".format(weights.sum()))
    device = data['dynamics'][2].get('device', 'cpu')

    if batch_size is None:
        # All orientations share dim, so the first one sets the batch size
        dim = get_effective_basis(rotate_molecule(data, euler_angles[0])).dim
        batch_size = default_batch_size(dim, device, n)
    print("Powder average over {} orientations in batches of {} on {}".format(n, batch_size, device))

    total = None
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        t0 = time.time()
        effs = [get_effective_basis(rotate_molecule(data, euler_angles[i])) for i in range(start, stop)]
        lio = powder_liouville(effs, weights[start:stop], data['dynamics'], labels=range(start + 1, stop + 1))
        del effs
        lio.get_initial_rho(from_file=False)
        lio.evolve_rho()
        part = pd.read_csv(lio.fmag)
        if total is None:
            total = part
        else:
            total[['Mx', 'My', 'Mz']] += part[['Mx', 'My', 'Mz']]
            total = pd.concat([total, part.drop(columns=['t', 'B', 'Mx', 'My', 'Mz'])], axis=1)
        print("Orientations {}-{} of {} done in {:.1f} s".format(start + 1, stop, n, time.time() - t0))
        del lio.L, lio.L0
        if device == 'gpu':
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
    total.to_csv(lio.fmag, index=False)
    print("Powder averaged magnetization saved to {}".format(lio.fmag))
    return total
