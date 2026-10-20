"""
tools/tool_powder.py rotates every tensor of the molecule with the orientation (anisotropy,
g, exchange), not only the anisotropy.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cases import working_directory

TOOL = Path(__file__).resolve().parents[1] / "tools" / "tool_powder.py"
ANGLES = [(0.0, 0.0, 0.0), (0.0, 54.7, 45.0), (0.0, 90.0, 120.0), (0.0, 150.0, 300.0)]
G_ISOTROPIC = "gs: [1.24, 0.0, 0.0, 0.0, 1.24, 0.0, 0.0, 0.0, 1.24]"
G_ANISOTROPIC = "gs: [1.10, 0.0, 0.0, 0.0, 1.30, 0.0, 0.0, 0.0, 1.60]"


def load_powder(workdir, monkeypatch, g=G_ISOTROPIC):
    """A tool_powder.powder object in workdir, with ANGLES in points_and_weights.txt."""
    np.savetxt(workdir / "points_and_weights.txt",
               np.column_stack([np.array(ANGLES), np.full(len(ANGLES), 1 / len(ANGLES))]))
    with working_directory(workdir):
        spec = importlib.util.spec_from_file_location("tool_powder", TOOL)
        tp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tp)
        assert G_ISOTROPIC in tp.input
        monkeypatch.setattr(tp, "input", tp.input.replace(G_ISOTROPIC, g))
        return tp, tp.powder()


def test_frames(workdir, monkeypatch):
    """The anisotropy frame is emat as before (now unrounded), and the g frame rotates with it."""
    tp, p = load_powder(workdir, monkeypatch)
    for i in range(len(ANGLES)):
        p.get_emat(i)
        data = p.get_input_data()
        frame = lambda item: np.array(item["reference_frame"]).reshape(3, 3)
        np.testing.assert_allclose(frame(data["anisotropy"][0]), p.emat, rtol=0, atol=1e-15)
        np.testing.assert_allclose(frame(data["gfactor"][0]), p.emat, rtol=0, atol=1e-15)


@pytest.mark.parametrize("g", [G_ISOTROPIC, G_ANISOTROPIC])
def test_rotating_molecule_is_rotating_field(g, workdir, monkeypatch):
    """
    The molecule rotated by R in a field along z has the magnetization along z of the
    unrotated molecule in a field along n = R^T z (= emat[:, 2]), projected on n. With an
    anisotropic g this holds only if g rotates with the molecule.
    """
    from qdmag.core.common import (many_spins, eigen_handy, get_h_exchange, get_h_anisotropy,
                                   get_h_Zeeman_Mv_eff, get_partition_function, get_magnetic_moment_Mv_tot)
    from qdmag.core.constants import factor_ex

    tp, p = load_powder(workdir, monkeypatch, g)
    p.emat = np.eye(3)
    data = p.get_input_data()
    spins = many_spins(data["spins"], len(data["spins"]), data["gfactor"])
    h0 = get_h_exchange(spins, data.get("exchange", []), factor_ex) + get_h_anisotropy(spins, data["anisotropy"])

    def M_along_field(Bvec):
        eigen = eigen_handy(h0 + get_h_Zeeman_Mv_eff(spins.Mv_tot, list(Bvec), "cartesian"))
        Z, _ = get_partition_function(eigen, p.T)
        return np.asarray(get_magnetic_moment_Mv_tot(spins.Mv_tot, eigen, p.T, Z)) @ (Bvec / np.linalg.norm(Bvec))

    Bs = [0.5, 3.0]
    for alpha, beta, gamma in ANGLES:
        Mz = p.get_M_eq_one_orientation(alpha, beta, gamma, Bs)
        n = p.emat[:, 2]
        expected = [M_along_field(B * n) for B in Bs]
        np.testing.assert_allclose(Mz, expected, rtol=1e-10, atol=1e-12)

    if g == G_ANISOTROPIC:
        # Rotating the anisotropy alone (the earlier behaviour) violates the identity
        alpha, beta, gamma = ANGLES[2]
        p.get_emat_from_angles(alpha, beta, gamma)
        emat = p.emat
        monkeypatch.setattr(tp, "rotate_frames", lambda d, e: (
            d["anisotropy"][0].__setitem__("reference_frame", e.ravel().tolist()) or d))
        Mz_old = p.get_M_eq_one_orientation(alpha, beta, gamma, Bs)
        expected = [M_along_field(B * emat[:, 2]) for B in Bs]
        assert np.max(np.abs(Mz_old - np.array(expected))) > 1e-3
