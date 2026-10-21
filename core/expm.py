"""
Dense matrix exponential written against an array module xp (numpy or cupy), used as
the GPU expm in core/backend.py.

cupyx.scipy.linalg.expm squares once more than needed (s = ceil(log2(||A||_1/theta_13)) + 1
as a safety margin), which costs accuracy for the large ||L deltat|| of the staircase
solver: over the regression runs trace(rho) drifts 3-13x faster than with
scipy.linalg.expm. This version replaces the margin by the backward error correction
of Al-Mohy and Higham (2009), as scipy.linalg.expm does. For the Liouvillians of the
examples it never squares more often than SciPy (once less for 2-spin at 2.5 T, and more
accurate there); on the GPU it cuts the trace drift by 3-6x, to within 1.5-4x of SciPy on
the CPU (journal 2026-10-19_gpu_plan.md, Section 14).

    [13/13] Pade approximant with scaling and squaring:
        N. J. Higham, SIAM J. Matrix Anal. Appl. 26, 1179 (2005), Algorithm 2.3.
    Choice of s:
        A. H. Al-Mohy and N. J. Higham, SIAM J. Matrix Anal. Appl. 31, 970 (2009),
        Algorithm 5.1, with upper bounds for ||A^8|| and ||A^10|| (see scaling_exponent).

Degree 13 is used for every A. The lower degrees of Algorithm 5.1 only save matrix
products when ||A||_1 < theta_13 = 5.37, which never happens for L * deltat here.
"""
import math

import numpy as np

# Coefficients b_0, ..., b_13 of the [13/13] Pade approximant (Higham 2005, Table 2.3)
_b13 = (64764752532480000., 32382376266240000., 7771770303897600.,
        1187353796428800., 129060195264000., 10559470521600.,
        670442572800., 33522128640., 1323241920., 40840800.,
        960960., 16380., 182., 1.)
_theta13 = 5.371920351148152
# 1/|c_27| of the backward error series of the [13/13] approximant (Al-Mohy and Higham 2009, Eq. (5.3))
_abs_c27_recip = 113250775606021113483283660800000000.
_unit_roundoff = 2.0**-53


def _onenorm(A, xp):
    return float(xp.abs(A).sum(axis=0).max())


def _ell(A, nrmA, xp, m=13):
    """
    Number of extra squarings needed so that the truncation error of the degree m
    approximant of exp(A) is below unit roundoff (Al-Mohy and Higham 2009, Eq. (5.3)):
        ell = max(ceil(log2(alpha / u) / (2m)), 0),
        alpha = |c_{2m+1}| * || |A|^(2m+1) ||_1 / ||A||_1.
    || |A|^p ||_1 is computed exactly, since |A| is nonnegative: its column sums are
    1^T |A|^p, which takes p products of a vector with |A|.
    """
    if nrmA == 0.0:
        return 0
    absA = xp.abs(A)
    v = xp.ones(A.shape[0], dtype=absA.dtype)
    for _ in range(2*m + 1):
        v = v @ absA
    alpha = float(v.max()) / (nrmA * _abs_c27_recip)
    if alpha == 0.0:
        return 0
    return max(int(math.ceil(math.log2(alpha / _unit_roundoff) / (2*m))), 0)


def scaling_exponent(A, xp=np, A4=None, A6=None):
    """
    Number of squarings s for the [13/13] approximant of exp(A): A is scaled by 2^-s.

    Al-Mohy and Higham (2009), Algorithm 5.1, take s from eta_5 = min(max(d6, d8), max(d8, d10)),
    d_p = ||A^p||_1^(1/p). Here d4 and d6 are exact (from A^4 and A^6, which the approximant
    needs anyway) and d8 and d10 are replaced by the upper bounds d8 <= d4 and
    d10 <= (||A^4|| ||A^6||)^(1/10), which keeps the error bound valid without the two extra
    matrix products. Then the backward error correction ell is added.
    """
    nrmA = _onenorm(A, xp)
    eta = nrmA
    if A4 is not None and A6 is not None:
        n4, n6 = _onenorm(A4, xp), _onenorm(A6, xp)
        d4, d6 = n4**0.25, n6**(1/6)
        eta = min(eta, min(max(d6, d4), max(d4, (n4 * n6)**0.1)))
    s = max(int(math.ceil(math.log2(eta / _theta13))), 0) if eta > _theta13 else 0
    return s + _ell(A * 2.0**-s, nrmA * 2.0**-s, xp)


def expm(A, xp=np):
    """
    exp(A) for a dense square A, on the device of xp (numpy or cupy).
    """
    n = A.shape[0]
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    s = scaling_exponent(A, xp, A4, A6)

    # Scaling by a power of two is exact, so the powers need not be recomputed.
    A, A2, A4, A6 = A * 2.0**-s, A2 * 4.0**-s, A4 * 16.0**-s, A6 * 64.0**-s
    b = _b13
    Id = xp.eye(n, dtype=A.dtype)
    U = A @ (A6 @ (b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*Id)
    V = A6 @ (b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*Id
    E = xp.linalg.solve(V - U, V + U)
    for _ in range(s):
        E = E @ E
    return E


def expm_stack(A, xp=np):
    """
    exp(A[b]) for a stack A of shape (nb, n, n), as expm does for each matrix: the same s
    for each member (from its own norms), and the members that share an s evaluated
    together with batched products and one batched solve.
    """
    n = A.shape[-1]
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    ss = np.array([scaling_exponent(A[b], xp, A4[b], A6[b]) for b in range(A.shape[0])])
    E = xp.empty_like(A)
    b = _b13
    Id = xp.eye(n, dtype=A.dtype)
    for s in np.unique(ss):
        idx = xp.asarray(np.nonzero(ss == s)[0])
        X, X2, X4, X6 = A[idx] * 2.0**-s, A2[idx] * 4.0**-s, A4[idx] * 16.0**-s, A6[idx] * 64.0**-s
        U = X @ (X6 @ (b[13]*X6 + b[11]*X4 + b[9]*X2) + b[7]*X6 + b[5]*X4 + b[3]*X2 + b[1]*Id)
        V = X6 @ (b[12]*X6 + b[10]*X4 + b[8]*X2) + b[6]*X6 + b[4]*X4 + b[2]*X2 + b[0]*Id
        Es = xp.linalg.solve(V - U, V + U)
        for _ in range(int(s)):
            Es = Es @ Es
        E[idx] = Es
    return E
