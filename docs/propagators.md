# Staircase propagators: choosing one, and the measurements behind the defaults

Reference detail for the `propagator` key of `tool_staircase.py`; see the README for the
list of options. All numbers were measured on an Apple M3 with NumPy 2.2.5 / SciPy 1.15.2.

## Cost versus the time step

Which to use is governed by `||L*deltat||_1`, since only `Pade` is insensitive to it. Measured on the 1-spin example (`dimds = 578`, `krylov_m = 30`), one propagation costs (median of 3, fastest in bold):

| `deltat` (ps) | `\|\|L*deltat\|\|_1` | `Pade` [s] | `Taylor` [s] | `Krylov` [s] | `max\|Krylov - Pade\|` |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 5.4e-01 | 0.0257 | 0.0024 | **0.0006** | 1.1e-16 |
| 0.1 | 5.4e+00 | 0.0527 | 0.0039 | **0.0007** | 6.1e-16 |
| 1 | 5.4e+01 | 0.0684 | 0.0129 | **0.0010** | 1.6e-15 |
| 10 | 5.4e+02 | 0.0955 | 0.1016 | **0.0046** | 1.7e-14 |
| 100 | 5.4e+03 | 0.0848 | 0.7148 | **0.0365** | 2.9e-13 |
| 1000 | 5.4e+04 | **0.1512** | 6.8399 | 0.3587 | 9.2e-13 |
| 10000 | 5.4e+05 | **0.0770** | 39.1419 | 20.2686 | 9.9e-12 |

All three agree to machine precision throughout. `Pade` is essentially flat — its cost does not depend on `deltat` — while `Taylor` and `Krylov` both grow with `||L*deltat||_1`, so the crossover falls between the 100 ps and 1000 ps rows, near `||L*deltat||_1 ~ 1e4`. Below it `Krylov` is the fastest of the three, by up to 40x; above it `Pade` wins, and at the production step of `deltat = 1e4 ps` it is some 260x faster than `Krylov` and 500x faster than `Taylor`. That is why `Pade` is the default. Check with `np.linalg.norm(lio.L * lio.deltat, 1)` before committing to a long run.

Note that `Krylov` grows *faster* than linearly in `||L*deltat||_1` — 56x between the last two rows for a 10x increase in norm. The number of Arnoldi substeps grows linearly, but the substeps also get more expensive: the thermal initial state is close to the steady state, so the early substeps hit a happy breakdown and terminate well short of `krylov_m`, and that discount disappears once the state has evolved. Extrapolating the small-`deltat` rows linearly would therefore understate `Krylov` badly at a production time step.

Timings on a laptop vary substantially with machine load — the `Pade` column here scatters between 0.026 and 0.151 s despite being nominally constant, and a run of this table taken during a heavy 20-process sweep gave 105 s rather than 20 s for the last `Krylov` cell. The orderings and the crossover are stable; the individual numbers should be read as indicative.

## Sparse superoperators

The `_sparse` propagators change how `L` is stored and rebuilt, not the mathematics. Selecting `Taylor_sparse` or `Krylov_sparse` makes `liouville` build `L0`, `LA`, `LC` and `L` with `scipy.sparse` and skip the dense superoperator entirely, which is what makes large effective bases reachable — a dense `L` for the Mn trimer full space (`dim = 216`, `dimds = 93312`) would need 65 GiB.

The rebuild of `L` at every stair is also much cheaper, because the sparse path assembles `A` and `C` from Kronecker products instead of looping in Python, and skips `get_indices_nzC` entirely. Measured in a real stair loop on the 1-spin example (`dimds = 578`, `deltat = 1 ps`, median of 15 interleaved rounds):

| `propagator` | rebuild `L` [s] | propagate [s] | total per stair [s] |
| --- | --- | --- | --- |
| `Pade` | 0.0257 | 0.0766 | 0.1023 |
| `Taylor` | 0.0314 | 0.0153 | 0.0467 |
| `Krylov` | 0.0248 | 0.0133 | 0.0381 |
| `Taylor_sparse` | **0.0064** | 0.0283 | 0.0347 |
| `Krylov_sparse` | **0.0066** | 0.0223 | **0.0288** |

All five agree to ~1e-13 after 100 stairs. The sparse rebuild is about 4x faster than the dense one here, and the advantage grows with the basis size — 12.8x on `examples/3-spin/staircase/26` (`dim = 26`), since it is the `dims**2` Python loops that are eliminated.

The sparse *propagation*, however, is a loss at these densities: both `_sparse` variants propagate more slowly than their dense twins, because sparse matrix-vector products do not beat dense BLAS until the density falls well below ~5%. And it does not fall with `dim`. Although `A` is sparse (`2*dim-1` nonzeros per row, so `~2/dim`), the dissipator is not: since `C = kron(X Rhbar, Id) - kron(Rhbar, X.T)` and `Rhbar` is essentially dense, the density of `C` — and hence of `L` — tracks the density of the spin-transition operator `X`, not `dim`. Measured across the four staircase examples (`dim` = 4, 16, 17, 26), `L` stays at 22-34% dense with no downward trend. Use the `_sparse` propagators when `L` cannot be stored densely at all, not as a general speed-up.

## Vectorized `construct_A`

`construct_A` has a vectorized twin, `construct_A_vectorized`, which computes the same superoperator as `kron(H, Id) - kron(Id, H.T)` rather than by a Python double loop over `dims**2`. It is bitwise identical and about 80x faster (0.0636 s to 0.0008 s at `dimds = 578`), and it is what the dense path uses; the original loop is kept for reference. This cuts the per-stair rebuild of a dense `L` from about 0.086 s to 0.023 s, so existing `Pade` runs at the production `deltat` get roughly 1.6x faster end to end.
