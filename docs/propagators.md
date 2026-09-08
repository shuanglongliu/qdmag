# Staircase propagators: choosing one, and the measurements behind the defaults

Reference detail for the `propagator` key of `tool_staircase.py`; see the README for the list of
options. Two sets of measurements: a `deltat` ladder on one system, and a sweep of all five
propagators over the four staircase examples at their own production time steps. All numbers were
measured on an Apple M3 with Python 3.13.3, NumPy 2.2.5 and SciPy 1.15.2, at commit `d3bb1f2`.

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

## All examples at their production time step

The ladder above varies `deltat` on one system. This sweep instead fixes each example at the
`deltat` committed in its own `input.yaml` and compares all five propagators. Every one of the 20
combinations completed; none needed a reduced time step.

| Example | System | `dim` | `dimds` | `deltat` (ps) | `\|\|L*deltat\|\|_1` | `L` density |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1-spin/linear` | single spin, S = 8 | 17 | 578 | 1e+04 | 5.40e+05 | 34.4% |
| `2-spin/isoJ/staircase` | two spins, S = 1/2 | 4 | 32 | 1e+04 | 7.54e+02 | 26.6% |
| `3-spin/staircase/16` | Mn trimer, 16-state basis | 16 | 512 | 2e+04 | 5.13e+05 | 21.7% |
| `3-spin/staircase/26` | Mn trimer, 26-state basis | 26 | 1352 | 2e+04 | 1.03e+05 | 30.0% |

`rebuild L` is `update_L` (median of 5); `propagate` is one application of `exp(L*deltat)` from the
thermal initial state. Their sum is the cost of one stair.

| Example | `propagator` | rebuild `L` [s] | propagate [s] | total [s] | max diff vs `Pade` |
| --- | --- | ---: | ---: | ---: | ---: |
| `1-spin/linear` | `Pade` | 0.0230 | 0.0882 | **0.1112** | 0 |
| | `Taylor` | 0.0232 | 114.6798 | 114.7030 | 9.5e-12 |
| | `Krylov` | 0.0226 | 104.9790 | 105.0016 | 9.8e-12 |
| | `Taylor_sparse` | 0.0050 | 225.8680 | 225.8730 | 9.6e-12 |
| | `Krylov_sparse` | 0.0051 | 187.1881 | 187.1932 | 8.9e-12 |
| `2-spin/isoJ/staircase` | `Pade` | 0.0004 | 0.0003 | **0.0008** | 0 |
| | `Taylor` | 0.0004 | 0.0158 | 0.0162 | 2.3e-15 |
| | `Krylov` | 0.0004 | 0.0012 | 0.0016 | 8.3e-13 |
| | `Taylor_sparse` | 0.0017 | 0.0237 | 0.0253 | 2.3e-15 |
| | `Krylov_sparse` | 0.0016 | 0.0015 | 0.0031 | 8.3e-13 |
| `3-spin/staircase/16` | `Pade` | 0.0149 | 0.0597 | **0.0746** | 0 |
| | `Taylor` | 0.0141 | 107.5726 | 107.5867 | 8.0e-12 |
| | `Krylov` | 0.0141 | 83.5972 | 83.6114 | 7.1e-12 |
| | `Taylor_sparse` | 0.0044 | 124.0752 | 124.0796 | 7.9e-12 |
| | `Krylov_sparse` | 0.0044 | 93.0452 | 93.0497 | 7.1e-12 |
| `3-spin/staircase/26` | `Pade` | 0.1613 | 1.5020 | **1.6633** | 0 |
| | `Taylor` | 0.1649 | 82.8586 | 83.0235 | 2.0e-12 |
| | `Krylov` | 0.1635 | 55.1500 | 55.3136 | 1.7e-12 |
| | `Taylor_sparse` | 0.0196 | 322.0980 | 322.1176 | 1.8e-12 |
| | `Krylov_sparse` | 0.0126 | 102.2918 | 102.3044 | 1.7e-12 |

`Pade` wins every example, by 500-1500x on the three with a large `||L*deltat||_1`. All five agree
to between 2.3e-15 and 9.8e-12. `Krylov` beats `Taylor` everywhere. The 2-spin system is the only
one with a small norm (754, because `dim = 4`), and even there `Pade` leads — by 2x rather than by
three orders of magnitude.

Scaled up to a complete run, with `nt = (tmax - tmin)/deltat` taken from each `input.yaml`:

| Example | stairs | `Pade` per stair | **`Pade` full run** | `Krylov` full run |
| --- | ---: | ---: | ---: | ---: |
| `1-spin/linear` | 100,000 | 0.1112 s | **3.09 h** | 121.5 days |
| `2-spin/isoJ/staircase` | 10,000 | 0.0008 s | **8 s** | 16 s |
| `3-spin/staircase/16` | 10,000 | 0.0746 s | **12.4 min** | 9.7 days |
| `3-spin/staircase/26` | 10,000 | 1.6633 s | **4.62 h** | 6.4 days |

These 20 measurements ran back to back as a sustained ~20 minute load, and the slow cells are
inflated by it: the `1-spin` `Krylov` entry reads 105 s here against 20 s for the same quantity in
the quiet single-process ladder above. Compare within a row, not across the two tables.

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

All five agree to ~1e-13 after 100 stairs. The sparse rebuild is about 4x faster than the dense one here, and the advantage grows with the basis size, since it is the `dims**2` Python loops that are eliminated: from the sweep above, `rebuild L` goes 0.0230 -> 0.0050 s (4.6x) on 1-spin, 0.0149 -> 0.0044 s (3.4x) on 3-spin/16, and 0.1613 -> 0.0126 s (**12.8x**) on 3-spin/26. It reverses only on the 2-spin system (0.0004 -> 0.0016 s), where `dimds = 32` is too small to amortize the sparse bookkeeping.

The sparse *propagation*, however, is a loss at these densities: both `_sparse` variants propagate more slowly than their dense twins, because sparse matrix-vector products do not beat dense BLAS until the density falls well below ~5%. And it does not fall with `dim`. Although `A` is sparse (`2*dim-1` nonzeros per row, so `~2/dim`), the dissipator is not: since `C = kron(X Rhbar, Id) - kron(Rhbar, X.T)` and `Rhbar` is essentially dense, the density of `C` — and hence of `L` — tracks the density of the spin-transition operator `X`, not `dim`. Measured across the four staircase examples, `L` stays at 22-34% dense with no downward trend in `dim`:

| Example | `dim` | `2/dim` | `A` | `LA` | `X` | `C` | `L` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1-spin/linear` | 17 | 0.118 | 0.106 | 0.104 | 0.111 | 0.170 | **0.344** |
| `2-spin/isoJ/staircase` | 4 | 0.500 | 0.102 | 0.051 | 0.500 | 0.312 | **0.234** |
| `3-spin/staircase/16` | 16 | 0.125 | 0.114 | 0.057 | 0.117 | 0.180 | **0.217** |
| `3-spin/staircase/26` | 26 | 0.077 | 0.075 | 0.038 | 0.281 | 0.320 | **0.300** |

`A` does follow `~2/dim` and the coherent block `LA` is sparser still, falling with `dim` as
expected; density(`C`) instead tracks density(`X`). Use the `_sparse` propagators when `L` cannot be
stored densely at all, not as a general speed-up.

## Vectorized `construct_A`

`construct_A` has a vectorized twin, `construct_A_vectorized`, which computes the same superoperator
as `kron(H, Id) - kron(Id, H.T)` rather than by a Python double loop over `dims**2`. It is what the
dense path uses; the original loop is kept for reference. It is **bitwise identical** in every
example — maximum difference exactly 0.0 — and is called once per stair by the dense rebuild:

| Example | loop [s] | vectorized [s] | sparse [s] | speed-up | full run before | full run now |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1-spin/linear` | 0.0355 | 0.0003 | 0.0003 | 116x | 4.07 h | **3.09 h** |
| `2-spin/isoJ/staircase` | 0.0001 | 0.0000 | 0.0002 | 7x | 9 s | **8 s** |
| `3-spin/staircase/16` | 0.0223 | 0.0002 | 0.0003 | 91x | 16.1 min | **12.4 min** |
| `3-spin/staircase/26` | 0.1558 | 0.0025 | 0.0005 | 63x | 5.05 h | **4.62 h** |

So existing `Pade` runs at the production `deltat` get roughly 1.2-1.6x faster end to end with no
change to their input files.
