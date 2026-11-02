# Theory

## Spin Hamiltonian

The spin Hamiltonian has three terms:

$$\hat{H} = \hat{H}_\text{ex} + \hat{H}_\text{ZFS} + \hat{H}_\text{Zee}$$

**Exchange interaction:**

$$\hat{H}_\text{ex} = -2 \sum_{ij} \hat{\boldsymbol{S}}_i^\mathrm{T} \boldsymbol{J}_{ij} \hat{\boldsymbol{S}}_j$$

The 3×3 exchange coupling matrix **J** can be decomposed into isotropic, traceless anisotropic, antisymmetric, and symmetric off-diagonal parts.

**Zero-field splitting:**

$$\hat{H}_\text{ZFS} = \sum_{i,kq} B_{i,k}^q \hat{O}_{i,k}^q(S_i)$$

where $\hat{O}_{i,k}^q$ are the extended Stevens operators. The order $k$ takes values $2, 4, \ldots, 2S$ and the component $q$ runs from $-k$ to $k$.

**Zeeman term:**

$$\hat{H}_\text{Zee} = \sum_i \mu_B \boldsymbol{B}^\mathrm{T} \mathbf{g}_i \hat{\boldsymbol{S}}_i$$

The magnetic field is assumed to be along the $z$-axis in the numerical implementation.

## Quantum Master Equation

The generalized Lindblad equation reads:

$$\frac{d\rho(t)}{dt} = \frac{1}{i\hbar}[\hat{H}, \rho(t)] - \Gamma\rho(t)$$

where the dissipator $\Gamma\rho(t) = ([X, R\rho(t)] + [X, R\rho(t)]^\dagger) \lambda^2\pi/\hbar$ accounts for spin-phonon coupling. The phonon spectral density is taken as $I(\omega) = I_0 \omega^\alpha \theta(\omega)$. The adjustable parameters are the prefactor $I_0$, the spin-phonon coupling constant $\lambda$ (typically a few $\textrm{cm}^{-1}$), and the spectral-density exponent $\alpha$ (default 2, giving a super-Ohmic bath; $\alpha < 1$ is sub-Ohmic and $\alpha = 1$ is Ohmic).

## Staircase Approximation

The continuous B(t) profile is replaced by a staircase function. Within each step the Hamiltonian is constant and the propagator has the exact closed-form solution:

$$\rho(t + \Delta t) = \exp(\mathcal{L} \Delta t) \rho(t)$$

This maintains numerical stability for time steps orders of magnitude larger than those required by Runge–Kutta methods, making millisecond simulations feasible.

## Effective Hamiltonian

For large multinuclear systems where the full Hilbert space dimension N = Πᵢ(2Sᵢ+1) is too large, an effective Hamiltonian of dimension n ≪ N is constructed by selecting thermally relevant basis states. The basis states are chosen as the lowest-energy eigenstates of isotropic exchange interaction with a perturbative Zeeman term for each value of the total $S_z$ projection by default. The small perturbative Zeeman field ensures the eigenstates are also eigenstates of the total $\hat{S}_z$ operator.

The pool of candidate states is built in a unique basis, so that the effective basis does not depend on the linear-algebra library, the number of threads or the CPU. The pool Hamiltonian conserves the total spin, so each pool state is a member $|S, M\rangle$ of a spin multiplet. The top state $M = S$ of each multiplet is taken from the $M$ sector of the product basis (where $\hat{S}_z$ is diagonal), with its largest component real and positive, and the other members follow from the Condon–Shortley convention, $|S, M-1\rangle = \hat{S}_-|S, M\rangle / \sqrt{(S+M)(S-M+1)}$. Multiplets of equal energy and equal $S$ are told apart by the intermediate spins $\hat{S}_{12}^2$, $\hat{S}_{123}^2$, …. This matters because the operator $X$ that couples the spins to the phonons is defined in this basis ($X_{ij} = 1$ for states with $|M_i - M_j| = 1$): a different phase or a different basis within a degenerate level would change the relaxation. A warning is printed when `states:` contains only part of a degenerate level.

## Powder Averaging

For powder samples, the net magnetization is averaged over all molecular orientations. Two angles are enough for this, rather than three. The external field lies along the global $z$ axis, so the magnetization depends on the orientation only through the direction $\mathbf{n}$ of the field in the ZFS reference frame,

$$\mathbf{n}(\beta, \gamma) = (-\sin\beta\cos\gamma, \, \sin\beta\sin\gamma, \, \cos\beta)$$

which does not contain the first Euler angle $\alpha$. Turning the molecule by $\alpha$ leaves the magnetization unchanged. The average is then a single Lebedev quadrature over the unit sphere,

$$\overline{M} = \frac{1}{4\pi} \int M(\mathbf{n}) \, d\Omega = \sum_i w_i M(\mathbf{n}_i), \qquad \sum_i w_i = 1$$

and each Lebedev point is mapped back onto the Euler angles that the drivers consume. Quadrature points and weights are saved to a text file for inspection and record.

Convergence with the Lebedev degree can be checked with `powder.convergence_test` in `tool_powder.py`, which powder averages the equilibrium magnetization over a series of degrees. 

## Case studies

Three case studies are included to demonstrate the package:

| System | Description | Hilbert space |
|--------|-------------|---------------|
| Ho(pzdo)₄ | Mononuclear Ho³⁺ complex, J = 8, ZFS up to 12th order | Full (17 states) |
| Spin-1/2 dimer | Two coupled S = 1/2 spins, four exchange coupling types | Full (4 states) |
| (CH₆N₃)₂MnCl₄ | Mn trimer, S = 5/2 local spins | Effective (16 or 26 states from full 216) |

The input files are described in [Examples](examples.md).
