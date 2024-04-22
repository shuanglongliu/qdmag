# spin\_dynamics

Codes for solving the time-dependent Schrodinger equation, the von Neumann equation, and the quantum master equation based on spin Hamiltonians. 

Set the variable "__file__" in __init__.py to "path/to/spin\_dynamics/" after downloading the code. The forward slash at the end of the path is required.

The exchange term and the Zeeman term commute if the exchange couplings are isotropic and the g tensors are isotropic and identical. When the exchange term and the Zeeman term commute, we adopt the eigenstates of h\_ex + h\_zee(Bz = 1e-4 T) as basis functions, which are referred to as the perturbed basis. On the perturbed basis, both h\_ex and h\_zee (or Mz\_tot) is diagonal and real. 

