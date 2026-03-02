# spin\_dynamics

Codes for solving time-dependent magnetization of magnetic molecules under a pulsed magnetic field based on [a generalized Lindblad equation][1,2]. The molecule is modeled as a spin system coupled to a phonon bath. 

Set the variable "\_\_file\_\_" in \_\_init\_\_.py to "path/to/spin\_dynamics/" after downloading the code. The forward slash at the end of the path is required.

The exchange term and the Zeeman term commute if the exchange couplings are isotropic and the g tensors are isotropic and identical. When the exchange term and the Zeeman term commute, we adopt the eigenstates of h\_ex + h\_zee(Bz = 1e-4 T) as basis functions, which are referred to as the perturbed basis. On the perturbed basis, both h\_ex and h\_zee (or Mz\_tot) is diagonal and real. 

# Tools

## tool_staircare.py

An example code for solving the Liouville-form quantum master equation using the staircase approximation.

# References

[1] K. Saito, S. Takesue, and S. Miyashita, Energy transport in the integrable system in contact with various types of phonon reservoirs, Phys. Rev. E 61, 2397 (2000). 
[2] Hiroki Nakano and Seiji Miyashita, Magnetization Process of Nanoscale Iron Cluster, J. Phys. Soc. Jpn. 70, 2151–2157 (2001). 
