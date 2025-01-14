
Tesla2meV = 0.057883820027408574
Kelvin2meV = 0.08617330341520024
meV2wavenumber = 8.065543937338
Tesla2wavenumber = 0.46686449369202904 # Tesla2meV * meV2wavenumber
Kelvin2wavenumber = 0.6950345649208562 # Kelvin2meV * meV2wavenumber
x_mu_B = 9.2740100783e-24 # Bohr magneton in SI unit, i.e. J T^-1
x_mu0 = 1.25663706212e-6 # Vacuum permeability in T m A^-1 = N A^-2
N_A = 6.02214076e23 # Avogadro constant
x_hbar = 1.054571817e-34 # Planck constant over 2 pi in J s 
x_e = 1.602176634e-19 # elementary charge in C
wavenumber2J = 1.602176634e-22 # 1/meV2wavenumber / 1000 * x_e
ps2s = 1.0e-12 # pico second

const1 = 1.5192674488095106 # wavenumber2J * ps2s / x_hbar


# Conversion factor adopted by the PHI code.
mu_B_per_Tesla_2_cm3_per_mol_phi = N_A*x_mu_B/10

