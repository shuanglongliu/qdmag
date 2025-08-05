import os
from spin_dynamics.core.common import read_input, many_spins
from spin_dynamics.core.effective_basis import effective_basis
from spin_dynamics.core.liouville import liouville

if __name__ == "__main__":

    # Read input parameters
    Ss, nS, exchange, anisotropy, gfactors, BT_Bgrid, BT_Tgrid, dynamics, states, n_thread = read_input()

    # Set the number of threads
    os.environ['OMP_NUM_THREADS'] = str(n_thread)

    # Spin system
    spins = many_spins(Ss, nS, gfactors)

    # Set up the effective basis
    eff = effective_basis(spins, exchange, anisotropy, dynamics, states)

    # Set up the quantum master equation
    lio = liouville(eff, dynamics)
    lio.get_initial_rho(from_file=False)
    # lio.get_initial_rho(from_file=True, 
    #      fname="./output/T_0.6K_I0_1.00e-14_lambdaa_10.00/Bt_linear_sweep_rate_5.0e-08/rho/0.000-0.100ps_dt0.001ps.h5",
    #      t_init=lio.tmin)
    lio.evolve_rho(method="staircase")
