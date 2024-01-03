import os
import sys
import time
import ray
from common import read_input, many_spins
from fitting import fit_magnetization






if __name__ == "__main__":

    # Command line options

    exp_technique = sys.argv[1]
    nT = len(sys.argv) - 2
    Ts_exp = [float(sys.argv[i]) for i in range(2, len(sys.argv))]


    # Ray initialization

    num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
    ray.init(num_cpus=num_cpus)


    # Spin system

    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, fit_problem = read_input()
    spins = many_spins(Ss, nS, gfactor, dipole, positions)


    # Fit spin Hamiltonian parameters

    fit_magnetization(exp_technique, Ts_exp, spins, exchange, anisotropy, ext_field, fit_problem)


    # Ray finalization

    ray.shutdown()


