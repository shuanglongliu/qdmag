import os
import sys
import time
import ray
from common import read_input, many_spins
from common import set_exchange, set_anisotropy
from common import get_h_exchange, get_h_anisotropy
from common import get_M_vs_B






if __name__ == "__main__":

    # Command line options

    J1, J2, D1, EoD1, D2, EoD2 = [float(sys.argv[i]) for i in range(1, len(sys.argv))]


    # Ray initialization

    num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK'))
    ray.init(num_cpus=num_cpus)


    # Spin system

    Ss, nS, positions, exchange, anisotropy, gfactor, dipole, ext_field, BET_Bgrid, BET_Egrid, BET_BEgrid, BET_Tgrid, fit_problem = read_input()
    spins = many_spins(Ss, nS, gfactor, dipole, positions)

    # Calculate magnetization

    fit_problem[0]["current_values"] = [J1, J2]
    fit_problem[1]["current_values"] = [D1, EoD1, D2, EoD2]

    exchange = set_exchange(exchange, fit_problem[0])
    anisotropy = set_anisotropy(anisotropy, fit_problem[1])

    h_ex = get_h_exchange(spins, exchange, -2)
    h_ani = get_h_anisotropy(spins, anisotropy)

    get_M_vs_B(spins, h_ex, h_ani, BET_Bgrid)


    # Ray finalization

    ray.shutdown()


