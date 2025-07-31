import sys
import numpy as np
from spin_dynamics.core.data import input

def get_input(*args):
    T, lambdaa, I0, \
        Bt_type, sweep_rate, times, fields, omega, amplitude, theta_B, phi_B, \
        tmin, tmax, deltat,  \
        save_mag, nt_mag, save_rho, nt_rho, multiphonon, imbalance, states, n_thread = args

    with open("input.yaml", "w") as f:
        f.write(input.format(T=T, lambdaa=lambdaa, I0=I0, \
             Bt_type=Bt_type, sweep_rate=sweep_rate, times=times, fields=fields, omega=omega, amplitude=amplitude, theta_B=theta_B, phi_B=phi_B, \
             tmin=tmin, tmax=tmax, deltat=deltat, \
             save_mag=save_mag, nt_mag=nt_mag, save_rho=save_rho, nt_rho=nt_rho, \
             multiphonon=multiphonon, imbalance=imbalance, states=states, n_thread=n_thread))
    return

if __name__ == "__main__":
    # Get arguments from the command line
    T       =    np.float64( sys.argv[1 ] )
    lambdaa =    np.float64( sys.argv[2 ] )
    I0      =    np.float64( sys.argv[3 ] )
                                         
    Bt_type                = sys.argv[4 ]
    sweep_rate = np.float64( sys.argv[5 ] )
    times =                  sys.argv[6 ]
    fields =                 sys.argv[7 ]
    omega =      np.float64( sys.argv[8 ] )
    amplitude =  np.float64( sys.argv[9 ] )
    theta_B =    np.float64( sys.argv[10] )
    phi_B =      np.float64( sys.argv[11] )

    tmin =       np.float64( sys.argv[12] )
    tmax =       np.float64( sys.argv[13] )
    deltat =     np.float64( sys.argv[14] )

    save_mag =               sys.argv[15]
    nt_mag =            int( sys.argv[16] )
    save_rho =               sys.argv[17]
    nt_rho =            int( sys.argv[18] )

    multiphonon =            sys.argv[19]
    imbalance =              sys.argv[20]

    states =                 sys.argv[21]

    n_thread =          int( sys.argv[22] )

    # Example: python tool_input.py 0.6 10.0 1e-10 linear 10.0 0.0,1.0e+9,10.0e+9 0.0,10.0,100.0 0.2 65.0 0.0 0.0 0.0 5e+9 1e+6 true 1 true 10 false false [200,150,88,30,10,0,1,2,3,4,5,17,41,99,173,215] 16
    get_input(T, lambdaa, I0, Bt_type, sweep_rate, times, fields, omega, amplitude, theta_B, phi_B, tmin, tmax, deltat, save_mag, nt_mag, save_rho, nt_rho, \
        multiphonon, imbalance, states, n_thread)


