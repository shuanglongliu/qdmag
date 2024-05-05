import os

__version__ = '0.0.00'

__file__ = os.getcwd()

path_segments = __file__.split("spin_dynamics")
n_path_segment = len(path_segments)

if n_path_segment > 2:
    __file__ = ""
    for i in range(n_path_segment-2):
        __file__ = __file__ + path_segments[i] + "spin_dynamics"
        __file__ = __file__ + path_segments[i+1] + "spin_dynamics"
    __file__ = __file__ + "/"
else:
    __file__ = __file__.split("spin_dynamics")[0] + "spin_dynamics/"


