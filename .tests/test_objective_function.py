import numpy as np

import sys
import os

# Add parent directory to sys.path to allow importing modules from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import objective_function

r = objective_function(
    np.array([0.15, 1.0, 1.0, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001])
)
print(r)

# BUCKLING is high
