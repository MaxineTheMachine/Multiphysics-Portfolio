import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#clean up this later !!!!!!!!

import numpy as np
from physics.mechanics import CantileverBeam
from physics.optics import WaveguideModeSolver
from physics.coupling import MemsPhotonicSystem

# MEMS beam
beam = CantileverBeam(
    length=200e-6,
    width=20e-6,
    thickness=2e-6,
    youngs_modulus=160e9,
    density=2330,
)

# Optical solver
optics = WaveguideModeSolver(
    wavelength=1.55e-6,
    x_range=(-2e-6, 2e-6),
    z_range=(-2e-6, 2e-6),
)

# Coupled system
system = MemsPhotonicSystem(
    beam=beam,
    optical_solver=optics,
    waveguide_width=0.5e-6,
    waveguide_height=0.22e-6,
    n_core=3.48,
    n_clad=1.44,
    z0=0.0,
)

# Sweep load
loads = np.linspace(0, 2e-3, 5)
results = system.optical_response_vs_load(loads)

for r in results:
    print(
        f"Load={r['load']:.2e} N/m | "
        f"z-shift={r['z_shift']*1e9:.2f} nm | "
        f"neff={r['neff'][0]:.6f}"
    )
