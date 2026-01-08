import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#clean up this later !!!!!!!!




def slab_waveguide(x, z):
    core_width = 0.5e-6
    n_core = 3.48
    n_clad = 1.44

    if abs(x) < core_width / 2:
        return n_core
    return n_clad

from physics.optics import WaveguideModeSolver

solver = WaveguideModeSolver(
    wavelength=1.55e-6,
    x_range=(-2e-6, 2e-6),
    z_range=(-2e-6, 2e-6),
)

neff, field = solver.solve_mode(slab_waveguide)

print("Effective index:", neff)
