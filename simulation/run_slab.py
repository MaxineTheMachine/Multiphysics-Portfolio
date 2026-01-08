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

from physics import optics

import matplotlib.pyplot as plt
import numpy as np

x = solver.x
z = solver.z

# create 2D grids for plotting and evaluation
Xg, Zg = np.meshgrid(x, z, indexing='xy')  # shapes (nz, nx)
n_map = np.vectorize(slab_waveguide)(Xg, Zg)

# sanity checks: pcolormesh expects a 2D C array of shape (len(z), len(x))
assert n_map.ndim == 2, f"n_map must be 2D; got shape {n_map.shape}"
assert n_map.shape == (z.size, x.size), f"n_map shape {n_map.shape} != (len(z), len(x))"

plt.figure()
plt.pcolormesh(
    x * 1e6,
    z * 1e6,
    n_map,
    shading="auto"
)
plt.colorbar(label="Refractive Index")
plt.xlabel("x (µm)")
plt.ylabel("z (µm)")
plt.title("Waveguide Refractive Index Profile")
plt.show()


#Optical mode density

Ey = field[:].reshape(solver.nx, solver.nz) 
intensity = np.abs(Ey) ** 2

plt.figure()
plt.pcolormesh(
    x * 1e6,
    z * 1e6,
    intensity,
    shading="auto"
)
plt.colorbar(label="|E|² (a.u.)")
plt.xlabel("x (µm)")
plt.ylabel("z (µm)")
plt.title("Fundamental TE Mode Intensity")
plt.show()

