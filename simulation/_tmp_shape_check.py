import numpy as np
from physics.optics import WaveguideModeSolver

def slab_waveguide(x, z):
    core_width = 0.5e-6
    n_core = 3.48
    n_clad = 1.44
    return n_core if abs(x) < core_width / 2 else n_clad

solver = WaveguideModeSolver(1.55e-6, (-2e-6, 2e-6), (-2e-6, 2e-6))
x = solver.x
z = solver.z
Xg, Zg = np.meshgrid(x, z, indexing='xy')
n_map = np.vectorize(slab_waveguide)(Xg, Zg)
print('x.shape', x.shape)
print('z.shape', z.shape)
print('Xg.shape', Xg.shape)
print('Zg.shape', Zg.shape)
print('n_map.shape', n_map.shape)
