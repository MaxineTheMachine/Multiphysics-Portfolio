import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#clean up this later !!!!!!!!

from physics.mechanics import CantileverBeam

beam = CantileverBeam(
    length=200e-6,
    width=20e-6,
    thickness=2e-6,
    youngs_modulus=160e9,
    density=2330,
)

x, w = beam.solve_deflection(load_type="uniform", load_value=1e-3)

print("Max deflection:", beam.max_deflection())
print("Resonant frequency (Hz):", beam.resonant_frequency())

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x * 1e6, w * 1e9)
plt.xlabel("Beam position x (µm)")
plt.ylabel("Deflection w(x) (nm)")
plt.title("MEMS Cantilever Deflection")
plt.grid(True)
plt.show()


#Tip Deflection vs Load
import numpy as np

loads = np.linspace(0, 2e-3, 10)
tip_deflections = []

for L in loads:
    x, w = beam.solve_deflection(load_type="uniform", load_value=L)
    tip_deflections.append(w[-1])

plt.figure()
plt.plot(loads * 1e3, np.array(tip_deflections) * 1e9, "o-")
plt.xlabel("Uniform Load (mN/m)")
plt.ylabel("Tip Deflection (nm)")
plt.title("MEMS Load–Deflection Curve")
plt.grid(True)
plt.show()


