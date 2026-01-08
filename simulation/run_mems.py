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
