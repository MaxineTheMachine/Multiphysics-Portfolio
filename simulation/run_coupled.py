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


# Plot Deflected Beam with Waveguide Location
import matplotlib.pyplot as plt

x_beam = results[-1]['x_beam']
w = results[-1]['w']
z_shift = results[-1]['z_shift']

plt.figure()
plt.plot(x_beam * 1e6, w * 1e9, label="Beam deflection")

plt.axhline(
    (system.z0 + z_shift) * 1e9,
    linestyle="--",
    label="Waveguide height"
)

plt.xlabel("x (µm)")
plt.ylabel("z displacement (nm)")
plt.title("MEMS Deflection with Integrated Waveguide")
plt.legend()
plt.grid(True)
plt.show()


#effective index vs load

loads = []
neffs = []

for r in results:
    loads.append(r["load"])
    neffs.append(r["neff"][0])

plt.figure()
plt.plot(
    np.array(loads) * 1e3,
    neffs,
    "o-"
)
plt.xlabel("Uniform Load (mN/m)")
plt.ylabel("Effective Index n_eff")
plt.title("Optomechanical Tuning via MEMS Deflection")
plt.grid(True)
plt.show()



#optical mode visualization

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, r in zip(axes, [results[0], results[-1]]):
    Ey = r["field"][:].reshape(optics.nx, optics.nz)
    x = optics.x
    z = optics.z

    # create 2D grids for plotting and evaluation
    ax.pcolormesh(
        x * 1e6,
        z * 1e6,
        np.abs(Ey) ** 2,
        shading="auto"
    )
    ax.set_title(f"Load = {r['load']*1e3:.2f} mN/m")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")

plt.suptitle("Optical Mode Shift Due to MEMS Actuation")
plt.show()
