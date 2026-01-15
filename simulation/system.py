from physics.mechanics import CantileverBeam
from physics.optics import WaveguideModeSolver
from physics.coupling import MemsPhotonicSystem


def build_default_system():
    """
    Factory function to construct the default coupled MEMS–photonics system.

    Returns
    -------
    MemsPhotonicSystem
        Fully initialized coupled system with standard parameters.
    """

    # MEMS cantilever parameters
    beam = CantileverBeam(
        length=200e-6,        # m
        width=20e-6,          # m
        thickness=2e-6,       # m
        youngs_modulus=160e9, # Pa (silicon)
        density=2330,         # kg/m^3
    )

    # Optical waveguide solver
    optical_solver = WaveguideModeSolver(
        wavelength=1.55e-6,   # m
        x_range=(-2e-6, 2e-6),
        z_range=(-2e-6, 2e-6),
        nx=200,
        nz=200,
    )

    # Coupled system
    system = MemsPhotonicSystem(
        beam=beam,
        optical_solver=optical_solver,
        waveguide_width=0.5e-6,
        waveguide_height=0.22e-6,
        n_core=3.48,          # Si
        n_clad=1.44,          # SiO2
        z0=0.0,
    )

    return system
