import numpy as np
from physics.mechanics import CantileverBeam
from physics.optics import WaveguideModeSolver


class MemsPhotonicSystem:
    """
    Coupled MEMS–photonics simulation.
    """

    def __init__(
        self,
        beam: CantileverBeam,
        optical_solver: WaveguideModeSolver,
        waveguide_width,
        n_core,
        n_clad,
        waveguide_height=0.22e-6,
        z0=0.0,
    ):
        self.beam = beam
        self.optical_solver = optical_solver

        self.wg_width = waveguide_width
        self.wg_height = waveguide_height
        self.n_core = n_core
        self.n_clad = n_clad
        self.z0 = z0

    def waveguide_geometry(self, z_shift):
        """
        Returns refractive index function for shifted waveguide.
        """

        def n_fn(x, z):
            if (
                abs(x) < self.wg_width / 2
                and abs(z - (self.z0 + z_shift)) < self.wg_height / 2
            ):
                return self.n_core
            return self.n_clad

        return n_fn

    def solve_static_response(self, load_value, x_probe=None):
        """
        Compute optical response for a given MEMS load.
        """

        # Solve MEMS deflection
        x_beam, w = self.beam.solve_deflection(
            load_type="uniform", load_value=load_value
        )

        # Probe location (default: beam midpoint)
        if x_probe is None:
            x_probe = self.beam.L / 2

        z_shift = np.interp(x_probe, x_beam, w)

        # Optical solve
        n_fn = self.waveguide_geometry(z_shift)
        neff, field = self.optical_solver.solve_mode(n_fn)

        return {
            "load": load_value,
            "x_beam": x_beam,
            "w": w,
            "z_shift": z_shift,
            "neff": neff,
            "field": field,
        }

    def optical_response_vs_load(self, loads):
        """
        Sweep MEMS load and record optical response.
        """

        results = []

        for load in loads:
            result = self.solve_static_response(load)
            results.append(result)

        return results
