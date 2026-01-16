class MEMSOptical2DSystem:
    def __init__(self, mems, optics, n_core, decay_length):
        self.mems = mems
        self.optics = optics
        self.n_core = n_core
        self.decay = decay_length

    def solve(self, pressure):
        z = self.mems.solve_static(pressure)
        n_map = self.n_core * np.exp(-z / self.decay)

        beta, mode = self.optics.solve_mode(
            n_map,
            self.mems.dx,
            self.mems.dy
        )

        return {
            "z": z,
            "n_map": n_map,
            "beta": beta,
            "mode": mode
        }
