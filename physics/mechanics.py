import numpy as np
from scipy.linalg import solve


class CantileverBeam:
    """
    Euler–Bernoulli cantilever beam model with finite-difference solver.
    """

    def __init__(self, length, width, thickness, youngs_modulus, density):
        self.L = length
        self.w = width
        self.t = thickness
        self.E = youngs_modulus
        self.rho = density

        self.I = (self.w * self.t**3) / 12.0

    def solve_deflection(self, num_nodes=200, load_type="uniform", load_value=1.0):
        """
        Solve beam deflection w(x).

        Parameters
        ----------
        num_nodes : int
            Number of discretization points
        load_type : str
            'uniform' or 'tip'
        load_value : float
            Load magnitude (N/m for uniform, N for tip)

        Returns
        -------
        x : ndarray
            Spatial coordinates
        w : ndarray
            Deflection profile
        """

        L = self.L
        E = self.E
        I = self.I

        x = np.linspace(0, L, num_nodes)
        dx = x[1] - x[0]

        K = np.zeros((num_nodes, num_nodes))
        f = np.zeros(num_nodes)

        # Interior finite-difference stencil
        for i in range(2, num_nodes - 2):
            K[i, i-2:i+3] = [1, -4, 6, -4, 1]

        K *= (E * I) / dx**4

        # Load vector
        if load_type == "uniform":
            f[:] = load_value
        elif load_type == "tip":
            f[-1] = load_value / dx
        else:
            raise ValueError("Unsupported load type")

        # Boundary conditions
        # w(0) = 0
        K[0, :] = 0
        K[0, 0] = 1
        f[0] = 0

        # w'(0) = 0
        K[1, :] = 0
        K[1, 0:3] = [-3, 4, -1]
        f[1] = 0

        # w''(L) = 0
        K[-2, :] = 0
        K[-2, -3:] = [1, -2, 1]
        f[-2] = 0

        # w'''(L) = 0
        K[-1, :] = 0
        K[-1, -4:] = [-1, 3, -3, 1]
        f[-1] = 0

        w = solve(K, f)

        return x, w

    def max_deflection(self, *args, **kwargs):
        _, w = self.solve_deflection(*args, **kwargs)
        return np.max(np.abs(w))

    def resonant_frequency(self):
        """
        Fundamental resonant frequency (Hz)
        """
        beta = 1.875
        return (beta**2 / (2 * np.pi)) * np.sqrt(
            (self.E * self.I) / (self.rho * self.w * self.t * self.L**4)
        )
