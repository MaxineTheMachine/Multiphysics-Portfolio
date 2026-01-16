# Physical Model

# Replace Euler–Bernoulli beam with Kirchhoff–Love plate theory:


import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class MEMSPlate:
    def __init__(self, Lx, Ly, Nx, Ny, D):
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.D = D
        self.Nx = Nx
        self.Ny = Ny

    def solve_static(self, pressure):
        # pressure: (Ny, Nx)
        N = self.Nx * self.Ny

        laplacian = diags(
            [1, -4, 1],
            offsets=[-1, 0, 1],
            shape=(self.Nx, self.Nx)
        ) / self.dx**2

        biharm = laplacian @ laplacian
        A = self.D * biharm

        z = spsolve(A, pressure.flatten())
        return z.reshape(self.Ny, self.Nx)
