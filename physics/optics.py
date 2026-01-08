import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigs


class WaveguideModeSolver:
    """
    Scalar finite-difference eigenmode solver for 2D waveguides.
    """

    def __init__(self, wavelength, x_range, z_range, nx=200, nz=200):
        self.lambda0 = wavelength
        self.k0 = 2 * np.pi / wavelength

        self.x = np.linspace(*x_range, nx)
        self.z = np.linspace(*z_range, nz)
        self.dx = self.x[1] - self.x[0]
        self.dz = self.z[1] - self.z[0]

        self.nx = nx
        self.nz = nz

    def solve_mode(self, refractive_index_fn, num_modes=1):
        """
        Solve for guided modes.

        Parameters
        ----------
        refractive_index_fn : callable
            n(x, z) refractive index profile
        num_modes : int
            Number of eigenmodes to compute
        """

        nx, nz = self.nx, self.nz
        dx, dz = self.dx, self.dz

        # Build refractive index grid
        n = np.zeros((nx, nz))
        for i, xi in enumerate(self.x):
            for j, zj in enumerate(self.z):
                n[i, j] = refractive_index_fn(xi, zj)

        n_flat = n.flatten()

        # Laplacian operators (keep everything sparse)
        Dx2 = diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
        Dz2 = diags([1, -2, 1], [-1, 0, 1], shape=(nz, nz)) / dz**2

        # Use sparse Kronecker products to form 2D Laplacian without densifying
        L = kron(eye(nz, format="csr"), Dx2, format="csr") + kron(Dz2, eye(nx, format="csr"), format="csr")

        # Sanity checks
        n_elems = nx * nz
        n_flat = n.flatten()
        assert n_flat.size == n_elems, f"refractive index vector size {n_flat.size} != {n_elems}"

        # Helmholtz operator (diagonal) - ensure explicit shape matches L
        diag_vals = (self.k0 * n_flat) ** 2
        diag = diags(diag_vals, 0, shape=(n_elems, n_elems), format="csr")
        assert diag.shape == L.shape, f"Shape mismatch: diag {diag.shape} vs L {L.shape}"

        A = L + diag

        # Solve eigenproblem
        eigvals, eigvecs = eigs(A, k=num_modes, which="LR")

        beta2 = np.real(eigvals)
        beta = np.sqrt(beta2)

        self.beta = beta
        self.neff = beta / self.k0

        self.field = eigvecs[:, 0].reshape((nx, nz))

        return self.neff, self.field
