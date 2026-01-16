
#refractive index field

# n_eff(x,y) = n_core * exp(-z(x,y) / decay_length)


from scipy.sparse.linalg import eigs

class Waveguide2D:
    def __init__(self, wavelength, n_clad):
        self.k0 = 2 * np.pi / wavelength
        self.n_clad = n_clad

    def solve_mode(self, n_map, dx, dy):
        Ny, Nx = n_map.shape
        N = Nx * Ny

        lap = build_2d_laplacian(Nx, Ny, dx, dy)
        V = diags((self.k0 * n_map.flatten())**2)

        H = lap + V
        vals, vecs = eigs(H, k=1, which="LR")

        beta = np.sqrt(vals[0])
        mode = vecs[:, 0].reshape(Ny, Nx)
        return beta.real, mode
