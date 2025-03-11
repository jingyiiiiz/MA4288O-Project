# File: market/heston.py
import numpy as np

class HestonModel:
    def __init__(self, 
                 s0=1.0,
                 v0=0.04,
                 kappa=1.5,
                 theta=0.04,
                 xi=0.5,
                 rho=-0.7,
                 r=0.0,
                 dt=1/365,
                 seed=1234):
        """
        A basic Heston model for frictionless simulation.
        All transaction-cost references removed.
        """
        self.s0 = s0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.r = r
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, n_paths=10000, n_steps=30):
        """
        Returns:
          S: shape (n_paths, n_steps+1)
          V: shape (n_paths, n_steps+1)
        """
        S = np.zeros((n_paths, n_steps + 1))
        V = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = self.s0
        V[:, 0] = self.v0

        # Cholesky
        cov = np.array([[1.0, self.rho],[self.rho, 1.0]])
        L = np.linalg.cholesky(cov)

        for t in range(n_steps):
            # Draw correlated normals
            Z = self.rng.normal(size=(n_paths, 2))
            dW = Z @ L.T
            dW1 = dW[:, 0] * np.sqrt(self.dt)
            dW2 = dW[:, 1] * np.sqrt(self.dt)

            vt = V[:, t]
            st = S[:, t]

            V[:, t+1] = np.maximum(
                vt + self.kappa * (self.theta - vt) * self.dt + self.xi * np.sqrt(vt) * dW2,
                1e-8
            )

            # Underlying price
            S[:, t+1] = st * np.exp((self.r - 0.5*vt)*self.dt + np.sqrt(vt)*dW1)

        return S, V
    
def make_varianceswap_paths(S_paths, V_paths, dt):
    """
    Creates a naive daily mark-to-market for a variance swap from t=0..T
    shape => (n_paths, n_steps+1)

    For demonstration, we'll do a toy approach:
    vsw[k] ~ expected integral of variance from k..end
    We'll do an extremely naive estimate:
    vsw[k] = V_paths[:,k] * (time_left)
    """
    n_paths, n_steps_plus = V_paths.shape
    n_steps = n_steps_plus - 1
    vsw = np.zeros_like(V_paths)  # same shape

    for i in range(n_paths):
        for k in range(n_steps+1):
            time_left = (n_steps - k)*dt
            vsw[i, k] = V_paths[i, k] * time_left
    return vsw

