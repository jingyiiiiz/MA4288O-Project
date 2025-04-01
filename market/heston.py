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

        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        L = np.linalg.cholesky(cov)

        for t in range(n_steps):
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
            S[:, t+1] = st * np.exp((self.r - 0.5*vt)*self.dt + np.sqrt(vt)*dW1)

        return S, V

def simulate_two_assets(n_paths=10000, n_steps=30, s0_1=100, s0_2=100, 
                        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, 
                        rho1=-0.7, rho2=-0.3, r=0.0, dt=1/365, seed=123):
    """
    Simulates two correlated assets under shared stochastic volatility.

    Returns:
      S: shape (n_paths, n_steps+1, 2)
      V: shape (n_paths, n_steps+1)
    """
    rng = np.random.default_rng(seed)
    S = np.zeros((n_paths, n_steps + 1, 2))
    V = np.zeros((n_paths, n_steps + 1))

    S[:, 0, 0] = s0_1
    S[:, 0, 1] = s0_2
    V[:, 0] = v0

    Z = rng.normal(size=(n_paths, n_steps, 3))  # 3 Brownian motions

    for t in range(n_steps):
        dW1 = Z[:, t, 0] * np.sqrt(dt)
        dW2 = Z[:, t, 1] * np.sqrt(dt)
        dWv = Z[:, t, 2] * np.sqrt(dt)

        vt = V[:, t]
        V[:, t+1] = np.maximum(
            vt + kappa * (theta - vt) * dt + xi * np.sqrt(vt) * dWv,
            1e-8
        )

        # Asset 1
        S[:, t+1, 0] = S[:, t, 0] * np.exp((r - 0.5*vt)*dt + np.sqrt(vt)*(rho1*dWv + np.sqrt(1 - rho1**2)*dW1))
        # Asset 2
        S[:, t+1, 1] = S[:, t, 1] * np.exp((r - 0.5*vt)*dt + np.sqrt(vt)*(rho2*dWv + np.sqrt(1 - rho2**2)*dW2))

    return S, V

def make_varianceswap_paths(S_paths, V_paths, dt):
    """
    Creates a naive daily mark-to-market for a variance swap from t=0..T.
    For demonstration, we use a simple estimate:
    vsw[k] = V_paths[:,k] * (time_left)
    """
    n_paths, n_steps_plus = V_paths.shape
    n_steps = n_steps_plus - 1
    vsw = np.zeros_like(V_paths)

    for i in range(n_paths):
        for k in range(n_steps+1):
            time_left = (n_steps - k)*dt
            vsw[i, k] = V_paths[i, k] * time_left
    return vsw
