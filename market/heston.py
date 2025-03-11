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

import numpy as np

class MultiAssetHestonModel:
    def __init__(self, num_assets=5, s0=100.0, v0=0.04, kappa=1.5, theta=0.04,
                 xi=0.5, rho=-0.7, r=0.0, dt=1/365, seed=1234):
        """
        Multi-asset Heston model for independent assets.

        Args:
        - num_assets: Number of independent Heston processes
        """
        self.num_assets = num_assets
        self.s0 = np.full(num_assets, s0)
        self.v0 = np.full(num_assets, v0)
        self.kappa = np.full(num_assets, kappa)
        self.theta = np.full(num_assets, theta)
        self.xi = np.full(num_assets, xi)
        self.rho = np.full(num_assets, rho)
        self.r = r
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, n_paths=100000, n_steps=30):
        """
        Simulate multiple independent Heston paths.

        Returns:
        - S: shape (n_paths, n_steps+1, num_assets)
        - V: shape (n_paths, n_steps+1, num_assets)
        """
        S = np.zeros((n_paths, n_steps + 1, self.num_assets))
        V = np.zeros((n_paths, n_steps + 1, self.num_assets))

        S[:, 0, :] = self.s0
        V[:, 0, :] = self.v0

        for t in range(n_steps):
            Z = self.rng.normal(size=(n_paths, self.num_assets, 2))  # 2 correlated noise terms per asset
            dW1 = Z[:, :, 0] * np.sqrt(self.dt)
            dW2 = Z[:, :, 1] * np.sqrt(self.dt)

            vt = V[:, t, :]
            st = S[:, t, :]

            V[:, t+1, :] = np.maximum(
                vt + self.kappa * (self.theta - vt) * self.dt + self.xi * np.sqrt(vt) * dW2,
                1e-8
            )
            S[:, t+1, :] = st * np.exp((self.r - 0.5 * vt) * self.dt + np.sqrt(vt) * dW1)

        return S, V
