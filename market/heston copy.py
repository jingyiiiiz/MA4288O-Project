# market/heston.py

import numpy as np

class HestonModel:
    def __init__(self, s0, v0, kappa, theta, xi, rho, r, dt, seed=None):
        self.s0 = s0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.r = r
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, n_paths, n_steps):
        S = np.zeros((n_paths, n_steps + 1))
        V = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = self.s0
        V[:, 0] = self.v0

        for t in range(n_steps):
            z1 = self.rng.standard_normal(n_paths)
            z2 = self.rng.standard_normal(n_paths)
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            V[:, t+1] = (
                V[:, t] + self.kappa * (self.theta - V[:, t]) * self.dt
                + self.xi * np.sqrt(np.maximum(V[:, t], 0)) * np.sqrt(self.dt) * w2
            )
            V[:, t+1] = np.maximum(V[:, t+1], 0)  # enforce non-negativity

            S[:, t+1] = (
                S[:, t] * np.exp(
                    (self.r - 0.5 * V[:, t]) * self.dt + np.sqrt(V[:, t]) * np.sqrt(self.dt) * w1
                )
            )

        return S, V