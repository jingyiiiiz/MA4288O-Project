import numpy as np

def simulate_five_heston_paths(n_paths=100000, n_steps=30, dt=1/365):
    # each call returns (S_i, V_i), shape (n_paths, n_steps+1)
    # We’ll store them in shape (n_paths, n_steps+1, 5) for the spot paths
    from market.heston import HestonModel

    S_arr = []
    # Example parameters for each underlying:
    param_list = [
      dict(s0=100.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, r=0.0, dt=dt, seed=1234),
      dict(s0=95.0,  v0=0.02, kappa=1.2, theta=0.02, xi=0.3, rho=-0.4, r=0.0, dt=dt, seed=4321),
      dict(s0=105.0, v0=0.06, kappa=1.8, theta=0.06, xi=0.6, rho=-0.8, r=0.0, dt=dt, seed=9999),
      dict(s0=80.0,  v0=0.03, kappa=1.5, theta=0.03, xi=0.5, rho=-0.7, r=0.0, dt=dt, seed=2023),
      dict(s0=120.0, v0=0.05, kappa=1.3, theta=0.05, xi=0.4, rho=-0.9, r=0.0, dt=dt, seed=7777),
    ]

    for params in param_list:
        model = HestonModel(**params)
        S_i, V_i = model.simulate_paths(n_paths, n_steps)
        S_arr.append(S_i)  
        # or store V_i if you need it for advanced payoffs

    # shape => (5, n_paths, n_steps+1)
    S_arr = np.stack(S_arr, axis=0)
    # reorder => (n_paths, n_steps+1, 5)
    S_arr = np.transpose(S_arr, (1,2,0))
    return S_arr  # shape (n_paths, n_steps+1, 5)
