import numpy as np

def basket_call_payoff(S_paths, K=100.0):
    """
    S_paths: shape (n_paths, n_steps+1, 5)
    returns shape (n_paths,)
    final basket = average of final spots across 5 underlyings
    """
    final_spots = S_paths[:, -1, :]  # shape (n_paths, 5)
    basket = final_spots.mean(axis=1)  # (n_paths,)
    return np.maximum(basket - K, 0.0)
