# File: payoff/european_option.py
import numpy as np

def european_call_payoff(S_paths, K=1.0):
    """
    Computes payoff of a European call option on the final price in S_paths.

    Args:
      S_paths (np.ndarray): shape (n_paths, n_steps+1), each row is a path of the underlying.
      K (float): strike price

    Returns:
      payoff (np.ndarray): shape (n_paths,), each entry is max(S_T - K, 0).
    """
    final_prices = S_paths[:, -1]
    return np.maximum(final_prices - K, 0.0)