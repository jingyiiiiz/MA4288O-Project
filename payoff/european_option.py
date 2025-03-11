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

# import numpy as np

def multi_asset_european_call_payoff(S_paths, K=100.0):
    """
    Computes the sum of European call option payoffs across multiple underlyings.

    Args:
      S_paths (np.ndarray): shape (n_paths, n_steps+1, num_assets), simulated price paths.
      K (float): Strike price.

    Returns:
      payoff (np.ndarray): shape (n_paths,), sum of call option payoffs over all assets.
    """
    final_prices = S_paths[:, -1, :]  # Final prices of each asset
    individual_payoffs = np.maximum(final_prices - K, 0.0)
    return np.sum(individual_payoffs, axis=1)  # Sum payoffs over all assets
