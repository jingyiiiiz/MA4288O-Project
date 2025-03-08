# File: optimizer/loss_functions.py
import torch

def exponential_utility_loss(S, deltas, payoff, lam=1.0):
    """
    frictionless PnL = -payoff + sum_{k=0..n-1} deltas_k * (S_{k+1} - S_{k})
    We minimize E[ exp(-lam * PnL) ].
    """
    # S shape: (batch_size, steps+1)
    # deltas shape: (batch_size, steps)
    # payoff shape: (batch_size,)

    S_diff = S[:, 1:] - S[:, :-1]  # shape (batch_size, steps)
    gains = (deltas * S_diff).sum(dim=1)  # shape (batch_size,)

    pnl = -payoff + gains
    loss = torch.exp(-lam * pnl).mean()
    return loss
