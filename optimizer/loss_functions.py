# File: optimizer/loss_functions.py
import torch
import torch.nn.functional as F

def exponential_utility_loss(S, deltas, payoff, lam=1.0):
    """
    Computes the exponential utility loss function:
    E[exp(-lam * PnL)] where PnL = -payoff + sum_{k=0}^{n-1} deltas_k * (S_{k+1} - S_k).
    
    Args:
    - S (torch.Tensor): Stock prices over time (batch_size, steps+1)
    - deltas (torch.Tensor): Hedge positions over time (batch_size, steps)
    - payoff (torch.Tensor): Final option payoff (batch_size,)
    - lam (float): Risk aversion parameter (higher = more conservative hedge)
    
    Returns:
    - loss (torch.Tensor): The mean exponential utility loss.
    """
    S_diff = S[:, 1:] - S[:, :-1]
    gains = torch.sum(deltas * S_diff, dim=1)
    pnl = -payoff + gains
    loss = torch.exp(-lam * pnl).mean()
    return loss

def cvar_loss_canonical(pnl, alpha=0.5):
    """
    Computes the CVaR (Conditional Value at Risk) loss:
    CVaR_{alpha}(X) = w + 1/(1-alpha) * E[max(X - w, 0)]
    where X = -pnl (we minimize negative PnL, i.e., maximize positive PnL).
    
    Args:
    - pnl (torch.Tensor): Profit and Loss values (batch_size,)
    - alpha (float): CVaR risk parameter (lower alpha = more conservative risk management)
    
    Returns:
    - loss (torch.Tensor): The computed CVaR loss.
    - w (torch.Tensor): The estimated quantile threshold.
    """
    X = -pnl
    w = X.median()  # Approximate the quantile threshold
    loss = w + (1.0 / (1.0 - alpha)) * torch.mean(F.relu(X - w))
    return loss, w
