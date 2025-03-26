# utils/multi_networks.py (for example)

import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentHedgeModel5D(nn.Module):
    """
    Recurrent approach for 5 underlyings:
      delta_k = F(S_k1, S_k2, S_k3, S_k4, S_k5, delta_{k-1})
    => out dimension = 5 for each step
    """
    def __init__(self, steps=30, in_dim=5+5, hidden_dim=64, out_dim=5):
        super().__init__()
        self.steps = steps
        self.day_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            for _ in range(steps)
        ])

    def forward(self, S):
        """
        S: shape (batch_size, steps+1, 5)
        returns shape (batch_size, steps, 5)
        """
        batch_size = S.shape[0]
        # Start with zero deltas for each of the 5 assets
        delta_prev = torch.zeros(batch_size, 5, device=S.device)
        deltas_list = []
        for k in range(self.steps):
            # slice out S_k => shape (batch_size, 5)
            S_k = S[:, k, :]
            # cat with delta_prev => shape (batch_size, 10)
            x = torch.cat([S_k, delta_prev], dim=1)
            delta_k = self.day_nets[k](x)  # (batch_size, 5)
            deltas_list.append(delta_k)
            delta_prev = delta_k
        # shape => (batch_size, steps, 5)
        return torch.stack(deltas_list, dim=1)
