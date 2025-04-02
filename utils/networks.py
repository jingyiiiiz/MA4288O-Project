# File: utils/networks.py

import torch
import torch.nn as nn

class RecurrentHedgeModel(nn.Module):
    def __init__(self, steps, input_dim=2, hidden_dim=32):
        super().__init__()
        self.steps = steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # delta_prev (1D) + h_t (hidden_dim)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + 1 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, S):
        batch_size = S.shape[0]
        delta_prev = torch.zeros(batch_size, 1, device=S.device)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=S.device)

        deltas = []
        for t in range(self.steps):
            S_t = S[:, t, :]  # shape: (batch, input_dim)
            input_t = torch.cat([S_t, delta_prev, h_t], dim=1)
            delta_t = self.shared_net(input_t)
            deltas.append(delta_t)
            delta_prev = delta_t

        return torch.stack(deltas, dim=1).squeeze(-1)  # shape: (batch, steps)
