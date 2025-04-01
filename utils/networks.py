# File: utils/networks.py

import torch
import torch.nn as nn

class RecurrentHedgeModel(nn.Module):
    """
    Standard recurrent hedge model for single-asset inputs.
    delta_k = f(S_k, delta_{k-1}, h_{k-1})
    Output: shape (batch_size, steps, 1)
    """
    def __init__(self, steps=30, input_dim=1, hidden_dim=32):
        super().__init__()
        self.steps = steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Shared feedforward net per time step
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim + 1 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Hidden state update (simple GRU-style recurrence)
        self.gru = nn.GRUCell(input_dim + 1 + hidden_dim, hidden_dim)

    def forward(self, S):
        """
        S: shape (batch_size, steps+1, input_dim)  # e.g. (N, 31, 1)
        Returns: shape (batch_size, steps, 1)
        """
        batch_size = S.shape[0]
        h_t = torch.zeros(batch_size, self.hidden_dim, device=S.device)
        delta_prev = torch.zeros(batch_size, 1, device=S.device)
        deltas = []

        for t in range(self.steps):
            S_t = S[:, t, :]  # ✅ keep shape (batch_size, input_dim)
            input_t = torch.cat([S_t, delta_prev, h_t], dim=1)  # (batch, input+1+hidden)
            delta_t = self.shared_net(input_t)  # (batch_size, 1)
            deltas.append(delta_t)
            delta_prev = delta_t
            h_t = self.gru(input_t, h_t)

        return torch.stack(deltas, dim=1)  # (batch_size, steps, 1)
