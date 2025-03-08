import torch
import torch.nn as nn
import torch.nn.functional as F

class HedgeNetwork(nn.Module):
    """
    A small feedforward sub-network used by RecurrentHedgeModel for each day.
    """
    def __init__(self, in_dim=2, hidden_dim=16, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        x: shape (batch_size, in_dim)
        returns shape (batch_size, out_dim)
        """
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out

class RecurrentHedgeModel(nn.Module):
    """
    Recurrent approach: delta_k = F( S_k, delta_{k-1} ).
    Creates `steps` sub-networks, each a HedgeNetwork,
    to produce daily hedges for each time step.
    """
    def __init__(self, steps=30, in_dim=2, hidden_dim=16, out_dim=1):
        super().__init__()
        self.steps = steps
        # Build one HedgeNetwork per day (time step)
        self.day_nets = nn.ModuleList([
            HedgeNetwork(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
            for _ in range(steps)
        ])

    def forward(self, S):
        """
        S shape: (batch_size, steps+1)
        Returns deltas shape: (batch_size, steps)
        """
        batch_size = S.shape[0]
        deltas_list = []
        # Start with zero position
        delta_prev = torch.zeros(batch_size, 1, device=S.device)

        for k in range(self.steps):
            # S_k is (batch_size,)
            S_k = S[:, k].unsqueeze(1)  # shape => (batch_size, 1)
            # Combine with previous delta
            inp = torch.cat([S_k, delta_prev], dim=1)  # shape => (batch_size, 2)
            delta_k = self.day_nets[k](inp)           # shape => (batch_size, 1)
            deltas_list.append(delta_k)
            delta_prev = delta_k

        # Combine into (batch_size, steps)
        deltas = torch.cat(deltas_list, dim=1)
        return deltas


class SimpleHedgeModel(nn.Module):
    """
    Simpler approach: delta_k = F(S_k), ignoring previous delta.
    We'll define a single-later approach for demonstration:
       - each day uses the *same* MLP, applied pointwise to S_k
    """
    def __init__(self, steps=30, hidden_dim=16):
        super().__init__()
        self.steps = steps
        self.hidden_dim = hidden_dim

        # We'll define layers for a small MLP that goes from (1 -> hidden_dim -> 1)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, S):
        """
        S shape: (batch_size, steps+1).
        We'll produce deltas for k=0..steps-1 by applying the same net to S_k.
        """
        batch_size = S.shape[0]
        # We only need S_0..S_{steps-1}
        S_input = S[:, :-1]  # shape => (batch_size, steps)

        # Flatten so each S_k becomes a row
        S_flat = S_input.reshape(-1, 1)  # (batch_size*steps, 1)

        h = F.relu(self.fc1(S_flat))
        out_flat = self.fc2(h)  # => (batch_size*steps, 1)

        # Reshape back to (batch_size, steps)
        deltas = out_flat.reshape(batch_size, self.steps)
        return deltas