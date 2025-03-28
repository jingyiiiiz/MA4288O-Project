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
    def __init__(self, steps=30, hidden_dim=32):
        super(RecurrentHedgeModel, self).__init__()
        self.steps = steps
        self.hidden_dim = hidden_dim
        self.shared_net = SharedHedgeNetwork(input_dim=3, hidden_dim=hidden_dim, output_dim=1)
        self.a_init = nn.Parameter(torch.zeros(1))  # Trainable initial delta

    def forward(self, S):
        batch_size = S.size(0)
        deltas = []
        h_t = torch.zeros(batch_size, 1, device=S.device)
        delta_prev = self.a_init.expand(batch_size, 1)

        for t in range(self.steps):
            S_t = S[:, t].unsqueeze(1)
            input_t = torch.cat([S_t, delta_prev, h_t], dim=1)
            delta_t = self.shared_net(input_t)
            deltas.append(delta_t)
            h_t = 0.8 * h_t + 0.2 * delta_t  # Track recurrent state
            delta_prev = delta_t

        return torch.cat(deltas, dim=1)


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
    
class SharedHedgeNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1):
        super(SharedHedgeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MultiAssetHedgeModel(nn.Module):
    """
    Recurrent approach for two instruments, e.g. underlying + var swap.
    delta_k = F_k( S_k, VSW_k, delta_{k-1} ).
    """
    def __init__(self, steps=30, in_dim=4, hidden_dim=32, out_dim=2):
        """
        steps: number of rebalancing times
        in_dim=4 => e.g. [S_k, vsw_k, delta_S_{k-1}, delta_V_{k-1}]
        out_dim=2 => [delta_S_k, delta_VSW_k]
        """
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

    def forward(self, market):
        """
        market shape => (batch_size, steps+1, 2) i.e. [S_k, vsw_k]
        returns => deltas (batch_size, steps, 2)
        """
        batch_size = market.shape[0]
        delta_prev = torch.zeros(batch_size, 2, device=market.device)
        deltas_list = []

        for k in range(self.steps):
            S_k = market[:, k, 0].unsqueeze(1)
            VSW_k = market[:, k, 1].unsqueeze(1)
            x = torch.cat([S_k, VSW_k, delta_prev], dim=1)  # shape (batch, 4)
            delta_k = self.day_nets[k](x)                  # shape (batch,2)
            deltas_list.append(delta_k)
            delta_prev = delta_k

        return torch.stack(deltas_list, dim=1)  # (batch_size, steps, 2)
