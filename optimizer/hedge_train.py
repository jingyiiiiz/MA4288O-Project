# optimizer/hedge_train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DeepHedgeCVaRTrainer:
    def __init__(self, alpha=0.5, lr=1e-3, n_epochs=50, init_delta=None):
        self.alpha = alpha
        self.lr = lr
        self.n_epochs = n_epochs
        self.init_delta = init_delta  # scalar float (e.g. BS delta) or None

    def train(self, S, Z):
        device = S.device
        batch_size, n_steps = S.shape[0], S.shape[1] - 1

        # Simple model: one learnable hedge per time step
        hedge = nn.Parameter(torch.zeros(n_steps, device=device))
        if self.init_delta is not None:
            hedge.data.fill_(self.init_delta)

        opt = optim.Adam([hedge], lr=self.lr)

        q_history = []
        loss_history = []

        for epoch in range(self.n_epochs):
            opt.zero_grad()

            dS = S[:, 1:] - S[:, :-1]  # shape: [batch_size, n_steps]
            gains = torch.sum(hedge * dS, dim=1)  # [batch_size]
            q = torch.mean(Z - gains)  # Risk-adjusted price estimate

            losses = torch.clamp(q - (Z - gains), min=0.0)
            sorted_losses, _ = torch.sort(losses)
            cvar_idx = int(self.alpha * batch_size)
            cvar_loss = torch.mean(sorted_losses[:cvar_idx])

            cvar_loss.backward()
            opt.step()

            # Logging
            loss_history.append(cvar_loss.item())
            q_history.append(q.item())

        # Final evaluation
        with torch.no_grad():
            dS = S[:, 1:] - S[:, :-1]
            gains = torch.sum(hedge * dS, dim=1)
            q_final = torch.mean(Z - gains).item()
            pnl = (q_final - Z + gains).cpu().numpy()

        return q_final, pnl, loss_history, q_history
