###############################################################################
# File: optimizer/hedge_train.py
###############################################################################

import torch
import torch.optim as optim
import torch.nn.functional as F

def cvar_loss_canonical(pnl, alpha=0.5):
    """
    Computes the standard CVaR_{alpha} of negative PnL:
        CVaR_{alpha}(X) = w + 1/(1-alpha) * E[ (X - w)_+ ]
    for X = -pnl (i.e. "loss" is negative PnL).
    
    In this function:
    - We approximate w each mini-batch by the median of X. 
      Alternatively, you can treat w as a trainable parameter.
    
    Returns: (loss, w)
        loss: the scalar to backprop
        w   : the chosen threshold/quantile
    """
    X = -pnl
    w = X.median()  # approximate a quantile
    loss = w + (1.0/(1.0 - alpha)) * torch.mean(F.relu(X - w))
    return loss, w

class DeepHedgeCVaRTrainer:
    """
    A simple trainer class that optimizes a neural-network model for 
    CVaR (alpha) under frictionless PnL = p0 - payoff + sum(delta * (S_{k+1}-S_k)).
    
    Usage:
       model = YourHedgeModel(...)  # e.g. RecurrentHedgeModel, SimpleHedgeModel
       trainer = DeepHedgeCVaRTrainer(model, alpha=0.5, lr=1e-3)
       p0 = trainer.train(S_tensor, Z_tensor, p0_init=0.0, n_epochs=10, batch_size=1024)
    """
    def __init__(self, model, alpha=0.5, lr=1e-3):
        self.model = model
        self.alpha = alpha
        self.lr = lr
        self.opt = None
        self.p0 = None  # We'll treat p0 as a learnable parameter

    def train(self, S_tensor, Z_tensor, p0_init=0.0, n_epochs=10, batch_size=2048):
        """
        Minimizes CVaR_{alpha} of negative PnL:
          PnL = p0 - Z + sum_{k=0..steps-1} [ delta_k(S_{k+1} - S_k) ].

        Args:
          S_tensor: shape (n_scenarios, steps+1), float
          Z_tensor: shape (n_scenarios,), float
          p0_init: initial guess for the hedge price p0
          n_epochs: training epochs
          batch_size: mini-batch size

        Returns:
          The learned p0 as a float
        """
        device = S_tensor.device

        # define p0 as a trainable parameter
        self.p0 = torch.tensor([p0_init], requires_grad=True, device=device)
        param_list = list(self.model.parameters()) + [self.p0]
        self.opt = optim.Adam(param_list, lr=self.lr)

        dataset_size = S_tensor.shape[0]
        steps = self.model.steps  # the model is assumed to have an attribute .steps

        for epoch in range(n_epochs):
            # Shuffle for each epoch
            idx = torch.randperm(dataset_size, device=device)
            S_shuffled = S_tensor[idx]
            Z_shuffled = Z_tensor[idx]

            for start in range(0, dataset_size, batch_size):
                end = min(start+batch_size, dataset_size)
                Sb = S_shuffled[start:end]
                Zb = Z_shuffled[start:end]

                # Forward pass: 
                #   deltas_b => shape (mini_batch, steps)
                deltas_b = self.model(Sb)
                Sdiff_b = Sb[:,1:] - Sb[:,:-1]  # shape (mini_batch, steps)
                gains_b = torch.sum(deltas_b * Sdiff_b, dim=1)  # (mini_batch,)
                pnl_b = self.p0 - Zb + gains_b

                loss_b, w_b = cvar_loss_canonical(pnl_b, alpha=self.alpha)

                self.opt.zero_grad()
                loss_b.backward()
                self.opt.step()

            # End-of-epoch progress check
            with torch.no_grad():
                deltas_all = self.model(S_tensor)
                Sdiff_all = S_tensor[:,1:] - S_tensor[:,:-1]
                gains_all = torch.sum(deltas_all * Sdiff_all, dim=1)
                pnl_all = self.p0 - Z_tensor + gains_all
                loss_all, w_all = cvar_loss_canonical(pnl_all, alpha=self.alpha)

            print(f"Epoch {epoch+1}/{n_epochs} | CVaR loss: {loss_all.item():.4f}, "
                  f"w={w_all.item():.4f}, p0={self.p0.item():.4f}")

        # Return the final learned p0
        return self.p0.item()
