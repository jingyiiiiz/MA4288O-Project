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
    Trainer class for Deep Hedging using configurable optimizers and loss functions.
    Supports both CVaR and Exponential Utility loss.
    """
    def __init__(self, model, optimizer_config=None, loss_function='cvar', alpha=0.5, lam=1.0):
        self.model = model
        self.alpha = alpha  # CVaR risk parameter
        self.lam = lam  # Exponential utility risk aversion
        self.optimizer_config = optimizer_config if optimizer_config else {'name': 'adam', 'learning_rate': 1e-3}
        self.loss_function = loss_function
        self.p0 = None  # Trainable initial price
        self.optimizer = self.configure_optimizer()

    def configure_optimizer(self):
        optimizer_name = self.optimizer_config.get('name', 'adam')
        learning_rate = self.optimizer_config.get('learning_rate', 1e-3)
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def compute_loss(self, pnl, deltas, S, payoff):
        from .loss_functions import cvar_loss_canonical, exponential_utility_loss
        
        if self.loss_function == 'cvar':
            loss, _ = cvar_loss_canonical(pnl, alpha=self.alpha)
        elif self.loss_function == 'exp_utility':
            loss = exponential_utility_loss(S, deltas, payoff, lam=self.lam)
        else:
            raise ValueError("Invalid loss function. Choose 'cvar' or 'exp_utility'")
        
        return loss

    def train(self, S_tensor, Z_tensor, p0_init=0.0, n_epochs=10, batch_size=4096):
        """
        Trains the model using the specified loss function and optimizer.
        Accepts either a float or torch.Tensor for p0_init.
        """
        device = S_tensor.device
        dataset_size = S_tensor.shape[0]

        if isinstance(p0_init, torch.Tensor):
            self.p0 = p0_init.clone().detach().to(device).requires_grad_()
        else:
            self.p0 = torch.tensor([p0_init], requires_grad=True, device=device)
        
        param_list = list(self.model.parameters()) + [self.p0]
        self.optimizer = self.configure_optimizer()

        for epoch in range(n_epochs):
            idx = torch.randperm(dataset_size, device=device)
            S_shuffled, Z_shuffled = S_tensor[idx], Z_tensor[idx]
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                Sb, Zb = S_shuffled[start:end], Z_shuffled[start:end]
                
                deltas_b = self.model(Sb)
                Sdiff_b = Sb[:, 1:] - Sb[:, :-1]
                gains_b = torch.sum(deltas_b * Sdiff_b, dim=1)
                pnl_b = self.p0 - Zb + gains_b
                
                loss_b = self.compute_loss(pnl_b, deltas_b, Sb, Zb)
                
                self.optimizer.zero_grad()
                loss_b.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                self.optimizer.step()
            
            with torch.no_grad():
                deltas_all = self.model(S_tensor)
                Sdiff_all = S_tensor[:, 1:] - S_tensor[:, :-1]
                gains_all = torch.sum(deltas_all * Sdiff_all, dim=1)
                pnl_all = self.p0 - Z_tensor + gains_all
                loss_all = self.compute_loss(pnl_all, deltas_all, S_tensor, Z_tensor)
                
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss_all.item():.4f} | p0: {self.p0.item():.4f}")

        return self.p0.item()
