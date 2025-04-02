import torch
import torch.optim as optim
import torch.nn.functional as F

def cvar_loss_canonical(pnl, alpha=0.5):
    X = -pnl
    w = X.median()
    loss = w + (1.0 / (1.0 - alpha)) * torch.mean(F.relu(X - w))
    return loss, w

class DeepHedgeCVaRTrainer:
    def __init__(self, model, optimizer_config=None, loss_function='cvar', alpha=0.5, lam=1.0):
        self.model = model
        self.alpha = alpha
        self.lam = lam
        self.optimizer_config = optimizer_config if optimizer_config else {'name': 'adam', 'learning_rate': 1e-3}
        self.loss_function = loss_function
        self.p0 = None
        self.optimizer = self.configure_optimizer()

    def configure_optimizer(self):
        name = self.optimizer_config.get('name', 'adam')
        lr = self.optimizer_config.get('learning_rate', 1e-3)
        if name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def compute_loss(self, pnl, deltas, S, payoff):
        if self.loss_function == 'cvar':
            loss, _ = cvar_loss_canonical(pnl, alpha=self.alpha)
        else:
            raise ValueError("Only 'cvar' loss is implemented in this version.")
        return loss

    def train(self, S_tensor, Z_tensor, p0_init=0.0, n_epochs=10, batch_size=4096):
        device = S_tensor.device
        dataset_size = S_tensor.shape[0]
        self.p0 = torch.tensor([p0_init], requires_grad=True, device=device)

        param_list = list(self.model.parameters()) + [self.p0]
        self.optimizer = self.configure_optimizer()

        for epoch in range(n_epochs):
            idx = torch.randperm(dataset_size, device=device)
            S_shuffled = S_tensor[idx]
            Z_shuffled = Z_tensor[idx]

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                Sb = S_shuffled[start:end]
                Zb = Z_shuffled[start:end]

                deltas_b = self.model(Sb)[:, :-1]                  # Trim to steps-1
                Sdiff_b = Sb[:, 1:, 0] - Sb[:, :-1, 0]             # Only S component
                gains_b = torch.sum(deltas_b * Sdiff_b, dim=1)
                pnl_b = self.p0 - Zb + gains_b

                loss_b = self.compute_loss(pnl_b, deltas_b, Sb, Zb)

                self.optimizer.zero_grad()
                loss_b.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            with torch.no_grad():
                deltas_all = self.model(S_tensor)[:, :-1]
                Sdiff_all = S_tensor[:, 1:, 0] - S_tensor[:, :-1, 0]
                gains_all = torch.sum(deltas_all * Sdiff_all, dim=1)
                pnl_all = self.p0 - Z_tensor + gains_all
                loss_all = self.compute_loss(pnl_all, deltas_all, S_tensor, Z_tensor)

            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss_all.item():.4f} | p0: {self.p0.item():.4f}")

        return self.p0.item(), pnl_all.detach().cpu().numpy(), loss_all.item(), self.p0.item()
