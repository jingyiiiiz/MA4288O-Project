import torch

def compute_5D_pnl(p0, payoff, deltas, S):
    """
    p0: float
    payoff: shape (batch_size,)
    deltas: shape (batch_size, steps, 5)
    S:      shape (batch_size, steps+1, 5)
    returns shape (batch_size,) final PnL
    """
    # Sdiff => shape (batch_size, steps, 5)
    Sdiff = S[:, 1:, :] - S[:, :-1, :]
    # sum_{k=0..steps-1} sum_{i=1..5} deltas[k,i]*Sdiff[k,i]
    gains = torch.sum(deltas * Sdiff, dim=(1,2))  # reduce over steps & asset-dim
    return p0 - payoff + gains

def train_5D_cvar(model, S_tensor, payoff_tensor, alpha=0.5, lr=1e-3, p0_init=0.0,
                  n_epochs=10, batch_size=2048):
    """
    Basic CVaR trainer for 5D hedge
    """
    device = S_tensor.device
    p0 = torch.tensor([p0_init], requires_grad=True, device=device)
    param_list = list(model.parameters()) + [p0]
    opt = torch.optim.Adam(param_list, lr=lr)

    dataset_size = S_tensor.shape[0]

    for epoch in range(n_epochs):
        idx = torch.randperm(dataset_size, device=device)
        S_shuffled = S_tensor[idx]
        Z_shuffled = payoff_tensor[idx]

        for start in range(0, dataset_size, batch_size):
            end = min(start+batch_size, dataset_size)
            Sb = S_shuffled[start:end]
            Zb = Z_shuffled[start:end]

            deltas_b = model(Sb)  # (batch, steps, 5)
            pnl_b = compute_5D_pnl(p0, Zb, deltas_b, Sb)
            
            loss_b, w_b = cvar_loss_canonical(pnl_b, alpha=alpha)
            
            opt.zero_grad()
            loss_b.backward()
            opt.step()

        # optional: print progress

    return p0.item()
