"""
Nonlinear autoencoder for OLR dimensionality reduction.  <-- THE ONLY FILE THE AGENT EDITS.

Everything is fair game: architecture, latent dim, depth/width, activations, normalization,
dropout, optimizer, LR schedule, batch size, regularization, etc. The one rule: it must run
without crashing and finish within prepare.TIME_BUDGET seconds of training.

The metric is prepare.evaluate_recon(...) on the "val" split: area-weighted NRMSE^2 = 1 - R^2
(lower is better). It is computed in prepare.py's frozen Y space, so it is directly comparable
to the PCA / mean-field baselines and to every other experiment.

Run:  python train.py > run.log 2>&1   then   grep "^val_nrmse2:" run.log
"""
import time
t_start = time.time()

import numpy as np
import torch
import torch.nn as nn

import prepare

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

LATENT_DIM = 16
HIDDEN = [256, 64]       # encoder widths; decoder mirrors it. [] = single linear layer.
ACT = "gelu"            # relu | gelu | tanh
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
NOISE_STD = 0.3         # denoising AE: Gaussian input corruption during training
TIE_STD_INPUT = False   # (reserved) placeholder for future input-norm experiments

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

torch.manual_seed(prepare.SEED)
np.random.seed(prepare.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Ytr = prepare.get_split("train")            # (n, D) float32, centered + area-weighted
D = Ytr.shape[1]
Xtr = torch.from_numpy(Ytr).to(device)
print(f"device={device}  train={Xtr.shape}  D={D}  latent={LATENT_DIM}")

_ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class AE(nn.Module):
    def __init__(self, D, latent, hidden, act, dropout):
        super().__init__()
        Act = _ACTS[act]
        # Encoder: D -> hidden... -> latent
        enc, d = [], D
        for h in hidden:
            enc += [nn.Linear(d, h), nn.BatchNorm1d(h), Act()]
            if dropout > 0:
                enc += [nn.Dropout(dropout)]
            d = h
        enc += [nn.Linear(d, latent)]
        self.encoder = nn.Sequential(*enc)
        # Decoder: latent -> reverse(hidden)... -> D
        dec, d = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.BatchNorm1d(h), Act()]
            if dropout > 0:
                dec += [nn.Dropout(dropout)]
            d = h
        dec += [nn.Linear(d, D)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = AE(D, LATENT_DIM, HIDDEN, ACT, DROPOUT).to(device)
num_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()  # plain MSE in Y space == area-weighted MSE in physical space

# ---------------------------------------------------------------------------
# Train until the fixed wall-clock budget is spent
# ---------------------------------------------------------------------------

n = Xtr.shape[0]
steps_per_epoch = (n + BATCH_SIZE - 1) // BATCH_SIZE
# Estimate total steps from budget: ~120s, rough epoch time from a quick probe
EST_EPOCHS = 600  # conservative estimate; scheduler wraps if exceeded
total_steps = EST_EPOCHS * steps_per_epoch
warmup_steps = total_steps // 20  # 5% linear warmup
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-5)
scheduler_warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

t_train0 = time.time()
epoch = 0
while time.time() - t_train0 < prepare.TIME_BUDGET:
    model.train()
    perm = torch.randperm(n, device=device)
    for i in range(0, n, BATCH_SIZE):
        xb = Xtr[perm[i:i + BATCH_SIZE]]
        opt.zero_grad(set_to_none=True)
        xb_in = xb + NOISE_STD * torch.randn_like(xb) if NOISE_STD > 0 else xb
        loss = loss_fn(model(xb_in), xb)
        loss.backward()
        opt.step()
        scheduler.step()
    epoch += 1
training_seconds = time.time() - t_train0


# ---------------------------------------------------------------------------
# Evaluate with the frozen metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct(Y):
    model.eval()
    yb = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(device)
    out = model(yb).cpu().numpy()
    return out


val_nrmse2, val_r2 = prepare.evaluate_recon(reconstruct, "val")

# PCA reference at the same latent dim, for context (linear bar).
Yc = Ytr - Ytr.mean(axis=0)
_, _, VT = np.linalg.svd(Yc, full_matrices=False)
Vk = VT[:LATENT_DIM]
pca_fn = lambda Y: (Y - Ytr.mean(axis=0)) @ Vk.T @ Vk + Ytr.mean(axis=0)
pca_ref_nrmse2, _ = prepare.evaluate_recon(pca_fn, "val")

# ---------------------------------------------------------------------------
# Summary (grep-able)
# ---------------------------------------------------------------------------
print("---")
print(f"val_nrmse2:       {val_nrmse2:.6f}")
print(f"val_r2:           {val_r2:.6f}")
print(f"pca_ref_nrmse2:   {pca_ref_nrmse2:.6f}")
print(f"latent_dim:       {LATENT_DIM}")
print(f"num_params:       {num_params}")
print(f"num_epochs:       {epoch}")
print(f"batch_size:       {BATCH_SIZE}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {time.time() - t_start:.1f}")
print(f"device:           {device}")
