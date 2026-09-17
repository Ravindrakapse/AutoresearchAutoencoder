"""
Nonlinear autoencoder for OLR dimensionality reduction.  <-- THE ONLY FILE THE AGENT EDITS.

HARD RULE (enforced by prepare.evaluate_ae): the model must be a single encoder that maps the
input to EXACTLY LATENT_DIM numbers, and a decoder that reconstructs from ONLY those numbers.
No skip / residual / parallel path from the input to the output — that would widen the effective
bottleneck and make the comparison against PCA at LATENT_DIM dishonest. evaluate_ae asserts the
code width, so such tricks fail loudly instead of silently inflating the latent dimension.

Everything else is fair game: encoder/decoder depth & width, activations, normalization, dropout,
noise, optimizer, LR schedule, batch size, regularization, weight init, latent size itself
(but then it is compared to PCA at that same size). Must run within prepare.TIME_BUDGET seconds.

Metric: prepare.evaluate_ae(encode, decode, LATENT_DIM) on "val" = area-weighted NRMSE^2 = 1-R^2,
lower better, directly comparable to PCA at the SAME LATENT_DIM.

Run:  python train.py > run.log 2>&1   then   grep "^val_nrmse2:" run.log
"""
import time
t_start = time.time()

import math
import numpy as np
import torch
import torch.nn as nn

import prepare

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

LATENT_DIM = 16
HIDDEN = [256, 64]      # encoder widths; decoder mirrors. bottleneck is always LATENT_DIM.
ACT = "silu"            # relu | gelu | tanh | silu
DROPOUT = 0.1
BATCH_NORM = True
NOISE_STD = 0.3         # denoising AE: Gaussian input corruption during training
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_FRAC = 0.05      # fraction of budget for linear LR warmup
ETA_MIN = 1e-5          # cosine annealing floor

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

_ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


class AE(nn.Module):
    """Strict autoencoder: encoder -> LATENT_DIM code -> decoder. No input->output skip."""

    def __init__(self, D, latent, hidden, act, dropout, batch_norm):
        super().__init__()
        Act = _ACTS[act]

        def block(i, o):
            layers = [nn.Linear(i, o)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(o))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return layers

        enc, d = [], D
        for h in hidden:
            enc += block(d, h); d = h
        enc += [nn.Linear(d, latent)]           # -> exactly latent numbers
        self.encoder = nn.Sequential(*enc)

        dec, d = [], latent
        for h in reversed(hidden):
            dec += block(d, h); d = h
        dec += [nn.Linear(d, D)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))      # reconstruction goes ONLY through the code


model = AE(D, LATENT_DIM, HIDDEN, ACT, DROPOUT, BATCH_NORM).to(device)
num_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()  # plain MSE in Y space == area-weighted MSE in physical space

# ---------------------------------------------------------------------------
# Train until the fixed wall-clock budget is spent (LR: linear warmup + cosine anneal)
# ---------------------------------------------------------------------------

n = Xtr.shape[0]
t_train0 = time.time()
epoch = 0
while True:
    progress = (time.time() - t_train0) / prepare.TIME_BUDGET
    if progress >= 1.0:
        break
    if progress < WARMUP_FRAC:
        lr = LR * progress / WARMUP_FRAC
    else:
        p = (progress - WARMUP_FRAC) / (1.0 - WARMUP_FRAC)
        lr = ETA_MIN + 0.5 * (LR - ETA_MIN) * (1 + math.cos(math.pi * p))
    for g in opt.param_groups:
        g["lr"] = lr

    model.train()
    perm = torch.randperm(n, device=device)
    for i in range(0, n, BATCH_SIZE):
        xb = Xtr[perm[i:i + BATCH_SIZE]]
        if xb.shape[0] < 2 and BATCH_NORM:      # BN needs >=2 samples
            continue
        opt.zero_grad(set_to_none=True)
        xb_in = xb + NOISE_STD * torch.randn_like(xb) if NOISE_STD > 0 else xb
        loss = loss_fn(model(xb_in), xb)
        loss.backward()
        opt.step()
    epoch += 1
training_seconds = time.time() - t_train0

# ---------------------------------------------------------------------------
# Evaluate with the frozen STRICT metric (forces reconstruction through the code only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_fn(Y):
    model.eval()
    return model.encode(torch.from_numpy(np.asarray(Y, np.float32)).to(device)).cpu().numpy()

@torch.no_grad()
def decode_fn(Z):
    model.eval()
    return model.decode(torch.from_numpy(np.asarray(Z, np.float32)).to(device)).cpu().numpy()

val_nrmse2, val_r2 = prepare.evaluate_ae(encode_fn, decode_fn, LATENT_DIM, "val")

# PCA reference at the SAME latent dim (fair linear bar).
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
