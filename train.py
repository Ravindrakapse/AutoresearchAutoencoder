"""
Nonlinear autoencoder for OLR dimensionality reduction.  <-- THE ONLY FILE THE AGENT EDITS.

HARD RULE (enforced by prepare.evaluate_ae): the model must be a single encoder that maps the
input to EXACTLY LATENT_DIM numbers, and a decoder that reconstructs from ONLY those numbers.
No skip / residual / parallel path from the input to the output — that would widen the effective
bottleneck and make the comparison against PCA at LATENT_DIM dishonest. evaluate_ae asserts the
code width, so such tricks fail loudly instead of silently inflating the latent dimension.

The latent dim is FROZEN in prepare.py (LATENT_DIM=16) and must NOT be changed here — a bigger
latent trivially lowers error and is not a real result. Everything else is fair game:
encoder/decoder depth & width, activations, normalization, dropout, noise, optimizer, LR
schedule, batch size, regularization, weight init. Must run within prepare.TIME_BUDGET seconds.

Metric: prepare.evaluate_ae(encode, decode) on "val" = area-weighted NRMSE^2 = 1-R^2,
lower better, directly comparable to PCA at the frozen prepare.LATENT_DIM.

Run:  python train.py > run.log 2>&1   then   grep "^val_nrmse2:" run.log
"""
import time
t_start = time.time()

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

LATENT_DIM = prepare.LATENT_DIM   # FROZEN in prepare.py (=16). Do NOT hardcode another value:
                                  # evaluate_ae asserts the code width equals prepare.LATENT_DIM.
HIDDEN = [128]           # encoder widths in PCA-reduced space; decoder mirrors.
PCA_PRE_DIM = 256        # fixed PCA pre-reduction: D -> PCA_PRE_DIM before the learnable AE
ACT = "silu"            # relu | gelu | tanh | silu
DROPOUT = 0.1
BATCH_NORM = True
NOISE_STD = 0.0         # no noise: in PCA space, uniform noise disproportionately corrupts low-variance components
BATCH_SIZE = 128
LR = 4e-3
WEIGHT_DECAY = 0
WARMUP_FRAC = 0.05      # fraction of budget for linear LR warmup
ETA_MIN = 1e-5          # cosine annealing floor
SWA_START_FRAC = 0.75   # start averaging weights after this fraction of training

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

# --- PCA pre-reduction: D -> PCA_PRE_DIM (fixed, not learned) ---
Ytr_np = Xtr.cpu().numpy()
Ytr_c = Ytr_np - Ytr_np.mean(axis=0)
_, _, VT_pre = np.linalg.svd(Ytr_c, full_matrices=False)
Vk_pre = torch.from_numpy(VT_pre[:PCA_PRE_DIM].astype(np.float32)).to(device)  # (PCA_PRE_DIM, D)
Xtr = Xtr @ Vk_pre.T  # (n, PCA_PRE_DIM)
D_eff = PCA_PRE_DIM
print(f"PCA pre-reduction: {D} -> {D_eff}")

_ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


class TiedLinear(nn.Module):
    """Decoder linear layer sharing (transposed) weight with an encoder layer."""
    def __init__(self, tied_to):
        super().__init__()
        self._ref = [tied_to]  # list avoids registering as submodule
        self.bias = nn.Parameter(torch.zeros(tied_to.in_features))

    def forward(self, x):
        return F.linear(x, self._ref[0].weight.t(), self.bias)


class AE(nn.Module):
    """Strict autoencoder: encoder -> LATENT_DIM code -> decoder (tied weights). No input->output skip."""

    def __init__(self, D, latent, hidden, act, dropout, batch_norm):
        super().__init__()
        Act = _ACTS[act]

        # --- Encoder ---
        enc_linears = []
        enc = []
        d = D
        for h in hidden:
            lin = nn.Linear(d, h)
            enc_linears.append(lin)
            enc.append(lin)
            if batch_norm:
                enc.append(nn.BatchNorm1d(h))
            enc.append(Act())
            if dropout > 0:
                enc.append(nn.Dropout(dropout))
            d = h
        final_enc = nn.Linear(d, latent)
        enc_linears.append(final_enc)
        enc.append(final_enc)
        self.encoder = nn.Sequential(*enc)

        # --- Decoder (tied weights = encoder weights transposed) ---
        dec = []
        reversed_lins = list(reversed(enc_linears))
        for lin in reversed_lins[:-1]:
            dec.append(TiedLinear(lin))
            if batch_norm:
                dec.append(nn.BatchNorm1d(lin.in_features))
            dec.append(Act())
            if dropout > 0:
                dec.append(nn.Dropout(dropout))
        dec.append(TiedLinear(reversed_lins[-1]))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))      # reconstruction goes ONLY through the code


model = AE(D_eff, LATENT_DIM, HIDDEN, ACT, DROPOUT, BATCH_NORM).to(device)
num_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()  # plain MSE in Y space == area-weighted MSE in physical space

# ---------------------------------------------------------------------------
# Train until the fixed wall-clock budget is spent (LR: linear warmup + cosine anneal)
# ---------------------------------------------------------------------------

n = Xtr.shape[0]
t_train0 = time.time()
epoch = 0
swa_params = None   # SWA: running equal-weight average of parameters
swa_n = 0
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
    # SWA: accumulate equal-weight parameter average over late training
    if progress >= SWA_START_FRAC:
        if swa_params is None:
            swa_params = [p.data.clone() for p in model.parameters()]
            swa_n = 1
        else:
            swa_n += 1
            for sp, p in zip(swa_params, model.parameters()):
                sp += (p.data - sp) / swa_n
    epoch += 1
training_seconds = time.time() - t_train0

# SWA: load averaged weights and refresh BN running statistics
if swa_params is not None:
    for sp, p in zip(swa_params, model.parameters()):
        p.data.copy_(sp)
    model.train()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            xb = Xtr[i:i + BATCH_SIZE]
            if xb.shape[0] >= 2:
                model(xb)
    print(f"SWA: averaged {swa_n} checkpoints")

# ---------------------------------------------------------------------------
# Evaluate with the frozen STRICT metric (forces reconstruction through the code only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_fn(Y):
    model.eval()
    Yt = torch.from_numpy(np.asarray(Y, np.float32)).to(device)
    return model.encode(Yt @ Vk_pre.T).cpu().numpy()

@torch.no_grad()
def decode_fn(Z):
    model.eval()
    Zt = torch.from_numpy(np.asarray(Z, np.float32)).to(device)
    return (model.decode(Zt) @ Vk_pre).cpu().numpy()

val_nrmse2, val_r2 = prepare.evaluate_ae(encode_fn, decode_fn, "val")

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
