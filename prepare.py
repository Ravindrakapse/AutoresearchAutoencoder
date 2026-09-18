"""
Fixed data preparation + evaluation for the OLR autoencoder autoresearch loop.

This file is FROZEN — do not modify it during experimentation. It defines:
  - which data is loaded and how it is split (train / val / test, by time block)
  - the preprocessing (center on train mean field, standardize, sqrt(area) weighting)
  - the single ground-truth metric: area-weighted NRMSE^2 (= 1 - R^2), lower is better
  - the fixed training TIME_BUDGET

Design trick — the "Y space":
  Everything downstream (PCA, linear AE, nonlinear AE) works on a transformed field Y:
      Xc = X - mean_field          (center on TRAIN per-cell temporal mean)
      Xz = Xc / std                (scalar std from TRAIN, gives O(1) inputs for NNs)
      Y  = Xz * s                  (s = sqrt(normalized cos-lat area weight) per cell)
  Because s^2 is the (normalized) grid-cell area, a PLAIN unweighted MSE in Y space
  equals the cos(lat) AREA-WEIGHTED MSE in physical space, and a plain SVD of Y is the
  area-weighted EOF/PCA. So every method uses ordinary MSE and stays directly comparable,
  with area weighting baked in for free. Y is centered, so the "mean field" baseline is
  simply Yhat = 0, and the metric denominator is the area-weighted variance.

Usage:
    python prepare.py            # sanity-check: prints shapes, splits, baseline bar
"""
import glob
import os

import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
# 2.5 deg global weekly OLR (cos-lat weighted coarsen). Globs so it picks up more years.
WEEKLY_GLOB = os.path.join(HERE, "..", "OLR_data", "weekly_olr", "era5_olr_weekly_*_c10.nc")

TIME_BUDGET = 120.0   # training wall-clock budget in seconds (excludes startup/eval)
SEED = 42

# FROZEN bottleneck size. The research question is "nonlinear AE vs linear PCA at a fixed,
# small latent dim". It lives here (not train.py) so an experiment CANNOT change it: a bigger
# latent trivially lowers error and is not a real result. evaluate_ae asserts the encoder
# outputs exactly this many numbers.
LATENT_DIM = 16

# Temporal block split (no leakage — OLR is strongly autocorrelated, never split randomly).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15       # test = remaining 0.15, held out, touched only for final reporting

# ---------------------------------------------------------------------------
# Load + build the dataset (cached at module level)
# ---------------------------------------------------------------------------

_CACHE = {}


def _pick_weekly_file():
    files = sorted(glob.glob(WEEKLY_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No weekly OLR file matching {WEEKLY_GLOB}. "
            f"Run OLR_data/make_weekly.py --coarsen 10 first."
        )
    # Prefer the widest year span (filename ...weekly_<start>-<end>_c10.nc); sorted() puts
    # the longest range last for a fixed start year, good enough — just take the last.
    return files[-1]


def _build():
    if _CACHE:
        return _CACHE
    import xarray as xr

    path = _pick_weekly_file()
    ds = xr.open_dataset(path)
    olr = ds["olr"]  # (time, latitude, longitude), W m-2
    lat = olr["latitude"].values
    lon = olr["longitude"].values
    time = olr["time"].values
    X = olr.values.astype(np.float64)          # (N, nlat, nlon)
    N, nlat, nlon = X.shape
    D = nlat * nlon
    Xflat = X.reshape(N, D)

    # Temporal block split (chronological).
    n_train = int(round(N * TRAIN_FRAC))
    n_val = int(round(N * VAL_FRAC))
    idx = {
        "train": np.arange(0, n_train),
        "val": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, N),
        "all": np.arange(0, N),
    }

    # Transform params fit on TRAIN ONLY.
    tr = Xflat[idx["train"]]
    mean_field = tr.mean(axis=0)                # (D,) per-cell temporal mean
    std = tr.std()                             # scalar std of centered train data
    if std == 0:
        std = 1.0

    # sqrt(area) weight per cell, normalized so mean(s^2)=1 (keeps Y at O(1)).
    w = np.cos(np.deg2rad(lat))                # (nlat,)
    w2d = np.repeat(w[:, None], nlon, axis=1).reshape(D)  # (D,)
    w2d = w2d * (D / w2d.sum())                # normalize: mean weight = 1
    s = np.sqrt(w2d)                           # (D,)

    def to_Y(Xf):
        return (((Xf - mean_field) / std) * s).astype(np.float32)

    Y = to_Y(Xflat)                            # (N, D) working representation

    _CACHE.update(dict(
        path=path, N=N, nlat=nlat, nlon=nlon, D=D, lat=lat, lon=lon, time=time,
        idx=idx, mean_field=mean_field, std=std, s=s, Y=Y,
    ))
    return _CACHE


# ---------------------------------------------------------------------------
# Public API (used by baselines.py and train.py)
# ---------------------------------------------------------------------------

def get_split(split):
    """Return the working representation Y for a split: (n, D) float32, centered+weighted."""
    c = _build()
    assert split in c["idx"], f"bad split {split}"
    return c["Y"][c["idx"][split]]


def meta():
    """Return dict with N, nlat, nlon, D, lat, lon, time, split sizes, data path."""
    c = _build()
    return dict(
        N=c["N"], nlat=c["nlat"], nlon=c["nlon"], D=c["D"],
        lat=c["lat"], lon=c["lon"], time=c["time"], path=c["path"],
        n_train=len(c["idx"]["train"]), n_val=len(c["idx"]["val"]),
        n_test=len(c["idx"]["test"]),
    )


def inverse_transform(Y):
    """Map working representation Y (n, D) back to physical OLR field (n, nlat, nlon) in W m-2."""
    c = _build()
    Xf = (np.asarray(Y, dtype=np.float64) / c["s"]) * c["std"] + c["mean_field"]
    return Xf.reshape(-1, c["nlat"], c["nlon"])


def evaluate_recon(reconstruct_fn, split="val"):
    """
    FIXED METRIC. reconstruct_fn: (n, D) Y -> (n, D) Yhat in the same working space.
    Returns (nrmse2, r2) where:
      nrmse2 = mean_n ||Y - Yhat||^2  /  mean_n ||Y||^2   (area-weighted, since baked into Y)
      r2     = 1 - nrmse2
    Y is centered, so the mean-field predictor (Yhat=0) scores exactly nrmse2 = 1.0.
    Lower nrmse2 is better.
    """
    Y = np.asarray(get_split(split), dtype=np.float64)
    Yhat = np.asarray(reconstruct_fn(Y.astype(np.float32)), dtype=np.float64)
    assert Yhat.shape == Y.shape, f"reconstruct_fn shape {Yhat.shape} != {Y.shape}"
    num = ((Y - Yhat) ** 2).sum(axis=1).mean()
    den = (Y ** 2).sum(axis=1).mean()
    nrmse2 = num / den
    return float(nrmse2), float(1.0 - nrmse2)


def evaluate_ae(encode_fn, decode_fn, split="val"):
    """
    STRICT autoencoder metric — the fair way to compare against PCA at the FROZEN latent dim.

    encode_fn: (n, D) Y  -> (n, LATENT_DIM) code
    decode_fn: (n, LATENT_DIM) code -> (n, D) Yhat

    Reconstruction is forced through a single code of EXACTLY LATENT_DIM numbers (the frozen
    constant above): decode_fn only ever sees the code, never the input, so no skip/residual
    path can widen the bottleneck; and the width is asserted against LATENT_DIM, so an
    experiment cannot inflate the latent (via a parallel skip OR by declaring a bigger latent)
    to trivially lower the error. Keeps "AE vs PCA at latent LATENT_DIM" honest and fixed.

    Returns (nrmse2, r2) — same area-weighted metric as evaluate_recon.
    """
    Y = np.asarray(get_split(split), dtype=np.float64)
    Z = np.asarray(encode_fn(Y.astype(np.float32)))
    assert Z.ndim == 2 and Z.shape[0] == Y.shape[0], \
        f"encode_fn must return (n, LATENT_DIM); got {Z.shape}"
    assert Z.shape[1] == LATENT_DIM, (
        f"code width {Z.shape[1]} != frozen LATENT_DIM {LATENT_DIM}. The latent dimension is "
        f"fixed in prepare.py and must not be changed; encoder must output exactly LATENT_DIM "
        f"numbers — no extra/skip/parallel dimensions, no larger latent."
    )
    Yhat = np.asarray(decode_fn(Z.astype(np.float32)), dtype=np.float64)
    assert Yhat.shape == Y.shape, f"decode_fn shape {Yhat.shape} != {Y.shape}"
    num = ((Y - Yhat) ** 2).sum(axis=1).mean()
    den = (Y ** 2).sum(axis=1).mean()
    nrmse2 = num / den
    return float(nrmse2), float(1.0 - nrmse2)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    m = meta()
    print(f"data file : {m['path']}")
    print(f"grid      : {m['nlat']} x {m['nlon']}  (D={m['D']})")
    print(f"samples   : N={m['N']}  train={m['n_train']} val={m['n_val']} test={m['n_test']}")
    for sp in ("train", "val", "test"):
        Y = get_split(sp)
        print(f"  {sp:5s} Y shape {Y.shape}  mean {Y.mean():+.4f}  var {Y.var():.4f}")
    # Baseline: mean field predictor (Yhat = 0) must score nrmse2 = 1.0, r2 = 0.
    n0, r0 = evaluate_recon(lambda Y: np.zeros_like(Y), "val")
    print(f"mean-field baseline (val): nrmse2={n0:.4f}  r2={r0:+.4f}  (expect 1.0 / 0.0)")
