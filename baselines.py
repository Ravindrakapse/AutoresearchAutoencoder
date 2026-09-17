"""
Baseline ladder for the OLR dimensionality-reduction problem. Run once to establish
the bar the nonlinear autoencoder must beat.

  1. Mean field   — reconstruct every week as the train mean map (Yhat = 0). nrmse2 = 1.
  2. PCA / EOF     — linear, orthogonal modes via SVD of the (area-weighted) train data.
                     This is the classic OLR analysis and the REAL bar for a given latent dim.
  3. Linear AE     — an AE with no nonlinearity spans the same subspace as PCA, so PCA is
                     its optimum; we report PCA as the linear-AE reference.

Everything runs in prepare.py's frozen "Y space", so PCA here is an area-weighted EOF and
its nrmse2 is directly comparable to whatever train.py's autoencoder reports.

Usage:
    python baselines.py                 # table over default latent dims, on val + test
    python baselines.py --dims 2 4 8 16 32 64
"""
import argparse

import numpy as np

import prepare


def fit_pca(Ytrain):
    """Return (mean, components VT) for PCA in Y space. Ytrain: (n, D)."""
    mu = Ytrain.mean(axis=0)               # ~0 already (Y centered on train mean field)
    U, S, VT = np.linalg.svd(Ytrain - mu, full_matrices=False)
    return mu, VT, S                       # VT: (min(n,D), D) principal axes


def pca_reconstructor(mu, VT, k):
    """Build a reconstruct_fn using the top-k PCA modes."""
    Vk = VT[:k]                            # (k, D)

    def reconstruct(Y):
        Z = (Y - mu) @ Vk.T                # encode  (n, k)
        return (Z @ Vk) + mu              # decode  (n, D)

    return reconstruct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128])
    args = p.parse_args()

    m = prepare.meta()
    print(f"data: {m['path']}")
    print(f"grid {m['nlat']}x{m['nlon']} (D={m['D']}), "
          f"N={m['N']} (train {m['n_train']}, val {m['n_val']}, test {m['n_test']})")
    print()

    Ytr = prepare.get_split("train")
    max_k = min(m["n_train"], m["D"])
    dims = [k for k in args.dims if k <= max_k]

    # 1. Mean field
    nv, rv = prepare.evaluate_recon(lambda Y: np.zeros_like(Y), "val")
    nt, rt = prepare.evaluate_recon(lambda Y: np.zeros_like(Y), "test")
    print(f"{'method':16s} {'k':>4s} {'val_nrmse2':>11s} {'val_r2':>8s} "
          f"{'test_nrmse2':>12s} {'test_r2':>8s}")
    print("-" * 64)
    print(f"{'mean_field':16s} {0:4d} {nv:11.4f} {rv:8.4f} {nt:12.4f} {rt:8.4f}")

    # 2. PCA / EOF
    mu, VT, S = fit_pca(Ytr)
    for k in dims:
        fn = pca_reconstructor(mu, VT, k)
        nv, rv = prepare.evaluate_recon(fn, "val")
        nt, rt = prepare.evaluate_recon(fn, "test")
        print(f"{'pca':16s} {k:4d} {nv:11.4f} {rv:8.4f} {nt:12.4f} {rt:8.4f}")

    # Variance explained by the leading modes (on train), for context.
    var = (S ** 2)
    var = var / var.sum()
    cum = np.cumsum(var)
    print()
    print("train variance explained (cumulative):")
    for k in dims:
        if k <= len(cum):
            print(f"  k={k:4d}: {100*cum[k-1]:6.2f}%")
    print()
    print("Note: a linear AE optimizes to this same PCA subspace — treat PCA as the")
    print("linear-AE reference. The nonlinear AE (train.py) must beat PCA at equal k.")


if __name__ == "__main__":
    main()
