# AutoresearchAutoencoder

LLM-driven ("autoresearch", after Karpathy) training of an **autoencoder for OLR
dimensionality reduction**. Adapts the three-file autoresearch loop to a climate field:
an agent iteratively edits the model, trains for a fixed time budget, keeps improvements,
discards regressions, and repeats.

## Problem

ERA5 Outgoing Longwave Radiation (OLR), weekly, 2.5° global (72×144 grid). Find a low-
dimensional latent representation from which the full field can be reconstructed. Does a
nonlinear autoencoder capture more structure than linear EOF/PCA at the same latent dimension?

## Files

| file | role | edited by |
|------|------|-----------|
| `prepare.py` | frozen: load weekly OLR, temporal train/val/test split, centering + standardization + sqrt(area) weighting, the fixed metric `evaluate_recon` (area-weighted NRMSE² = 1−R²), `TIME_BUDGET` | nobody |
| `baselines.py` | mean field + PCA/EOF ladder — the bar the AE must beat | nobody |
| `train.py` | the nonlinear autoencoder — architecture, optimizer, training loop | **the agent** |
| `program.md` | agent instructions (open-ended — the agent explores architectures itself) | the human |
| `results.tsv` | experiment log (tracked, committed each iteration) | the agent |

## The metric

Everything runs in a transformed **Y space** (centered on the train mean field, standardized,
multiplied by sqrt of normalized cos-lat area weight). A plain unweighted MSE in Y space equals
the cos(lat) **area-weighted** MSE in physical W/m², and a plain SVD is an area-weighted EOF.
So PCA, linear AE and nonlinear AE all use ordinary MSE and are directly comparable, with area
weighting baked in. Metric: `val_nrmse2 = mean‖Y−Ŷ‖² / mean‖Y‖²` (mean-field ⇒ 1.0, perfect ⇒ 0).

## Data

Weekly OLR fields (2.5° global, cos-lat area-weighted) are prepared once by the data-prep
script and loaded automatically by `prepare.py`, which picks up more years as they are added.

## Quick start

```bash
pip install -r requirements.txt
python prepare.py            # sanity: shapes, splits, mean-field baseline (~1.0)
python baselines.py          # PCA/EOF bar over latent dims
python train.py > run.log 2>&1   # one autoencoder experiment (fixed time budget)
grep "^val_nrmse2:" run.log
```

GPU: `train.py` auto-uses CUDA if available, else CPU.

## Running the agent

Point a coding agent at `program.md` in this repo:

> Have a look at program.md and let's kick off a new experiment — do the setup first.

Then it runs the keep/discard loop autonomously on a dedicated `autoresearch/<tag>` branch.
