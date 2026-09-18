# program.md — OLR autoencoder autoresearch

Autonomous research loop: you (the agent) improve a nonlinear autoencoder that compresses
weekly global OLR fields into a small latent vector and reconstructs them. You edit ONE file
(`train.py`), run it for a fixed time budget, keep the change if the metric improves, discard
it otherwise, and repeat — indefinitely.

## The task

- Data: ERA5 weekly OLR, 2.5° global (72×144 grid, D=10368), one field per week.
- Goal: lowest **val_nrmse2** = area-weighted NRMSE² = 1 − R² on the validation split.
  Lower is better. 1.0 = trivial mean-field reconstruction, 0.0 = perfect.
- All methods run in `prepare.py`'s frozen "Y space" (centered, standardized, sqrt(area)
  weighted), so plain MSE = area-weighted MSE and the metric is directly comparable across
  every experiment and against the PCA/mean-field baselines.

## Setup (do once, then start the loop)

1. **Run tag**: propose one from today's date (e.g. `sep17`). Branch `autoresearch/<tag>`
   must not already exist. Create it: `git checkout -b autoresearch/<tag>`.
2. **Read in-scope files**: `README.md`, `prepare.py` (frozen), `train.py` (you edit),
   `baselines.py` (the bar).
3. **Verify data**: `python prepare.py` should print grid/splits and a mean-field baseline
   of nrmse2≈1.0. If it errors about a missing file, tell the human to run
   `OLR_data/make_weekly.py --coarsen 10`.
4. **Establish the bar**: `python baselines.py > baselines.log 2>&1`, read the PCA table.
   PCA at a given latent dim is what the nonlinear AE must beat at the SAME latent dim.
5. **Init results**: create `results.tsv` with just the header row (leave it untracked).
6. Confirm, then start experimenting.

## Rules

**You CAN**: edit `train.py` freely — architecture, depth/width, activations, normalization,
dropout, optimizer, LR schedule, batch size, regularization, weight init, etc. (NOT the latent
dim — it is frozen in prepare.py, see below.)

**You CANNOT**:
- Modify `prepare.py` — it holds the fixed split, preprocessing, TIME_BUDGET, and the metrics
  `evaluate_ae` / `evaluate_recon` (the ground truth). Do not touch it.
- Change the metric or the split (both live in `prepare.py`; `baselines.py` only reports them).
- Change what data is used or how val/test are defined.
- Add heavyweight dependencies. numpy + torch + xarray only.
- **Inflate the bottleneck.** `LATENT_DIM` is **FROZEN in prepare.py (=16)** — you may NOT
  change it anywhere. The research question is nonlinear AE vs linear PCA at this ONE fixed,
  small latent; a bigger latent trivially lowers error and is not a result. The model MUST be a
  single encoder → exactly `prepare.LATENT_DIM` numbers → decoder, scored via
  `prepare.evaluate_ae(encode, decode)`. No skip / residual / parallel path from input to
  output. `evaluate_ae` asserts the encoder output width == `prepare.LATENT_DIM`, so inflating
  the latent (bigger code OR a skip) crashes the run. A PCA pre-reduction of the *input* is fine
  as long as the code still passes through exactly LATENT_DIM and reconstruction is scored in
  full space.
- **Touch the harness via git.** `prepare.py` and `program.md` are fixed. NEVER `git reset`,
  `checkout`, or `revert` to a commit that changes them. On a discard, undo ONLY your own
  experiment commit with `git reset --hard HEAD~1` (see the loop) — never jump to an older hash.

**Fixed time budget**: `train.py` trains for `prepare.TIME_BUDGET` seconds (wall clock), so
you never optimize for speed — a bigger/better model in the same budget is a fair win. If a
run exceeds ~3× the budget total, kill it and treat as failure.

**Simplicity criterion**: all else equal, simpler is better. A tiny gain that adds ugly
complexity is not worth it; an equal-or-better result from *deleting* code is a great win.

**First run**: run `train.py` unmodified to record the baseline nonlinear AE.

**Never split by anything but time.** Never leak val/test into training or preprocessing.

## Output format

`train.py` prints a grep-able summary:

```
---
val_nrmse2:       0.184000
val_r2:           0.816000
pca_ref_nrmse2:   0.201000
latent_dim:       16
num_params:       ...
num_epochs:       ...
training_seconds: 120.0
total_seconds:    128.4
device:           cuda
```

Extract with: `grep "^val_nrmse2:\|^val_r2:\|^pca_ref_nrmse2:" run.log`.
If the grep is empty the run crashed — `tail -n 50 run.log` for the traceback.

## Logging (results.tsv, tab-separated, keep untracked)

Columns: `commit  val_nrmse2  val_r2  status  description`

- commit: short git hash (7 chars)
- val_nrmse2 / val_r2: metric (use 0.0 / 0.0 for crashes)
- status: `keep`, `discard`, or `crash`
- description: short text of what the experiment tried

```
commit	val_nrmse2	val_r2	status	description
a1b2c3d	0.184000	0.816000	keep	baseline nonlinear AE, latent 16
b2c3d4e	0.171000	0.829000	keep	deeper encoder [512,256,64]
c3d4e5f	0.190000	0.810000	discard	tanh activation, worse
```

## The loop

LOOP FOREVER:
1. Check git state (branch/commit).
2. Edit `train.py` with one experimental idea.
3. `git commit`.
4. `python train.py > run.log 2>&1` (redirect — do NOT flood context with training output).
5. `grep "^val_nrmse2:\|^pca_ref_nrmse2:" run.log`.
6. Empty grep → crashed → `tail -n 50 run.log`, fix if trivial (typo/import), else log `crash`.
7. Record in `results.tsv`.
8. Improved (lower val_nrmse2 than the best **kept** experiment so far — the baseline AE if
   none yet) → keep the commit, advance. This holds even while still **above** PCA: banking
   every real improvement is what lets the hill-climb ratchet down toward (and past) the bar.
9. Equal or worse than the best kept so far → `git reset --hard HEAD~1` (undo ONLY this
   experiment's commit). Never `git reset` to an older commit hash — that can silently revert
   the frozen harness (prepare.py / program.md).

**PCA is the GOAL to beat, not the keep/discard gate.** Never discard an experiment merely
because it hasn't beaten PCA yet — compare only against your own best kept result.

**NEVER STOP** once the loop starts. Do not ask the human whether to continue — they may be
away. If out of ideas, think harder: re-read the files, combine near-misses, try more radical
architectures. Run until manually interrupted.

## Approach

The scientific question: does a nonlinear model capture more of the OLR field than linear
EOFs at the same latent dim? First goal is to cleanly beat PCA at a fixed latent dim; then
push further. How you get there is up to you — explore the space of architectures, objectives,
regularizers, optimizers and training schedules yourself. Note the main practical constraint:
the input is high-dimensional (D=10368) with ~3,140 training weeks, so a large model still
overfits — the unmodified baseline AE already loses to PCA. Capacity control and regularization
are the dominant lever.

If you introduce any auxiliary objective, remember the FIXED metric is reconstruction
(`evaluate_recon` on val) — an auxiliary term only counts if it improves that.

**Web search** is allowed for inspiration when you want fresh directions or are stuck —
architectures, regularizers, autoencoder / dimensionality-reduction methods for spatial
fields. But it is optional, not required each loop; implement only within the allowed deps
(numpy / torch / xarray), and never change the metric, the split, or the data.

Keep each experiment a single, describable change so results.tsv stays a clean research log.
