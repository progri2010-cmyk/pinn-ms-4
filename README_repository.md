# pinn-ms-4

Code, data-generation scripts and evaluation pipeline for the paper

**Phase-Wrapping-Aware Multiple Shooting for Learning Divergence-Free Magnetic Fields from Radiation-Belt Electron Trajectories**
Genti Progri, Department of Physics, University of Elbasan "Aleksandër Xhuvani"

Every number, table and figure in the paper can be regenerated from this repository. Everything runs on a CPU. One training run takes about two minutes on one core. The full study comprises 475 trained models and takes roughly 20 core-hours. Evaluating the drift-orbit horizon of Table 4 takes another 6–10 minutes per model.

## Environment

Python ≥ 3.10 is required. The results in the paper were produced with Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0, matplotlib 3.11 and torch 2.15 (CPU build).

```
pip install -r requirements.txt
```

Run every command from the repository root. Seeds are fixed in the scripts:
- model seed: the `SEED` argument;
- data noise: seed 0;
- validation electrons: seed 7;
- extra noise realisations: seeds 1001–1009.

Regenerating the data therefore reproduces the data sets bit for bit on the same platform.

## Layout

| Path | Contents |
|---|---|
| `physics/fields.py` | Normalised dipole and storm field. Selected with `RB_TRUTH=storm\|prc`; `RB_EPS` sets ε (default 0.02). |
| `physics/ring_current.py` | Ring-current truth field (7 loops, regularised Biot–Savart, divergence-free). |
| `physics/boris.py` | Boris integrator: numpy (ground truth) and differentiable torch, with the `exact_phase` option of eq. (4). |
| `physics/invariants.py` | μ, J per bounce, guiding centre, dipole L of the guiding centre, drift phase. |
| `models/` | Neural residual (curl of an MLP vector potential), harmonic residual, hybrid, second-order neural ODE. |
| `training/mic.py`, `training/run.py` | Multiple-shooting trainer (trainable nodes, continuity, curriculum, two-stage hybrid). |
| `data/` | `generate.py` (training/validation/10× test sets), `generate_long.py` (3400-unit drift-orbit test set), `noise_realizations.py` (9 extra noise draws). |
| `evaluation/` | Rollouts, metrics, statistics, figures. |
| `run_*.py` | Experiment batches: each script trains one group of configurations, uses two parallel workers and skips finished runs. |
| `paper/` | Manuscript source, tables and the docx build script. |

## Reproducing the paper

### 1. Data

```
python -m data.generate                       # storm field, data/dataset.npz
python -m data.generate 0.5 ; python -m data.generate 2      # weak / strong storm (Section 5.2)
RB_TRUTH=prc python -m data.generate          # ring-current field
RB_EPS=0.01 python -m data.generate           # ε = 0.01 test (Section 5.2)
python -m data.noise_realizations             # data/dataset_nr1..9.npz (Section 5.1)
RB_TRUTH=storm python -m data.generate_long ; RB_TRUTH=prc python -m data.generate_long   # Table 4
```

### 2. Training

A single run:

```
python -m training.run NAME SCHEDULE SEED key=value ... [data=path]
```

Example, the two-stage hybrid with exact phase:

```
python -m training.run xp_shyb_l2_lam10_K48 uniform 0 K=48 epochs=300 model=hybrid l_max=2 lam_reg=10.0 exact_phase=True stage1=100
```

The batches that produce each result:

| Result in the paper | Scripts |
|---|---|
| Table 1 (storm field, 10 seeds) | `run_grid.py`, `run_grid4.py`, `run_grid5.py`, `run_grid7.py`, `run_rev_seeds.py`, `run_final.py` (neural ODE) |
| Noise realisations (Section 5.1) | `run_noise.py` |
| Figure 2, phase-wrapping criterion | `run_threshold.py`, `run_grid2.py`, `run_final.py` (exact-phase runs), `run_eps.py` (ε = 0.01) |
| Curriculum and K = 96 (Section 5.2) | `run_grid2.py` |
| Capacity and regularisation (Section 5.3) | `run_grid3.py` |
| Table 2 and Table 3, ring-current field | `run_grid6.py`, `run_grid7.py`, `run_grid8.py`, `run_grid9.py`, `run_grid10.py` (22 seeds and best-of-three), `run_grid11.py` |
| Stage-1 control (Section 5.5) | `run_grid12.py`, `run_grid13.py`, `run_final.py` |

### 3. Evaluation, statistics and figures

```
RB_TRUTH=storm python -m evaluation.eval_truth data/dataset.npz PREFIX ...     # 10× horizon metrics, Tables 1–3
RB_TRUTH=prc   python -m evaluation.eval_truth data/dataset_prc.npz PREFIX ...
python -m evaluation.phase_criterion                                          # true-field mismatch Δφ0, eq. (6)
python -m evaluation.phase_surrogate                                          # data-only estimate Δφ_est, eq. (7)
python -m evaluation.phase_est_direction                                      # robustness of Δφ_est to field direction
RB_TRUTH=storm python -m evaluation.long_horizon xp_harm_l2_K48 xp_shyb_l2_lam10_K48 xp_mlp_K48 --maxseed 9   # Table 4
RB_TRUTH=prc   python -m evaluation.long_horizon prc_shyb_l2_lam01_K64 prc_mlp_K64 --maxseed 9
python -m evaluation.drift_diagnostics ; python -m evaluation.head_to_head
python -m evaluation.figures ; python -m evaluation.fig_phase2                 # all figures
```

Metrics are cached per model as JSON in `outputs/metrics_*/`, so an interrupted evaluation resumes where it stopped.

### Statistical tests

The unit of analysis is the mean over the test electrons of one trained model.

- Paired comparisons use the exact two-sided Wilcoxon signed-rank test and pair-resampling bootstrap intervals. A comparison is paired when the two models share the seed and architecture, and hence the initial weights, or share the noise realisation.
- Independent samples use the two-sided Mann–Whitney U test.

## Trained models

The 475 trained models (`outputs/*.pt`, `outputs/*/*.pt`) are distributed as a release asset of this repository. With them, every evaluation above runs without retraining.

## Licence and citation

Please cite the paper if you use this code. Licence: MIT.
