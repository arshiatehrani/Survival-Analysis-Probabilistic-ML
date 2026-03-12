# Probabilistic Neural Networks for Survival Analysis

> Forked from [thecml/baysurv](https://github.com/thecml/baysurv) — the official implementation of *"Efficient Training of Probabilistic Neural Networks for Survival Analysis"* (IEEE JBHI 2024).

This fork extends the original codebase with GPU acceleration, modernized dependencies, HPC cluster support, experiment tracking, and a refactored model pipeline.

## What's Changed

**Dependency & Compatibility Updates**
- Upgraded to TensorFlow 2.19 + Keras 3.x + TensorFlow Probability 0.25
- Replaced `auton-survival` dependency with native [pycox](https://github.com/havakv/pycox) wrappers (DSM → DeepHitSingle, DCPH → CoxPH, DCM → LogisticHazard)
- Fixed scikit-learn, SciPy, and Keras 3.x API breaking changes (`sparse_output`, `simpson`, `reset_state`, `save_weights`)

**GPU Acceleration**
- All neural network models now run on GPU by default (PyTorch and TensorFlow)
- BayCox/BayMTLR: automatic CUDA device selection
- BNN models (MLP, SNGP, MCD, VI): GPU with memory growth enabled
- Pycox models (DSM, DCM): networks moved to GPU via `net.to(device)`

**HPC Cluster Support (Digital Alliance of Canada)**
- SLURM job script (`run_baysurv_job.sh`) with GPU allocation, module loading, and virtualenv setup
- Optimized for Compute Canada wheelhouse (`requirements_cc.txt`)
- Data staging to fast local scratch (`$SLURM_TMPDIR`)
- Suppressed noisy TF/rpy2/LMOD warnings for clean logs

**CLI & Experiment Tracking**
- Argparse support for selective training: `--datasets`, `--models`, `--epochs`
- Optional [Weights & Biases](https://wandb.ai) integration (`--wandb`) with per-epoch loss curves for BNN models
- Progress bars (tqdm) and detailed logging: dataset stats, training time, inference time, evaluation metrics

## Project Structure

```
├── train_sota_models.py       # Train SOTA baseline models
├── train_bnn_models.py        # Train BNN models (main contribution)
├── run_baysurv_job.sh         # SLURM job script for HPC clusters
├── paths.py                   # Project path configuration
├── configs/                   # Per-model, per-dataset hyperparameter YAML configs
│   ├── cox/ coxnet/ coxboost/ rsf/
│   ├── dsm/ dcph/ dcm/
│   ├── baycox/ baymtlr/
│   └── mlp/
├── tools/
│   ├── sota_builder.py        # SOTA model constructors (pycox wrappers)
│   ├── baysurv_builder.py     # BNN model constructors
│   ├── baysurv_trainer.py     # BNN training loop with wandb support
│   ├── evaluator.py           # Survival evaluation metrics
│   └── Evaluations/           # Concordance, Brier score, calibration, etc.
├── utility/
│   ├── loss.py                # Cox PH loss functions
│   ├── bnn_isd_models.py      # BayesCox, BayesMTLR architectures
│   ├── survival.py            # Survival curve utilities
│   ├── training.py            # Data loading, scaling, splitting
│   └── config.py              # YAML config loader
├── data/                      # Dataset files (CSV/feather)
├── tuning/                    # Hyperparameter tuning scripts (wandb sweeps)
├── requirements.txt           # Original dependencies
└── requirements_cc.txt        # Compute Canada cluster dependencies
```

## Models

### SOTA Baselines (`train_sota_models.py`)

| Model | Type | Framework |
|-------|------|-----------|
| Cox PH | Classical | scikit-survival |
| CoxNet | Regularized Cox | scikit-survival |
| CoxBoost | Gradient-boosted Cox | scikit-survival |
| RSF | Random Survival Forest | scikit-survival |
| DSM | Deep Survival Machines | pycox (DeepHitSingle) |
| DCM | Deep Cox Mixtures | pycox (LogisticHazard) |
| BayCox | Bayesian Cox | PyTorch |
| BayMTLR | Bayesian MTLR | PyTorch |

### BNN Models (`train_bnn_models.py`)

| Model | Type | Framework |
|-------|------|-----------|
| MLP | Deterministic baseline | TensorFlow |
| SNGP | Spectral-normalized GP | TensorFlow |
| MCD (1/2/3) | MC Dropout (10%/20%/50%) | TensorFlow |
| VI | Variational Inference | TensorFlow Probability |

## Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| SUPPORT | ~9,100 | Seriously ill hospitalized patients |
| SEER | ~36,000 | Cancer registry (breast cancer) |
| METABRIC | ~1,900 | Breast cancer genomics |
| MIMIC | ~42,000 | ICU patient records |

## Quick Start

### Local Setup

```bash
conda create -n baysurv python=3.11
conda activate baysurv
pip install -r requirements.txt
mkdir -p models results
```

### Training

```bash
# Train all models on all datasets (default)
python train_sota_models.py
python train_bnn_models.py

# Train specific models/datasets
python train_sota_models.py --datasets SUPPORT METABRIC --models cox dsm baycox
python train_bnn_models.py --datasets SUPPORT --models mlp vi --epochs 50

# With experiment tracking
python train_sota_models.py --wandb
python train_bnn_models.py --wandb --wandb-project my-experiment
```

### HPC Cluster (SLURM)

```bash
# Edit run_baysurv_job.sh to select models/datasets, then:
sbatch run_baysurv_job.sh

# Monitor progress
tail -f slurm-*.out
```

### Results

After training, results are saved to:
- `results/sota_results.csv` — SOTA model metrics
- `results/baysurv_test_results.csv` — BNN model metrics
- `results/baysurv_training_results.csv` — BNN training curves
- `models/` — Saved model weights

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| CI | Concordance Index (time-dependent) |
| IBS | Integrated Brier Score |
| INBLL | Integrated Negative Binomial Log-Likelihood |
| MAE | Mean Absolute Error (Hinge / Pseudo-obs) |
| D-Calib | D-Calibration p-value |
| KM-MSE | Kaplan-Meier calibration |
| C-Calib | Coverage calibration (Bayesian models) |
| ICI | Integrated Calibration Index |

## Citation

Original paper:
```bibtex
@article{lillelund_efficient_2025,
    author={Lillelund, Christian M. and Magris, Martin and Pedersen, Christian F.},
    journal={IEEE Journal of Biomedical and Health Informatics}, 
    title={{Efficient Training of Probabilistic Neural Networks for Survival Analysis}},
    volume={29},
    number={9},
    pages={6157--6166},
    year={2025},
}
```

## License

See [LICENSE](LICENSE).
