# Probabilistic Neural Networks for Survival Analysis

> Forked from [thecml/baysurv](https://github.com/thecml/baysurv) — the official implementation of *"Efficient Training of Probabilistic Neural Networks for Survival Analysis"* (IEEE JBHI 2025, Vol. 29, No. 9).

This fork extends the original codebase with GPU acceleration, modernized dependencies, HPC cluster support, comprehensive evaluation output, experiment tracking, and full compatibility with the paper's Tables II, III, and IV.

## What's Changed

**Dependency & Compatibility Updates**
- Upgraded to TensorFlow 2.19 + Keras 3.x + TensorFlow Probability 0.25
- Replaced `auton-survival` dependency with native [pycox](https://github.com/havakv/pycox) wrappers (DSM, DCM)
- Replaced `tf-models-official` SNGP layers with custom Keras 3-compatible `SpectralNormalization` and `RandomFeatureGaussianProcess` implementations
- Fixed scikit-learn, SciPy, pandas 2.x, and Keras 3.x API breaking changes
- Removed unused DCPH model code from the main training pipeline

**GPU Acceleration**
- All neural network models now run on GPU by default (PyTorch and TensorFlow)
- BayCox/BayMTLR: automatic CUDA device selection
- BNN models (MLP, SNGP, MCD, VI): TensorFlow GPU with memory growth
- Pycox models (DSM, DCM): PyTorch networks moved to GPU

**HPC Cluster Support (Digital Alliance of Canada)**
- SLURM job script (`run_baysurv_job.sh`) with GPU allocation, module loading, and virtualenv setup
- MIG (Multi-Instance GPU) slices for faster queue times: 1g.10gb, 2g.20gb, 3g.40gb, or full H100
- Optimized for Compute Canada wheelhouse (`requirements_cc.txt`)
- Data staging to fast local scratch (`$SLURM_TMPDIR`)
- Suppressed noisy TF/rpy2/LMOD warnings for clean SLURM logs
- Unbuffered Python output for real-time progress monitoring

**Full Paper Metrics Output**
- All metrics from Tables II and III: CI, IBS, MAE_H, MAE_PO, ICI, D-Cal, C-Cal
- Extended metrics: MSE_Hinge, RMSE_Hinge, X-Cal, AUC@25/50/75, 1-Cal@25/50/75
- Model parameter counts (NParams) for Table IV
- Credible Interval (CrI) plots for VI/MCD/BayCox/BayMTLR (Figure 2)
- C-Cal shows `-` for deterministic models (cox, coxnet, coxboost, rsf, dsm, dcm); only probabilistic models report it

**Training Progress**
- Single-line ASCII progress bars (`#` and `-`) for BayCox, BayMTLR, and BNN models
- BayCox/BayMTLR: Train Total, KL, nll; Val Total, nll
- BNN models: Train/Val loss and variance
- All stdout captured to `results/sota_training_log.txt` and `results/bnn_training_log.txt`

**CLI & Experiment Tracking**
- Argparse support: `--datasets`, `--models`, `--epochs`, `--cri-samples`, `--no-early-stop`
- Optional [Weights & Biases](https://wandb.ai) integration (`--wandb`) with per-epoch loss curves
- Detailed logging: dataset characteristics, split sizes, training time, metrics, model save paths

## All 14 Models

The paper evaluates **14 models** split across two training scripts. Without any argparse arguments, running both scripts trains all 14 models on all 4 datasets.

### SOTA Baselines — `train_sota_models.py` (8 models)

| Model | Paper Reference | Framework | Epochs | Config |
|-------|----------------|-----------|--------|--------|
| CoxPH | [13] Cox 1972 | scikit-survival | N/A | `configs/cox/` |
| CoxNet | [22] Simon et al. | scikit-survival | N/A | `configs/coxnet/` |
| CoxBoost | [31] Hothorn et al. | scikit-survival | N/A | `configs/coxboost/` |
| RSF | [24] Ishwaran et al. | scikit-survival | N/A | `configs/rsf/` |
| DSM | [4] Nagpal et al. | pycox (DeepHit) | per config | `configs/dsm/` |
| DCM | [5] Nagpal et al. | pycox (LogHazard) | per config | `configs/dcm/` |
| BayCox | [7] Qi et al. | PyTorch | **5000** (patience 50) | `configs/baycox/` |
| BayMTLR | [7] Qi et al. | PyTorch | **1000** (patience 50) | `configs/baymtlr/` |

### BNN Models — `train_bnn_models.py` (6 models)

| Model | Paper Section | Framework | Epochs | Config |
|-------|--------------|-----------|--------|--------|
| MLP Baseline | III-D(i) | TensorFlow | **100** (patience 5) | `configs/mlp/` |
| + VI | III-C | TF Probability | **100** (patience 5) | `configs/mlp/` |
| + MCD (p=0.1) | III-C | TensorFlow | **100** (patience 5) | `configs/mlp/` |
| + MCD (p=0.2) | III-C | TensorFlow | **100** (patience 5) | `configs/mlp/` |
| + MCD (p=0.5) | III-C | TensorFlow | **100** (patience 5) | `configs/mlp/` |
| + SNGP | III-C | TensorFlow | **100** (patience 5) | `configs/mlp/` |

**Why different epoch counts?** The 6 BNN models (MLP, VI, MCD, SNGP) share the *same* hyperparameters for fair comparison (as stated in the paper). BayCox and BayMTLR are *literature benchmarks* with their own independently tuned configs. All models use early stopping, so they converge at different epochs regardless of the maximum.

## Datasets

| Dataset | Samples | Covariates | Censoring | Description |
|---------|---------|------------|-----------|-------------|
| METABRIC | 1,902 | 9 | 42% | Breast cancer genomics [18] |
| SEER | 4,024 | 28 | 85% | Cancer registry — breast cancer [19] |
| SUPPORT | 8,873 | 14 | 32% | Hospitalized adults survival [20] |
| MIMIC-IV | 38,520 | 91 | 67% | ICU patient records [21] |

## Paper Metrics (Tables II & III)

### Table II — Prediction Performance

| Metric | Description | Direction |
|--------|-------------|-----------|
| CI_td | Time-dependent Concordance Index | ↑ higher is better |
| MAE_H | Mean Absolute Error (hinge loss) | ↓ lower is better |
| MAE_PO | Mean Absolute Error (pseudo-obs) | ↓ lower is better |
| IBS | Integrated Brier Score | ↓ lower is better |

### Table III — Calibration Performance

| Metric | Description | Direction |
|--------|-------------|-----------|
| ICI | Integrated Calibration Index | ↓ lower is better |
| D-Cal | Distribution calibration (p-value, >0.05 = calibrated) | ↑ higher is better |
| C-Cal | Coverage calibration (only for VI, MCD, BayCox, BayMTLR) | ↑ higher is better |

D-Cal values marked with `*` indicate the model is NOT D-calibrated (p ≤ 0.05). C-Cal shows `-` for deterministic models (cox, coxnet, coxboost, rsf, dsm, dcm).

### Extended Metrics

| Metric | Description |
|--------|-------------|
| MSE_Hinge, RMSE_Hinge | Squared / root squared error (hinge) |
| X-Cal | X-calibration score |
| AUC@25, AUC@50, AUC@75 | Time-dependent AUC at percentiles |
| 1-Cal@25, 1-Cal@50, 1-Cal@75 | One-calibration at percentiles |

AUC and 1-Cal may show `N/A` when the test set has insufficient class balance at that time point.

## Quick Start

### Local Setup

```bash
conda create -n baysurv python=3.11
conda activate baysurv
pip install -r requirements.txt
mkdir -p models results
```

### Training All 14 Models

```bash
# Train all SOTA models (8) on all datasets
python train_sota_models.py

# Train all BNN models (6) on all datasets
python train_bnn_models.py
```

### Training Specific Models/Datasets

```bash
# Only Cox and BayCox on SUPPORT
python train_sota_models.py --datasets SUPPORT --models cox baycox

# Only MLP and VI on SUPPORT with 50 epochs
python train_bnn_models.py --datasets SUPPORT --models mlp vi --epochs 50

# All models on METABRIC only
python train_sota_models.py --datasets METABRIC
python train_bnn_models.py --datasets METABRIC
```

### BNN-Specific Flags

```bash
# CrI plot: 1000 MC samples (paper default)
python train_bnn_models.py --cri-samples 1000

# Disable early stopping, run all epochs
python train_bnn_models.py --no-early-stop --epochs 200
```

### With Experiment Tracking

```bash
export WANDB_API_KEY="your-key"
python train_sota_models.py --wandb
python train_bnn_models.py --wandb --wandb-project my-experiment
```

### HPC Cluster (SLURM)

```bash
# Edit run_baysurv_job.sh: GPU block (MIG 1g/2g/3g or full H100), PROJECT_DIR
sbatch run_baysurv_job.sh

# Monitor progress (tail interprets \r for in-place updates)
tail -f slurm-*.out
```

## Output Files

After training, results are saved to:

| File | Contents |
|------|----------|
| `results/sota_results.csv` | Table II & III + extended metrics for SOTA models |
| `results/baysurv_test_results.csv` | Table II & III + extended metrics for BNN models |
| `results/baysurv_training_results.csv` | Per-epoch train/valid loss & variance (BNN) |
| `results/sota_training_log.txt` | Full stdout from SOTA training |
| `results/bnn_training_log.txt` | Full stdout from BNN training |
| `results/*_survival_curves.pdf` | Individual survival curves per model |
| `results/*_brier_curve.pdf` | Time-dependent Brier score BS(t) per model |
| `results/*_pred_vs_actual.pdf` | Predicted vs actual survival time scatter |
| `results/*_time_histogram.pdf` | Predicted survival time distribution |
| `results/*_calibration.pdf` | Calibration curves (all models per dataset) |
| `results/*_training_curves.pdf` | Train/valid loss curves (BNN only) |
| `results/*_cri_sample*.pdf` | Credible interval plots (Figure 2, BNN only) |
| `models/*.joblib` | Saved scikit-survival models |
| `models/*.pt` | Saved PyTorch models (BayCox, BayMTLR) |
| `models/*/` | Saved TF/Keras model checkpoints (BNN) |

### CSV Column Reference

The results CSV files contain:

```
CI, IBS, MAEHinge, MAEPseudo, DCalib, KM, INBLL, CCalib, ICI,
MSE_Hinge, RMSE_Hinge, X_Cal, AUC_25, AUC_50, AUC_75,
One_Cal_25, One_Cal_50, One_Cal_75,
NParams, TrainTime, TestTime, ModelName, DatasetName
```

### Reading Results

```python
import pandas as pd

# Load SOTA results
sota = pd.read_csv('results/sota_results.csv')
print(sota.pivot_table(values='CI', index='ModelName', columns='DatasetName'))

# Load BNN results
bnn = pd.read_csv('results/baysurv_test_results.csv')
print(bnn.pivot_table(values='CI', index='ModelName', columns='DatasetName'))
```

## Project Structure

```
├── train_sota_models.py       # Train 8 SOTA baseline models
├── train_bnn_models.py        # Train 6 BNN models (paper's main contribution)
├── run_baysurv_job.sh         # SLURM job script for HPC clusters
├── paths.py                   # Project path configuration
├── configs/                   # Per-model, per-dataset YAML hyperparameters
│   ├── cox/ coxnet/ coxboost/ rsf/    # Classical/tree models
│   ├── dsm/ dcm/                       # Deep learning baselines
│   ├── baycox/ baymtlr/                # BNN literature benchmarks
│   └── mlp/                            # Shared config for MLP/VI/MCD/SNGP
├── tools/
│   ├── sota_builder.py        # SOTA model constructors (pycox wrappers)
│   ├── baysurv_builder.py     # BNN models + SpectralNorm + RFGP layers
│   ├── baysurv_trainer.py     # BNN training loop (TF/Keras)
│   ├── bnn_isd_trainer.py     # BayCox/BayMTLR training loop (PyTorch)
│   ├── results_generator.py   # Generic evaluation, plots, TeeLogger
│   ├── evaluator.py           # Survival evaluation metrics
│   ├── data_loader.py         # Dataset loaders
│   └── Evaluations/           # Concordance, Brier, AUC, calibration modules
├── utility/
│   ├── loss.py                # Cox PH loss functions
│   ├── bnn_isd_models.py      # BayesCox, BayesMTLR architectures
│   ├── survival.py            # Survival curve computation
│   ├── training.py            # Data loading, scaling, splitting
│   ├── plot.py                # Calibration and training curve plotting
│   └── config.py              # YAML config loader
├── tuning/                    # Hyperparameter tuning scripts (wandb sweeps)
│   ├── tune_sota_models.py    # Tune all SOTA models
│   ├── tune_mlp_model.py      # Tune MLP (shared config for all BNN models)
│   └── tune_mcd_model.py      # Tune MCD specifically
├── data/                      # Dataset files (CSV/feather)
├── models/                    # Saved model weights (created at runtime)
├── results/                   # Saved metrics and plots (created at runtime)
├── requirements.txt           # Local dependencies
└── requirements_cc.txt        # Compute Canada cluster dependencies
```

## Hyperparameter Tuning

Per the paper (Appendix B): *"We use Bayesian optimization [25] to tune hyperparameters over ten iterations on the validation set, adopting the hyperparameters leading to the highest concordance index (CItd)."* Hyperparameters are tuned **per dataset**.

### Two modes

| Mode | Description |
|------|--------------|
| **Pre-tuned (default)** | Use configs in `configs/mlp/*.yaml` (paper Table V). No tuning step. |
| **Bayesian optimization** | Run tuning first, save best config, then train. Requires `WANDB_API_KEY`. |

### Automated workflow (Bayesian optimization)

```bash
# Single command: tune then train (saves best config automatically)
python train_bnn_models.py --datasets SUPPORT --models vi --tune

# Or via job script:
TUNE_MODE=1 sbatch run_baysurv_job.sh
```

### Manual tuning (optional)

```bash
# Tune and save config for one dataset
python tuning/tune_mlp_model.py --dataset SUPPORT --iterations 10 --save-config

# Then train (uses the saved config)
python train_bnn_models.py --datasets SUPPORT --models vi
```

The MLP config is shared across all 6 BNN models (MLP, VI, MCD×3, SNGP) for fair comparison.

## Citation

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
