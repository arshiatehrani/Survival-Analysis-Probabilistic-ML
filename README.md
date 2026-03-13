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
- Optimized for Compute Canada wheelhouse (`requirements_cc.txt`)
- Data staging to fast local scratch (`$SLURM_TMPDIR`)
- Suppressed noisy TF/rpy2/LMOD warnings for clean SLURM logs
- Unbuffered Python output for real-time progress monitoring

**Full Paper Metrics Output**
- All metrics from Tables II and III are now printed during training:
  CI, IBS, MAE_H, MAE_PO, ICI, D-Cal, C-Cal
- Model parameter counts printed after each model (Table IV)
- Credible Interval (CrI) plots saved for VI/MCD models (Figure 2 from paper)
- Results saved to CSV files for post-analysis

**CLI & Experiment Tracking**
- Argparse support for selective training: `--datasets`, `--models`, `--epochs`
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

D-Cal values marked with `*` in the output indicate the model is NOT D-calibrated (p ≤ 0.05).

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

### With Experiment Tracking

```bash
export WANDB_API_KEY="your-key"
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

## Output Files

After training, results are saved to:

| File | Contents |
|------|----------|
| `results/sota_results.csv` | All Table II & III metrics for SOTA models |
| `results/baysurv_test_results.csv` | All Table II & III metrics for BNN models |
| `results/baysurv_training_results.csv` | Per-epoch training loss & variance curves |
| `results/*_cri_sample*.pdf` | Credible interval plots for VI/MCD (Figure 2) |
| `models/*.joblib` | Saved scikit-survival models |
| `models/*.pt` | Saved PyTorch models (BayCox, BayMTLR) |
| `models/*/weights.weights.h5` | Saved TF/Keras models (BNN) |

### CSV Column Reference

The results CSV files contain these columns:

```
CI, IBS, MAEHinge, MAEPseudo, DCalib, KM, INBLL, CCalib, ICI,
TrainTime, TestTime, ModelName, DatasetName
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
│   ├── baysurv_trainer.py     # BNN training loop with wandb support
│   ├── evaluator.py           # Survival evaluation metrics
│   ├── data_loader.py         # Dataset loaders
│   └── Evaluations/           # Concordance, Brier, calibration modules
├── utility/
│   ├── loss.py                # Cox PH loss functions
│   ├── bnn_isd_models.py      # BayesCox, BayesMTLR architectures
│   ├── survival.py            # Survival curve computation
│   ├── training.py            # Data loading, scaling, splitting
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

The configs in `configs/` contain pre-tuned hyperparameters from the paper (Appendix B, Table V). To re-tune:

```bash
# Tune SOTA models (uses wandb sweeps, 10 iterations)
python tuning/tune_sota_models.py --dataset SUPPORT --model cox

# Tune BNN/MLP hyperparameters (100 iterations)
python tuning/tune_mlp_model.py --dataset SUPPORT
```

Per the paper, the MLP config is shared across all 6 BNN models (MLP, VI, MCD×3, SNGP) for fair comparison.

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
