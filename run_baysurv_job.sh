#!/bin/bash
#SBATCH --job-name=elec888_train
#SBATCH --time=0-08:00:00
#SBATCH --account=def-bakhshai
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=arshia.tehrani1380@gmail.com
#SBATCH --mail-type=ALL
#
# --- GPU config: uncomment ONE block below ---
#
# [ALT] MIG 3g.40gb (3/8 H100, 40GB VRAM) -- faster queue, plenty for current models
##SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
##SBATCH --cpus-per-task=12
##SBATCH --mem=64G
#
# [ACTIVE] MIG 2g.20gb (2/8 H100, 20GB VRAM) -- safe for all 14 models on all datasets
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#
# [ALT] MIG 1g.10gb (1/8 H100, 10GB VRAM) -- fastest queue, may OOM on BNN models
##SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
##SBATCH --cpus-per-task=6
##SBATCH --mem=32G
#
# [ALT] Full H100 (80GB VRAM) -- for larger/novel models later
##SBATCH --gpus=h100:1
##SBATCH --cpus-per-task=12
##SBATCH --mem=128G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"

# Wandb: set your API key here to enable --wandb tracking from the cluster.
# Get your key from https://wandb.ai/authorize
# export WANDB_API_KEY="your-key-here"
#
# GPU fail-fast (see section 4): training expects a usable GPU. To force-run on CPU anyway:
#   ALLOW_CPU=1 sbatch run_baysurv_job.sh

echo "Job started on $(date)"
nvidia-smi || echo "WARNING: nvidia-smi failed before modules/venv (driver or no GPU node?)"

############################
# 1. Load required modules #
############################

module load python/3.11.5
module load cuda/12.6
module load cudnn
module load arrow
module load opencv/4.13.0
module load r/4.5.0

unset -f module ml which 2>/dev/null

############################
# 2. Copy data to fast local scratch #
############################

PROJECT_DIR=/home/arshiat/projects/elec888/Survival-Analysis-Probabilistic-ML

cp -r "$PROJECT_DIR/data" "$SLURM_TMPDIR/data"
export BAYSURV_DATA_DIR="$SLURM_TMPDIR/data"
echo "Data copied to $SLURM_TMPDIR/data"

############################
# 3. Create venv and install #
############################

VENV_DIR="$SLURM_TMPDIR/baysurv_env"
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --no-index --upgrade pip

cd "$PROJECT_DIR"
pip install --no-index -r requirements_cc.txt

# TFP 0.25 needs tf-keras
pip install --no-index tf-keras

pip check

############################
# 4. Sanity checks + fail-fast if no usable GPU #
############################
# train_bnn_models.py needs TensorFlow on GPU; train_sota_models.py needs PyTorch CUDA for
# BayCox, BayMTLR, DSM, DCM, etc. Exit before long runs if either stack sees no GPU.

python -V

if [ "${ALLOW_CPU:-0}" = "1" ]; then
  echo "ALLOW_CPU=1: skipping strict GPU checks (BNN/SOTA may run on CPU)."
  python -c "import tensorflow as tf; print('TF:', tf.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))"
  python -c "import torch; print('Torch cuda_available:', torch.cuda.is_available())"
else
  echo "Verifying GPU (set ALLOW_CPU=1 to skip)..."
  if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "ERROR: No GPU visible to nvidia-smi. Use a GPU partition and #SBATCH --gpus=..., or set ALLOW_CPU=1." >&2
    exit 1
  fi
  python - <<'PY' || { echo "ERROR: GPU verification failed (see messages above)." >&2; exit 1; }
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print(
        "ERROR: TensorFlow reports zero GPUs — train_bnn_models.py would use CPU.",
        file=sys.stderr,
    )
    sys.exit(1)
print("OK: TensorFlow sees", len(gpus), "GPU(s):", gpus)

try:
    import torch
except ImportError as exc:
    print("ERROR: PyTorch not importable:", exc, file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print(
        "ERROR: torch.cuda.is_available() is False — SOTA torch models (BayCox, BayMTLR, DSM, DCM, …) would use CPU.",
        file=sys.stderr,
    )
    sys.exit(1)
print("OK: PyTorch CUDA:", torch.cuda.get_device_name(0))
PY
fi

echo "Current directory: $(pwd)"
echo "Data files in $BAYSURV_DATA_DIR:"
ls "$BAYSURV_DATA_DIR"

mkdir -p models results

############################
# 5. Run training scripts #
############################
#
# There are 14 models total, split across two scripts:
#
#   train_sota_models.py (8 models):
#     cox, coxnet, coxboost, rsf, dsm, dcm, baycox, baymtlr
#
#   train_bnn_models.py (6 models):
#     mlp, sngp, mcd1, mcd2, mcd3, vi
#
# Datasets: SUPPORT, SEER, METABRIC, MIMIC
#
# Set DATASETS (space-separated) to train on a subset; default is all four.
# Examples:
#   export DATASETS="SEER MIMIC"
#   sbatch run_baysurv_job.sh
#   sbatch --export=ALL,DATASETS="SEER MIMIC" run_baysurv_job.sh
#
# DEBUG_MODE=1 and TUNE_MODE=1 still use SUPPORT only (fast / tuning paths).
#
# If DATASETS is unset, default is all four datasets above.
# Add --wandb to enable experiment tracking (requires WANDB_API_KEY above).
#
# Examples - run specific models/datasets:
#   python train_sota_models.py --datasets SUPPORT --models cox baycox
#   python train_sota_models.py --models dsm dcm
#   python train_bnn_models.py --datasets SUPPORT --models mlp vi
#   python train_bnn_models.py --epochs 50
#
# Examples - BNN-specific flags:
#   python train_bnn_models.py --cri-samples 1000      # MC samples for CrI plot (paper=1000, default=1000)
#   python train_bnn_models.py --cri-samples 100       # Faster but lower quality CrI plots
#   python train_bnn_models.py --cri-plot-samples 0,42,100   # CrI plots for specific samples (VI/MCD/BayCox/BayMTLR)
#   python train_bnn_models.py --cri-plot-all          # CrI plots for all test samples (many PDFs)
#   python train_bnn_models.py --cri-plot-random      # Use random sample instead of 42 (when single sample)
#   python train_bnn_models.py --no-early-stop         # Run all epochs (ignore config early_stop)
#   python train_bnn_models.py --no-early-stop --epochs 200
#
# Examples - with wandb:
#   python train_sota_models.py --wandb
#   python train_bnn_models.py --wandb --wandb-project my-project
#
# Outputs saved to:
#   results/sota_results.csv           - Table II & III metrics + extended metrics for SOTA models
#   results/baysurv_test_results.csv   - Table II & III metrics + extended metrics for BNN models
#   results/baysurv_training_results.csv - Per-epoch loss & variance curves
#   results/sota_training_log.txt      - Full stdout log from SOTA training
#   results/bnn_training_log.txt       - Full stdout log from BNN training
#   results/*_survival_curves.pdf      - Individual survival curves per model
#   results/*_brier_curve.pdf          - Time-dependent Brier score BS(t) per model
#   results/*_pred_vs_actual.pdf       - Predicted vs actual survival time scatter
#   results/*_time_histogram.pdf       - Predicted survival time distribution
#   results/*_calibration.pdf          - Calibration curves (all models per dataset)
#   results/*_training_curves.pdf      - Train/valid loss curves per dataset (BNN only)
#   results/*_cri_sample*.pdf          - Credible interval plots (Figure 2, VI/MCD/BayCox/BayMTLR)
#   models/                            - Saved model weights

############################
# TUNE_MODE: 0 = use pre-tuned configs (configs/mlp/*.yaml) [default]
#            1 = run Bayesian optimization first, then train
#
# For TUNE_MODE=1: set WANDB_API_KEY above (https://wandb.ai/authorize)
# To use: TUNE_MODE=1 sbatch run_baysurv_job.sh
############################
TUNE_MODE=${TUNE_MODE:-0}
echo "TUNE_MODE=$TUNE_MODE (0=pre-tuned, 1=Bayesian optimization)"

# DEBUG_MODE: 0=normal, 1=fast debug (MLP on SUPPORT). For debug: DEBUG_MODE=1 sbatch run_baysurv_job.sh
DEBUG_MODE=${DEBUG_MODE:-0}
echo "DEBUG_MODE=$DEBUG_MODE | TUNE_MODE=$TUNE_MODE"

# Paper reproduction datasets (both train_sota_models.py and train_bnn_models.py)
DATASETS="${DATASETS:-SUPPORT SEER METABRIC MIMIC}"
echo "DATASETS=$DATASETS"

# RESUME_DIR: set to an existing results/<run_dir> path to append new model results
# Example: RESUME_DIR=results/20260330_155147_MIMI_mlp_sngp_mcd1_mcd2 sbatch run_baysurv_job.sh
RESUME_FLAG=""
if [ -n "${RESUME_DIR:-}" ]; then
  echo "RESUME_DIR=$RESUME_DIR"
  RESUME_FLAG="--resume-dir $RESUME_DIR"
fi

# BNN_MODELS: space-separated list of BNN models to train (default: all)
# Example: BNN_MODELS="transformer_mcd saint_mcd" sbatch run_baysurv_job.sh
MODELS_FLAG=""
if [ -n "${BNN_MODELS:-}" ]; then
  echo "BNN_MODELS=$BNN_MODELS"
  MODELS_FLAG="--models $BNN_MODELS"
fi

# BNN_ONLY: set to 1 to skip train_sota_models.py and only run train_bnn_models.py
# Example: BNN_ONLY=1 sbatch run_baysurv_job.sh
BNN_ONLY=${BNN_ONLY:-0}

if [ "$DEBUG_MODE" = "1" ]; then
  echo ">>> DEBUG: Running MLP on SUPPORT at $(date)"
  python train_bnn_models.py --datasets SUPPORT --models mlp $RESUME_FLAG
elif [ "$TUNE_MODE" = "1" ]; then
  echo "Starting Bayesian optimization + training at $(date)"
  python train_bnn_models.py --datasets SUPPORT --models vi --tune --tune-iterations 10 $RESUME_FLAG
else
  if [ "$BNN_ONLY" = "1" ]; then
    echo "BNN_ONLY=1: skipping train_sota_models.py"
  else
    echo "Starting train_sota_models.py at $(date)"
    python train_sota_models.py --datasets $DATASETS
  fi

  echo "Starting train_bnn_models.py at $(date) (pre-tuned configs)"
  python train_bnn_models.py --datasets $DATASETS $MODELS_FLAG $RESUME_FLAG
fi

echo "Job finished on $(date)"