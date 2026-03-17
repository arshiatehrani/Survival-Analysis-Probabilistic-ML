#!/bin/bash
#SBATCH --job-name=calib_loss
#SBATCH --time=0-04:00:00
#SBATCH --account=def-bakhshai
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=arshia.tehrani1380@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
######################################################################
# Calibration-Aware Loss Experiment — Multi-Seed Runner
#
# Runs all loss configurations across specified datasets with 10 seeds.
# Results are written to: results/{EXPERIMENT_NAME}/{dataset}/{loss_config}/seed_{seed}/
#
# Usage:
#   sbatch run_exp_calibration.sh                       # defaults: METABRIC, 10 seeds
#   DATASET="METABRIC SUPPORT" sbatch run_exp_calibration.sh
#   SEEDS="0 1 2 3 4" EPOCHS=50 sbatch run_exp_calibration.sh
######################################################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"

# ---------- Configurable parameters ----------
DATASETS="${DATASET:-METABRIC}"              # Space-separated list
MODEL="${MODEL:-mcd1}"
EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"       # 10 seeds by default
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(date +%Y%m%d)_calibration_loss}"

echo "Calibration-aware loss experiment started on $(date)"
echo "  Datasets:   $DATASETS"
echo "  Model:      $MODEL"
echo "  Epochs:     $EPOCHS"
echo "  Seeds:      $SEEDS"
echo "  Experiment: $EXPERIMENT_NAME"

nvidia-smi

# ---------- Environment setup ----------
PROJECT_DIR=/home/arshiat/projects/elec888/Survival-Analysis-Probabilistic-ML

# Copy data to local scratch
cp -r "$PROJECT_DIR/data" "$SLURM_TMPDIR/data"
export BAYSURV_DATA_DIR="$SLURM_TMPDIR/data"
echo "Data copied to $SLURM_TMPDIR/data"

# Load modules
module load python/3.11.5
module load cuda/12.6
module load cudnn
module load arrow
module load opencv/4.13.0
module load r/4.5.0
export FONTCONFIG_PATH=/etc/fonts

unset -f module ml which 2>/dev/null

# Virtual environment
cd "$PROJECT_DIR"
VENV_DIR="$SLURM_TMPDIR/baysurv_env"
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
pip install --no-index tf-keras

pip check

mkdir -p models results

# Quick sanity check
python -c "
import sys; print(f'Python {sys.version}')
import tensorflow as tf; print(f'TF: {tf.__version__} | GPUs: {tf.config.list_physical_devices(\"GPU\")}')
print(f'Datasets: $DATASETS | Model: $MODEL | Epochs: $EPOCHS')
"

# ---------- Helper function ----------
run_exp() {
    local DATASET="$1"
    local LOSS="$2"
    local LAMBDA="$3"
    local MU="$4"
    local SEED="$5"

    echo ""
    echo "------------------------------------------------------------"
    echo ">>> [$DATASET] $LOSS (λ=$LAMBDA, μ=$MU) seed=$SEED at $(date)"
    echo "------------------------------------------------------------"

    python experiments/exp_calibration_loss.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --loss-type "$LOSS" \
        --lambda-val "$LAMBDA" \
        --mu-val "$MU" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --experiment-name "$EXPERIMENT_NAME" \
        --n-samples-test 100

    echo ">>> Finished [$DATASET] $LOSS λ=$LAMBDA μ=$MU seed=$SEED (exit=$?)"
}

# ---------- Run all experiments ----------
for DS in $DATASETS; do
    echo ""
    echo "########################################"
    echo "# Starting experiments on $DS"
    echo "########################################"

    for SEED in $SEEDS; do
        echo ""
        echo "========================================"
        echo "  SEED $SEED"
        echo "========================================"

        # 1. Baseline: pure Cox PH
        run_exp "$DS" "cox" 0.3 0.0 "$SEED"

        # 2. Pure calibration losses
        run_exp "$DS" "ibs"  0.3 0.0 "$SEED"
        run_exp "$DS" "crps" 0.3 0.0 "$SEED"

        # 3. Joint losses: lambda sweep
        for LAM in 0.1 0.3 0.5 0.7; do
            run_exp "$DS" "joint_ibs"  "$LAM" 0.0 "$SEED"
            run_exp "$DS" "joint_crps" "$LAM" 0.0 "$SEED"
        done

        # 4. Joint + marginal KL: best lambda with mu sweep
        run_exp "$DS" "joint_crps_kl" 0.3 0.05 "$SEED"
        run_exp "$DS" "joint_crps_kl" 0.3 0.1  "$SEED"
    done
done

# ---------- Compare results ----------
echo ""
echo "########################################"
echo "# Generating comparison report"
echo "########################################"

for DS in $DATASETS; do
    python experiments/compare_runs.py \
        --experiment-name "$EXPERIMENT_NAME" \
        --dataset "$DS"
done

# Also run without dataset filter for cross-dataset comparison
python experiments/compare_runs.py --experiment-name "$EXPERIMENT_NAME"

echo ""
echo "All experiments complete at $(date)"
echo "Results: $SRC_DIR/results/$EXPERIMENT_NAME/"
