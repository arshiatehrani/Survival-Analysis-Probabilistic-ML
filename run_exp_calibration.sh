#!/bin/bash
#SBATCH --account=def-bakhshai
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --job-name=calib_loss
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
SRC_DIR="$HOME/projects/elec888/Survival-Analysis-Probabilistic-ML"
LOCAL_DIR="$SLURM_TMPDIR"

# Copy data to local scratch
if [ -d "$SRC_DIR/data" ]; then
    cp -r "$SRC_DIR/data" "$LOCAL_DIR/data"
    echo "Data copied to $LOCAL_DIR/data"
fi

# Load modules
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 cudnn/9.5.1.17 arrow/17.0.0 python/3.11 scipy-stack/2024b
export FONTCONFIG_PATH=/etc/fonts

# Virtual environment
cd "$SRC_DIR"
virtualenv --no-download "$LOCAL_DIR/baysurv_env"
source "$LOCAL_DIR/baysurv_env/bin/activate"
pip install --upgrade pip --no-index
pip install -r requirements_cc.txt --no-index
pip install tf-keras --no-index

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
