#!/bin/bash
#SBATCH --job-name=unc_train
#SBATCH --time=0-00:30:00
#SBATCH --account=def-bakhshai
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=arshia.tehrani1380@gmail.com
#SBATCH --mail-type=ALL
#
# --- GPU config: uncomment ONE block below ---
#
# [ACTIVE] MIG 1g.10gb (1/8 H100, 10GB VRAM) -- fastest queue, fine for single-model runs
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#
# [ALT] MIG 2g.20gb (2/8 H100, 20GB VRAM) -- safe for all stochastic models
##SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
##SBATCH --cpus-per-task=6
##SBATCH --mem=64G
#
# [ALT] MIG 3g.40gb (3/8 H100, 40GB VRAM) -- faster compute, good for sweeps
##SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
##SBATCH --cpus-per-task=12
##SBATCH --mem=64G
#
# [ALT] Full H100 (80GB VRAM) -- overkill for current models
##SBATCH --gpus=h100:1
##SBATCH --cpus-per-task=12
##SBATCH --mem=128G
######################################################################
# Uncertainty-Aware Training Experiment (SPCL for Survival Analysis)
#
# Trains stochastic BNN models with self-paced curriculum learning.
# Results: results/{EXPERIMENT_NAME}/{dataset}/{unc_config}/seed_{seed}/
#
# Usage:
#   sbatch run_exp_uncertainty.sh                                      # defaults
#   DATASET="METABRIC" sbatch run_exp_uncertainty.sh                   # one dataset
#   DATASET="METABRIC SUPPORT SEER MIMIC" sbatch run_exp_uncertainty.sh  # all datasets
#   SEEDS="0 1 2 3 4" sbatch run_exp_uncertainty.sh                    # more seeds
#   MODEL=vi LOSS=cox sbatch run_exp_uncertainty.sh                    # different model/loss
#
# Available datasets: SUPPORT, SEER, METABRIC, MIMIC
# Available models:   mcd1, mcd2, mcd3, vi
# Available losses:   cox, ibs, crps, joint_ibs, joint_crps, joint_crps_kl
#
# Flags reference (defaults from the original uncertainty code):
#   --unc-mode        none|soft|curriculum|both   (default: soft)
#   --warmup-epochs   int                         (default: 2)
#   --mc-passes       int                         (default: 5)
#   --temperature     float                       (default: 2.0)
#   --curriculum-start float                      (default: 0.55)
#   --curriculum-end   float                      (default: 1.0)
#   --loss-type       cox|ibs|crps|joint_ibs|joint_crps|joint_crps_kl
#   --lambda-val      float                       (default: 0.3)
#   --mu-val          float                       (default: 0.0)
######################################################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"

# ---------- Configurable parameters ----------
DATASETS="${DATASET:-METABRIC}"               # Space-separated: SUPPORT SEER METABRIC MIMIC
MODEL="${MODEL:-mcd1}"                        # mcd1 | mcd2 | mcd3 | vi
LOSS="${LOSS:-crps}"                          # cox | ibs | crps | joint_ibs | joint_crps | joint_crps_kl
EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-0}"                           # Single seed to start; expand: "0 1 2 3 4 5 6 7 8 9"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(date +%Y%m%d)_uncertainty_training}"

echo "Uncertainty-aware training experiment started on $(date)"
echo "  Datasets:     $DATASETS"
echo "  Model:        $MODEL"
echo "  Base loss:    $LOSS"
echo "  Epochs:       $EPOCHS"
echo "  Seeds:        $SEEDS"
echo "  Experiment:   $EXPERIMENT_NAME"

nvidia-smi

# ---------- Environment setup ----------
PROJECT_DIR=/home/arshiat/projects/elec888/Survival-Analysis-Probabilistic-ML

cp -r "$PROJECT_DIR/data" "$SLURM_TMPDIR/data"
export BAYSURV_DATA_DIR="$SLURM_TMPDIR/data"
echo "Data copied to $SLURM_TMPDIR/data"

module load python/3.11.5
module load cuda/12.6
module load cudnn
module load arrow
module load opencv/4.13.0
module load r/4.5.0
export FONTCONFIG_PATH=/etc/fonts

unset -f module ml which 2>/dev/null

cd "$PROJECT_DIR"
VENV_DIR="$SLURM_TMPDIR/baysurv_env"
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
pip install --no-index tf-keras

pip check

mkdir -p models results

python -c "
import sys; print(f'Python {sys.version}')
import tensorflow as tf; print(f'TF: {tf.__version__} | GPUs: {tf.config.list_physical_devices(\"GPU\")}')
print(f'Datasets: $DATASETS | Model: $MODEL | Loss: $LOSS | Epochs: $EPOCHS')
"

# ---------- Helper function ----------
run_unc_exp() {
    local DATASET="$1"
    local UNC_MODE="$2"
    local LOSS_TYPE="$3"
    local LAMBDA="$4"
    local TEMP="$5"
    local WARMUP="$6"
    local MC_PASSES="$7"
    local CURR_START="$8"
    local CURR_END="$9"
    local SEED="${10}"

    echo ""
    echo "------------------------------------------------------------"
    echo ">>> [$DATASET] mode=$UNC_MODE loss=$LOSS_TYPE T=$TEMP warmup=$WARMUP seed=$SEED at $(date)"
    echo "------------------------------------------------------------"

    python experiments/exp_uncertainty_training.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --loss-type "$LOSS_TYPE" \
        --lambda-val "$LAMBDA" \
        --unc-mode "$UNC_MODE" \
        --warmup-epochs "$WARMUP" \
        --mc-passes "$MC_PASSES" \
        --temperature "$TEMP" \
        --curriculum-start "$CURR_START" \
        --curriculum-end "$CURR_END" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --experiment-name "$EXPERIMENT_NAME" \
        --n-samples-test 100

    echo ">>> Finished [$DATASET] mode=$UNC_MODE loss=$LOSS_TYPE seed=$SEED (exit=$?)"
}

# ---------- Run experiments ----------
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

        # 1. Baseline: no uncertainty (standard training)
        run_unc_exp "$DS" "none" "$LOSS" 0.3 2.0 2 5 0.55 1.0 "$SEED"

        # 2. Soft weighting (default hyperparams from OG uncertainty code)
        run_unc_exp "$DS" "soft" "$LOSS" 0.3 2.0 2 5 0.55 1.0 "$SEED"

        # 3. Hard curriculum
        run_unc_exp "$DS" "curriculum" "$LOSS" 0.3 2.0 2 5 0.55 1.0 "$SEED"

        # 4. Both (soft + curriculum)
        run_unc_exp "$DS" "both" "$LOSS" 0.3 2.0 2 5 0.55 1.0 "$SEED"

        # ---- SWEEPS (uncomment when ready) ----

        ## Temperature sweep (soft)
        # for T in 1.0 2.0 5.0; do
        #     run_unc_exp "$DS" "soft" "$LOSS" 0.3 "$T" 2 5 0.55 1.0 "$SEED"
        # done

        ## Curriculum start sweep
        # for CS in 0.3 0.55 0.75; do
        #     run_unc_exp "$DS" "curriculum" "$LOSS" 0.3 2.0 2 5 "$CS" 1.0 "$SEED"
        # done

        ## Warmup epoch sensitivity
        # for W in 2 5 10; do
        #     run_unc_exp "$DS" "soft" "$LOSS" 0.3 2.0 "$W" 5 0.55 1.0 "$SEED"
        # done

        ## MC passes sensitivity
        # for MC in 3 5 10; do
        #     run_unc_exp "$DS" "soft" "$LOSS" 0.3 2.0 2 "$MC" 0.55 1.0 "$SEED"
        # done

    done
done

# ---------- Compare results ----------
echo ""
echo "########################################"
echo "# Generating comparison report"
echo "########################################"

for DS in $DATASETS; do
    python experiments/compare_uncertainty_runs.py \
        --experiment-name "$EXPERIMENT_NAME" \
        --dataset "$DS"
done

python experiments/compare_uncertainty_runs.py --experiment-name "$EXPERIMENT_NAME"

echo ""
echo "All experiments complete at $(date)"
echo "Results: results/$EXPERIMENT_NAME/"
