#!/bin/bash
#SBATCH --job-name=mc_samples
#SBATCH --time=0-02:00:00
#SBATCH --account=def-bakhshai
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=arshia.tehrani1380@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
######################################################################
# MC Sample Efficiency Experiment — Multi-Seed Runner
#
# Loads trained checkpoints from a calibration-loss experiment and
# evaluates inference time and accuracy across varying MC samples.
#
# Usage examples:
#   SOURCE_EXPERIMENT=20260316_calibration_loss DATASET=METABRIC sbatch run_exp_mc_samples.sh
######################################################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"

# ---------- Configurable parameters ----------
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:?ERROR: Set SOURCE_EXPERIMENT to the calibration-loss experiment name}"
DATASETS="${DATASET:-METABRIC}"
MODEL="${MODEL:-mcd1}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
LOSS_CONFIGS="${LOSS_CONFIGS:-cox}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SOURCE_EXPERIMENT}_mc_samples}"

echo "MC Sample Efficiency experiment started on $(date)"
echo "  Source:     $SOURCE_EXPERIMENT"
echo "  Datasets:   $DATASETS"
echo "  Model:      $MODEL"
echo "  Configs:    $LOSS_CONFIGS"
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
print(f'Source experiment: $SOURCE_EXPERIMENT')
"

# ---------- Run Sample Efficiency Evaluation ----------
for DS in $DATASETS; do
    echo ""
    echo "########################################"
    echo "# MC Sample Efficiency on $DS"
    echo "########################################"

    for LC in $LOSS_CONFIGS; do
        for SEED in $SEEDS; do
            # Check if source checkpoint exists
            SRC_DIR="$PROJECT_DIR/results/$SOURCE_EXPERIMENT/$DS/$LC/seed_$SEED"
            if [ ! -f "$SRC_DIR/config.json" ]; then
                echo "  SKIP: No checkpoint for $DS/$LC/seed_$SEED"
                continue
            fi

            echo ""
            echo ">>> [$DS] $LC seed=$SEED at $(date)"

            python experiments/exp_mc_samples.py \
                --source-experiment "$SOURCE_EXPERIMENT" \
                --dataset "$DS" \
                --model "$MODEL" \
                --loss-config "$LC" \
                --seed "$SEED" \
                --experiment-name "$EXPERIMENT_NAME"

            echo ">>> Finished [$DS] $LC seed=$SEED (exit=$?)"
        done
    done
done

# ---------- Merge and plot efficiency curves ----------
echo ""
echo "########################################"
echo "# Generating efficiency report/plots"
echo "########################################"

for DS in $DATASETS; do
    python experiments/plot_mc_efficiency.py \
        --experiment-name "$EXPERIMENT_NAME" \
        --dataset "$DS"
done

# Also run without dataset filter for cross-dataset comparison
python experiments/plot_mc_efficiency.py --experiment-name "$EXPERIMENT_NAME"

echo ""
echo "All MC efficiency experiments complete at $(date)"
echo "Results: $PROJECT_DIR/results/$EXPERIMENT_NAME/"
