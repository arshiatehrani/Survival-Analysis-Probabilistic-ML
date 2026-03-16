#!/bin/bash
#SBATCH --job-name=calibloss_exp
#SBATCH --time=0-02:00:00
#SBATCH --account=def-bakhshai
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=arshia.tehrani1380@gmail.com
#SBATCH --mail-type=ALL
#
# MIG 1g.10gb — sufficient for single mcd1 model
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"

echo "Calibration-aware loss experiment started on $(date)"
nvidia-smi

############################
# 1. Load modules
############################
module load python/3.11.5
module load cuda/12.6
module load cudnn
module load arrow
module load opencv/4.13.0
module load r/4.5.0
unset -f module ml which 2>/dev/null

############################
# 2. Copy data to scratch
############################
PROJECT_DIR=/home/arshiat/projects/elec888/Survival-Analysis-Probabilistic-ML
cp -r "$PROJECT_DIR/data" "$SLURM_TMPDIR/data"
export BAYSURV_DATA_DIR="$SLURM_TMPDIR/data"
echo "Data copied to $SLURM_TMPDIR/data"

############################
# 3. Create venv
############################
VENV_DIR="$SLURM_TMPDIR/baysurv_env"
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --no-index --upgrade pip
cd "$PROJECT_DIR"
pip install --no-index -r requirements_cc.txt
pip install --no-index tf-keras
pip check

############################
# 4. Sanity checks
############################
python -V
python -c "import tensorflow as tf; print('TF:', tf.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))"
mkdir -p results

############################
# 5. Run experiments
############################
#
# Choose dataset(s) below. Default: METABRIC (fast) + SUPPORT (medium)
# To run only one dataset: DATASET="METABRIC" sbatch run_exp_calibration.sh
#
DATASETS="${DATASET:-METABRIC SUPPORT}"
MODEL="${MODEL:-mcd1}"
EPOCHS="${EPOCHS:-100}"

echo "Datasets: $DATASETS | Model: $MODEL | Epochs: $EPOCHS"

run_exp() {
    local dataset=$1
    local loss=$2
    local lam=${3:-0.3}
    local mu=${4:-0.0}
    echo ""
    echo "============================================================"
    echo ">>> [$dataset] $loss (λ=$lam, μ=$mu) at $(date)"
    echo "============================================================"
    python experiments/exp_calibration_loss.py \
        --dataset "$dataset" \
        --model "$MODEL" \
        --loss-type "$loss" \
        --lambda-val "$lam" \
        --mu-val "$mu" \
        --epochs "$EPOCHS"
}

for DS in $DATASETS; do
    echo ""
    echo "########################################"
    echo "# Starting experiments on $DS"
    echo "########################################"

    # Experiment 0: Baseline
    run_exp "$DS" cox

    # Experiment 1: IBS only
    run_exp "$DS" ibs

    # Experiment 2: CRPS only
    run_exp "$DS" crps

    # Experiment 3: Joint Cox+IBS sweep
    for LAM in 0.1 0.3 0.5 0.7; do
        run_exp "$DS" joint_ibs "$LAM"
    done

    # Experiment 4: Joint Cox+CRPS sweep
    for LAM in 0.1 0.3 0.5 0.7; do
        run_exp "$DS" joint_crps "$LAM"
    done

    # Experiment 5: Joint Cox+CRPS+KL
    run_exp "$DS" joint_crps_kl 0.3 0.05
    run_exp "$DS" joint_crps_kl 0.3 0.1
done

############################
# 6. Generate comparison
############################
echo ""
echo "============================================================"
echo ">>> Generating comparison report at $(date)"
echo "============================================================"
python experiments/compare_runs.py

echo "All experiments finished on $(date)"
