#!/bin/bash
#SBATCH --job-name=elec888_train
#SBATCH --time=0-01:00:00
#SBATCH --account=def-bakhshai
#SBATCH --mem=16G
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=arshia.tehrani1380@gmail.com
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"

echo "Job started on $(date)"
nvidia-smi

############################
# 1. Load required modules #
############################

module load python/3.11.5
module load cuda/12.6
module load cudnn
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
# SNGP model needs tf-models-official; --no-deps prevents it from downgrading TF
pip install --no-index --no-deps tf-models-official

pip check

############################
# 4. Sanity checks #
############################

python -V
python -c "import tensorflow as tf; print('TF:', tf.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))"

echo "Current directory: $(pwd)"
echo "Data files in $BAYSURV_DATA_DIR:"
ls "$BAYSURV_DATA_DIR"

mkdir -p models results

############################
# 5. Run training scripts #
############################
# Without arguments, all datasets and models run (default behavior).
# Use --datasets and --models to select subsets.
#
# SOTA models: cox, coxnet, coxboost, rsf, dsm, dcm, baycox, baymtlr
# BNN models:  mlp, sngp, mcd1, mcd2, mcd3, vi
# Datasets:    SUPPORT, SEER, METABRIC, MIMIC
#
# Examples:
#   python train_sota_models.py --datasets SUPPORT --models cox baycox
#   python train_sota_models.py --models dsm dcm
#   python train_sota_models.py --datasets SUPPORT METABRIC
#   python train_bnn_models.py --datasets SUPPORT --models mlp vi
#   python train_bnn_models.py --epochs 50

echo "Starting train_sota_models.py at $(date)"
python train_sota_models.py

echo "Starting train_bnn_models.py at $(date)"
python train_bnn_models.py

echo "Job finished on $(date)"
