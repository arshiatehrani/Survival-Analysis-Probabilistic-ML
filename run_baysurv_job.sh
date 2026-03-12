#!/bin/bash
#SBATCH --job-name=elec888_train
#SBATCH --time=0-00:30:00
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

echo "Starting train_sota_models.py at $(date)"
python train_sota_models.py

echo "Starting train_bnn_models.py at $(date)"
python train_bnn_models.py

echo "Job finished on $(date)"
