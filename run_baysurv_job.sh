#!/bin/bash
#SBATCH --job-name=elec888_train
#SBATCH --time=0-00:30:00             # walltime (D-HH:MM:SS)
#SBATCH --account=def-bakhshai
#SBATCH --mem=32000                   # 32 GB RAM
#SBATCH --gpus-per-node=h100:1        # 1 H100 GPU
#SBATCH --cpus-per-task=8             # CPU cores
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=arshia.tehrani1380@gmail.com
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started on $(date)"
nvidia-smi

############################
# 1. Load required modules #
############################

# Adjust these module names to match your cluster configuration
module load python/3.11.5
module load cuda
module load cudnn
module load opencv/4.13.0
module load r/4.5.0

############################
# 2. Create venv and install from requirements_cc.txt (no network) #
############################

VENV_DIR="$SLURM_TMPDIR/baysurv_env"d
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --no-index --upgrade pip

# Project dir must be current so requirements_cc.txt is found
cd /home/arshiat/projects/elec888/Survival-Analysis-Probabilistic-ML
pip install --no-index -r requirements_cc.txt

python -V
python -c "import tensorflow as tf; print('TF version:', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"

############################
# 3. Project folder is already current #
############################

echo "Current directory: $(pwd)"
echo "Data files in ./data:"
ls data

############################
# 4. Run training scripts  #
############################

echo "Starting train_sota_models.py at $(date)"
python train_sota_models.py

echo "Starting train_bnn_models.py at $(date)"
python train_bnn_models.py

echo "Job finished on $(date)"

