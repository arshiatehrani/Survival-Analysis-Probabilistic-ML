#!/bin/bash
#SBATCH --job-name=diagnose_auc
#SBATCH --time=0-00:15:00
#SBATCH --account=def-bakhshai
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-user=arshia.tehrani1380@gmail.com
#SBATCH --mail-type=ALL
#
# CPU-only job for AUC/1-Cal diagnostic (no GPU needed).
# Runs diagnose_auc_onecal.py on SUPPORT, SEER, METABRIC, MIMIC.

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:rpy2.rinterface"

echo "Job started on $(date)"

############################
# 1. Load required modules #
############################

module load python/3.11.5
module load arrow
module load r/4.5.0

unset -f module ml which 2>/dev/null

############################
# 2. Copy data to scratch  #
############################

PROJECT_DIR=/home/arshiat/projects/elec888/Survival-Analysis-Probabilistic-ML

cp -r "$PROJECT_DIR/data" "$SLURM_TMPDIR/data"
export BAYSURV_DATA_DIR="$SLURM_TMPDIR/data"
echo "Data copied to $SLURM_TMPDIR/data"

############################
# 3. Create venv and install#
############################

VENV_DIR="$SLURM_TMPDIR/diag_env"
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --no-index --upgrade pip

cd "$PROJECT_DIR"
pip install --no-index -r requirements_cc.txt
pip install --no-index tf-keras

############################
# 4. Run diagnostic on all #
############################

DATASETS="SUPPORT SEER METABRIC MIMIC"

for dataset in $DATASETS; do
  echo ""
  echo "========== Diagnosing $dataset =========="
  python misc/diagnose_auc_onecal.py --dataset "$dataset"
  echo ""
done

echo "Job finished on $(date)"
