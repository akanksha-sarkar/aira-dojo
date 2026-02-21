#!/bin/bash
#SBATCH --job-name=run_1
#SBATCH --partition=jjs533,gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|a6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/share/j_sun/as2637/logs/%j.out
#SBATCH --chdir=/home/as2637/sciduc/aira-dojo

set -e

echo "=== SLURM INFO ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=================="

eval "$(command conda shell.bash hook 2>/dev/null || true)"
conda activate aira-dojo || {
    echo "❌ Failed to activate conda env"
    exit 1
}

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load apptainer-1.4.0

echo "=== APPTAINER VERSION ==="
apptainer --version
echo "========================="

# (Optional but recommended) hard check
if ! apptainer --version | grep -q "1.4.0"; then
    echo "❌ Incorrect Apptainer version loaded"
    exit 1
fi



# -------------------------
# Experiment config
# -------------------------
DOMAIN="cub_ssl"
K_SEED="k5seed42"
AGENT="aide"
EXP_NUM="2"
DATA_DIR="/share/j_sun/as2637/sciduc_data/cub_ssl"
CACHE_DIR="/share/j_sun/as2637"
SUBSET="k5"

LOG_DIR="/share/j_sun/as2637/logs"
EXPERIMENT_DIR="${LOG_DIR}/${DOMAIN}/${K_SEED}/${AGENT}/${EXP_NUM}"

if [ -d "$EXPERIMENT_DIR" ]; then
    echo "❌ Experiment directory $EXPERIMENT_DIR already exists. Exiting."
    exit 1
fi

mkdir -p "$EXPERIMENT_DIR"

JOB_OUT="${EXPERIMENT_DIR}/${SLURM_JOB_ID}.out"

# Move Slurm's default output log if present (but only if real file)
if [ -f "${LOG_DIR}/${SLURM_JOB_ID}.out" ]; then
    mv "${LOG_DIR}/${SLURM_JOB_ID}.out" "$JOB_OUT"
fi

export EXPERIMENT_DIR

export WANDB_ENTITY=as2637-cornell-university
export WANDB_PROJECT=sciduc
nvidia-smi

echo "✅ Running job ${SLURM_JOB_ID}"
echo "Logging to: $JOB_OUT"

# -------------------------
# Run experiment
# -------------------------
python -m dojo.main_run \
  +_exp=run_example \
  task=sciduc/${DOMAIN} \
  task.subset=${SUBSET} \
  +task.data_dir=${DATA_DIR} \
  task.cache_dir=${CACHE_DIR} \
  logger.use_wandb=True
