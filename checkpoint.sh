#!/bin/bash
#SBATCH --job-name=kitti
#SBATCH --partition=jjs533,gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|6000ada"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/share/j_sun/as2637/logs/%j.out
#SBATCH --chdir=/home/as2637/sciduc/aira-dojo
#SBATCH --requeue

set -e

echo "=== SLURM INFO ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=================="

source /home/as2637/miniconda/etc/profile.d/conda.sh
conda activate aira-dojo

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
# Original defaults (can be changed if you want to re-use the script)
DOMAIN="cub"
K="k5"
SEED="seed0"
AGENT="aide"
SETTING="setC"

# Directory of the previous run whose checkpoint you want to resume from.
# For your current case:
#   /share/j_sun/as2637/logs/cub/k5/seed0/aide/30
RESUME_EXPERIMENT_DIR="/share/j_sun/as2637/logs/cub/k5/seed0/aide/30"

# Use the resume directory as the experiment dir (so dojo finds checkpoint/)
EXPERIMENT_DIR="${RESUME_EXPERIMENT_DIR}"

echo "Resuming from experiment dir: $EXPERIMENT_DIR"
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
# solver.time_limit_secs = max wall time for the agent/search (this run stops when hit).
# solver.execution_timeout = max time per single program execution (unchanged from run_example: 2h).


python -m dojo.main_run \
    +_exp=run_example \
    task=agentSSL/_default \
    task.name=${DOMAIN} \
    task.setting=${SETTING} \
    task.subset=${K} \
    task.seed=${SEED}
