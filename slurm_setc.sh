#!/bin/bash
#SBATCH --job-name=run_1
#SBATCH --partition=jjs533,gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|a6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
# Keep Slurm's own capture in a stable place (small “bootstrap” log)
#SBATCH --output=/share/j_sun/as2637/logs/slurm_bootstrap/%j.out
#SBATCH --chdir=/home/as2637/sciduc/aira-dojo
#SBATCH --requeue

set -euo pipefail

# -------------------------
# Experiment config
# -------------------------
DOMAIN="resisc45"      # or "cub", etc.
K="k5"
SEED="seed0"
AGENT="aide"
SETTING="setC"

ROOT="/share/j_sun/as2637"
LOG_ROOT="${ROOT}/logs"          # for canonical logs (like screenshot)
RUN_ROOT="${ROOT}/${DOMAIN}"     # optional: if you want under dataset folder; change as you prefer

# If you want the screenshot style under logs:
#   /share/j_sun/as2637/logs/resisc45/k5/seed0/aide/<exp_num>/<jobid>.out
BASE_EXP_NUM=30
while true; do
  EXP_NUM="${BASE_EXP_NUM}"
  EXPERIMENT_DIR="${LOG_ROOT}/${DOMAIN}/${K}/${SEED}/${AGENT}/${EXP_NUM}"
  if [ ! -d "${EXPERIMENT_DIR}" ]; then
    break
  fi
  BASE_EXP_NUM=$((BASE_EXP_NUM + 1))
done

mkdir -p "${EXPERIMENT_DIR}"

JOB_OUT="${EXPERIMENT_DIR}/${SLURM_JOB_ID}.out"

# **Key line**: after this, EVERYTHING prints into JOB_OUT
# (and also to the slurm stdout stream via tee, so `squeue`/`tail` still feels normal)
exec > >(tee -a "${JOB_OUT}") 2>&1

echo "Using experiment number: ${EXP_NUM}"
echo "Logging to: ${JOB_OUT}"
echo

echo "=== SLURM INFO ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "=================="
echo

# -------------------------
# Environment
# -------------------------
eval "$(command conda shell.bash hook 2>/dev/null || true)"
conda activate aira-dojo || { echo "❌ Failed to activate conda env"; exit 1; }

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
module load apptainer-1.4.0

echo "=== APPTAINER VERSION ==="
apptainer --version
echo "========================="

if ! apptainer --version | grep -q "1.4.0"; then
  echo "❌ Incorrect Apptainer version loaded"
  exit 1
fi

export EXPERIMENT_DIR
export WANDB_ENTITY=as2637-cornell-university
export WANDB_PROJECT=sciduc

nvidia-smi
echo
echo "✅ Running job ${SLURM_JOB_ID}"
echo

# -------------------------
# Run experiment
# -------------------------
python -m dojo.main_run \
  +_exp=run_example \
  task=agentSSL/_default \
  task.name="${DOMAIN}" \
  task.setting="${SETTING}" \
  task.subset="${K}" \
  task.seed="${SEED}"