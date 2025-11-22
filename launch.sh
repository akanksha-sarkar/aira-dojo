#!/bin/bash
#BSUB -J run_1
#BSUB -q gpu_h100
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 25:00
#BSUB -o /groups/branson/home/line2/aira-dojo/shared/logs/%J.out
#BSUB -cwd /groups/branson/home/line2/aira-dojo
DOMAIN="bucktales" # bucktales or wildfin
K_SEED=k5seed42 # k5seed42, k10seed42, k20seed42, k100
AGENT="aira" # aira or aide
EXP_NUM="1" # Experiment number
source ~/.bashrc
conda activate aira-dojo || { echo "❌ Failed to activate conda env"; exit 1; }
TERM_OUT="/groups/branson/home/line2/aira-dojo/shared/logs/${LSB_JOBID}.out"
LOG_DIR="/groups/branson/home/line2/aira-dojo/shared/logs"
EXPERIMENT_DIR="${LOG_DIR}/${DOMAIN}/${K_SEED}/${AGENT}/${EXP_NUM}"

# If directory already exists, abort
if [ -d "$EXPERIMENT_DIR" ]; then
    echo "❌ Experiment directory $EXPERIMENT_DIR already exists. Exiting to avoid overwriting."
    exit 1
fi

# Otherwise, create it safely
mkdir -p "$EXPERIMENT_DIR"

JOB_OUT="${LOG_DIR}/${DOMAIN}/${K_SEED}/${AGENT}/${EXP_NUM}/${LSB_JOBID}.out"

# Rename the generic log once the job starts
if [ -f "$TERM_OUT" ]; then
    mv "${TERM_OUT}" "$JOB_OUT"
fi

export EXPERIMENT_DIR

echo "✅ Running job: $LSB_JOBNAME (ID: $LSB_JOBID)"
echo "Logging to: $JOB_OUT"

python -m dojo.main_run +_exp=run_example task=sciduc/${DOMAIN} logger.use_wandb=False