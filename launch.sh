#!/bin/bash
#BSUB -J run_1
#BSUB -q gpu_h100
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 25:00
#BSUB -o /groups/branson/home/line2/aira-dojo/shared/logs/%J.out
#BSUB -cwd /groups/branson/home/line2/aira-dojo

source ~/.bashrc
conda activate aira-dojo || { echo "❌ Failed to activate conda env"; exit 1; }

LOG_DIR="/groups/branson/home/line2/aira-dojo/shared/logs"
EXPERIMENT_DIR="${LOG_DIR}/${LSB_JOBNAME}"

# If directory already exists, abort
if [ -d "$EXPERIMENT_DIR" ]; then
    echo "❌ Experiment directory $EXPERIMENT_DIR already exists. Exiting to avoid overwriting."
    exit 1
fi

# Otherwise, create it safely
mkdir -p "$EXPERIMENT_DIR"

JOB_OUT="${LOG_DIR}/${LSB_JOBNAME}/${LSB_JOBID}.out"

# Rename the generic log once the job starts
if [ -f "${LOG_DIR}/${LSB_JOBID}.out" ]; then
    mv "${LOG_DIR}/${LSB_JOBID}.out" "$JOB_OUT"
fi

export EXPERIMENT_DIR

echo "✅ Running job: $LSB_JOBNAME (ID: $LSB_JOBID)"
echo "Logging to: $JOB_OUT"

python -m dojo.main_run +_exp=run_example task=sciduc/bucktales logger.use_wandb=False