#!/bin/bash
#BSUB -J run_1
#BSUB -q gpu_a100
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 24:00
#BSUB -o /groups/branson/home/line2/aira-dojo/shared/logs/%J.out
#BSUB -cwd /groups/branson/home/line2/aira-dojo

# --- Environment setup ---
source ~/.bashrc
conda activate aira-dojo || { echo "❌ Failed to activate conda env"; exit 1; }

echo "✅ Running on $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

python -m dojo.main_run +_exp=run_example task=sciduc/bucktales logger.use_wandb=False
