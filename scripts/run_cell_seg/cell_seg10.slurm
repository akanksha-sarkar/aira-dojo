#!/bin/bash
#SBATCH --partition=jjs533,gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=cellseg_k10
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Environment setup
source ~/.bashrc
conda activate base   # change if you use a different env

# Move to project directory
cd ~/sciduc/aira-dojo

# Your exact command
python -m dojo.main_run \
    +_exp=run_example \
    task=sciduc/cell_seg \
    task.subset=k10 \
    logger.use_wandb=False