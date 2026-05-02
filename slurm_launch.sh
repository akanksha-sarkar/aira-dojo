#!/bin/bash
#SBATCH --job-name=dtd_claude
#SBATCH --partition=jjs533,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="6000ada"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --exclude=bhattacharjee-compute-02,snavely-compute-09,genai-large-01,lil-compute-05,unicorn-compute-01,lancer-compute-01,abdelfattah-compute-02,portal-compute-02,ma-compute-02,kuleshov-compute-02,unicorn-compute-04
#SBATCH --output=/share/j_sun/as2637/logs/%j.out
#SBATCH --error=/share/j_sun/as2637/logs/%j.err
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
DOMAIN="dtd"
K="k3"
SEED="seed0"
AGENT="aira"
# Fixed run id: every sbatch uses this same directory so checkpoint always resumes.
# Change this when you intentionally want a brand-new experiment.
EXP_NUM="110"
#DATA_DIR="/share/j_sun/agentSSL/resisc45"
CACHE_DIR="/share/j_sun/as2637"
SETTING="aSSL_backbone_unsupervised_debug"


LOG_DIR="/share/j_sun/as2637/logs"
EXPERIMENT_DIR="${LOG_DIR}/${DOMAIN}/${SETTING}/${K}/${SEED}/${AGENT}/${EXP_NUM}"

echo "Experiment dir (always same path → resume from checkpoint if present): ${EXPERIMENT_DIR}"
mkdir -p "$EXPERIMENT_DIR"

JOB_OUT="${EXPERIMENT_DIR}/${SLURM_JOB_ID}.out"
JOB_ERR="${EXPERIMENT_DIR}/${SLURM_JOB_ID}.err"

# Slurm writes stdout to the fixed path from #SBATCH --output at submit time.
# Once we know the full EXPERIMENT_DIR, move that file into the experiment folder.
SLURM_STDOUT_STAGING="/share/j_sun/as2637/logs/${SLURM_JOB_ID}.out"
if [ -f "$SLURM_STDOUT_STAGING" ] && [ "$SLURM_STDOUT_STAGING" != "$JOB_OUT" ]; then
    mv "$SLURM_STDOUT_STAGING" "$JOB_OUT"
fi

# Same idea for stderr (separate stream/file from stdout in Slurm).
SLURM_STDERR_STAGING="/share/j_sun/as2637/logs/${SLURM_JOB_ID}.err"
if [ -f "$SLURM_STDERR_STAGING" ] && [ "$SLURM_STDERR_STAGING" != "$JOB_ERR" ]; then
    mv "$SLURM_STDERR_STAGING" "$JOB_ERR"
fi

export EXPERIMENT_DIR

export WANDB_ENTITY=as2637-cornell-university
export WANDB_PROJECT=sciduc
nvidia-smi

echo "✅ Running job ${SLURM_JOB_ID}"
echo "Logging to: $JOB_OUT"

# -------------------------
# Environment for Dojo (before main_run)
# -------------------------
# Repo root is #SBATCH --chdir. `set -a` exports every KEY=value from the file. These override
# the same names from Python `load_dotenv()` (dotenv defaults to override=False).
# ENV_CLAUDE_FILE="${ENV_CLAUDE_FILE:-.env_claude}"
# if [ -f "$ENV_CLAUDE_FILE" ]; then
#     set -a
#     # shellcheck disable=SC1090
#     . "$ENV_CLAUDE_FILE"
#     set +a
#     echo "Loaded environment file: $ENV_CLAUDE_FILE"
# else
#     echo "WARNING: $ENV_CLAUDE_FILE not found (cwd=$(pwd)); continuing without it." >&2
# fi

# -------------------------
# Run experiment
# -------------------------
# solver.time_limit_secs = max wall time for the agent/search (this run stops when hit).
# solver.execution_timeout = max time per single program execution (unchanged from run_example: 2h).


python -m dojo.main_run \
    +_exp=warm_draft_run_example_5.4\
    task=agentSSL/_default \
    task.name=${DOMAIN} \
    task.setting=${SETTING} \
    task.subset=${K} \
    task.seed=${SEED}
