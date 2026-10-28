#!/bin/bash
#SBATCH --job-name=backtranslation_check
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=<gpu partition>
#SBATCH --gres=gpu:1
#SBATCH --output=logs/backtranslation_check_%j.out
#SBATCH --error=logs/backtranslation_check_%j.err

# --- environment: edit for your cluster; placeholders are in angle brackets ------------
source "${CONDA_SH:-<path to conda>/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-<env name>}"
export HF_HOME="${HF_HOME:-<scratch dir>/huggingface}"
export TMPDIR="${TMPDIR:-<scratch dir>/tmp}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# ----------------------------------------------------------------------------------------
# Submit from the repository root after `mkdir -p logs`; SLURM writes the log there.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

#   sbatch data/translation/run_check_backtranslation.sh \
#       --translations-dir <dir with <language>.jsonl files> \
#       --languages "Bhojpuri" "Central Kurdish" "Finnish" "Haitian Creole" "Slovak" "Yoruba"
python -u data/translation/check_backtranslation.py "$@"
