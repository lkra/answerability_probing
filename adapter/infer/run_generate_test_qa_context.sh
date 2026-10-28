#!/bin/bash
#SBATCH --job-name=qa_context_test_infer
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=<gpu partition>
#SBATCH --gres=gpu:1
#SBATCH --output=logs/qa_context_test_infer_%j.out
#SBATCH --error=logs/qa_context_test_infer_%j.err

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

#   sbatch adapter/infer/run_generate_test_qa_context.sh --limit 16 --out-path adapter/runs/qa_context_smoke.jsonl
#   sbatch adapter/infer/run_generate_test_qa_context.sh --checkpoint adapter/runs/v8/outputs/checkpoint-12400 \
#       --out-path adapter/runs/v8/qa_context_test_out.jsonl        # resumes if --out-path exists
#   sbatch adapter/infer/run_generate_test_qa_context.sh --lang deu
python -u adapter/infer/generate_test_qa_context.py "$@"
