#!/bin/bash
#SBATCH --job-name=qa_context_adapter
#SBATCH --time=16:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=<gpu partition>
#SBATCH --gres=gpu:1
#SBATCH --output=logs/qa_context_adapter_%j.out
#SBATCH --error=logs/qa_context_adapter_%j.err

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

# Train Adapter 1 (--variant v8) or Adapter 2 (--variant v4):
#   sbatch adapter/train/run_finetune.sh --variant v8 --phase train
#   sbatch adapter/train/run_finetune.sh --variant v8 --phase save --checkpoint adapter/runs/v8/outputs/checkpoint-<N>
# The recipe is passed explicitly so the log shows it; arguments after it override (smoke test: --batch-size 1).
echo "nproc: $(nproc), SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
python -u adapter/train/qwen35_9b_qa_context_finetune.py \
    --dataset-num-proc "${SLURM_CPUS_PER_TASK:-16}" \
    --max-seq-length 4096 --batch-size 8 --grad-accum 1 --learning-rate 1e-4 --epochs 1 \
    "$@"
