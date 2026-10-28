#!/bin/bash
#SBATCH --job-name=translate_questions
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=<gpu partition>
#SBATCH --gres=gpu:1
#SBATCH --output=logs/translate_questions_%j.out
#SBATCH --error=logs/translate_questions_%j.err

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

# Modes, chosen by the first argument:
#   sbatch data/translation/run_translate_questions.sh test      German, 4 questions -> data/translated/translated_test/
#   sbatch data/translation/run_translate_questions.sh subset    1,000 questions, all languages -> data/translated/translated/
#   sbatch data/translation/run_translate_questions.sh           all questions, all languages -> data/translated/translated/
#   sbatch data/translation/run_translate_questions.sh fromdir   --input-dir data/translated/to-translate -> data/translated/finetune/
# Further arguments are forwarded, e.g.
#   sbatch data/translation/run_translate_questions.sh fromdir --model hymt2 --output-dir data/translated/finetune/hymt2
# Runs into the same output dir resume by id, so run one model per output dir.
case "$1" in
    test)    shift; OUTPUT_DIR=data/translated/translated_test; RUN_ARGS="--languages German --limit 4 --batch-size 2" ;;
    subset)  shift; OUTPUT_DIR=data/translated/translated;      RUN_ARGS="--limit 1000" ;;
    fromdir) shift; OUTPUT_DIR=data/translated/finetune;        RUN_ARGS="--input-dir data/translated/to-translate" ;;
    *)              OUTPUT_DIR=data/translated/translated;      RUN_ARGS="" ;;
esac

python -u data/translation/translate_questions.py --output-dir "$OUTPUT_DIR" $RUN_ARGS "$@"
