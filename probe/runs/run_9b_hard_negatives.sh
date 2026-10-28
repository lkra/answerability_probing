#!/usr/bin/env bash
# Extract the 9B bare cache for BELEBELE swap negatives (data/built/belebele_answerability_hard_b8.jsonl),
# item-matched to the belebele8 cache: 128 items, 8 languages, 2,048 rows. Same settings as
# run_9b_xlingual_extract.sh. Then: python3 probe/probing/negatives_control.py
set -o pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
A=probe
C=probe/caches/qwen35-9b/bare
LAYERS=0,4,8,12,16,19,22,26,29,32
B8LANGS=yor_Latn,swh_Latn,ckb_Arab,hat_Latn,por_Latn,eng_Latn,zho_Hans,pes_Arab

echo "=== BELEBELE swap (hard) negatives, item-matched to belebele8 $(date -Is) ==="
$PY -u $A/probing/extract_activations.py \
    --model Qwen/Qwen3.5-9B --prompt-style bare \
    --token-budget 4096 --require-gpu --save-layers $LAYERS \
    --source belebele \
    --belebele data/built/belebele_answerability_hard_b8.jsonl \
    --belebele-languages $B8LANGS \
    --out $C/belebele-hard || { echo "EXTRACTION FAILED"; exit 1; }

echo "=== DONE $(date -Is) ==="
echo "Now run (CPU, no GPU needed):"
echo "  \$PY $A/probing/negatives_control.py"
