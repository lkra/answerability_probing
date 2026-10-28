#!/usr/bin/env bash
# Tier II (translated TyDi questions) and item-matched BELEBELE at Qwen3.5-9B, for both
# probe arms. Consumes the caches from run_9b_xlingual_extract.sh.
#
# Invocation verified: with --xl-acts .../belebele it reproduces the committed
# belebele_crosslingual_{tydi,tydi_udhr}.json to CPU-vs-GPU arithmetic noise (max |d| on
# any AUROC 1e-5), so the two arms below are the same two arms those records used:
#   tydi        --train-acts train                 (32,899 rows)
#   tydi+udhr   --train-acts train udhr            (50,179 rows)
set -o pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
A=probe
C=probe/caches/qwen35-9b/bare
R=$A/runs/qwen35-9b/bare

EV="$PY -u $A/probing/eval_crosslingual.py --layer 19"

for ARM in tydi tydi_udhr; do
  TR="$C/train"; [ "$ARM" = tydi_udhr ] && TR="$C/train $C/udhr"

  echo "=== TyDi/cross ($ARM) $(date -Is) ==="
  $EV --train-acts $TR --mono-acts $C/validation --xl-acts $C/translation \
      --layer-profile --out $R/crosslingual_$ARM.json || { echo "FAILED xl $ARM"; exit 1; }

  echo "=== BELEBELE-8 item-matched ($ARM) $(date -Is) ==="
  $EV --train-acts $TR --mono-acts $C/validation --xl-acts $C/belebele8 \
      --belebele data/built/belebele_answerability_delete_perlang.jsonl --mono-from-diagonal \
      --out $R/belebele8_crosslingual_$ARM.json || { echo "FAILED b8 $ARM"; exit 1; }
done

echo "=== EVAL DONE $(date -Is) ==="
