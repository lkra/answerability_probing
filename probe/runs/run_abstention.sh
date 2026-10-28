#!/usr/bin/env bash
# Abstention pipeline with a qa-context-prompt cache: extract -> generate -> probe -> analyse, so
# probe and generator read the same prompt. Its row set is what run_abstention_9b.sh uses.
#
#   bash probe/runs/run_abstention.sh Qwen/Qwen3.5-4B
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:?usage: run_abstention.sh <hf-model-id>   e.g. Qwen/Qwen3.5-4B}"

PY=${PY:-python3}
A=probe
SLUG=$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z' | sed 's|.*/||; s|qwen3\.5-|qwen35-|')
C=probe/caches/$SLUG/qa-context
G=probe/generation/$SLUG
R=$A/runs/$SLUG/qa-context
mkdir -p "$G" "$R"

TOKEN_BUDGET="${TOKEN_BUDGET:-$($PY -c "
from transformers import AutoConfig; import sys; sys.path.insert(0,'$A/probing')
from _loading import text_config
h = text_config(AutoConfig.from_pretrained('$MODEL')).hidden_size
print(8192 if h <= 2560 else 4096 if h <= 4096 else 2048)" 2>/dev/null)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--model $MODEL --token-budget $TOKEN_BUDGET --require-gpu"
echo "model=$MODEL slug=$SLUG token-budget=$TOKEN_BUDGET"

# ---- extract -------------------------------------------------------------------------
EX="$PY $A/probing/extract_activations.py --prompt-style qa-context $COMMON"
echo "=== extract 1/4: TyDi train ===";        $EX --split train      --out $C/train
echo "=== extract 2/4: TyDi validation ===";   $EX --split validation --out $C/validation
echo "=== extract 3/4: cross-lingual ===";     $EX --source translation --limit-per-lang 90 --out $C/translation
echo "=== extract 4/4: Belebele ===";          $EX --source belebele    --limit-per-lang 30 --out $C/belebele

# ---- generate ------------------------------------------------------------------------
# --ids-from <cache>/meta.jsonl is what makes "extract only the rows you generate for"
# safe: both sides cover the identical row set, so the join later drops nothing.
for SET in validation translation belebele; do
  case $SET in
    validation) SRC="--source tydi --split validation" ;;
    *)          SRC="--source $SET" ;;
  esac
  echo "=== generate: $SET ==="
  $PY $A/probing/generate_answers.py $SRC $COMMON \
      --ids-from $C/${SET/validation/validation}/meta.jsonl --out $G/$SET.jsonl
done

# ---- probe + analyse -----------------------------------------------------------------
echo "=== probe: sweep + score monolingual validation ==="
$PY $A/probing/probe_predictions.py --train-acts $C/train --eval-acts $C/validation \
    --layer sweep --out $G/probe_validation.jsonl
LAYER=$($PY -c "import json;print(json.loads(open('$G/probe_validation.jsonl').readline())['layer'])")
echo "=== operating layer: $LAYER ==="

# The layer is swept once and reused: re-sweeping per eval set would pick a different
# layer for each and make the three numbers incomparable.
for SET in translation belebele; do
  echo "=== probe: score $SET at layer $LAYER ==="
  $PY $A/probing/probe_predictions.py --train-acts $C/train --eval-acts $C/$SET \
      --layer "$LAYER" --out $G/probe_$SET.jsonl
done

for SET in validation translation belebele; do
  echo ""; echo "################ ABSTENTION ANALYSIS: $SET ################"
  $PY $A/probing/analyze_abstention.py --generations $G/$SET.jsonl --probe $G/probe_$SET.jsonl \
      --out $G/abstention_$SET.json --dump-unmatched 5 || echo "(analysis failed for $SET)"
done

echo "=== DONE ==="; wc -l $G/*.jsonl
