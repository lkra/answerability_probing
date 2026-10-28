#!/usr/bin/env bash
# Extraction and probing for one model: TyDi train/validation, the cross-lingual TyDi set and
# BELEBELE/unanswerable; then the layer sweep, leave-one-language-out probe and cross-lingual
# evaluations. Variants: bare = "{passage}\n\n{question}" with no instruction (all 11 TyDi
# languages); qa-context = the organisers' template (8 languages).
#
#   bash probe/runs/run_probing.sh Qwen/Qwen3.5-9B bare
#
# Writes probe/caches/<slug>/<variant>/{train,validation,translation,belebele}/ and
# probe/runs/<slug>/<variant>/{lolo_train,crosslingual,belebele_crosslingual}.json
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:?usage: run_probing.sh <hf-model-id> [variant ...]   e.g. Qwen/Qwen3.5-9B bare}"
shift
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(bare qa-context)

PY=${PY:-python3}
A=probe
SLUG=$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z' | sed 's|.*/||; s|qwen3\.5-|qwen35-|')
LAYERS=$($PY -c "
from transformers import AutoConfig; import sys; sys.path.insert(0,'$A/probing')
from _loading import text_config
print(','.join(str(i) for i in range(text_config(AutoConfig.from_pretrained('$MODEL')).num_hidden_layers + 1)))" 2>/dev/null)

# --token-budget governs peak memory, and the right value scales with model size: Qwen3.5's
# gated-delta-rule layers run a torch fallback (flash-linear-attention/causal-conv1d are not
# installed) whose intermediates are large. 8192 is measured-safe for 4B on a 24 GB card;
# a 9B in bf16 leaves ~6 GB after weights, so it needs roughly half that. Override with
# TOKEN_BUDGET= if a run OOMs.
TOKEN_BUDGET="${TOKEN_BUDGET:-$($PY -c "
from transformers import AutoConfig; import sys; sys.path.insert(0,'$A/probing')
from _loading import text_config
h = text_config(AutoConfig.from_pretrained('$MODEL')).hidden_size
print(8192 if h <= 2560 else 4096 if h <= 4096 else 2048)" 2>/dev/null)}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "model=$MODEL slug=$SLUG variants=${VARIANTS[*]} token-budget=$TOKEN_BUDGET"

for V in "${VARIANTS[@]}"; do
  C=probe/caches/$SLUG/$V
  R=$A/runs/$SLUG/$V
  mkdir -p "$R"
  EX="$PY $A/probing/extract_activations.py --model $MODEL --prompt-style $V \
      --token-budget $TOKEN_BUDGET --require-gpu"

  echo "=== [$V] extract 1/5: TyDi train ==="
  $EX --split train      --out $C/train
  echo "=== [$V] extract 2/5: TyDi validation ==="
  $EX --split validation --out $C/validation
  echo "=== [$V] extract 3/5: translation (full) ==="
  $EX --source translation --out $C/translation
  echo "=== [$V] extract 4/4: Belebele delete_strict ==="
  $EX --source belebele --belebele data/built/belebele_answerability_delete_strict.jsonl --out $C/belebele

  echo "=== [$V] probe: leave-one-language-out + layer sweep + controls ==="
  $PY $A/probing/train_probes.py --acts $C/train --layers "$LAYERS" --out $R/lolo_train.json
  echo "=== [$V] probe: cross-lingual ==="
  $PY $A/probing/eval_crosslingual.py --train-acts $C/train --mono-acts $C/validation \
      --xl-acts $C/translation --lolo $R/lolo_train.json --layer-profile \
      --out $R/crosslingual.json
  echo "=== [$V] probe: belebele ==="
  $PY $A/probing/eval_crosslingual.py --train-acts $C/train --mono-acts $C/validation \
      --xl-acts $C/belebele --belebele data/built/belebele_answerability_delete_strict.jsonl \
      --mono-from-diagonal --lolo $R/lolo_train.json --out $R/belebele_crosslingual.json
done

echo "=== DONE ==="; ls $A/runs/$SLUG/*/
