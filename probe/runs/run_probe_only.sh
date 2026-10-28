#!/usr/bin/env bash
# Layer sweep and probe for one model on TyDi only: extract train (ROWS per language, default
# 3000; ROWS=0 for all) and validation with the bare prompt, then train_probes.py.
#
#   bash probe/runs/run_probe_only.sh Qwen/Qwen3.5-9B
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:?usage: run_probe_only.sh <hf-model-id>   e.g. Qwen/Qwen3.5-9B}"
ROWS="${ROWS:-3000}"

PY=${PY:-python3}
A=probe
SLUG=$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z' | sed 's|.*/||; s|qwen3\.5-|qwen35-|')
C=probe/caches/$SLUG/bare
R=$A/runs/$SLUG/bare
mkdir -p "$R"

read -r NLAYERS HIDDEN <<<"$($PY -c "
from transformers import AutoConfig; import sys; sys.path.insert(0,'$A/probing')
from _loading import text_config
c = text_config(AutoConfig.from_pretrained('$MODEL')); print(c.num_hidden_layers, c.hidden_size)" 2>/dev/null)"
LAYERS=$($PY -c "print(','.join(str(i) for i in range($NLAYERS + 1)))")
TOKEN_BUDGET="${TOKEN_BUDGET:-$([ "$HIDDEN" -le 2560 ] && echo 8192 || { [ "$HIDDEN" -le 4096 ] && echo 4096 || echo 2048; })}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "model=$MODEL  layers=$NLAYERS  hidden=$HIDDEN  token-budget=$TOKEN_BUDGET  rows/lang=${ROWS:-all}"

LIM=""; [ "$ROWS" != "0" ] && LIM="--limit-per-lang $ROWS"
EX="$PY $A/probing/extract_activations.py --model $MODEL --prompt-style bare \
    --token-budget $TOKEN_BUDGET --require-gpu"

echo "=== 1/3: extract TyDi train ==="
$EX --split train $LIM --out $C/train
echo "=== 2/3: extract TyDi validation ==="
$EX --split validation --out $C/validation

# train_probes gives the leave-one-language-out sweep (which layer generalises across
# languages); probe_predictions picks the operating layer on held-out TRAIN rows and fits
# the deployable probe, and prints the depth-matched full-vs-linear attention contrast.
echo "=== 3/3: layer sweep + probe ==="
$PY $A/probing/train_probes.py --acts $C/train --layers "$LAYERS" --out $R/lolo_train.json
$PY $A/probing/probe_predictions.py --train-acts $C/train --eval-acts $C/validation \
    --layer sweep --out $R/probe_predictions.jsonl

echo
echo "=== SUMMARY ==="
$PY - <<PY
import json
d = json.load(open("$R/lolo_train.json"))
best = d["best_layer"]; f = d["folds"][str(best)]; L = d["languages"]
m = sum(f[l]["auroc"] for l in L) / len(L)
o = sum(d["baselines"][l]["overlap_auroc"] for l in L) / len(L)
print(f"model            {d['config']['model']}")
print(f"LOLO best layer  {best} of {d['layers'][-1]}   (relative depth {best/d['layers'][-1]:.2f})")
print(f"mean AUROC       {m:.4f}  vs overlap baseline {o:.4f}")
print(f"shuffled labels  {d['shuffled_label_acc']:.3f} +- {d.get('shuffled_label_acc_sd', float('nan')):.3f}")
print()
print("Qwen3.5-4B reference: best layer 17 of 32, mean .974, overlap .797, shuffled .497.")
print("4B and 9B share layer count (32) and full-attention positions, so the layer index")
print("is directly comparable -- no relative-depth rescaling needed.")
PY
echo "probe scores: $R/probe_predictions.jsonl"
