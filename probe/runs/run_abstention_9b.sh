#!/usr/bin/env bash
# Abstention analysis with the shipped 9B probe directions: score the bare caches with
# probe_directions.npz, order the rows, generate base-model answers under the qa-context prompt,
# analyse. Row set: the ids of the 4B qa-context caches (probe/caches/qwen35-4b/qa-context).
#
#   PILOT=1 bash probe/runs/run_abstention_9b.sh    # 64 rows
#   PHASE=1 bash probe/runs/run_abstention_9b.sh    # first ~6k rows, cell-balanced
#   PHASE=2 bash probe/runs/run_abstention_9b.sh    # the rest; nothing is regenerated
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PHASE="${PHASE:-1}"
PILOT="${PILOT:-0}"
PHASE1_ROWS="${PHASE1_ROWS:-2000}"
PROBES="${PROBES:-tydi,tydi_udhr}"

PY=${PY:-python3}
A=probe
C=probe/caches/qwen35-9b/bare              # what gets scored (activations already exist)
T=probe/caches/qwen35-4b/qa-context        # what defines the row set
G=probe/generation/qwen35-9b
mkdir -p "$G"

# 4096 is what the extraction runs used on this card for a hidden size of 4096. Generation
# holds a KV cache on top of an 18.5 GB model, so if it OOMs, halve this rather than
# lowering --batch-size: the budget is rows x padded length, which is what actually governs
# peak prefill. That is what PILOT=1 is for.
TOKEN_BUDGET="${TOKEN_BUDGET:-4096}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--model $MODEL --token-budget $TOKEN_BUDGET --require-gpu"

echo "model=$MODEL  phase=$PHASE  pilot=$PILOT  token-budget=$TOKEN_BUDGET  probes=$PROBES"

for SET in validation translation belebele; do
  case $SET in
    validation) SRC="--source tydi --split validation" ;;
    *)          SRC="--source $SET" ;;
  esac

  # ---- score, once, over the whole row set --------------------------------------------
  # Deliberately not restricted to the phase-1 ids: scoring is CPU-only and cheap, and
  # covering every row now means phase 2 needs no rescoring and the two phases are joined
  # against identical probe files.
  if [ ! -f "$G/probe_tydi_$SET.jsonl" ]; then
    echo "=== score $SET with the shipped directions ==="
    $PY $A/probing/score_bare_cache.py --acts $C/$SET --ids-from $T/$SET/meta.jsonl \
        --out "$G/probe_{probe}_$SET.jsonl"
  fi

  # ---- order ---------------------------------------------------------------------------
  if [ ! -f "$G/ids_${SET}_all.jsonl" ]; then
    echo "=== order $SET ==="
    $PY $A/probing/order_generation_rows.py --scores "$G/probe_{probe}_$SET.jsonl" \
        --probes "$PROBES" --set $SET --phase1 "$PHASE1_ROWS" --out-dir $G
  fi

  # ---- generate ------------------------------------------------------------------------
  # --max-len stays at the qa-context default of 1200 and does NOT match the bare cache's
  # 1024. That is the arm, not an oversight: the model is shown what the organizers show it.
  if [ "$PILOT" = "1" ]; then
    head -n 64 "$G/ids_${SET}_phase1.jsonl" > "$G/ids_${SET}_pilot.jsonl"
    IDS="$G/ids_${SET}_pilot.jsonl"; OUT="$G/pilot_$SET.jsonl"
  elif [ "$PHASE" = "1" ]; then
    IDS="$G/ids_${SET}_phase1.jsonl"; OUT="$G/$SET.jsonl"
  else
    IDS="$G/ids_${SET}_all.jsonl";    OUT="$G/$SET.jsonl"
  fi
  echo "=== generate $SET (phase $PHASE) from $(basename $IDS) ==="
  $PY $A/probing/generate_answers.py $SRC $COMMON --ids-from "$IDS" --out "$OUT"

  if [ "$PILOT" = "1" ]; then
    echo "=== pilot done for $SET: $TOKEN_BUDGET fits. Re-run with PHASE=1. ==="
    exit 0
  fi

  # ---- analyse, once per probe ---------------------------------------------------------
  for P in ${PROBES//,/ }; do
    echo ""; echo "######## ABSTENTION: $SET, probe=$P, phase=$PHASE ########"
    $PY $A/probing/analyze_abstention.py --generations "$OUT" --probe "$G/probe_${P}_$SET.jsonl" \
        --out "$G/abstention_${P}_$SET.json" --dump-unmatched 5 \
        || echo "(analysis failed for $SET/$P)"
  done
done

echo "=== DONE (phase $PHASE) ==="; wc -l $G/*.jsonl
if [ "$PHASE" = "1" ]; then
  echo
  echo "Interim numbers: read them PER BIN. The prefix is balanced across confidence bins,"
  echo "so aggregate rates over-weight the rare ones. Run PHASE=2 for the aggregates."
fi
