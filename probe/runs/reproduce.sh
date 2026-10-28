#!/usr/bin/env bash
# Every probe result for Qwen3.5-9B, in order, from the repository root:
#
#   bash probe/runs/reproduce.sh             # everything (extraction and base-model answers need a GPU)
#   bash probe/runs/reproduce.sh --cpu-only  # skip the GPU steps; reuse the activation caches
#
# Records land in probe/runs/qwen35-9b/bare/, activation caches in probe/caches/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
R=probe/runs/qwen35-9b/bare
C=probe/caches/qwen35-9b/bare
GPU=1; [ "${1:-}" = "--cpu-only" ] && GPU=0

if [ "$GPU" = 1 ]; then
  echo "=== activation caches (GPU)"
  bash probe/runs/run_9b_xlingual_extract.sh   # TyDi, BELEBELE, UDHR, BELEBELE-8, TyDi/cross
  bash probe/runs/run_9b_hard_negatives.sh     # BELEBELE-8 swap
fi

echo "=== layer sweep, TyDi vs TyDi+UDHR, leave-one-language-out"
"$PY" probe/probing/train_probes.py --acts $C/train --layers 0,4,8,12,16,19,22,26,29,32 \
    --out $R/lolo_train.json
"$PY" probe/probing/train_probe_multi.py --layer 19 --folds 5 --bootstrap 2000 \
    --out $R/arms.json --save-oof $R/arms_oof.npz
"$PY" probe/probing/train_probe_multi.py --mode lolo --layer 19 --folds 5 --lolo-folds 5 \
    --out $R/lolo_udhr_multi.json
"$PY" probe/probing/train_probe_multi.py --mode lolo --layer 19 --folds 5 --lolo-folds 5 \
    --lolo-probes udhr,tydi+udhr,tydi+udhr+beleb --lolo-removal question-udhr,question-all,union-all \
    --lolo-budget fixed --out $R/lolo_udhr_union.json
"$PY" probe/probing/train_probe_multi.py --mode lolo-tydi --layer 19 --lolo-budget fixed --lolo-budget-seeds 4 \
    --out $R/lolo_tydi_union.json

echo "=== cross-lingual and BELEBELE-8, both probes"
bash probe/runs/run_9b_xlingual_eval.sh

echo "=== per-language offsets, controls"
"$PY" probe/probing/calibrate_offsets.py --calib-acts $C/udhr     --out $R/offsets_udhr.json
"$PY" probe/probing/calibrate_offsets.py --calib-acts $C/belebele --out $R/offsets_belebele.json
"$PY" probe/probing/export_probe_directions.py --device cpu
"$PY" probe/probing/lexical_baselines.py
"$PY" probe/probing/langid_control.py
"$PY" probe/probing/coverage_split.py --dataset all --group both
"$PY" probe/probing/belebele_calibration.py
"$PY" probe/probing/negatives_control.py

echo "=== shipped probe on the test set (the only step that reads it)"
"$PY" probe/probing/train_probe_multi.py --mode final --layer 19 --folds 5 --final-probe tydi+udhr \
    --final-rule globalF1 --calib-fold 0 --bias-from cv-mean \
    --out $R/mist_test_predictions_tydi_udhr.jsonl

if [ "$GPU" = 1 ]; then
  # Base-model answers vs probe scores. The row set comes from probe/caches/qwen35-4b/qa-context
  # (run_abstention.sh Qwen/Qwen3.5-4B).
  echo "=== base-model abstention (GPU)"
  PHASE=1 bash probe/runs/run_abstention_9b.sh
  PHASE=2 bash probe/runs/run_abstention_9b.sh
fi
