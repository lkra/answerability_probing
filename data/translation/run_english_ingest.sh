#!/usr/bin/env bash
# Append one English shard to data/translated/Hy-MT2-7B-FP8.jsonl, extend the 9B translation cache
# with the new rows (same extractor settings as the existing cache) and re-run eval_crosslingual.py
# for both probes.
#
#   RECIPE=matched bash data/translation/run_english_ingest.sh          # default
#   DRY=1 RECIPE=matched bash data/translation/run_english_ingest.sh    # check, write nothing
set -euo pipefail
cd "$(dirname "$0")/../.."

RECIPE="${RECIPE:-matched}"
DRY="${DRY:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
A=probe
T=data/translated
C=probe/caches/qwen35-9b/bare
R=$A/runs/qwen35-9b/bare
SHARD="$T/shards_eng_$RECIPE/english.jsonl"

[ -f "$SHARD" ] || { echo "no shard at $SHARD -- run run_english_translation.sh first"; exit 1; }

echo "=== before ==="
echo "compiled: $(wc -l < $T/Hy-MT2-7B-FP8.jsonl) rows"
echo "cache:    $(wc -l < $C/translation/meta.jsonl) rows"

echo "=== 1/4: append $RECIPE shard (dry run first, always) ==="
$PY data/translation/append_language_to_compiled.py --shard "$SHARD" --lang eng_Latn --dry-run
if [ "$DRY" = "1" ]; then echo "DRY=1, stopping."; exit 0; fi
$PY data/translation/append_language_to_compiled.py --shard "$SHARD" --lang eng_Latn

echo "=== 2/4: extend the 9B bare translation cache (append-only, resumes by id) ==="
$PY -u $A/probing/extract_activations.py --source translation --prompt-style bare \
    --model Qwen/Qwen3.5-9B --require-gpu --token-budget 4096 --max-len 1024 \
    --save-layers 0,4,8,12,16,19,22,26,29,32 --out $C/translation

echo "=== 3/4: re-run the cross-lingual evaluation, both arms ==="
for ARM in tydi tydi_udhr; do
  TR="$C/train"; [ "$ARM" = tydi_udhr ] && TR="$C/train $C/udhr"
  $PY -u $A/probing/eval_crosslingual.py --layer 19 --train-acts $TR \
      --mono-acts $C/validation --xl-acts $C/translation --layer-profile \
      --out $R/crosslingual_${ARM}_eng.json || { echo "FAILED $ARM"; exit 1; }
done

echo "=== 4/4: what changed ==="
echo "compiled: $(wc -l < $T/Hy-MT2-7B-FP8.jsonl) rows"
echo "cache:    $(wc -l < $C/translation/meta.jsonl) rows"
$PY - <<'PY'
import json, collections
p = "probe/caches/qwen35-9b/bare/translation/meta.jsonl"
c = collections.Counter(json.loads(l)["question_lang"] for l in open(p))
print(f"{sum(c.values()):,} rows, {len(c)} question languages; eng_Latn = {c.get('eng_Latn', 0):,}")
PY
echo
echo "New results are in *_eng.json, ALONGSIDE the 17-language originals -- nothing was"
echo "overwritten, so the two are directly comparable. The `matched` shard is the one that was ingested"
echo "for the list of paper locations these numbers would feed."
