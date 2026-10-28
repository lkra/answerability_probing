#!/usr/bin/env bash
# Translate the TyDi eval questions into English under two decoding recipes: matched (the settings
# the other 17 languages were translated with: max_tokens 256, top_p 1.0, no top_k, repetition
# penalty 1.0) and current (Tencent's 7B recipe). Writes data/translated/shards_eng_<recipe>/english.jsonl;
# nothing is merged (see append_language_to_compiled.py). The matched shard is the one ingested.
#
#   SMOKE=1 bash data/translation/run_english_translation.sh   # 8 rows
#   bash data/translation/run_english_translation.sh           # 6,171 rows
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
T=data/translated
LIMIT=""; [ "${SMOKE:-0}" = "1" ] && LIMIT="--limit 8"
BS="${BATCH_SIZE:-16}"

run () {   # $1 = tag, rest = decoding flags
  local tag="$1"; shift
  local out="$T/shards_eng_$tag"
  echo "=== English, recipe=$tag $(date -Is) ==="
  $PY -u data/translation/translate_questions.py \
      --input "$T/questions_to_translate.tsv" \
      --output-dir "$out" --languages English --model hymt2 \
      --batch-size "$BS" $LIMIT "$@"
  echo "--- $out/english.jsonl: $(wc -l < "$out/english.jsonl") rows"
}

# The old defaults, recovered from e673413^ rather than guessed. Passing them explicitly is
# safe: translate_questions.py fills decoding flags from MODELS only where the flag is None,
# so an explicit value always wins (:335).
run matched --max-tokens 256 --temperature 0.7 --top-p 1.0 --top-k -1 --repetition-penalty 1.0
run current

echo
echo "=== side by side ==="
$PY - <<'PY'
import json
from pathlib import Path
T = Path("data/translated")
shards = {t: T / f"shards_eng_{t}" / "english.jsonl" for t in ("matched", "current")}
rows = {t: {json.loads(l)["id"]: json.loads(l)["question"] for l in open(p, encoding="utf-8")}
        for t, p in shards.items() if p.exists()}
if len(rows) == 2:
    ids = sorted(set(rows["matched"]) & set(rows["current"]))
    same = sum(rows["matched"][i].strip() == rows["current"][i].strip() for i in ids)
    print(f"{len(ids):,} shared ids; identical under both recipes: {same:,} "
          f"({100*same/max(len(ids),1):.1f}%)")
    shown = 0
    for i in ids:
        if rows["matched"][i].strip() != rows["current"][i].strip() and shown < 5:
            print(f"\n  [{i}]\n    matched: {rows['matched'][i][:160]}"
                  f"\n    current: {rows['current'][i][:160]}")
            shown += 1
PY
echo "=== DONE $(date -Is) ==="
echo "Merge is NOT automatic. To ingest one of them:"
echo "  $PY data/translation/append_language_to_compiled.py --shard $T/shards_eng_<tag>/english.jsonl \\"
echo "      --lang eng_Latn --dry-run"
