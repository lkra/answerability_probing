#!/usr/bin/env bash
# Build every constructed dataset from data/upstream + data/translated into data/built.
# CPU only, deterministic. Run from the repository root:
#
#   pip install -r requirements-data.txt
#   python3 data/fetch_sources.py        # once: upstream corpora at the revisions data/sources.json pins
#   data/build_all.sh
#
# Steps that need something not present yet are skipped with a message rather than failing, so the
# adapter files can be built from the translation bundle alone once data/built/tydiqa_answerability
# exists. Expected row counts are asserted at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
B=data/built; mkdir -p "$B"

echo "=== 0. organisers' prompt templates (from data/test/tests.jsonl)"
"$PY" data/extract_qa_templates.py

echo; echo "=== 0b. UDHR rows, rendered from the committed items, passages and localisations"
"$PY" data/udhr/build_rows.py
"$PY" data/udhr/build_rows.py --answerable-only \
    --out data/udhr/data/udhr_qa_train_answerable.jsonl --stats-out data/udhr/data/udhr_qa_answerable_stats.json
"$PY" data/udhr/build_rows.py --answerable-only --ctx-answerable 25 \
    --out data/udhr/data/udhr_qa_train_answerable_ctx25.jsonl --stats-out data/udhr/data/udhr_qa_answerable_ctx25_stats.json

if [ -d data/upstream/answerable_tydiqa ]; then
  echo; echo "=== 1. TyDiQA answerability set (probe train/validation, passages for the adapter rows)"
  "$PY" data/tydi/build_tydiqa.py
else
  echo; echo "=== 1. skipped: data/upstream/answerable_tydiqa missing (run data/fetch_sources.py); using data/built/tydiqa_answerability as is"
fi

echo; echo "=== 2. TyDiQA/cross23 rows: sentence form with span fallback (Adapter 2's TyDi source)"
"$PY" data/tydi/build_qa_context_train.py --out $B/qa_context_tydi_train.jsonl --stats-out $B/qa_context_tydi_stats.json

echo; echo "=== 3. TyDiQA/cross23 rows: adaptive target length (sentence if <=120 chars, else span)"
"$PY" data/tydi/build_qa_context_train.py --answer-form adaptive --max-sentence-chars 120 \
    --out $B/qa_context_tydi_adaptive_train.jsonl --stats-out $B/qa_context_tydi_adaptive_stats.json

echo; echo "=== 4. Adapter 1 (V8): adaptive targets made well-formed"
"$PY" data/tydi/build_v8_wellformed.py

if [ -d data/upstream/belebele ]; then
  echo; echo "=== 5. BELEBELE generation set"
  "$PY" data/belebele/build_belebele_qa_context.py
  echo; echo "=== 6. BELEBELE/unanswerable probe set (answer sentence deleted, strict alignment)"
  "$PY" data/belebele/build_belebele.py --negatives delete --delete-align strict --pairs all --output $B/belebele_answerability_delete_strict.jsonl
  echo; echo "=== 6b. BELEBELE delete negatives, per-language alignment (item-matched 8-language evaluation)"
  "$PY" data/belebele/build_belebele.py --negatives delete --delete-align per-language --pairs all --output $B/belebele_answerability_delete_perlang.jsonl
  echo; echo "=== 6c. BELEBELE swap negatives on the same items and languages (negative-construction control)"
  "$PY" data/belebele/build_belebele.py --negatives hard --languages yor_Latn,swh_Latn,ckb_Arab,hat_Latn,por_Latn,eng_Latn,zho_Hans,pes_Arab --pairs same --output $B/belebele_answerability_hard8.jsonl
  "$PY" data/belebele/restrict_hard_to_belebele8.py
else
  echo; echo "=== 5-6. skipped: data/upstream/belebele missing (run data/fetch_sources.py)"
fi

echo; echo "=== 7. Adapter 2 (V4): 40/50/10 TyDi/BELEBELE/UDHR mix, cap 4 per completion, 0.35 distinct floor"
"$PY" data/mix/build_qa_context_mix.py

echo; echo "=== Row-count checks"
status=0
check() { local n; n=$(wc -l < "$1" | tr -d ' '); if [ "$n" -eq "$2" ]; then echo "  ok    $1  $n"; else echo "  FAIL  $1  $n (expected $2)"; status=1; fi; }
[ -f $B/tydiqa_answerability/train.jsonl ] && check $B/tydiqa_answerability/train.jsonl 116067
check $B/qa_context_tydi_train.jsonl 100466
check $B/qa_context_tydi_adaptive_train.jsonl 100466
check $B/qa_context_v8_train.jsonl 100466
[ -f $B/belebele_qa_context_train.jsonl ] && check $B/belebele_qa_context_train.jsonl 54924
[ -f $B/belebele_answerability_delete_strict.jsonl ] && check $B/belebele_answerability_delete_strict.jsonl 22464
[ -f $B/belebele_answerability_delete_perlang.jsonl ] && check $B/belebele_answerability_delete_perlang.jsonl 430110
[ -f $B/belebele_answerability_hard_b8.jsonl ] && check $B/belebele_answerability_hard_b8.jsonl 2560
check $B/qa_context_v4_train.jsonl 60000
check data/udhr/data/udhr_qa_train.jsonl 17280
check data/udhr/data/udhr_qa_train_answerable.jsonl 13776
check data/udhr/data/udhr_qa_train_answerable_ctx25.jsonl 43008
exit $status
