#!/usr/bin/env bash
# Rebuild udhr_qa end to end.
#
# The deterministic steps are safe to re-run at any time: same inputs, same
# bytes out. The three model passes are NOT re-run by this script, because
# re-running them would replace artefacts that were reviewed and are the record
# of what was actually built. They are marked below with their inputs and
# outputs; see README.md ("The three model passes") for how to drive them.
#
# Usage:
#   ./data/udhr/run_all.sh            # deterministic steps only (default)
#   ./data/udhr/run_all.sh --check    # also re-run verify.py and fail on regression
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}

echo "=== 1. fetch the UDHR corpus (pinned by sha256) ==="
$PY udhr/fetch_udhr.py

echo
echo "=== 2. parse + prove the language mapping against the real test set ==="
$PY udhr/parse_udhr.py

echo
echo "=== 3. cut passages from the unused articles ==="
$PY udhr/build_passages.py

echo
echo "=== 4. export question-authoring jobs and design exemplars ==="
$PY udhr/export_question_jobs.py

# --- MODEL PASS 1 -----------------------------------------------------------
# data/question_jobs.json + data/design_exemplars.json
#   -> data/questions_raw/<passage_id>.json      (author)
#   -> data/questions_verified/<passage_id>.json (independent review)
# Skipped here; the existing files are the artefact of record.

echo
echo "=== 5. merge questions, re-checking mechanically what can be re-checked ==="
$PY udhr/merge_questions.py

echo
echo "=== 6. propose transplanted unanswerables ==="
$PY udhr/build_transplants.py

# --- MODEL PASS 2 -----------------------------------------------------------
# data/transplant_jobs/<passage_id>.json
#   -> data/transplants_verified/<passage_id>.json

echo
echo "=== 7. merge verified transplants ==="
$PY udhr/merge_transplants.py

echo
echo "=== 8. export localisation jobs (24 languages) ==="
$PY udhr/build_translation_inputs.py

# --- MODEL PASS 3 -----------------------------------------------------------
# data/translation_inputs/<lang>.<half>.json
#   -> data/translation_outputs/<lang>.<half>.json

echo
echo "=== 9. merge localisations, measure grounding ==="
$PY udhr/merge_translations.py

# --- MODEL PASS 4 (audit) ---------------------------------------------------
# export_backtranslation_jobs.py -> data/backtranslation_jobs/<lang>.json
#   -> data/backtranslation_outputs/<lang>.json   (blind back-translation)
#   then check_backtranslation.py scores it (needs sacrebleu)
# --- MODEL PASS 5 (repair) --------------------------------------------------
# export_repair_jobs.py -> data/repair_question_jobs/<lang>.json
#   -> data/repair_question_outputs/<lang>.json
#   then merge_repairs.py applies them to localised.json
# Both already applied; re-running would replace reviewed artefacts.

echo
echo "=== 10. render training rows ==="
$PY udhr/build_rows.py

echo
echo "=== 11. verify ==="
if [ "${1:-}" = "--check" ]; then
  $PY udhr/verify.py
else
  $PY udhr/verify.py || echo "(verify reported failures -- see above)"
fi

echo
echo "=== 12. answerable-only variant ==="
$PY udhr/build_rows.py --answerable-only \
    --out data/udhr/data/udhr_qa_train_answerable.jsonl \
    --stats-out data/udhr/data/udhr_qa_answerable_stats.json

echo
echo "done."
echo "  both labels     : data/udhr/data/udhr_qa_train.jsonl"
echo "  answerable only : data/udhr/data/udhr_qa_train_answerable.jsonl"
