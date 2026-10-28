#!/usr/bin/env bash
# Re-extract the Qwen3.5-9B `bare` caches needed for the TyDi/cross (Tier II) and
# item-matched BELEBELE (belebele8) evaluations, which the 9B run tree never had.
#
# Protocol matched to the existing 9B caches (run_probe_only.sh / the belebele+udhr
# extractions of 2026-07-31): token budget 4096, bare prompt, max_len 1024, last
# pooling, TyDi train subsampled to 3000 rows/language.
#
# Deviation from those caches: --save-layers, because the 9B translation cache at all
# 33 layers is 59 GB. The ten layers kept bracket the operating layer 19 and are enough
# for the operating point plus a cross-lingual layer profile.
set -o pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
A=probe
C=probe/caches/qwen35-9b/bare
LAYERS=0,4,8,12,16,19,22,26,29,32
TB=4096
B8LANGS=yor_Latn,swh_Latn,ckb_Arab,hat_Latn,por_Latn,eng_Latn,zho_Hans,pes_Arab

EX="$PY -u $A/probing/extract_activations.py --model Qwen/Qwen3.5-9B --prompt-style bare \
    --token-budget $TB --require-gpu --save-layers $LAYERS"

step () { echo "=== $1 $(date -Is) ==="; }

step "1/6 TyDi train (<=3000/lang)"
$EX --split train --limit-per-lang 3000 --out $C/train    || { echo "STEP 1 FAILED"; exit 1; }

step "2/6 BELEBELE delete_strict (verification cache)"
$EX --source belebele --out $C/belebele                   || { echo "STEP 2 FAILED"; exit 1; }

step "3/6 udhr calibration set"
$EX --source udhr --out $C/udhr                           || { echo "STEP 3 FAILED"; exit 1; }

step "4/6 TyDi validation (monolingual ceiling)"
$EX --split validation --out $C/validation                || { echo "STEP 4 FAILED"; exit 1; }

step "5/6 BELEBELE delete_perlang, 8 languages item-matched"
$EX --source belebele --belebele data/built/belebele_answerability_delete_perlang.jsonl \
    --belebele-languages $B8LANGS --out $C/belebele8      || { echo "STEP 5 FAILED"; exit 1; }

step "6/6 translation (TyDi validation passage + MT question)"
$EX --source translation --out $C/translation             || { echo "STEP 6 FAILED"; exit 1; }

echo "=== ALL EXTRACTIONS DONE $(date -Is) ==="
